"""SQL guard and table scoping (D5, D10, D11, А1, А2). No network.

Most assertions inspect the re-parsed scoped SQL (AST), not substrings, so they check what
BigQuery would actually execute.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from retail_analyst.config import get_settings, load_schema, load_users
from retail_analyst.contracts import AccessScope, ErrorKind, GuardResult, UserContext
from retail_analyst.sql.guard import guard_and_scope

SALT = "s3cret-Salt'with\\quote"
RAW_PREFIX = "bigquery-public-data.thelook_ecommerce."
SETTINGS = get_settings().model_copy(update={"id_salt": SALT})
SCHEMA = load_schema()
USERS = load_users()
LIMIT = SETTINGS.sql.default_limit


def make_user(column: str | None, values: list[str]) -> UserContext:
    return UserContext(user_id="t", display_name="T", role="test", access=AccessScope(column=column, values=values))


def guard(sql: str, user: str | UserContext = "anna") -> GuardResult:
    return guard_and_scope(sql, USERS[user] if isinstance(user, str) else user, SETTINGS, SCHEMA)


def ok(sql: str, user: str | UserContext = "anna") -> exp.Expression:
    """Assert the query passes; return the parsed scoped SQL after checking the core invariant."""
    result = guard(sql, user)
    assert result.ok, result.message
    assert result.error_kind is None
    statements = sqlglot.parse(result.scoped_sql, read="bigquery")
    assert len(statements) == 1
    tree = statements[0]
    assert_only_scoped_access(tree)
    return tree


def rejected(sql: str, kind: ErrorKind = ErrorKind.RULE_VIOLATION, user: str | UserContext = "anna") -> str:
    result = guard(sql, user)
    assert not result.ok, f"should be rejected: {sql}"
    assert result.error_kind == kind, result.message
    assert result.scoped_sql is None
    assert result.message
    return result.message


def assert_only_scoped_access(tree: exp.Expression) -> None:
    """Raw tables appear only inside our __scoped_* CTEs; no bare base-table names remain."""
    for table in tree.find_all(exp.Table):
        cte = table.find_ancestor(exp.CTE)
        if table.args.get("db"):
            assert cte is not None and cte.alias.startswith("__scoped_"), table.sql("bigquery")
            assert ".".join(p.name for p in table.parts).startswith(RAW_PREFIX)
        else:
            assert table.name not in SCHEMA, f"unscoped reference to {table.name}"


def ctes(tree: exp.Expression) -> dict[str, exp.Select]:
    return {c.alias: c.this for c in tree.args["with_"].expressions}


def scoped_refs(tree: exp.Expression) -> list[tuple[str, str]]:
    """(scoped CTE name, alias) for each reference to a scoped CTE outside our own CTEs."""
    out = []
    for table in tree.find_all(exp.Table):
        cte = table.find_ancestor(exp.CTE)
        if table.name.startswith("__scoped_") and not (cte and cte.alias.startswith("__scoped_")):
            out.append((table.name, table.alias))
    return out


def select_names(select: exp.Select) -> list[str]:
    return [e.alias_or_name for e in select.expressions]


# --------------------------------------------------------------------------------------
# Table references and aliases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        "orders",
        "`orders`",
        "ORDERS",
        "Orders",
        "thelook_ecommerce.orders",
        "`thelook_ecommerce.orders`",
        "`thelook_ecommerce`.`orders`",
        "THELOOK_ECOMMERCE.Orders",
        "bigquery-public-data.thelook_ecommerce.orders",
        "`bigquery-public-data.thelook_ecommerce.orders`",
        "`bigquery-public-data`.thelook_ecommerce.orders",
        "`bigquery-public-data.thelook_ecommerce`.orders",
        "`BIGQUERY-PUBLIC-DATA.THELOOK_ECOMMERCE.ORDERS`",
    ],
)
def test_all_spellings_of_a_base_table_are_scoped(ref):
    tree = ok(f"SELECT status FROM {ref} AS o")
    assert scoped_refs(tree) == [("__scoped_orders", "o")]
    assert guard(f"SELECT status FROM {ref} AS o").tables_used == ["orders"]


def test_alias_is_kept():
    tree = ok("SELECT o.status FROM orders o WHERE o.status = 'Complete'")
    assert scoped_refs(tree) == [("__scoped_orders", "o")]


def test_bare_table_gets_its_name_as_alias_so_qualified_columns_work():
    tree = ok("SELECT orders.status FROM thelook_ecommerce.orders")
    assert scoped_refs(tree) == [("__scoped_orders", "orders")]
    assert tree.find(exp.Column).table == "orders"


def test_joins_and_subqueries_are_all_scoped():
    sql = """
        SELECT p.brand, SUM(oi.sale_price) AS revenue
        FROM order_items AS oi
        JOIN products p ON p.id = oi.product_id
        WHERE oi.order_id IN (SELECT order_id FROM orders WHERE status = 'Complete')
          AND EXISTS (SELECT 1 FROM users u WHERE u.id = oi.user_id AND u.country = 'Brasil')
        GROUP BY p.brand
    """
    tree = ok(sql)
    assert sorted(scoped_refs(tree)) == [
        ("__scoped_order_items", "oi"),
        ("__scoped_orders", "orders"),
        ("__scoped_products", "p"),
        ("__scoped_users", "u"),
    ]
    assert guard(sql).tables_used == ["order_items", "orders", "products", "users"]


def test_parenthesised_query_is_accepted():
    tree = ok("(SELECT status FROM orders)")
    assert scoped_refs(tree) == [("__scoped_orders", "orders")]


def test_table_modifiers_stay_attached():
    tree = ok("SELECT M, F FROM orders PIVOT(COUNT(order_id) FOR gender IN ('M', 'F'))")
    pivot = tree.find(exp.Pivot)
    assert pivot is not None and pivot.parent.name == "__scoped_orders"


# --------------------------------------------------------------------------------------
# CTEs
# --------------------------------------------------------------------------------------


def test_model_cte_is_kept_after_scoped_ctes():
    tree = ok(
        "WITH rev AS (SELECT product_id, SUM(sale_price) AS revenue FROM order_items GROUP BY product_id) "
        "SELECT product_id, revenue FROM Rev ORDER BY revenue DESC"
    )
    names = list(ctes(tree))
    assert names == ["__scoped_products", "__scoped_order_items", "rev"]
    assert [t.name for t in ctes(tree)["rev"].find_all(exp.Table)] == ["__scoped_order_items"]


def test_nested_with_in_subquery():
    tree = ok("SELECT n FROM (WITH x AS (SELECT COUNT(*) AS n FROM orders) SELECT n FROM x)")
    assert ("__scoped_orders", "orders") in scoped_refs(tree)


def test_recursive_cte_is_allowed():
    ok("WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH orders AS (SELECT 1 AS status) SELECT status FROM orders",
        "WITH Orders AS (SELECT 1 AS status) SELECT status FROM orders",
        "WITH x AS (SELECT 1 AS a), users AS (SELECT 2 AS id) SELECT id FROM users",
        "SELECT id FROM (WITH products AS (SELECT 1 AS id) SELECT id FROM products)",
    ],
)
def test_cte_shadowing_a_base_table_is_rejected(sql):
    assert "reserved" in rejected(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH __scoped_orders AS (SELECT 1 AS a) SELECT a FROM __scoped_orders",
        "WITH __SCOPED_x AS (SELECT 1 AS a) SELECT a FROM __scoped_x",
    ],
)
def test_reserved_cte_prefix_is_rejected(sql):
    assert "reserved" in rejected(sql)


def test_direct_reference_to_scoped_cte_name_is_rejected():
    rejected("SELECT id FROM __scoped_products")


@pytest.mark.parametrize(
    "sql",
    [
        # CTE defined in a subquery is not visible outside: BigQuery would look up a real table.
        "SELECT a FROM (WITH events AS (SELECT 1 AS a) SELECT a FROM events) JOIN events USING (a)",
        # Non-recursive WITH: a CTE cannot see later CTEs.
        "WITH a AS (SELECT x FROM b), b AS (SELECT 1 AS x) SELECT x FROM a",
    ],
)
def test_cte_name_out_of_scope_is_treated_as_a_table(sql):
    assert "not available" in rejected(sql)


# --------------------------------------------------------------------------------------
# Statement shape: one read-only query
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["UNION ALL", "UNION DISTINCT", "INTERSECT DISTINCT", "EXCEPT DISTINCT"])
def test_set_operations_are_scoped_and_limited_once(op):
    tree = ok(f"SELECT status FROM orders {op} SELECT status FROM order_items")
    assert sorted(scoped_refs(tree)) == [("__scoped_order_items", "order_items"), ("__scoped_orders", "orders")]
    assert isinstance(tree, exp.SetOperation)
    assert tree.args["limit"].expression.this == str(LIMIT)
    assert len(list(tree.find_all(exp.Limit))) == 1


def test_line_comment_cannot_hide_a_statement():
    tree = ok("SELECT status FROM orders -- ; DROP TABLE orders")
    assert not list(tree.find_all(exp.Drop))
    assert "DROP" not in guard("SELECT status FROM orders -- ; DROP TABLE orders").scoped_sql


def test_block_comment_then_second_statement_is_rejected():
    assert "one SQL statement" in rejected("SELECT status FROM orders /* harmless */; DROP TABLE orders")


def test_comments_are_dropped_from_scoped_sql():
    # A comment containing "*/" must never be re-emitted where it could close early.
    sql = "SELECT status /* x */ FROM orders -- */ , (SELECT email FROM users) /*\n WHERE TRUE"
    result = guard(sql)
    assert result.ok
    assert "/*" not in result.scoped_sql and "--" not in result.scoped_sql
    assert "email" not in result.scoped_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT status FROM orders; SELECT status FROM orders",
        "SELECT 1; DROP TABLE orders",
        "CREATE TEMP FUNCTION f(x INT64) AS (x + 1); SELECT f(1)",
        "BEGIN SELECT 1; END",
    ],
)
def test_multiple_statements_are_rejected(sql):
    assert "one SQL statement" in rejected(sql)


def test_trailing_semicolon_is_fine():
    ok("SELECT status FROM orders;")


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (order_id) VALUES (1)",
        "INSERT INTO thelook_ecommerce.orders SELECT order_id FROM orders",
        "UPDATE orders SET status = 'x' WHERE TRUE",
        "DELETE FROM orders WHERE TRUE",
        "DROP TABLE orders",
        "TRUNCATE TABLE orders",
        "ALTER TABLE orders ADD COLUMN x INT64",
        "MERGE orders t USING products s ON t.order_id = s.id WHEN MATCHED THEN DELETE",
        "CREATE TABLE x AS SELECT status FROM orders",
        "CREATE TEMP FUNCTION f(x INT64) AS (x + 1)",
        "EXPORT DATA OPTIONS(uri='gs://b/*.csv', format='CSV') AS SELECT status FROM orders",
        "CALL mydataset.proc()",
        "DECLARE x INT64 DEFAULT 1",
        "SET x = 1",
        "EXECUTE IMMEDIATE 'DROP TABLE orders'",
        "ASSERT (SELECT COUNT(*) FROM orders) > 0",
    ],
)
def test_write_ddl_and_scripting_statements_are_rejected(sql):
    assert "read-only SELECT" in rejected(sql)


def test_pipe_syntax_is_rejected():
    assert "Pipe syntax" in rejected("FROM orders |> WHERE status = 'Complete' |> SELECT status")


@pytest.mark.parametrize("sql", ["SELEC status FROM orders", "SELECT status FROM orders WHERE (", "SELECT 'unclosed"])
def test_syntax_errors_are_invalid_sql(sql):
    rejected(sql, ErrorKind.INVALID_SQL)


@pytest.mark.parametrize("sql", ["", "   ", "-- only a comment"])
def test_empty_query_is_invalid_sql(sql):
    assert "empty" in rejected(sql, ErrorKind.INVALID_SQL)


# --------------------------------------------------------------------------------------
# SELECT * (cost, D11)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "SELECT o.* FROM orders o",
        "SELECT * EXCEPT (status) FROM orders",
        "SELECT o.* EXCEPT (status) FROM orders o",
        "SELECT * REPLACE (1 AS status) FROM orders",
        "SELECT status FROM (SELECT * FROM orders)",
        "WITH x AS (SELECT * FROM products) SELECT brand FROM x",
        "WITH x AS (SELECT id FROM products) SELECT * FROM x JOIN orders o ON TRUE",  # one source is a table
        "WITH x AS (SELECT * FROM products) SELECT * FROM x",  # inner star still over a table
    ],
)
def test_select_star_is_rejected(sql):
    assert "List the columns" in rejected(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH t AS (SELECT status, COUNT(*) AS n FROM orders GROUP BY status) SELECT * FROM t ORDER BY n DESC",
        "WITH a AS (SELECT brand FROM products), b AS (SELECT a.* FROM a) SELECT * FROM b",
        "SELECT * FROM (SELECT status FROM orders) AS s",
    ],
)
def test_select_star_over_own_cte_or_subquery_is_allowed(sql):
    # Columns are already chosen inside the CTE/subquery, so the star reads nothing extra (D11).
    ok(sql)


def test_count_star_is_allowed():
    ok("SELECT COUNT(*) AS n, COUNTIF(status = 'Complete') AS done FROM orders")


# --------------------------------------------------------------------------------------
# Forbidden functions (А2)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT predicted FROM ML.PREDICT(MODEL `p.d.m`, (SELECT status FROM orders))",
        "SELECT ML.PREDICT(MODEL m, (SELECT 1))",
        "SELECT ml.distance([1.0], [2.0])",
        "SELECT ML.GENERATE_EMBEDDING(MODEL m, (SELECT 1))",
        "SELECT result FROM ML.GENERATE_TEXT(MODEL m, (SELECT 1))",
        "SELECT AI.GENERATE('Summarise', connection_id => 'c')",
        "SELECT ai.classify(name, ['a', 'b']) FROM products",
        "SELECT AI.FORECAST(TABLE orders, data_col => 'num_of_item')",
        "SELECT x FROM AI.FORECAST(TABLE orders, data_col => 'num_of_item')",
        "SELECT EXTERNAL_QUERY('conn', 'SELECT 1')",
        "SELECT external_query('conn', 'SELECT 1')",
        "SELECT SAFE.EXTERNAL_QUERY('conn', 'SELECT 1')",
        "SELECT `ML.PREDICT`(1)",
        "SELECT VECTOR_SEARCH(TABLE orders, 'c', TABLE products)",
        "SELECT KEYS.NEW_KEYSET('AEAD_AES_GCM_256')",
        "SELECT AEAD.ENCRYPT(k, 'x', 'y') FROM products",
    ],
)
def test_forbidden_functions_are_rejected(sql):
    assert "not allowed" in rejected(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT mydataset.my_udf(status) FROM orders",
        "SELECT `other-project.ds.fn`(status) FROM orders",
        "SELECT x FROM thelook_ecommerce.some_tvf(1)",
    ],
)
def test_user_defined_functions_are_rejected(sql):
    # Persistent UDFs / table functions could read tables we do not scope.
    assert "not allowed" in rejected(sql)


def test_ordinary_and_namespaced_builtin_functions_are_allowed():
    ok(
        "SELECT DATE_TRUNC(DATE(created_at), MONTH) AS m, SAFE_DIVIDE(SUM(sale_price), COUNT(*)) AS aov, "
        "SAFE.DIVIDE(1, 2) AS d, NET.HOST('https://a.b/c') AS h, APPROX_COUNT_DISTINCT(user_id) AS buyers, "
        "CASE WHEN oi.status = 'Complete' THEN 1 ELSE 0 END AS c "
        "FROM order_items oi GROUP BY m, d, h, c"
    )


# --------------------------------------------------------------------------------------
# Other tables, datasets, projects
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        "thelook_ecommerce.INFORMATION_SCHEMA.TABLES",
        "`bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS`",
        "`bigquery-public-data`.thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS",
        "`region-us`.INFORMATION_SCHEMA.JOBS",
        "INFORMATION_SCHEMA.TABLES",
    ],
)
def test_information_schema_is_rejected(ref):
    assert "INFORMATION_SCHEMA" in rejected(f"SELECT table_name FROM {ref}")


@pytest.mark.parametrize(
    "ref",
    [
        "events",
        "inventory_items",
        "distribution_centers",
        "thelook_ecommerce.events",
        "`bigquery-public-data.thelook_ecommerce.inventory_items`",
        "`thelook_ecommerce.order*`",
        "thelook_ecommerce.`orders_backup`",
    ],
)
def test_other_tables_in_the_dataset_are_rejected(ref):
    assert "not available" in rejected(f"SELECT id FROM {ref}")


@pytest.mark.parametrize(
    "ref",
    [
        "`my-project.thelook_ecommerce.orders`",
        "`bigquery-public-data.other_dataset.orders`",
        "other_dataset.orders",
        "`bigquery-public-data.samples.shakespeare`",
        "a.b.c.d",
    ],
)
def test_other_projects_and_datasets_are_rejected(ref):
    rejected(f"SELECT status FROM {ref}")


def test_rejected_table_inside_subquery_or_cte():
    rejected("SELECT status FROM orders WHERE order_id IN (SELECT id FROM thelook_ecommerce.events)")
    rejected("WITH e AS (SELECT id FROM events) SELECT id FROM e")


# --------------------------------------------------------------------------------------
# PII (D10)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT email FROM users",
        "SELECT u.first_name, u.last_name FROM users u",
        "SELECT COUNT(*) FROM users u WHERE u.email LIKE '%@gmail.com'",
        "SELECT city FROM users GROUP BY city, street_address",
        "SELECT ST_ASTEXT(user_geom) FROM users",
        "SELECT AVG(latitude) FROM users",
    ],
)
def test_personal_columns_are_rejected(sql):
    assert "personal data" in rejected(sql)


def test_scoped_users_has_no_personal_columns_and_hashed_id():
    tree = ok("SELECT u.country, COUNT(DISTINCT u.id) AS n FROM users u GROUP BY 1", user="ceo")
    users = ctes(tree)["__scoped_users"]
    names = select_names(users)
    assert names == ["id", "age", "gender", "state", "city", "country", "traffic_source", "created_at"]
    assert_hashed(users, "id")


def assert_hashed(select: exp.Select, column: str) -> None:
    expr = next(e for e in select.expressions if e.alias_or_name == column)
    assert isinstance(expr, exp.Alias)
    pseudonym = expr.this
    assert "SHA256(CONCAT(" in pseudonym.sql("bigquery").upper()
    assert SALT in [lit.this for lit in pseudonym.find_all(exp.Literal) if lit.is_string]
    assert pseudonym.find(exp.Column).name == column


def test_customer_ids_are_hashed_in_every_table():
    tree = ok(
        "SELECT o.user_id, oi.user_id AS u2, u.id FROM orders o JOIN order_items oi USING (order_id) "
        "JOIN users u ON u.id = o.user_id"
    )
    scoped = ctes(tree)
    assert_hashed(scoped["__scoped_orders"], "user_id")
    assert_hashed(scoped["__scoped_order_items"], "user_id")
    assert_hashed(scoped["__scoped_users"], "id")
    # Non-customer ids stay raw.
    assert "order_id" in select_names(scoped["__scoped_orders"])
    assert "product_id" in select_names(scoped["__scoped_order_items"])


def test_cast_of_customer_id_still_yields_pseudonym():
    # А1: hashing happens at the source, so any expression over user_id sees the hash.
    tree = ok("SELECT CAST(user_id AS STRING) AS customer FROM order_items")
    assert scoped_refs(tree) == [("__scoped_order_items", "order_items")]
    assert_hashed(ctes(tree)["__scoped_order_items"], "user_id")


# --------------------------------------------------------------------------------------
# Access filter (D10) and foreign-literal refusal
# --------------------------------------------------------------------------------------


def products_filter(tree: exp.Expression) -> exp.Expression | None:
    where = ctes(tree)["__scoped_products"].args.get("where")
    return where.this if where else None


def test_brand_manager_scope_filters_products_and_dependants():
    tree = ok("SELECT COUNT(*) FROM users")
    scoped = ctes(tree)
    assert list(scoped) == ["__scoped_products", "__scoped_users"]  # only what is needed + dependency
    flt = products_filter(tree)
    assert isinstance(flt, exp.In) and flt.this.name == "brand"
    assert [v.this for v in flt.expressions] == ["Calvin Klein", "Diesel", "Carhartt"]
    users_where = scoped["__scoped_users"].args["where"].this
    assert isinstance(users_where, exp.In) and users_where.this.name == "id"
    inner_tables = [".".join(p.name for p in t.parts) for t in users_where.find_all(exp.Table)]
    assert sorted(inner_tables) == ["__scoped_products", RAW_PREFIX + "order_items"]


def test_orders_and_order_items_are_filtered_through_scoped_products():
    scoped = ctes(ok("SELECT o.status, oi.sale_price FROM orders o JOIN order_items oi USING (order_id)"))
    assert list(scoped) == ["__scoped_products", "__scoped_order_items", "__scoped_orders"]
    items_where = scoped["__scoped_order_items"].args["where"].this
    assert items_where.this.name == "product_id"
    assert [t.name for t in items_where.find_all(exp.Table)] == ["__scoped_products"]
    orders_where = scoped["__scoped_orders"].args["where"].this
    assert orders_where.this.name == "order_id"
    assert sorted(t.name for t in orders_where.find_all(exp.Table)) == ["__scoped_products", "order_items"]


def test_category_manager_scope():
    tree = ok("SELECT category, SUM(retail_price) FROM products WHERE category = 'jeans' GROUP BY 1", user="mark")
    flt = products_filter(tree)
    assert flt.this.name == "category"
    assert [v.this for v in flt.expressions] == ["Jeans", "Outerwear & Coats"]
    msg = rejected("SELECT name FROM products WHERE category = 'Tops & Tees'", ErrorKind.FOREIGN_BRAND, "mark")
    assert msg.startswith("You don't have access to category 'Tops & Tees'")
    # Brand is not Mark's access column: any brand literal is fine (filtered by category anyway).
    ok("SELECT name FROM products WHERE brand = 'Nike'", user="mark")


def test_ceo_is_unrestricted_but_still_hashed_and_pii_free():
    tree = ok(
        "SELECT p.brand, COUNT(DISTINCT o.user_id) FROM orders o JOIN order_items oi USING (order_id) "
        "JOIN products p ON p.id = oi.product_id JOIN users u ON u.id = o.user_id "
        "WHERE p.brand = 'Nike' GROUP BY 1",
        user="ceo",
    )
    scoped = ctes(tree)
    assert list(scoped) == ["__scoped_products", "__scoped_order_items", "__scoped_orders", "__scoped_users"]
    assert all(body.args.get("where") is None for body in scoped.values())
    assert_hashed(scoped["__scoped_users"], "id")
    assert "email" not in select_names(scoped["__scoped_users"])


def test_ceo_without_products_needs_no_products_cte():
    assert list(ctes(ok("SELECT status FROM orders", user="ceo"))) == ["__scoped_orders"]


@pytest.mark.parametrize(
    "sql, value",
    [
        ("SELECT name FROM products WHERE brand = 'Nike'", "Nike"),
        ("SELECT name FROM products p WHERE 'Nike' = p.brand", "Nike"),
        ("SELECT name FROM products WHERE brand IN ('Diesel', 'Levi\\'s')", "Levi's"),
        ("SELECT name FROM products WHERE (brand = 'Nike')", "Nike"),
        ("SELECT name FROM products WHERE LOWER(brand) = 'nike'", "nike"),
        ("SELECT name FROM products WHERE brand LIKE 'Nik%'", "Nik%"),
        ("SELECT COUNTIF(p.brand = 'Nike') FROM products p", "Nike"),
        ("WITH x AS (SELECT id FROM products WHERE brand = 'Nike') SELECT id FROM x", "Nike"),
    ],
)
def test_foreign_brand_literal_is_refused(sql, value):
    msg = rejected(sql, ErrorKind.FOREIGN_BRAND)
    assert msg.startswith(f"You don't have access to brand '{value}'")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM products WHERE brand = 'calvin klein'",
        "SELECT name FROM products WHERE UPPER(brand) = 'DIESEL'",
        "SELECT name FROM products WHERE brand IN ('Diesel', 'CARHARTT')",
        "SELECT name FROM products WHERE brand LIKE '%klein'",
        "SELECT name FROM products WHERE brand LIKE 'Carh_rtt'",
        "SELECT name FROM products WHERE brand != 'Nike'",
        "SELECT name FROM products WHERE NOT brand = 'Nike'",
        "SELECT name FROM products WHERE NOT (brand = 'Nike')",
        "SELECT name FROM products WHERE brand NOT IN ('Nike')",
        "SELECT name FROM products WHERE brand NOT LIKE 'Nik%'",
        "SELECT name FROM products WHERE name = 'Nike Air'",
    ],
)
def test_own_brand_literals_and_exclusions_are_ok(sql):
    ok(sql)


def test_config_values_cannot_inject_sql():
    evil = "x') OR TRUE OR brand IN ('y"
    tree = ok("SELECT name FROM products", user=make_user("brand", [evil, "O'Neill"]))
    flt = products_filter(tree)
    assert isinstance(flt, exp.In)
    assert [v.this for v in flt.expressions] == [evil, "O'Neill"]


def test_empty_access_list_sees_nothing():
    tree = ok("SELECT name FROM products", user=make_user("brand", []))
    assert isinstance(products_filter(tree), exp.Boolean) and products_filter(tree).this is False


# --------------------------------------------------------------------------------------
# LIMIT and output
# --------------------------------------------------------------------------------------


def test_default_limit_is_added():
    tree = ok("SELECT status FROM orders")
    assert tree.args["limit"].expression.this == str(LIMIT)


def test_existing_limit_is_kept():
    tree = ok("SELECT status FROM orders ORDER BY created_at DESC LIMIT 5")
    assert tree.args["limit"].expression.this == "5"
    assert len(list(tree.find_all(exp.Limit))) == 1


def test_inner_limit_does_not_count_as_outer_limit():
    tree = ok("SELECT status FROM (SELECT status FROM orders LIMIT 5)")
    assert tree.args["limit"].expression.this == str(LIMIT)


def test_tables_used_is_sorted_and_unique():
    result = guard("SELECT a.status FROM orders a JOIN orders b USING (order_id) JOIN products p ON TRUE")
    assert result.tables_used == ["orders", "products"]


def test_query_without_base_tables_passes():
    result = guard("SELECT 1 AS one")
    assert result.ok and result.tables_used == []
    assert "__scoped_" not in result.scoped_sql


FAILING = [
    "SELECT * FROM orders",
    "SELECT email FROM users",
    "DROP TABLE orders",
    "SELECT id FROM events",
    "SELECT AI.GENERATE('x')",
    "SELECT name FROM products WHERE brand = 'Nike'",
    "WITH orders AS (SELECT 1 AS a) SELECT a FROM orders",
    "SELECT 1; SELECT 2",
    "SELECT FROM WHERE",
    "SELECT status FROM orders FOR",
]


@pytest.mark.parametrize("sql", FAILING)
def test_messages_never_contain_salt_or_scoped_sql(sql):
    # Messages may quote the model's own SQL, never what the guard adds.
    result = guard(sql)
    assert not result.ok
    for secret in (SALT, "s3cret", "__scoped_", "SHA256"):
        assert secret not in result.message


def test_salt_is_escaped_in_scoped_sql():
    # The salt contains a quote and a backslash; it must round-trip as one string literal.
    result = guard("SELECT user_id FROM orders")
    tree = sqlglot.parse_one(result.scoped_sql, read="bigquery")
    assert SALT in [lit.this for lit in tree.find_all(exp.Literal) if lit.is_string]


OK_QUERIES = [
    "SELECT status, COUNT(*) AS n FROM orders GROUP BY status",
    "SELECT p.category, SUM(oi.sale_price - p.cost) AS margin FROM order_items oi "
    "JOIN products p ON p.id = oi.product_id GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    "WITH c AS (SELECT user_id, MIN(created_at) AS first FROM orders GROUP BY 1) "
    "SELECT u.traffic_source, COUNT(*) FROM c JOIN users u ON u.id = c.user_id GROUP BY 1",
    "SELECT status FROM orders UNION ALL SELECT status FROM order_items",
    "SELECT (SELECT COUNT(*) FROM users) AS customers, (SELECT COUNT(*) FROM orders) AS orders_count",
]


@pytest.mark.parametrize("user", ["anna", "mark", "ceo"])
@pytest.mark.parametrize("sql", OK_QUERIES)
def test_scoped_sql_reparses_and_only_touches_scoped_data(sql, user):
    ok(sql, user)  # `ok` re-parses and asserts the invariant
