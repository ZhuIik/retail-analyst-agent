"""Table scoping rewrite — the actual protection for per-user access and PII (D10, А1).

Every base-table reference in the model's query is replaced by a CTE `__scoped_<table>` that
contains only (a) rows in the user's access scope, (b) exposed columns, (c) salted hashes
instead of customer ids. The model's SQL can therefore only ever see scoped data, whatever
it computes on top.

All CTEs are built as sqlglot expressions, so config values (salt, access values, column
names) are emitted as escaped literals / identifiers and cannot inject SQL.
"""

from __future__ import annotations

from sqlglot import exp

from retail_analyst.contracts import AccessScope
from retail_analyst.sql.rules import DIALECT, SCOPED_PREFIX

# Dependency order: every scoped table is filtered through the user's products.
_ORDER = ("products", "order_items", "orders", "users")


def scoped_name(table: str) -> str:
    return f"{SCOPED_PREFIX}{table}"


def build_scoped_ctes(tables: set[str], access: AccessScope, dataset: str, salt: str, schema: dict) -> list[exp.CTE]:
    """CTEs for `tables` plus their dependencies, in dependency order."""
    needed = set(tables)
    if not access.is_unrestricted and needed & {"order_items", "orders", "users"}:
        needed.add("products")
    project, ds = dataset.split(".", 1)

    def raw(table: str) -> exp.Table:
        return exp.Table(
            this=exp.to_identifier(table, quoted=True),
            db=exp.to_identifier(ds, quoted=True),
            catalog=exp.to_identifier(project, quoted=True),
        )

    def scoped_product_ids() -> exp.Select:
        return exp.select("id").from_(scoped_name("products"))

    def items_of_scoped_products(column: str) -> exp.Select:
        in_scope = exp.column("product_id").isin(query=scoped_product_ids())
        return exp.select(column).from_(raw("order_items")).where(in_scope)

    if access.is_unrestricted:
        filters: dict[str, exp.Expression | None] = dict.fromkeys(_ORDER)
    else:
        filters = {
            "products": (exp.column(access.column).isin(*access.values) if access.values else exp.false()),
            "order_items": exp.column("product_id").isin(query=scoped_product_ids()),
            "orders": exp.column("order_id").isin(query=items_of_scoped_products("order_id")),
            "users": exp.column("id").isin(query=items_of_scoped_products("user_id")),
        }

    ctes = []
    for table in _ORDER:
        if table not in needed:
            continue
        body = exp.select(*_exposed_columns(schema[table], salt)).from_(raw(table))
        if filters[table] is not None:
            body = body.where(filters[table])
        ctes.append(exp.CTE(this=body, alias=exp.TableAlias(this=exp.to_identifier(scoped_name(table)))))
    return ctes


def _exposed_columns(table_spec: dict, salt: str) -> list[exp.Expression]:
    """Exposed columns from schema.yaml; `hashed` ones become salted pseudonyms (А1)."""
    columns: list[exp.Expression] = []
    for name, spec in table_spec["columns"].items():
        spec = spec or {}
        if not spec.get("exposed", True):
            continue
        col = exp.column(name)
        if spec.get("hashed"):
            hashed = exp.func(
                "FARM_FINGERPRINT",
                exp.func("CONCAT", exp.Literal.string(salt), exp.cast(col, "STRING"), dialect=DIALECT),
                dialect=DIALECT,
            )
            columns.append(exp.alias_(hashed, name))
        else:
            columns.append(col)
    return columns


def rewrite_tables(refs: list[tuple[exp.Table, str]]) -> None:
    """Point each base-table reference at its scoped CTE, keeping (or adding) the alias.

    The bare table name becomes the alias when there was none, so `orders.status` still works.
    Other table modifiers (PIVOT, hints, ...) stay attached to the node.
    """
    for table, name in refs:
        if not table.alias:
            table.set("alias", exp.TableAlias(this=exp.to_identifier(name)))
        table.set("this", exp.to_identifier(scoped_name(name)))
        table.set("db", None)
        table.set("catalog", None)
        table.meta.pop("quoted_table", None)  # sqlglot would otherwise backtick the CTE name


def prepend_ctes(root: exp.Query, ctes: list[exp.CTE]) -> None:
    """Put our CTEs before the model's own WITH entries (they may reference ours)."""
    if not ctes:
        return
    existing = root.args.get("with_")
    if existing is None:
        root.set("with_", exp.With(expressions=ctes))
    else:
        existing.set("expressions", ctes + list(existing.expressions))


def ensure_limit(root: exp.Query, default_limit: int) -> None:
    if root.args.get("limit") is None:
        root.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))
