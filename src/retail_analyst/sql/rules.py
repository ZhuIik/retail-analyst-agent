"""Static rules for model-written SQL (D5, D11, А2).

Every check works on the sqlglot AST (BigQuery dialect), never on the SQL text: comments,
backticks and letter case cannot hide anything from an AST walk. A failed check raises
`GuardViolation` with a message written for the model ("what to change"), never containing
internals such as the scoped SQL or the salt.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

from retail_analyst.contracts import ErrorKind

DIALECT = "bigquery"
SCOPED_PREFIX = "__scoped_"  # reserved for our scoped CTEs (D10)

# Built-in BigQuery function namespaces that are safe to call. Any other qualified call
# (`ds.fn(x)`, `` `project.ds.fn`(x) ``) is a persistent UDF, which can read tables we do
# not scope — so it is rejected (D10: the scoped CTEs must be the only route to data).
SAFE_FUNCTION_NAMESPACES = frozenset({"NET", "HLL_COUNT", "KLL_QUANTILES"})

# Statement kinds that must never appear anywhere in the tree (D5). The root-type whitelist
# already rejects them at the top; this catches anything sqlglot might nest.
_WRITE_OR_SCRIPT_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Command, exp.Export, exp.Declare, exp.Set, exp.Into, exp.Transaction, exp.Commit, exp.Rollback,
    exp.Use, exp.Copy, exp.LoadData, exp.Grant, exp.Revoke,
)  # fmt: skip


class GuardViolation(Exception):
    """A rule failed. `message` is shown to the model."""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


def _violation(message: str) -> GuardViolation:
    return GuardViolation(ErrorKind.RULE_VIOLATION, message)


# --------------------------------------------------------------------------------------
# Parsing and statement shape
# --------------------------------------------------------------------------------------


def parse_single_select(sql: str) -> exp.Query:
    """Parse and return the single read-only query, or raise GuardViolation."""
    try:
        tokens = sqlglot.Dialect.get_or_raise(DIALECT).tokenize(sql)
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise GuardViolation(ErrorKind.INVALID_SQL, _parse_error_message(e)) from None
    except Exception as e:  # TokenError, recursion on absurd nesting, ...
        raise GuardViolation(ErrorKind.INVALID_SQL, f"SQL could not be parsed: {str(e)[:200]}") from None

    if not statements:
        raise GuardViolation(ErrorKind.INVALID_SQL, "The query is empty.")
    if len(statements) > 1:
        raise _violation(
            f"Exactly one SQL statement is allowed, got {len(statements)}. Send a single SELECT without ';'."
        )
    # sqlglot rewrites pipe syntax into CTEs with `SELECT *`; reject it up front with a clear message.
    if any(t.token_type == TokenType.PIPE_GT for t in tokens):
        raise _violation("Pipe syntax (|>) is not supported. Write a standard SELECT query.")

    root = statements[0]
    # `(SELECT ...)` with nothing around it: unwrap so WITH and LIMIT can be attached.
    while isinstance(root, exp.Subquery) and all(k == "this" or not v for k, v in root.args.items()):
        root = root.this
    if not isinstance(root, exp.Select | exp.SetOperation):
        raise _violation(
            f"Only read-only SELECT queries are allowed (got {_statement_name(root)}). "
            "Data cannot be modified, exported or scripted."
        )
    if node := next(root.find_all(*_WRITE_OR_SCRIPT_NODES), None):
        raise _violation(f"Only read-only SELECT queries are allowed ({_statement_name(node)} found).")
    return root


def _parse_error_message(e: sqlglot.errors.ParseError) -> str:
    if not e.errors:
        return f"SQL syntax error: {str(e)[:200]}"
    err = e.errors[0]
    return f"SQL syntax error at line {err.get('line')}, column {err.get('col')}: {err.get('description', '')[:200]}"


def _statement_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Command):
        return str(node.this).upper()
    if isinstance(node, _WRITE_OR_SCRIPT_NODES):
        return type(node).__name__.upper()
    return "a non-SELECT statement"


# --------------------------------------------------------------------------------------
# CTEs and table references
# --------------------------------------------------------------------------------------


def check_cte_names(root: exp.Expression, base_tables: set[str]) -> None:
    """A model CTE named like a base table or our scoped CTEs would bypass scoping (D10)."""
    for cte in root.find_all(exp.CTE):
        name = cte.alias_or_name.lower()
        if name in base_tables:
            raise _violation(
                f"CTE name '{cte.alias_or_name}' is reserved (it is a table name). Rename the CTE, e.g. '{name}_cte'."
            )
        if name.startswith(SCOPED_PREFIX):
            raise _violation(f"CTE names starting with '{SCOPED_PREFIX}' are reserved. Rename the CTE.")


def cte_is_visible(table: exp.Table, name: str) -> bool:
    """True if an unqualified `name` at this position refers to a CTE in scope.

    Resolution follows BigQuery: CTEs of every enclosing WITH are visible; inside a CTE body,
    only earlier CTEs of the same WITH are visible (all of them, itself included, under
    RECURSIVE). A name that is not a CTE in scope would be resolved by BigQuery as a real
    table, so anything not proven to be a CTE is treated as a table reference.
    """
    node: exp.Expression = table
    while (parent := node.parent) is not None:
        visible: list[exp.Expression] = []
        if isinstance(parent, exp.With) and isinstance(node, exp.CTE):
            ctes = parent.expressions
            visible = ctes if parent.args.get("recursive") else ctes[: ctes.index(node)]
        elif isinstance(with_ := parent.args.get("with_"), exp.With) and node is not with_:
            visible = with_.expressions
        if any(c.alias_or_name.lower() == name for c in visible):
            return True
        node = parent
    return False


def resolve_tables(root: exp.Expression, base_tables: set[str], dataset: str) -> list[tuple[exp.Table, str]]:
    """Return (node, base table name) for every base-table reference; reject anything else.

    Accepted spellings (any case, any backticks): `orders`, `thelook_ecommerce.orders`,
    `bigquery-public-data.thelook_ecommerce.orders`. CTE references are left alone.
    """
    project, ds = (p.lower() for p in dataset.split(".", 1))
    allowed_qualifiers = ([], [ds], [project, ds])
    refs: list[tuple[exp.Table, str]] = []
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise _violation(
                f"Table functions and paths like '{table.sql(DIALECT, comments=False)[:80]}' are not allowed. "
                f"Query the tables {_table_list(base_tables)} directly."
            )
        parts = [p.name for p in table.parts]
        shown = ".".join(parts)
        if "INFORMATION_SCHEMA" in shown.upper():
            raise _violation(
                "INFORMATION_SCHEMA is not available. The tables and columns you may use are listed in the schema."
            )
        *qualifiers, name = (p.lower() for p in parts)
        if qualifiers in allowed_qualifiers and name in base_tables:
            refs.append((table, name))
        elif not qualifiers and cte_is_visible(table, name):
            continue
        else:
            raise _violation(f"Table '{shown}' is not available. Use only the tables {_table_list(base_tables)}.")
    return refs


def _table_list(base_tables: set[str]) -> str:
    return ", ".join(sorted(base_tables))


# --------------------------------------------------------------------------------------
# Functions (А2)
# --------------------------------------------------------------------------------------


def function_name(func: exp.Func) -> str:
    """The qualified name BigQuery will see for this call, e.g. 'ML.PREDICT', 'NET.HOST'.

    Derived from what the generator emits (the regenerated SQL is what runs), so a namespace
    that sqlglot keeps as a parent Dot (`AI.GENERATE(...)`) or table qualifier
    (`FROM ML.PREDICT(...)`) is included. A leading `SAFE.` is dropped: it only changes
    error handling, not what the function does.
    """
    if isinstance(func, exp.Anonymous):
        parts = [func.name]
    else:
        head = func.sql(DIALECT, comments=False).split("(", 1)[0].strip()
        # Constructs like CASE or infix-rendered functions have no "NAME(" head.
        parts = [head if head.replace(".", "").replace("_", "").isalnum() else func.sql_name()]
    node: exp.Expression = func
    while isinstance(node.parent, exp.Dot) and node.parent.expression is node:
        node = node.parent
        parts.insert(0, node.this.sql(DIALECT, comments=False))
    if isinstance(func.parent, exp.Table) and func.parent.this is func:
        parts[:0] = [q.name for q in (func.parent.args.get("catalog"), func.parent.args.get("db")) if q]
    return ".".join(parts).replace("`", "").upper().removeprefix("SAFE.")


def check_functions(root: exp.Expression, forbidden_prefixes: list[str]) -> None:
    prefixes = tuple(p.upper() for p in forbidden_prefixes)
    for func in root.find_all(exp.Func):
        name = function_name(func)
        if name.startswith(prefixes):
            raise _violation(
                f"Function {name} is not allowed (AI/ML, external and similar functions are disabled). "
                "Compute the result with standard SQL."
            )
        namespace = name.rpartition(".")[0]
        if namespace and namespace not in SAFE_FUNCTION_NAMESPACES:
            raise _violation(f"User-defined or qualified function {name} is not allowed. Use built-in functions.")


# --------------------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------------------


def check_no_star(root: exp.Expression) -> None:
    """`SELECT *` scans every column and BigQuery bills by columns read (D11)."""
    for star in root.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):  # COUNT(*) reads no columns
            raise _violation(
                "SELECT * (including t.* and * EXCEPT) is not allowed. List the columns you need explicitly."
            )


def personal_columns(schema: dict) -> set[str]:
    """Columns marked `exposed: false` anywhere, unless the same name is exposed elsewhere."""
    hidden, exposed = set(), set()
    for table in schema.values():
        for col, spec in table["columns"].items():
            (exposed if (spec or {}).get("exposed", True) else hidden).add(col.lower())
    return hidden - exposed


def check_personal_columns(root: exp.Expression, schema: dict) -> None:
    """Clearer message only: the scoped tables do not contain these columns at all (D10)."""
    hidden = personal_columns(schema)
    for column in root.find_all(exp.Column):
        if column.name.lower() in hidden:
            raise _violation(f"Column {column.name} is not available (personal data).")
