"""SQL guard and table scoping (D5, D10, D11, А1, А2).

    guard_and_scope(sql, user, settings, schema) -> GuardResult

Pipeline: parse (sqlglot, BigQuery) -> static rules -> demographics rule (k-anonymity) ->
foreign-literal check -> scoping rewrite -> default LIMIT -> self-check of the generated SQL.
The model proposes, the code
guarantees (D1): what goes to BigQuery is SQL *regenerated* from the checked AST (comments
dropped), never the model's original text.

Messages go to the model, so they never contain the scoped SQL or the salt.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from retail_analyst.config import Settings
from retail_analyst.contracts import ErrorKind, GuardResult, UserContext
from retail_analyst.sql import rules
from retail_analyst.sql.access import find_foreign_literal, foreign_literal_message
from retail_analyst.sql.demographics import check_demographics
from retail_analyst.sql.rules import DIALECT, SCOPED_PREFIX, GuardViolation
from retail_analyst.sql.scoping import build_scoped_ctes, ensure_limit, prepend_ctes, rewrite_tables

_SCOPING_FAILED = "The query could not be safely scoped. Simplify it (plain SELECT with joins, subqueries or CTEs)."


def guard_and_scope(sql: str, user: UserContext, settings: Settings, schema: dict) -> GuardResult:
    base_tables = set(schema)
    try:
        root = rules.parse_single_select(sql)
        rules.check_cte_names(root, base_tables)
        rules.check_functions(root, settings.sql.forbidden_function_prefixes)
        refs = rules.resolve_tables(root, base_tables, settings.dataset)
        rules.check_no_star(root)
        rules.check_personal_columns(root, schema)
        # Checked on the model's query, before scoping adds our own CTEs. The result rows are
        # filtered by group size in run_sql; here we only make sure the count is there.
        group_size_column = check_demographics(root, refs, schema, settings.sql.min_group_size)
    except GuardViolation as v:
        return GuardResult(ok=False, error_kind=v.kind, message=v.message)

    if (value := find_foreign_literal(root, user.access)) is not None:
        return GuardResult(
            ok=False, error_kind=ErrorKind.FOREIGN_BRAND, message=foreign_literal_message(user.access, value)
        )

    tables_used = sorted({name for _, name in refs})
    try:
        ctes = build_scoped_ctes(set(tables_used), user.access, settings.dataset, settings.id_salt, schema)
        rewrite_tables(refs)
        prepend_ctes(root, ctes)
        ensure_limit(root, settings.sql.default_limit)
        scoped_sql = root.sql(dialect=DIALECT, comments=False)
        _verify_scoped(scoped_sql, base_tables)
    except Exception:  # never echo internals: the exception text could contain the salt
        return GuardResult(ok=False, error_kind=ErrorKind.RULE_VIOLATION, message=_SCOPING_FAILED)

    return GuardResult(ok=True, scoped_sql=scoped_sql, tables_used=tables_used, group_size_column=group_size_column)


def _verify_scoped(scoped_sql: str, base_tables: set[str]) -> None:
    """Defence in depth: re-parse the output and check the invariant the rewrite promises.

    Raw (qualified) tables may appear only inside our `__scoped_*` CTEs, and no unqualified
    base-table name may remain anywhere (BigQuery could resolve it against a default dataset).
    """
    statements = [s for s in sqlglot.parse(scoped_sql, read=DIALECT) if s is not None]
    if len(statements) != 1:
        raise AssertionError("scoped SQL must be one statement")
    for table in statements[0].find_all(exp.Table):
        cte = table.find_ancestor(exp.CTE)
        inside_ours = cte is not None and cte.alias_or_name.startswith(SCOPED_PREFIX)
        if table.args.get("db") or table.args.get("catalog"):
            if not inside_ours or table.name not in base_tables:
                raise AssertionError("raw table outside a scoped CTE")
        elif table.name.lower() in base_tables:
            raise AssertionError("unscoped base table reference")
