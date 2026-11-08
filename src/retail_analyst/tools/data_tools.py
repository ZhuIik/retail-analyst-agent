"""Data tools: run_sql and describe_data (D11 pipeline, D13 empty-result diagnosis).

run_sql is where the two protection layers meet: the SQL guard (scoping, PII, read-only) and the
BigQuery client (dry-run cost check, byte cap, retries). describe_data reuses the same path, so
value lookups are scoped to the user too.
"""

from __future__ import annotations

from typing import Literal

import sqlglot.expressions as exp
from pydantic import BaseModel, Field

from retail_analyst.config import load_schema
from retail_analyst.contracts import AgentError, ErrorKind, TableData, ToolResult, ToolRisk
from retail_analyst.sql.guard import guard_and_scope
from retail_analyst.tools.registry import ToolContext, ToolOutcome, ToolRegistry, ToolSpec


class RunSqlArgs(BaseModel):
    sql: str = Field(description="One BigQuery Standard SQL SELECT over orders, order_items, products, users.")
    purpose: str = Field(description="Short description of what this query answers, e.g. 'monthly revenue 2026'.")


DescribeColumn = Literal["brand", "category", "department", "status", "traffic_source", "country", "state"]
_DESCRIBE_TABLE = {
    "brand": "products",
    "category": "products",
    "department": "products",
    "status": "order_items",
    "traffic_source": "users",
    "country": "users",
    "state": "users",
}


class DescribeDataArgs(BaseModel):
    column: DescribeColumn = Field(description="Which column's values to list.")
    search: str | None = Field(default=None, description="Optional case-insensitive substring, e.g. 'levi'.")


def _error(kind: ErrorKind, message: str) -> ToolOutcome:
    return ToolOutcome(ToolResult(status="error", error_kind=kind, message=message))


def _execute(sql: str, purpose: str, ctx: ToolContext) -> ToolOutcome:
    """Guard -> dry-run -> execute -> store. Fatal errors (quota, auth, region) propagate."""
    settings, trace = ctx.settings, ctx.trace
    guard = guard_and_scope(sql, ctx.user, settings, load_schema())
    trace.event("sql_guard", ok=guard.ok, error_kind=guard.error_kind, sql=sql, purpose=purpose)
    if not guard.ok:
        return _error(guard.error_kind or ErrorKind.RULE_VIOLATION, guard.message)

    bq = ctx.services["bq"]
    try:
        scanned = bq.dry_run(guard.scoped_sql)
        if scanned > settings.sql.dry_run_max_bytes:
            return _error(
                ErrorKind.TOO_EXPENSIVE,
                f"Query would scan {scanned / 1e6:.0f} MB (limit {settings.sql.dry_run_max_bytes / 1e6:.0f} MB). "
                "Select fewer columns, filter dates or aggregate.",
            )
        labels = {"user": ctx.user.user_id, "session": ctx.session_id, "trace": trace.trace_id or "none"}
        result = bq.execute(guard.scoped_sql, labels=labels)
    except AgentError as e:
        if e.kind.is_fatal:
            raise  # the graph traces it and ends the turn (D12)
        trace.event("error", kind=e.kind, message=e.user_message, detail=e.detail)
        return _error(e.kind, e.user_message)

    query_id = f"q{next(ctx.services['query_counter'])}"
    meta = result.meta
    trace.event("bq_job", query_id=query_id, job_id=meta.job_id, bytes=meta.bytes_processed,
                rows=meta.rows_total, ms=meta.elapsed_ms, attempts=meta.attempts)
    stored = {"sql": sql, "purpose": purpose, "columns": result.data.columns,
              "rows": result.data.rows, "rows_total": meta.rows_total}
    update = {"queries": {query_id: stored}}

    if meta.rows_total == 0:
        return ToolOutcome(
            ToolResult(status="empty", error_kind=ErrorKind.EMPTY_RESULT, query_id=query_id, message=_empty_hint(ctx)),
            update,
        )
    preview = TableData(columns=result.data.columns, rows=result.data.rows[: settings.sql.preview_rows])
    return ToolOutcome(
        ToolResult(status="ok", query_id=query_id, data=preview, rows_total=meta.rows_total, meta=meta), update
    )


def _empty_hint(ctx: ToolContext) -> str:
    """D13: zero rows is a diagnosis task, not an answer."""
    hint = "The query returned 0 rows. Before reporting 'none', check the filters: exact values (use describe_data)"
    prompt_ctx = ctx.services.get("prompt_ctx")
    if prompt_ctx and prompt_ctx.data_from:
        hint += f", the date range (data covers {prompt_ctx.data_from} to {prompt_ctx.data_to})"
    if not ctx.user.access.is_unrestricted:
        hint += f", and the user's scope ({ctx.user.access.column}: {', '.join(ctx.user.access.values)})"
    return hint + "."


def run_sql(args: RunSqlArgs, ctx: ToolContext) -> ToolOutcome:
    return _execute(args.sql, args.purpose, ctx)


def describe_data(args: DescribeDataArgs, ctx: ToolContext) -> ToolOutcome:
    col, table = args.column, _DESCRIBE_TABLE[args.column]
    where = f"{col} IS NOT NULL"
    if args.search:
        # STRPOS rather than LIKE: a substring search is not a filter on a foreign brand (D10).
        needle = exp.Literal.string(args.search.lower()).sql("bigquery")
        where += f" AND STRPOS(LOWER({col}), {needle}) > 0"
    sql = f"SELECT {col} AS value, COUNT(*) AS n FROM {table} WHERE {where} GROUP BY value ORDER BY n DESC LIMIT 30"
    return _execute(sql, f"describe {col}", ctx)


def register_data_tools(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(
        name="run_sql",
        description="Run one read-only BigQuery SQL query over the user's data and get the first rows of the result.",
        args_model=RunSqlArgs,
        handler=run_sql,
        risk=ToolRisk.READ,
    ))
    registry.register(ToolSpec(
        name="describe_data",
        description="List the most common exact values of a column (optionally matching a substring), "
        "e.g. to find the exact spelling of a brand before filtering.",
        args_model=DescribeDataArgs,
        handler=describe_data,
        risk=ToolRisk.READ,
    ))
