"""Shared data contracts.

Every module talks to the others only through the types in this file. Keeping them in one
place is what lets the SQL guard, the BigQuery layer, the report store and the agent graph be
built and tested independently.

Design rule (D1): the model proposes, the code guarantees. Nothing in these types is trusted
because the model said so: user identity comes from login, access scope from config, and
every write is filtered by owner in code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Identity and access (D10, П2)
# --------------------------------------------------------------------------------------

AccessColumn = Literal["brand", "category", "department"]


class AccessScope(BaseModel):
    """Which products a user may see. `column=None` means all products (e.g. the CEO).

    The column is a config value, not code: if the team says access is by category,
    only users.yaml changes.
    """

    column: AccessColumn | None = None
    values: list[str] = Field(default_factory=list)

    @property
    def is_unrestricted(self) -> bool:
        return self.column is None


class UserContext(BaseModel):
    """Who is asking. Built by code at login (`--user` in the prototype), never by the model."""

    user_id: str
    display_name: str
    role: str
    access: AccessScope


# --------------------------------------------------------------------------------------
# Errors (D12): who handles which error
# --------------------------------------------------------------------------------------


class ErrorKind(StrEnum):
    # The model can fix these by rewriting its query (up to N attempts).
    INVALID_SQL = "invalid_sql"  # syntax error, unknown column, type error
    RULE_VIOLATION = "rule_violation"  # not SELECT, forbidden table/function, SELECT *
    FOREIGN_BRAND = "foreign_brand"  # literal for a brand outside the user's scope
    TOO_EXPENSIVE = "too_expensive"  # dry-run above threshold / bytesBilledLimitExceeded
    EMPTY_RESULT = "empty_result"  # 0 rows: needs diagnosis, not "the answer is 0"
    # Code handles these; the model never spends a call on them.
    TRANSIENT = "transient"  # network, 5xx (except bytesBilledLimitExceeded), timeouts
    # Stop immediately with an honest message.
    QUOTA_EXHAUSTED = "quota_exhausted"  # daily LLM / BigQuery quota
    AUTH = "auth"  # missing or invalid credentials
    LOCATION_BLOCKED = "location_blocked"  # Google API not available from this region
    UNKNOWN = "unknown"

    @property
    def fixable_by_model(self) -> bool:
        return self in {
            ErrorKind.INVALID_SQL,
            ErrorKind.RULE_VIOLATION,
            ErrorKind.FOREIGN_BRAND,
            ErrorKind.TOO_EXPENSIVE,
            ErrorKind.EMPTY_RESULT,
        }

    @property
    def is_fatal(self) -> bool:
        return self in {ErrorKind.QUOTA_EXHAUSTED, ErrorKind.AUTH, ErrorKind.LOCATION_BLOCKED}


class AgentError(Exception):
    """Raised by infrastructure layers (BigQuery, LLM). Carries a classified kind.

    `user_message` is safe to show to the user; `detail` may contain internals and goes
    to the trace only. It must never contain the rewritten (scoped) SQL or the salt.
    """

    def __init__(self, kind: ErrorKind, user_message: str, detail: str = "") -> None:
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message
        self.detail = detail


# --------------------------------------------------------------------------------------
# SQL guard (D10, D11) — implemented in retail_analyst.sql
# --------------------------------------------------------------------------------------


class GuardResult(BaseModel):
    """Outcome of validating and scoping one model-written query.

    ok=True  -> `scoped_sql` is safe to send to BigQuery (single SELECT, allowed tables only,
                no forbidden functions, every base table replaced by the user's scoped CTE,
                customer ids replaced by salted hashes).
    ok=False -> `error_kind` + `message` explain to the model what to fix. `scoped_sql` is None.

    `group_size_column` is set when the query uses customer demographics: the executor must drop
    result rows where this column is NULL or below `sql.min_group_size` (k-anonymity).
    """

    ok: bool
    scoped_sql: str | None = None
    error_kind: ErrorKind | None = None
    message: str = ""
    tables_used: list[str] = Field(default_factory=list)
    group_size_column: str | None = None


# --------------------------------------------------------------------------------------
# BigQuery execution (D11, D17) — implemented in retail_analyst.bq
# --------------------------------------------------------------------------------------


class QueryMeta(BaseModel):
    job_id: str | None = None
    bytes_processed: int = 0
    bytes_billed: int = 0
    rows_total: int = 0
    elapsed_ms: int = 0
    attempts: int = 1  # >1 means transient retries happened


class TableData(BaseModel):
    """A query result as plain JSON-serialisable rows (safe for the checkpointer)."""

    columns: list[str]
    rows: list[list[Any]]


class ExecResult(BaseModel):
    data: TableData
    meta: QueryMeta


# --------------------------------------------------------------------------------------
# Tools (П1: registry with risk classes)
# --------------------------------------------------------------------------------------


class ToolRisk(StrEnum):
    READ = "read"  # runs immediately
    INTERNAL_WRITE = "internal_write"  # changes our data -> confirmation node (interrupt)
    EXTERNAL_WRITE = "external_write"  # sends data out -> confirmation + preview + audit (design only)
    EXTERNAL_READ = "external_read"  # brings untrusted data in -> wrapped as <data> (design only)


class ToolResult(BaseModel):
    """What a tool returns. `render_for_model()` is the only text the model ever sees."""

    status: Literal["ok", "error", "empty"]
    message: str = ""
    error_kind: ErrorKind | None = None
    query_id: str | None = None  # e.g. "q3"; facts in submit_answer cite these ids
    data: TableData | None = None  # preview only (already truncated)
    rows_total: int | None = None
    meta: QueryMeta | None = None

    def render_for_model(self) -> str:
        """Results are data, not instructions: wrap them in <data> tags (H10)."""
        head = f"status={self.status}"
        if self.query_id:
            head += f" query_id={self.query_id}"
        if self.error_kind:
            head += f" error={self.error_kind.value}"
        parts = [head]
        if self.message:
            parts.append(self.message)
        if self.data is not None:
            shown = len(self.data.rows)
            total = self.rows_total if self.rows_total is not None else shown
            parts.append(f"rows shown: {shown} of {total}")
            if shown < total:
                # Seen live: without this the model paged through results with extra queries (and calls).
                parts.append(
                    "Only the first rows are shown. Do not page through the rest: aggregate, rank or filter "
                    "in SQL so that what you need fits."
                )
            lines = [" | ".join(self.data.columns)]
            lines += [" | ".join("" if v is None else str(v) for v in row) for row in self.data.rows]
            parts.append("<data>\n" + "\n".join(lines) + "\n</data>")
        return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Final answer (D15): the model must answer through this tool (function-calling mode ANY)
# --------------------------------------------------------------------------------------

AnswerKind = Literal["answer", "clarification", "refusal", "info"]


class Fact(BaseModel):
    text: str = Field(description="One factual statement, with numbers copied exactly from query results.")
    query_ids: list[str] = Field(
        default_factory=list, description="Ids of the queries (e.g. ['q1']) whose results support this fact."
    )


class AnswerSubmission(BaseModel):
    """Arguments of the `submit_answer` tool. The final text is assembled by code, not the model."""

    kind: AnswerKind = Field(
        description=(
            "answer: analysis based on data; clarification: you need one missing detail to compute; "
            "refusal: off-topic, unsafe or about a specific person; info: questions about the data/schema "
            "or follow-up discussion without new numbers."
        )
    )
    message: str = Field(description="Main text for the user (summary or the clarifying question / refusal).")
    facts: list[Fact] = Field(default_factory=list)
    hypotheses: list[str] = Field(
        default_factory=list, description="Possible explanations ('why'). Never present them as facts."
    )
    recommendations: list[str] = Field(default_factory=list, description="Concrete action items.")
    assumptions: list[str] = Field(
        default_factory=list, description="Definitions and interpretations used (metric definitions, periods)."
    )
    offer_save: bool = Field(default=False, description="True for report-like answers worth saving.")
    suggested_title: str | None = Field(default=None, description="Short title if offer_save is true.")


# --------------------------------------------------------------------------------------
# Reports (D16) — implemented in retail_analyst.reports
# --------------------------------------------------------------------------------------


class ReportSummary(BaseModel):
    report_id: int
    title: str
    created_at: str  # ISO 8601
    session_id: str


class Report(ReportSummary):
    owner_id: str
    body: str
    question: str
    sql: list[str] = Field(default_factory=list)
    status: Literal["active", "deleted"] = "active"


class PendingDeletion(BaseModel):
    """Set in graph state by `delete_reports`; the confirm node pauses on it (interrupt).

    The ids are a snapshot: confirmation applies to exactly these reports, nothing else.
    `required_answer` is "yes" for 1-3 reports, or the exact count for bulk deletions.
    """

    report_ids: list[int]
    preview: list[ReportSummary]
    required_answer: str
    tool_call_id: str  # the delete_reports call; the confirm node answers it with the outcome


# --------------------------------------------------------------------------------------
# Tracing (D18) — implemented in retail_analyst.tracing
# --------------------------------------------------------------------------------------

TraceEventType = Literal[
    "turn_start",  # user message received
    "llm_call",  # model, prompt_version, tokens, latency, fallback_used
    "tool_call",  # tool name, args (SQL), result status, error_kind
    "sql_guard",  # guard verdict
    "bq_job",  # job id, bytes, rows, attempts
    "retry",  # transient retry
    "fallback",  # switched to a fallback model
    "interrupt",  # paused for confirmation
    "confirmation",  # user answer to a confirmation
    "number_check",  # unverified numbers in facts (non-blocking)
    "golden_examples",  # Golden Bucket trios added to the prompt (ids, scores)
    "pii_redaction",  # output regex hit
    "turn_end",  # final kind, llm_calls, total latency
    "error",  # any error with kind
]
