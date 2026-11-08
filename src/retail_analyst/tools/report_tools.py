"""Report tools (D16): save, find and (request to) delete saved reports.

The owner is always `ctx.user.user_id` (from login), never a tool argument. `delete_reports`
has no side effects: it only resolves the matching reports and asks the graph to confirm
(`pending_deletion`); the confirm node deletes exactly that snapshot after the user agrees.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from retail_analyst.contracts import PendingDeletion, ReportSummary, TableData, ToolResult, ToolRisk
from retail_analyst.reports.store import ReportStore
from retail_analyst.safety.pii import redact_pii
from retail_analyst.tools.registry import ToolContext, ToolOutcome, ToolRegistry, ToolSpec

_MAX_LISTED = 20


class SaveReportArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200, description="Short descriptive title for the report.")


class FindReportsArgs(BaseModel):
    text: str | None = Field(default=None, description="Words to look for in report titles and text.")
    this_conversation_only: bool = Field(
        default=False, description="True to search only reports saved in this conversation."
    )


class DeleteReportsArgs(BaseModel):
    text: str | None = Field(default=None, description="Words to look for in report titles and text.")
    report_ids: list[int] | None = Field(default=None, description="Exact report ids, if the user named them.")
    this_conversation_only: bool = Field(
        default=False, description="True to match only reports saved in this conversation."
    )


def _store(ctx: ToolContext) -> ReportStore:
    return ctx.services["reports"]


def _summary_table(reports: list[ReportSummary]) -> TableData:
    return TableData(
        columns=["report_id", "title", "created_at"],
        rows=[[r.report_id, r.title, r.created_at] for r in reports[:_MAX_LISTED]],
    )


def _save_report(args: SaveReportArgs, ctx: ToolContext) -> ToolOutcome:
    # Г1: the body is what the user saw, taken from state; the model passes only the title.
    body = ctx.state.get("last_answer_text")
    last = ctx.state.get("last_answer") or {}
    if not body:
        return ToolOutcome(ToolResult(status="error", message="There is no answer to save yet."))
    queries = ctx.state.get("queries") or {}
    sql = [queries[q]["sql"] for q in last.get("query_ids", []) if q in queries]
    # D14: the body was already redacted when shown; the question is the user's raw text.
    body, _ = redact_pii(body)
    question, _ = redact_pii(last.get("question", ""))
    report = _store(ctx).save(
        owner_id=ctx.user.user_id,
        title=args.title.strip(),
        body=body,
        question=question,
        sql=sql,
        session_id=ctx.session_id,
    )
    return ToolOutcome(ToolResult(status="ok", message=f"Saved report #{report.report_id} '{report.title}'."))


def _find(ctx: ToolContext, text: str | None, this_conversation_only: bool) -> list[ReportSummary]:
    session = ctx.session_id if this_conversation_only else None
    return _store(ctx).search(ctx.user.user_id, text=text, session_id=session)


def _find_reports(args: FindReportsArgs, ctx: ToolContext) -> ToolOutcome:
    found = _find(ctx, args.text, args.this_conversation_only)
    if not found:
        return ToolOutcome(ToolResult(status="empty", message="No saved reports match."))
    return ToolOutcome(ToolResult(status="ok", data=_summary_table(found), rows_total=len(found)))


def _delete_reports(args: DeleteReportsArgs, ctx: ToolContext) -> ToolOutcome:
    # No side effects here (D16): resolve the owner's matching reports, then let the confirm node
    # ask the user. Foreign or unknown ids simply do not match, because the store filters by owner.
    matches = _find(ctx, args.text, args.this_conversation_only)
    if args.report_ids is not None:
        wanted = set(args.report_ids)
        matches = [r for r in matches if r.report_id in wanted]
    if not matches:
        return ToolOutcome(ToolResult(status="empty", message="No saved reports of yours match; nothing to delete."))
    threshold = ctx.settings.reports.bulk_delete_threshold
    # Strictness scales with impact: bulk deletions require typing the exact count.
    required = "yes" if len(matches) <= threshold else str(len(matches))
    pending = PendingDeletion(
        report_ids=[r.report_id for r in matches],
        preview=matches,
        required_answer=required,
        tool_call_id="",  # bound by the graph's tools node, which knows the call id
    )
    return ToolOutcome(
        ToolResult(status="ok", message=f"Awaiting user confirmation to delete {len(matches)} report(s)."),
        state_update={"pending_deletion": pending.model_dump()},
    )


def register_report_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="save_report",
            description=(
                "Save the last answer shown to the user as a report. Only after the user agreed to save. "
                "Pass only a short title; the report text is taken from the conversation automatically."
            ),
            args_model=SaveReportArgs,
            handler=_save_report,
            # Additive and explicitly requested: the agent asks "Save?" first; confirmation is
            # reserved for destructive actions.
            risk=ToolRisk.READ,
        )
    )
    registry.register(
        ToolSpec(
            name="find_reports",
            description="List the user's saved reports (id, title, date), optionally filtered by words.",
            args_model=FindReportsArgs,
            handler=_find_reports,
            risk=ToolRisk.READ,
        )
    )
    registry.register(
        ToolSpec(
            name="delete_reports",
            description=(
                "Request deletion of the user's saved reports matching words and/or ids. Nothing is deleted "
                "by this call: the user sees a preview and must confirm; you will receive the outcome."
            ),
            args_model=DeleteReportsArgs,
            handler=_delete_reports,
            risk=ToolRisk.INTERNAL_WRITE,
        )
    )
