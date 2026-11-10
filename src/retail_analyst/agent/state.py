"""Graph state. Everything here must be JSON-serialisable (checkpointer)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user: dict[str, Any]  # UserContext.model_dump()
    session_id: str
    trace_id: str  # one per user message (turn)
    question: str  # the user's message for this turn
    # Query results by id ("q1", ...): {"sql": str, "columns": [...], "rows": [...], "rows_total": int}
    queries: Annotated[dict[str, dict[str, Any]], merge_dicts]
    llm_calls: int  # in the current turn; hard limit from settings
    sql_fix_attempts: int  # consecutive fixable SQL errors in the current turn
    pending_deletion: dict[str, Any] | None  # PendingDeletion.model_dump() or None
    last_answer: dict[str, Any] | None  # AnswerSubmission.model_dump() of the last shown answer
    last_answer_text: str | None  # the exact text shown to the user (saved as the report body)
    final_text: str | None  # output of the finalize node for this turn
