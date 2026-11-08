"""Test doubles for graph tests: a scripted chat model, a recording tracer, settings, a fake SQL tool.

No network, no credentials: the model's replies are scripted AIMessages with tool_calls.
"""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from retail_analyst.config import (
    AgentSettings,
    ModelSettings,
    ReportSettings,
    Settings,
    SqlSettings,
    TracingSettings,
)
from retail_analyst.contracts import AnswerSubmission, TableData, ToolResult
from retail_analyst.tools.registry import ToolContext, ToolOutcome, ToolSpec

_ids = itertools.count(1)


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted replies in order; an Exception in the script is raised instead."""

    script: list[Any]
    received: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.received.append(list(messages))
        if not self.script:
            raise AssertionError("the model was called more times than scripted")
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, **args: Any) -> AIMessage:
    """An AIMessage with one tool call."""
    return calls((name, args))


def calls(*items: tuple[str, dict[str, Any]]) -> AIMessage:
    """An AIMessage with several tool calls."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{next(_ids)}", "type": "tool_call"} for n, a in items],
    )


def submit(message: str = "Done.", **fields: Any) -> AIMessage:
    return call(
        "submit_answer", **AnswerSubmission(kind=fields.pop("kind", "info"), message=message, **fields).model_dump()
    )


class FakeTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(self, type: str, **payload: Any) -> None:
        self.events.append({"type": type, **payload})

    def of(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type]


def make_settings(db_path: str = "unused.sqlite", **agent: Any) -> Settings:
    return Settings(
        dataset="test.dataset",
        models=ModelSettings(analyst="fake-model"),
        agent=AgentSettings(**agent),
        sql=SqlSettings(),
        reports=ReportSettings(db_path=db_path, bulk_delete_threshold=3),
        tracing=TracingSettings(),
    )


class FakeSqlArgs(BaseModel):
    sql: str


def fake_sql_tool(rows: list[list[Any]], columns: list[str] | None = None) -> ToolSpec:
    """A stand-in for the real SQL tool: stores the result under the next query id (q1, q2, ...)."""

    def handler(args: FakeSqlArgs, ctx: ToolContext) -> ToolOutcome:
        qid = f"q{len(ctx.state.get('queries') or {}) + 1}"
        data = TableData(columns=columns or ["value"], rows=rows)
        stored = {"sql": args.sql, "columns": data.columns, "rows": data.rows, "rows_total": len(rows)}
        return ToolOutcome(
            ToolResult(status="ok", query_id=qid, data=data, rows_total=len(rows)),
            state_update={"queries": {qid: stored}},
        )

    return ToolSpec(name="run_sql", description="Run a SQL query.", args_model=FakeSqlArgs, handler=handler)
