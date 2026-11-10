"""History trimming before each agent call and tool definitions for bind_tools."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from retail_analyst.agent.graph import tool_definition, trim_history
from retail_analyst.contracts import AnswerSubmission


def _ai_call(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {}, "id": call_id, "type": "tool_call"}])


def _tool(call_id: str, qid: str | None = None) -> ToolMessage:
    return ToolMessage(
        content=f"rows for {call_id}", tool_call_id=call_id, name="run_sql", artifact={"query_id": qid} if qid else None
    )


def _turn(n: int) -> list:
    """Human, AI(call), Tool, AI(call), Tool: 5 messages."""
    return [
        HumanMessage(f"question {n}"),
        _ai_call(f"{n}a"),
        _tool(f"{n}a", f"q{n}"),
        _ai_call(f"{n}b"),
        _tool(f"{n}b"),
    ]


def test_window_starts_at_a_user_message_and_never_splits_tool_pairs():
    history = _turn(1) + _turn(2) + _turn(3)  # 15 messages
    trimmed = trim_history(history, keep=8)
    # The last 8 would start mid-turn 2 (a ToolMessage); the window moves forward to turn 3's question.
    assert isinstance(trimmed[0], HumanMessage) and trimmed[0].content == "question 3"
    assert trimmed == history[10:]


def test_older_tool_results_become_stubs_current_turn_kept_whole():
    history = _turn(1) + _turn(2)
    trimmed = trim_history(history, keep=12)
    assert len(trimmed) == 10
    old = [m.content for m in trimmed[:5] if isinstance(m, ToolMessage)]
    assert old == ["[result of q1 omitted]", "[result of run_sql omitted]"]
    assert [m.content for m in trimmed[5:] if isinstance(m, ToolMessage)] == ["rows for 2a", "rows for 2b"]
    assert history[2].content == "rows for 1a"  # state messages are not modified


def test_long_current_turn_is_kept_whole():
    history = _turn(1) + [HumanMessage("big")] + [m for i in range(5) for m in (_ai_call(f"x{i}"), _tool(f"x{i}"))]
    trimmed = trim_history(history, keep=4)
    assert trimmed[0].content == "big" and len(trimmed) == 11


def test_tool_definition_inlines_nested_models():
    definition = tool_definition("submit_answer", "Final answer.", AnswerSubmission)
    params = definition["parameters"]
    assert definition["description"] == "Final answer."
    assert "$defs" not in params and "title" not in params
    assert params["properties"]["facts"]["items"]["properties"]["text"]["type"] == "string"
