"""Regressions for auxiliary PII outputs, observability and execution budgets."""

import json

import pytest
from pydantic import BaseModel

from retail_analyst.contracts import ToolRisk
from retail_analyst.tools.registry import ToolRegistry, ToolSpec
from retail_analyst.tracing import Tracer
from tests.fakes import call, calls, submit
from tests.test_graph_scenarios import Harness, _tool_messages


def test_sql_budget_blocks_later_calls_in_same_batch_and_next_model_call(tmp_path):
    h = Harness(tmp_path, [
        calls(("bad_sql", {}), ("run_sql", {"sql": "SELECT 1"})),
        call("run_sql", sql="SELECT 2"), submit("Could not compute."),
    ], max_sql_fix_attempts=1)
    out = h.say("question")
    assert out.get("queries", {}) == {}
    assert sum("budget exhausted" in m.content for m in _tool_messages(out)) == 2
    assert out["final_text"] == "Could not compute."


def test_tool_budget_blocks_oversized_batch(tmp_path):
    h = Harness(tmp_path, [
        calls(*[("run_sql", {"sql": "SELECT 1"}) for _ in range(5)]), submit("Partial."),
    ], max_tool_calls_per_turn=2)
    out = h.say("question")
    assert len(out["queries"]) == 2
    assert sum("budget exhausted" in m.content for m in _tool_messages(out)) == 3


def test_trace_contains_clean_answer_and_model_result_preview(tmp_path):
    h = Harness(tmp_path, [call("run_sql", sql="SELECT 1"), submit("Contact bob@example.com")])
    tracer = Tracer(tmp_path, "session", "alice")
    tracer.new_turn()
    h.tracer.event = tracer.event
    h.say("question")
    events = tracer.last_turn()
    answer = next(e["payload"] for e in events if e["type"] == "final_answer")
    assert answer["answer"]["message"] == "Contact [redacted]"
    assert answer["final_text"] == "Contact [redacted]"
    assert answer["query_ids"] == ["q1"]
    tool = next(e["payload"] for e in events if e["type"] == "tool_call")
    assert "Texas" in tool["result_preview"]
    assert "bob@example.com" not in json.dumps(events)


def test_report_title_and_saved_sql_are_cleaned(tmp_path):
    h = Harness(tmp_path, [
        call("run_sql", sql="SELECT 'bob@example.com'"), submit("Done"),
        call("save_report", title="For bob@example.com"), submit("Saved"),
    ])
    h.say("question")
    h.say("save")
    [summary] = h.store.list("alice")
    [report] = h.store.get_many("alice", [summary.report_id])
    assert report.title == "For [redacted]"
    assert report.sql == ["SELECT '[redacted]'"]


@pytest.mark.parametrize("risk", [ToolRisk.EXTERNAL_READ, ToolRisk.EXTERNAL_WRITE])
def test_unimplemented_external_policy_is_rejected_before_handler(risk):
    def handler(*args):
        raise AssertionError("must never run")
    with pytest.raises(ValueError, match="execution policy"):
        ToolRegistry().register(ToolSpec("external", "external", BaseModel, handler, risk=risk))
