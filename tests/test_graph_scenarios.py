"""Whole-graph scenarios with a scripted model (D21, Б2). No network.

The deletion scenarios prove the D16 guarantee: nothing is deleted without the user's exact
confirmation, only the previewed snapshot is deleted, and never another owner's report.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai._function_utils import convert_to_genai_function_declarations
from langgraph.types import Command
from pydantic import BaseModel

from retail_analyst import llm
from retail_analyst.agent.graph import build_graph
from retail_analyst.contracts import AccessScope, AgentError, ErrorKind, ToolResult, UserContext
from retail_analyst.reports.store import ReportStore
from retail_analyst.tools.registry import ToolOutcome, ToolRegistry, ToolSpec
from retail_analyst.tools.report_tools import register_report_tools
from tests.fakes import FakeTracer, ScriptedChatModel, call, calls, fake_sql_tool, make_settings, submit

ALICE = UserContext(
    user_id="alice", display_name="Alice", role="brand_manager", access=AccessScope(column="brand", values=["Acme"])
)


class NoArgs(BaseModel):
    pass


def _bad_sql(args: BaseModel, ctx: Any) -> ToolOutcome:
    return ToolOutcome(ToolResult(status="error", error_kind=ErrorKind.INVALID_SQL, message="Unknown column x."))


def _quota_sql(args: BaseModel, ctx: Any) -> ToolOutcome:
    raise AgentError(ErrorKind.QUOTA_EXHAUSTED, "BigQuery daily quota is exhausted.")


class Harness:
    """One graph + scripted model + temp store + recording tracer; one thread per harness."""

    def __init__(self, tmp_path, script: list[Any], user: UserContext = ALICE, **agent_settings: Any) -> None:
        self.store = ReportStore(tmp_path / "reports.sqlite")
        self.tracer = FakeTracer()
        self.model = ScriptedChatModel(script=script)
        self.user = user
        registry = ToolRegistry()
        register_report_tools(registry)
        registry.register(fake_sql_tool(rows=[["Texas", 1234.5], ["Ohio", 99.0]], columns=["state", "revenue"]))
        registry.register(ToolSpec(name="bad_sql", description="Always fails.", args_model=NoArgs, handler=_bad_sql))
        registry.register(
            ToolSpec(name="quota_sql", description="Quota exhausted.", args_model=NoArgs, handler=_quota_sql)
        )
        registry.register(
            ToolSpec(
                name="admin_only", description="Admin.", args_model=NoArgs, handler=_bad_sql, roles=frozenset({"admin"})
            )
        )
        self.graph = build_graph(
            registry,
            lambda tools: self.model.bind_tools(tools),
            {"reports": self.store, "tracer": self.tracer},
            make_settings(**agent_settings),
            lambda state: "You are a test analyst.",
        )
        self.config = {"configurable": {"thread_id": "thread-1"}}

    def say(self, text: str) -> dict[str, Any]:
        return self.graph.invoke(
            {
                "messages": [HumanMessage(text)],
                "question": text,
                "user": self.user.model_dump(),
                "session_id": "s1",
                "trace_id": f"trace-{text}",
                "llm_calls": 0,
                "sql_fix_attempts": 0,
                "final_text": None,
            },
            self.config,
        )

    def resume(self, answer: Any) -> dict[str, Any]:
        return self.graph.invoke(Command(resume=answer), self.config)

    def state(self) -> dict[str, Any]:
        return self.graph.get_state(self.config).values

    def save(self, title: str, owner: str = "alice", session: str = "s1") -> int:
        return self.store.save(owner, title, f"body of {title}", "q", [], session).report_id

    def active_ids(self, owner: str = "alice") -> set[int]:
        return {r.report_id for r in self.store.list(owner)}


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_fatal_tool_error_ends_the_turn_without_another_model_call(tmp_path):
    # D12: quota/auth/region errors cannot be fixed by rewriting SQL; the model must not burn calls on them.
    h = Harness(tmp_path, [calls(("quota_sql", {}), ("run_sql", {"sql": "SELECT 1"}))])
    out = h.say("revenue?")
    assert out["final_text"] == "BigQuery daily quota is exhausted."
    assert h.model.script == [] and len(h.model.received) == 1  # exactly one model call
    replies = _tool_messages(out)
    assert len(replies) == 2  # every tool call answered: the next turn's history stays valid
    assert "Not executed" in replies[1].content
    assert out["llm_calls"] == 1


def _reply_to(state: dict[str, Any], tool: str) -> ToolMessage:
    """The ToolMessage answering the latest call of `tool` (there must be exactly one)."""
    call_id = [c["id"] for m in state["messages"] for c in getattr(m, "tool_calls", []) if c["name"] == tool][-1]
    [reply] = [m for m in _tool_messages(state) if m.tool_call_id == call_id]
    return reply


# --------------------------------------------------------------------------------------
# Deletion (D16)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["no", "yes please", "", "y", "2", {"answer": "yes"}])
def test_delete_pauses_and_anything_but_exact_answer_cancels(tmp_path, reply):
    h = Harness(tmp_path, [call("delete_reports", text="texas"), submit("Nothing was deleted.")])
    t1, t2 = h.save("Revenue Texas"), h.save("Texas returns")
    other = h.save("Margins")

    result = h.say("delete my Texas reports")

    # Paused at the confirm node: the store is untouched, no audit row, one model call so far.
    [pause] = result["__interrupt__"]
    assert pause.value["type"] == "confirm_deletion"
    assert pause.value["required_answer"] == "yes"
    assert {r["report_id"] for r in pause.value["preview"]} == {t1, t2}
    assert h.graph.get_state(h.config).next == ("confirm",)
    assert h.active_ids() == {t1, t2, other}
    assert h.store.audit_log("alice") == []
    assert len(h.model.received) == 1

    final = h.resume(reply)

    assert h.active_ids() == {t1, t2, other}  # 0 deletions without confirmation
    assert h.store.audit_log("alice") == []
    assert final["pending_deletion"] is None
    assert final["final_text"] == "Nothing was deleted."
    # The confirm node answered exactly the delete_reports call (once), so the history stays valid.
    assert "nothing was deleted" in _reply_to(final, "delete_reports").content
    assert h.tracer.of("confirmation")[0]["confirmed"] is False


def test_delete_confirmed_deletes_only_the_previewed_snapshot(tmp_path):
    h = Harness(tmp_path, [call("delete_reports", text="texas"), submit("Deleted 2 reports.")])
    t1, t2 = h.save("Revenue Texas"), h.save("Texas returns")
    other = h.save("Margins")

    h.say("delete my Texas reports")
    late = h.save("Texas late arrival")  # matches the text, but was not in the confirmed preview
    final = h.resume("  YES ")

    assert h.active_ids() == {other, late}
    [audit] = h.store.audit_log("alice")
    assert (audit["report_ids"], audit["trace_id"]) == ([t1, t2], "trace-delete my Texas reports")
    assert "Permanently deleted 2 of 2" in _reply_to(final, "delete_reports").content
    assert final["final_text"] == "Deleted 2 reports."
    assert h.tracer.of("confirmation")[0]["deleted"] == 2
    assert len(h.tracer.of("interrupt")) == 1  # traced once, although the confirm node ran twice


def test_bulk_delete_requires_the_exact_count(tmp_path):
    script = [
        call("delete_reports", text="weekly"),
        submit("Cancelled."),
        call("delete_reports", text="weekly"),
        submit("Deleted."),
    ]
    h = Harness(tmp_path, script)
    ids = {h.save(f"Weekly sales {i}") for i in range(4)}  # 4 > bulk_delete_threshold (3)

    first = h.say("delete all weekly reports")
    assert first["__interrupt__"][0].value["required_answer"] == "4"
    h.resume("yes")  # not enough for a bulk deletion
    assert h.active_ids() == ids

    h.say("delete all weekly reports")
    h.resume("4")
    assert h.active_ids() == set()
    assert len(h.store.audit_log("alice")) == 1


def test_other_owners_reports_are_never_deleted(tmp_path):
    h = Harness(tmp_path, [])
    b1, b2 = h.save("Bob revenue", owner="bob"), h.save("Bob margins", owner="bob")
    mine = h.save("Alice report")
    h.model.script = [
        call("delete_reports", report_ids=[b1, b2]),
        submit("No such reports."),
        call("delete_reports", text="bob"),
        submit("No such reports."),
        call("delete_reports", report_ids=[mine, b1]),
        submit("Deleted."),
    ]

    # Foreign ids passed by the model: nothing matches, no confirmation is even asked.
    result = h.say("delete reports 1 and 2")
    assert "__interrupt__" not in result
    assert "nothing to delete" in _reply_to(result, "delete_reports").content
    # Text that only matches another owner's titles: same.
    assert "__interrupt__" not in h.say("delete bob's reports")
    # Mixed ids: the preview (and the deletion) contain only the user's own report.
    paused = h.say("delete reports 3 and 1")
    assert [r["report_id"] for r in paused["__interrupt__"][0].value["preview"]] == [mine]
    h.resume("yes")

    assert h.active_ids("bob") == {b1, b2}
    assert h.active_ids() == set()
    assert h.store.audit_log("bob") == []
    # The model cannot even express an owner: no tool exposes such an argument.
    delete_def = next(t for t in h.model.bound_tools if t["name"] == "delete_reports")
    assert set(delete_def["parameters"]["properties"]) == {"text", "report_ids", "this_conversation_only"}


def test_zero_matches_needs_no_confirmation(tmp_path):
    h = Harness(tmp_path, [call("delete_reports", text="nothing like this"), submit("No matching reports.")])
    h.save("Revenue")
    result = h.say("delete the forecast report")
    assert "__interrupt__" not in result and result["final_text"] == "No matching reports."
    assert h.store.audit_log("alice") == []


# --------------------------------------------------------------------------------------
# Budgets and protocol errors
# --------------------------------------------------------------------------------------


def test_llm_call_limit_gives_honest_partial_answer_without_extra_call(tmp_path):
    script = [call("run_sql", sql="SELECT 1"), call("find_reports"), submit("never reached")]
    h = Harness(tmp_path, script, max_llm_calls_per_turn=2)

    result = h.say("complex question")

    assert len(h.model.received) == 2 and len(h.model.script) == 1  # no third model call
    assert "limit of 2 model calls" in result["final_text"]
    assert "q1" in result["final_text"]  # says what was completed
    assert result["llm_calls"] == 2
    assert result.get("last_answer") is None  # a stop message is not a saveable answer


def test_submit_answer_mixed_with_other_calls_is_rejected_and_model_retries(tmp_path):
    mixed = calls(("find_reports", {}), ("submit_answer", {"kind": "info", "message": "premature"}))
    h = Harness(tmp_path, [mixed, submit("You have no reports.")])

    result = h.say("list my reports")

    by_id = {m.tool_call_id: m for m in _tool_messages(result)}
    find_id, submit_id = (c["id"] for c in mixed.tool_calls)
    assert by_id[submit_id].status == "error" and "alone" in by_id[submit_id].content
    assert by_id[find_id].status == "success"
    assert result["final_text"] == "You have no reports."
    # The retry saw the error message.
    assert any(isinstance(m, ToolMessage) and "alone" in m.content for m in h.model.received[1])


def test_invalid_args_and_hidden_tools_return_fixable_errors(tmp_path):
    bad = call("submit_answer", kind="poem", message="x")
    h = Harness(tmp_path, [call("admin_only"), call("find_reports", this_conversation_only="maybe"), bad, submit("ok")])

    result = h.say("hi")

    errors = [m.content for m in _tool_messages(result) if m.status == "error"]
    assert "Unknown tool 'admin_only'" in errors[0]  # role filter enforced in code, not by the prompt
    assert "Invalid arguments for find_reports" in errors[1]
    assert "Invalid submit_answer arguments" in errors[2]
    assert result["final_text"] == "ok"
    assert all(t["name"] != "admin_only" for t in h.model.bound_tools)


def test_sql_fix_budget_tells_the_model_to_stop(tmp_path):
    h = Harness(tmp_path, [call("bad_sql"), call("bad_sql"), call("bad_sql"), submit("Could not compute.")])

    result = h.say("question")

    contents = [m.content for m in _tool_messages(result) if m.status == "error"]
    assert ["STOP" in c for c in contents] == [False, False, True]
    assert result["sql_fix_attempts"] == 3


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("429 daily quota"), "daily quota"),
        (AgentError(ErrorKind.AUTH, "Check your GOOGLE_API_KEY.", "401"), "Check your GOOGLE_API_KEY."),
    ],
)
def test_llm_failure_ends_turn_honestly(tmp_path, monkeypatch, exc, expected):
    monkeypatch.setattr(llm, "classify_llm_error", lambda e: ErrorKind.QUOTA_EXHAUSTED)
    h = Harness(tmp_path, [exc])
    result = h.say("question")
    assert expected in result["final_text"]
    [error] = h.tracer.of("error")
    assert error["source"] == "llm"


# --------------------------------------------------------------------------------------
# Answers and saving (D15, Г1)
# --------------------------------------------------------------------------------------


def test_save_report_stores_exactly_the_text_shown(tmp_path):
    answer = submit(
        "Texas leads revenue.",
        kind="answer",
        facts=[{"text": "Texas revenue is 1,234.5 and Ohio 7,777.", "query_ids": ["q1"]}],
        hypotheses=["Larger population."],
        assumptions=["Revenue = sale_price of completed orders."],
        offer_save=True,
        suggested_title="Revenue by state",
    )
    script = [
        call("run_sql", sql="SELECT state, revenue FROM t"),
        answer,
        call("save_report", title="Revenue by state"),
        submit("Saved."),
    ]
    h = Harness(tmp_path, script)

    shown = h.say("revenue by state contact me at bob@example.com")["final_text"]
    assert "Save this as a report" in shown and "**Hypotheses**" in shown
    h.say("yes, save it")

    [summary] = h.store.list("alice")
    [report] = h.store.get_many("alice", [summary.report_id])
    assert report.body == shown
    assert report.question == "revenue by state contact me at [redacted]"  # D14 net before storage
    assert report.sql == ["SELECT state, revenue FROM t"]
    assert report.session_id == "s1"
    # The number check flags 7,777 (not in q1's rows) in the trace only; the answer is unchanged.
    [check] = h.tracer.of("number_check")
    assert check["unverified"] == ["7,777"]


def test_save_without_an_answer_is_an_error(tmp_path):
    h = Harness(tmp_path, [call("save_report", title="x"), submit("Nothing to save yet.")])
    result = h.say("save it")
    assert "no answer to save" in _reply_to(result, "save_report").content
    assert h.store.list("alice") == []


def test_answer_text_is_pii_redacted(tmp_path):
    h = Harness(tmp_path, [submit("Write to jane.doe@example.com for details.")])
    result = h.say("who to contact")
    assert "jane.doe@example.com" not in result["final_text"]
    assert h.tracer.of("pii_redaction")[0]["hits"] == 1


def test_bound_tool_definitions_are_accepted_by_gemini_converter(tmp_path):
    h = Harness(tmp_path, [submit("hi")])
    h.say("hi")
    [tool] = convert_to_genai_function_declarations(h.model.bound_tools)
    decls = {d.name: d for d in tool.function_declarations}
    assert {"submit_answer", "save_report", "find_reports", "delete_reports", "run_sql"} <= set(decls)
    fact = decls["submit_answer"].parameters.properties["facts"].items
    assert set(fact.properties) == {"text", "query_ids"}  # nested model inlined, no $ref
