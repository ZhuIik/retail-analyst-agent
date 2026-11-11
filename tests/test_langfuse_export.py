"""Langfuse export is optional: no keys, a broken package or bad keys must never break the agent."""

from __future__ import annotations

import builtins

from retail_analyst.langfuse_export import LangfuseExport, _mask


def test_without_keys_export_is_off_and_harmless(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    export = LangfuseExport()
    assert not export.enabled
    assert export.run_config("s1", "anna", "t1") == {}
    assert export.check() is False
    export.flush()  # no-op, no exception


def test_import_failure_disables_export(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    export = LangfuseExport()
    assert not export.enabled and export.run_config("s1", "anna", "t1") == {}


def test_with_keys_the_run_config_carries_session_and_user(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")  # unreachable on purpose
    export = LangfuseExport()
    assert export.enabled
    config = export.run_config("s1", "anna", "t1")
    assert config["callbacks"] == [export.handler]
    assert config["metadata"]["langfuse_session_id"] == "s1"
    assert config["metadata"]["langfuse_user_id"] == "anna"
    export.flush()  # unreachable server: must not raise


def test_mask_redacts_pii_in_nested_payloads():
    data = {"input": ["contact a@b.com"], "output": "ok", "n": 3}
    assert _mask(data=data) == {"input": ["contact [redacted]"], "output": "ok", "n": 3}


def test_graph_runs_with_the_langfuse_callback_and_no_server(monkeypatch, tmp_path):
    # Wrong keys and no reachable server: the turn still completes normally.
    from langchain_core.messages import HumanMessage

    from tests.fakes import submit
    from tests.test_graph_scenarios import Harness

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-wrong")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-wrong")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    export = LangfuseExport()
    h = Harness(tmp_path, [submit("Hello.")])
    config = {**h.config, **export.run_config("s1", "alice", "t1")}
    state = {"messages": [HumanMessage("hi")], "question": "hi", "user": h.user.model_dump(), "session_id": "s1",
             "trace_id": "t1", "llm_calls": 0, "sql_fix_attempts": 0, "final_text": None}  # fmt: skip
    out = h.graph.invoke(state, config)
    assert out["final_text"] == "Hello."
    export.flush()
