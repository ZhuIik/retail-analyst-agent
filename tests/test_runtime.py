"""Hard process termination and checkpoint/interrupt continuity. No cloud access."""

import atexit
import pickle
import time

import pytest
from langgraph.types import Command

from retail_analyst.deadline import remaining_timeout, turn_deadline
from retail_analyst.runtime import (
    ProcessSession,
    WorkerTimeout,
    checkpoint_data,
    restore_checkpointer,
    supervised,
)
from tests.fakes import call, make_settings, submit
from tests.test_graph_scenarios import ALICE, Harness


def late_write(marker, result, end):
    marker.with_suffix(".started").write_text("started")
    time.sleep(10)
    marker.write_text("late side effect")
    with result.open("wb") as stream:
        pickle.dump("too late", stream)


def quick_result(result, end):
    with result.open("wb") as stream:
        pickle.dump({"ok": True}, stream)


def failed_worker(result, end):
    raise ValueError("a secret from an SDK error must not be printed")


def blocked_sdk_shutdown(result, end):
    atexit.register(time.sleep, 30)
    quick_result(result, end)


def test_supervisor_terminates_worker_not_just_waiting_thread(tmp_path):
    marker = tmp_path / "late.txt"
    started = time.monotonic()
    with pytest.raises(WorkerTimeout):
        supervised(late_write, (marker,), 4)
    assert time.monotonic() - started < 5.5
    assert marker.with_suffix(".started").is_file()  # worker really started
    assert not marker.exists()  # no delayed side effect after returning a timeout


def test_supervisor_returns_completed_work():
    assert supervised(quick_result, (), 10) == {"ok": True}


def test_worker_failure_does_not_echo_sdk_error():
    with pytest.raises(RuntimeError) as exc:
        supervised(failed_worker, (), 10)
    assert "secret" not in str(exc.value)


def test_completed_turn_does_not_wait_for_sdk_shutdown_hooks():
    assert supervised(blocked_sdk_shutdown, (), 10) == {"ok": True}


def test_confirmation_snapshot_can_resume_in_a_fresh_graph(tmp_path):
    h = Harness(tmp_path, [call("delete_reports", text="weekly"), submit("Deleted")])
    report_id = h.save("Weekly report")
    paused = h.say("delete weekly")
    assert paused["__interrupt__"]
    data = pickle.loads(pickle.dumps(checkpoint_data(h.graph.checkpointer)))
    fresh = Harness(tmp_path, [submit("Deleted")])
    restored = restore_checkpointer(data)
    fresh.graph.checkpointer.storage = restored.storage
    fresh.graph.checkpointer.writes = restored.writes
    fresh.graph.checkpointer.blobs = restored.blobs
    final = fresh.graph.invoke(Command(resume="yes"), fresh.config)
    assert final["final_text"] == "Deleted"
    assert report_id not in fresh.active_ids()


def test_timeout_discards_pending_confirmation_and_resets_context(tmp_path, monkeypatch):
    settings = make_settings(db_path=str(tmp_path / "reports.sqlite"))
    settings.tracing.dir = str(tmp_path / "traces")
    session = ProcessSession(ALICE, settings)
    session.awaiting = {"required_answer": "yes"}
    session._snapshot = {"old": "checkpoint"}

    def timeout(*args, **kwargs):
        raise WorkerTimeout()

    monkeypatch.setattr("retail_analyst.runtime.supervised", timeout)
    session.ask("yes")
    assert session.awaiting is None and session._snapshot is None
    assert "reset the conversation" in session.graph.get_state({}).values["final_text"]
    assert session.tracer.last_turn()[0]["payload"]["kind"] == "hard_turn_timeout"


def test_remaining_deadline_caps_rpc_budget_and_cannot_be_reset_by_retry():
    token = turn_deadline.set(time.monotonic() + 0.5)
    try:
        assert 0 < remaining_timeout(60) <= 0.5
        turn_deadline.set(time.monotonic() - 1)
        with pytest.raises(TimeoutError):
            remaining_timeout(60)
    finally:
        turn_deadline.reset(token)
