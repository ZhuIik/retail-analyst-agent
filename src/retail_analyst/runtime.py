"""Run each turn in a killable process. Only completed checkpoints survive a turn.

The supervisor includes process startup, session construction, retries and export in the
deadline. On timeout it terminates the worker; a late tool cannot mutate reports/checkpoints.
SQLite commits and external API requests completed before termination are not rolled back.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retail_analyst.config import Settings
from retail_analyst.contracts import UserContext
from retail_analyst.tracing import Tracer


class WorkerTimeout(TimeoutError):
    pass


def _worker_entry(target, args: tuple, result: Path, end: float) -> None:
    try:
        target(*args, result, end)
    except Exception as exc:
        # SDK exception strings can contain secrets; do not print an uncontrolled traceback.
        with result.open("wb") as stream:
            pickle.dump({"worker_error": type(exc).__name__}, stream)
    # SDK shutdown hooks/export threads must not hold a completed turn open indefinitely.
    os._exit(0)


def supervised(target, args: tuple, timeout: float) -> Any:
    """No threads/futures left running after a timeout; result files are private and ephemeral."""
    end = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="retail-turn-") as directory:
        result = Path(directory) / "result.pickle"
        worker = multiprocessing.get_context("spawn").Process(
            target=_worker_entry, args=(target, args, result, end)
        )
        worker.start()
        try:
            worker.join(max(0, end - time.monotonic()))
            if worker.is_alive() or time.monotonic() > end:
                raise WorkerTimeout("Turn deadline exceeded")
            if worker.exitcode != 0 or not result.is_file():
                raise RuntimeError("Analysis worker failed")
            with result.open("rb") as stream:
                data = pickle.load(stream)  # trusted child output, never supplied by an API caller
            if time.monotonic() > end:
                raise WorkerTimeout("Turn deadline exceeded")
            if isinstance(data, dict) and "worker_error" in data:
                raise RuntimeError(f"Analysis worker failed ({data['worker_error']})")
            return data
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(0.2)
            if worker.is_alive():
                worker.kill()
                worker.join(0.2)
            worker.close()


def checkpoint_data(checkpointer) -> dict:
    return {
        "storage": {thread: dict(namespaces) for thread, namespaces in checkpointer.storage.items()},
        "writes": dict(checkpointer.writes),
        "blobs": dict(checkpointer.blobs),
    }


def restore_checkpointer(data: dict):
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    checkpointer.storage.update({key: defaultdict(dict, value) for key, value in data["storage"].items()})
    checkpointer.writes.update(data["writes"])
    checkpointer.blobs.update(data["blobs"])
    return checkpointer


def _turn_worker(user: dict, settings: dict, snapshot: dict | None, request: dict, result: Path, end: float) -> None:
    from retail_analyst.api import HttpSession
    from retail_analyst.cli import Session
    from retail_analyst.deadline import turn_deadline
    from retail_analyst.faults import FaultInjector

    turn_deadline.set(end)
    saved = snapshot or {}
    session_type = Session if request["console"] else HttpSession
    session = session_type(
        UserContext.model_validate(user), Settings.model_validate(settings), saved.get("faults") or (
            FaultInjector(set(request["fault_modes"])) if request["fault_modes"] else None
        ),
        session_id=request["session_id"],
        checkpointer=restore_checkpointer(saved["checkpoints"]) if saved else None,
        prompt_ctx=saved.get("prompt_ctx"), query_start=saved.get("query_next", 1),
    )
    if saved:
        session.thread, session.started = saved["thread"], saved["started"]
        session.awaiting = saved["awaiting"]
        session.tracer.new_turn(request["trace_id"])
    session.ask(request["text"], trace_id=request["trace_id"])
    data = {
        "checkpoints": checkpoint_data(session.graph.checkpointer),
        "thread": session.thread, "started": session.started, "awaiting": session.awaiting,
        "prompt_ctx": session._services["prompt_ctx"],
        "query_next": next(session._services["query_counter"]),
        "faults": session._services["bq"]._faults,
        "values": session.graph.get_state(session.config).values,
    }
    with result.open("wb") as stream:
        pickle.dump(data, stream)
    # Local JSONL/checkpoint is already complete. Optional SaaS export is best effort.
    export_wait = min(0.5, max(0, end - time.monotonic() - 0.1))
    if export_wait and session.langfuse.enabled:
        export = threading.Thread(target=session.langfuse.flush, daemon=True)
        export.start()
        export.join(export_wait)


class ProcessSession:
    """CLI/HTTP adapter with a hard execution boundary and an in-memory checkpoint snapshot."""

    def __init__(self, user: UserContext, settings: Settings, faults=None, *, console: bool = False) -> None:
        self.user, self.settings, self.console = user, settings, console
        self.session_id = uuid.uuid4().hex
        self.tracer = Tracer(
            settings.resolve(settings.tracing.dir), self.session_id, user.user_id,
            sensitive_values=(settings.google_api_key, settings.id_salt),
        )
        self.langfuse = SimpleNamespace(enabled=False, flush=lambda: None)
        self._fault_modes = list(faults.modes) if faults else []
        self._snapshot: dict | None = None
        self.awaiting: dict | None = None
        self._values: dict = {}
        self.config: dict = {}
        self.graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(values=self._values))

    @property
    def reports(self):
        from retail_analyst.reports.store import ReportStore

        return ReportStore(self.settings.resolve(self.settings.reports.db_path))

    def ask(self, text: str) -> None:
        trace_id = self.tracer.trace_id if self.awaiting else self.tracer.new_turn()
        request = {
            "text": text, "trace_id": trace_id, "session_id": self.session_id,
            "console": self.console, "fault_modes": self._fault_modes,
        }
        try:
            data = supervised(
                _turn_worker, (self.user.model_dump(), self.settings.model_dump(), self._snapshot, request),
                self.settings.agent.turn_timeout_s,
            )
        except WorkerTimeout:
            self._snapshot, self.awaiting = None, None
            text = (
                f"The {self.settings.agent.turn_timeout_s}-second time limit was reached and analysis was stopped. "
                "I reset the conversation. Some operations may already have completed; "
                "check /reports before repeating a save or deletion."
            )
            self._values = {"final_text": text}
            self.tracer.event("error", kind="hard_turn_timeout", seconds=self.settings.agent.turn_timeout_s)
            self.tracer.event("final_answer", final_text=text, answer={"kind": "info", "message": text})
            if self.console:
                print(text)
            return
        except RuntimeError:
            self._snapshot, self.awaiting = None, None
            text = (
                "Analysis failed and I reset the conversation. "
                "Check /reports before repeating a save or deletion."
            )
            self._values = {"final_text": text}
            self.tracer.event("error", kind="worker_failed")
            self.tracer.event("final_answer", final_text=text, answer={"kind": "info", "message": text})
            if self.console:
                print(text)
            return
        self._snapshot, self.awaiting, self._values = data, data["awaiting"], data["values"]

    def show_reports(self) -> None:
        from retail_analyst.cli import Session

        Session.show_reports(self)

    def show_trace(self) -> None:
        from retail_analyst.cli import Session

        Session.show_trace(self)
