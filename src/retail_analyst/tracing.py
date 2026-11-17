"""JSONL tracing.

    class Tracer:
        def __init__(self, trace_dir: Path, session_id: str, user_id: str) -> None
        def new_turn(self) -> str                                   # returns a new trace_id
        def event(self, type: TraceEventType, **payload) -> None    # never raises
        def last_turn(self) -> list[dict]                           # events of the latest turn (/trace)
        @property trace_id -> str | None

One JSONL file per session (`<trace_dir>/<session_id>.jsonl`); every line has ts, session_id,
trace_id, user_id, type, payload. String payload values pass the PII regex and are
truncated. Production would emit the same events through OpenTelemetry.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from retail_analyst.contracts import TraceEventType
from retail_analyst.safety.pii import redact_pii

log = logging.getLogger(__name__)

MAX_STRING_CHARS = 4000


class Tracer:
    def __init__(
        self, trace_dir: Path, session_id: str, user_id: str, *, sensitive_values: tuple[str, ...] = ()
    ) -> None:
        self.path = Path(trace_dir) / f"{session_id}.jsonl"
        self.session_id = session_id
        self.user_id = user_id
        self._trace_id: str | None = None
        self._sensitive_values = tuple(v for v in sensitive_values if v)

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    def new_turn(self, trace_id: str | None = None) -> str:
        self._trace_id = trace_id or uuid.uuid4().hex[:12]
        return self._trace_id

    def event(self, type: TraceEventType, **payload: object) -> None:
        """Append one event. Tracing must never break the agent, so every failure is swallowed."""
        try:
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "session_id": self.session_id,
                "trace_id": self._trace_id,
                "user_id": self.user_id,
                "type": type,
                "payload": _clean(payload, self._sensitive_values),
            }
            line = json.dumps(record, ensure_ascii=False, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            log.warning("trace event %r dropped: %s", type, exc)

    def last_turn(self) -> list[dict]:
        """Events of the latest trace_id in this session's file (for the /trace command)."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line must not hide the rest
        if not events:
            return []
        latest = events[-1].get("trace_id")
        return [e for e in events if e.get("trace_id") == latest]


def _clean(value: Any, sensitive_values: tuple[str, ...] = ()) -> Any:
    """Redact PII from every string (also inside nested dicts/lists) and cap its length."""
    if isinstance(value, str):
        for secret in sensitive_values:
            value = value.replace(secret, "[secret]")
        text, _ = redact_pii(value)
        if len(text) > MAX_STRING_CHARS:
            text = f"{text[:MAX_STRING_CHARS]}... [truncated {len(text) - MAX_STRING_CHARS} chars]"
        return text
    if isinstance(value, BaseModel):
        return _clean(value.model_dump(mode="json"), sensitive_values)
    if isinstance(value, dict):
        return {_clean(str(k), sensitive_values): _clean(v, sensitive_values) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_clean(v, sensitive_values) for v in value]
    if value is None or isinstance(value, bool | int | float):
        return value
    return _clean(str(value), sensitive_values)
