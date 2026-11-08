"""JSONL tracer (D18): PII redaction, truncation, never raises, last_turn."""

from __future__ import annotations

import json

from retail_analyst.contracts import ErrorKind
from retail_analyst.tracing import MAX_STRING_CHARS, Tracer


def read_lines(tracer: Tracer) -> list[dict]:
    return [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]


def test_event_record_shape(tmp_path) -> None:
    tracer = Tracer(tmp_path, "sess1", "alice")
    trace_id = tracer.new_turn()
    tracer.event("turn_start", question="top brands?", n=3, kind=ErrorKind.TRANSIENT)
    [record] = read_lines(tracer)
    assert tracer.path == tmp_path / "sess1.jsonl"
    assert record["session_id"] == "sess1" and record["user_id"] == "alice"
    assert record["trace_id"] == trace_id == tracer.trace_id
    assert record["type"] == "turn_start"
    assert record["payload"] == {"question": "top brands?", "n": 3, "kind": "transient"}
    assert record["ts"].endswith("+00:00")


def test_pii_redacted_everywhere_and_long_strings_truncated(tmp_path) -> None:
    tracer = Tracer(tmp_path, "s", "u")
    tracer.new_turn()
    tracer.event(
        "tool_call",
        answer="mail john.doe@example.com",
        nested={"rows": ["call +1 415-555-0100"]},
        sql="x" * (MAX_STRING_CHARS + 50),
    )
    payload = read_lines(tracer)[0]["payload"]
    text = json.dumps(payload)
    assert "john.doe@example.com" not in text and "415-555-0100" not in text
    assert "[redacted]" in payload["answer"]
    assert payload["sql"].startswith("x" * MAX_STRING_CHARS) and "truncated 50 chars" in payload["sql"]


def test_event_never_raises(tmp_path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file in the way")
    tracer = Tracer(blocker, "s", "u")  # the trace dir is a file: writing must fail quietly
    tracer.event("error", message="boom")
    assert tracer.last_turn() == []

    ok = Tracer(tmp_path, "s2", "u")
    ok.event("error", obj=object())  # non-JSON values are stringified, not fatal
    assert len(read_lines(ok)) == 1


def test_last_turn_returns_latest_trace_only(tmp_path) -> None:
    tracer = Tracer(tmp_path, "s", "u")
    assert tracer.last_turn() == []
    tracer.new_turn()
    tracer.event("turn_start", q="first")
    tracer.event("turn_end", kind="answer")
    second = tracer.new_turn()
    tracer.event("turn_start", q="second")
    with tracer.path.open("a", encoding="utf-8") as f:
        f.write("{torn line")  # a crash mid-write must not hide the turn
    events = tracer.last_turn()
    assert [e["payload"]["q"] for e in events] == ["second"]
    assert all(e["trace_id"] == second for e in events)
