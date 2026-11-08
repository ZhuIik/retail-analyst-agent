"""Saved reports library (D16), SQLite in the prototype (Cloud SQL in production).

Every method takes `owner_id` and filters by it in SQL: the owner filter is added by code,
never by the model, so no tool argument can reach another user's reports. Deletion is soft
(status + timestamp) and always leaves an audit row.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from retail_analyst.contracts import Report, ReportSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    question    TEXT NOT NULL,
    sql_json    TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS reports_owner ON reports(owner_id, status);
CREATE TABLE IF NOT EXISTS audit_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT NOT NULL,
    owner_id         TEXT NOT NULL,
    action           TEXT NOT NULL,
    report_ids_json  TEXT NOT NULL,
    trace_id         TEXT NOT NULL
);
"""

_SUMMARY_COLS = "id, title, created_at, session_id"
_REPORT_COLS = "id, title, created_at, session_id, owner_id, body, question, sql_json, status"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _casefold(value: str | None) -> str | None:
    return value.casefold() if value is not None else None


def _summary(row: sqlite3.Row) -> ReportSummary:
    return ReportSummary(
        report_id=row["id"], title=row["title"], created_at=row["created_at"], session_id=row["session_id"]
    )


def _report(row: sqlite3.Row) -> Report:
    return Report(
        report_id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        session_id=row["session_id"],
        owner_id=row["owner_id"],
        body=row["body"],
        question=row["question"],
        sql=json.loads(row["sql_json"]),
        status=row["status"],
    )


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class ReportStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across threads, and the file is
        # released immediately (matters on Windows). `with conn` commits or rolls back.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # SQLite's lower()/LIKE only fold ASCII; Python casefold handles any language.
        conn.create_function("casefold", 1, _casefold, deterministic=True)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save(self, owner_id: str, title: str, body: str, question: str, sql: list[str], session_id: str) -> Report:
        created_at = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reports (owner_id, title, body, question, sql_json, session_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_id, title, body, question, json.dumps(sql), session_id, created_at),
            )
            report_id = cur.lastrowid
        assert report_id is not None
        return Report(
            report_id=report_id,
            title=title,
            created_at=created_at,
            session_id=session_id,
            owner_id=owner_id,
            body=body,
            question=question,
            sql=sql,
        )

    def list(self, owner_id: str, limit: int = 20) -> list[ReportSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_SUMMARY_COLS} FROM reports WHERE owner_id = ? AND status = 'active'"
                " ORDER BY id DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [_summary(r) for r in rows]

    def search(self, owner_id: str, text: str | None = None, session_id: str | None = None) -> list[ReportSummary]:
        """Active reports of the owner whose title or body contains every word of `text`.

        Matching is deliberately simple (D16): precision does not affect safety, because
        deletion always shows a preview and needs the user's confirmation.
        """
        where = ["owner_id = ?", "status = 'active'"]
        params: list[object] = [owner_id]
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        for word in (text or "").casefold().split():
            where.append("(instr(casefold(title), ?) > 0 OR instr(casefold(body), ?) > 0)")
            params += [word, word]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_SUMMARY_COLS} FROM reports WHERE {' AND '.join(where)} ORDER BY id DESC",
                params,
            ).fetchall()
        return [_summary(r) for r in rows]

    def get_many(self, owner_id: str, report_ids: list[int]) -> list[Report]:
        """The owner's reports among `report_ids` (any status); other owners' ids are ignored."""
        if not report_ids:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_REPORT_COLS} FROM reports WHERE owner_id = ?"
                f" AND id IN ({_placeholders(len(report_ids))}) ORDER BY id",
                [owner_id, *report_ids],
            ).fetchall()
        return [_report(r) for r in rows]

    def soft_delete(self, owner_id: str, report_ids: list[int], trace_id: str) -> int:
        """Soft-delete the owner's active reports among `report_ids`; return how many.

        The ids come from the confirmed snapshot (D16); the owner and status filters are
        re-applied here so a stale or foreign id can never be deleted.
        """
        if not report_ids:
            return 0
        now = _now()
        with self._connect() as conn:
            ids = [
                r["id"]
                for r in conn.execute(
                    f"SELECT id FROM reports WHERE owner_id = ? AND status = 'active'"
                    f" AND id IN ({_placeholders(len(report_ids))})",
                    [owner_id, *report_ids],
                )
            ]
            if ids:
                conn.execute(
                    f"UPDATE reports SET status = 'deleted', deleted_at = ?"
                    f" WHERE owner_id = ? AND id IN ({_placeholders(len(ids))})",
                    [now, owner_id, *ids],
                )
            conn.execute(
                "INSERT INTO audit_log (ts, owner_id, action, report_ids_json, trace_id) VALUES (?, ?, ?, ?, ?)",
                (now, owner_id, "soft_delete", json.dumps(ids), trace_id),
            )
        return len(ids)

    def audit_log(self, owner_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, ts, owner_id, action, report_ids_json, trace_id FROM audit_log"
                " WHERE owner_id = ? ORDER BY id",
                (owner_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "owner_id": r["owner_id"],
                "action": r["action"],
                "report_ids": json.loads(r["report_ids_json"]),
                "trace_id": r["trace_id"],
            }
            for r in rows
        ]
