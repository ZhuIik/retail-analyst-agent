"""ReportStore: owner isolation, search, permanent delete with an audit row."""

from __future__ import annotations

import pytest

from retail_analyst.reports.store import ReportStore


@pytest.fixture
def store(tmp_path):
    return ReportStore(tmp_path / "nested" / "reports.sqlite")  # parent dir is created


def _save(store: ReportStore, owner: str, title: str, body: str = "text", session: str = "s1") -> int:
    return store.save(owner, title, body, "question?", ["SELECT 1"], session).report_id


def test_save_and_get_roundtrip(store):
    rid = _save(store, "alice", "Revenue by state", body="Texas leads with 1,234.")
    [report] = store.get_many("alice", [rid])
    assert (report.title, report.body, report.sql) == ("Revenue by state", "Texas leads with 1,234.", ["SELECT 1"])
    assert report.question == "question?" and report.session_id == "s1" and report.created_at


def test_every_read_is_filtered_by_owner(store):
    a = _save(store, "alice", "Revenue Q1")
    b = _save(store, "bob", "Revenue Q1")
    assert [r.report_id for r in store.list("alice")] == [a]
    assert [r.report_id for r in store.search("alice", "revenue")] == [a]
    assert [r.report_id for r in store.get_many("alice", [a, b])] == [a]
    assert store.get_many("alice", []) == []


def test_search_is_case_insensitive_all_words_and_session_filter(store):
    r1 = _save(store, "alice", "Revenue in TEXAS", session="s1")
    r2 = _save(store, "alice", "Returns", body="return rate in texas", session="s2")
    _save(store, "alice", "Margins", session="s2")
    assert {r.report_id for r in store.search("alice", "texas")} == {r1, r2}
    assert [r.report_id for r in store.search("alice", "Texas revenue")] == [r1]
    assert [r.report_id for r in store.search("alice", "texas", session_id="s2")] == [r2]
    assert len(store.search("alice")) == 3
    assert store.search("alice", "%") == []  # no LIKE wildcards
    # Non-ASCII titles fold too (SQLite lower() would not).
    r4 = _save(store, "alice", "ВЫРУЧКА по штатам")
    assert [r.report_id for r in store.search("alice", "выручка")] == [r4]


def test_delete_is_permanent_owner_only_and_audited_without_content(store):
    a1 = _save(store, "alice", "A1")
    a2 = _save(store, "alice", "A2")
    b1 = _save(store, "bob", "B1")

    assert store.delete("alice", [a1, b1, 999], trace_id="t1") == 1
    assert [r.report_id for r in store.list("alice")] == [a2]
    assert [r.report_id for r in store.list("bob")] == [b1]  # bob's report untouched
    assert store.get_many("alice", [a1]) == []  # gone for good: no recovery window (client's rule)
    assert store.search("alice", "A1") == []

    # Deleting again is a no-op.
    assert store.delete("alice", [a1], trace_id="t2") == 0
    log = store.audit_log("alice")
    entries = [(e["action"], e["report_ids"], e["trace_id"]) for e in log]
    assert entries == [("delete", [a1], "t1"), ("delete", [], "t2")]
    assert set(log[0]) == {"id", "ts", "owner_id", "action", "report_ids", "trace_id"}  # no titles or bodies kept
    assert store.audit_log("bob") == []
    assert store.delete("alice", [], trace_id="t3") == 0


def test_persists_across_instances(tmp_path):
    path = tmp_path / "r.sqlite"
    rid = _save(ReportStore(path), "alice", "Kept")
    assert [r.report_id for r in ReportStore(path).list("alice")] == [rid]
