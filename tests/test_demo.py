"""Public persona selection cannot open another visitor's reports or bypass the daily cap."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from retail_analyst.api import create_app
from retail_analyst.config import load_users
from retail_analyst.demo import DemoAccess
from retail_analyst.reports.store import ReportStore
from tests.fakes import make_settings, submit
from tests.test_api import KEYS, auth, factory


def test_public_demo_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ANALYST_DEMO_ENABLED", raising=False)
    with TestClient(create_app(api_keys=KEYS)) as c:
        assert c.get("/v1/demo/users").status_code == 404
        assert c.post("/v1/demo/access", json={"user": "anna"}).status_code == 404


def enable_demo(monkeypatch, tmp_path, limit=50):
    settings = make_settings(db_path=str(tmp_path / "reports.sqlite"))
    monkeypatch.setenv("ANALYST_DEMO_ENABLED", "true")
    monkeypatch.setenv("ANALYST_DEMO_DAILY_TURNS", str(limit))
    monkeypatch.setattr("retail_analyst.api.get_settings", lambda: settings)


def test_persona_selection_isolates_even_visitors_with_same_persona(tmp_path, monkeypatch):
    enable_demo(monkeypatch, tmp_path)
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [submit("Hello")], sessions))
    store = ReportStore(tmp_path / "reports.sqlite")
    store.save("anna", "Existing private report", "body", "q", [], "s")
    with TestClient(app) as c:
        assert [u["id"] for u in c.get("/v1/demo/users").json()] == ["anna", "mark", "ceo"]
        assert c.post("/v1/demo/access", json={"user": "arbitrary-admin"}).status_code == 400
        a = c.post("/v1/demo/access", json={"user": "anna"}).json()["token"]
        b = c.post("/v1/demo/access", json={"user": "anna"}).json()["token"]
        ha, hb = ({"Authorization": "Bearer " + token} for token in (a, b))
        sid = c.post("/v1/chat", headers=ha, json={"message": "hi"}).json()["session_id"]
        assert c.post("/v1/chat", headers=hb, json={"message": "hi", "session_id": sid}).status_code == 404
        assert c.get(f"/v1/sessions/{sid}/trace", headers=hb).status_code == 404
        assert c.get("/v1/reports", headers=ha).json() == []
        visitor = sessions[0][0].user
        assert visitor.access == load_users()["anna"].access
        assert visitor.user_id != "anna"
        store.save(visitor.user_id, "My demo report", "body", "q", [], "s")
        assert len(c.get("/v1/reports", headers=ha).json()) == 1
        assert c.get("/v1/reports", headers=hb).json() == []
        assert c.get("/v1/reports", headers=auth()).json()[0]["title"] == "Existing private report"


def test_quota_survives_new_visitor_and_server_restart_before_session_start(tmp_path, monkeypatch):
    enable_demo(monkeypatch, tmp_path, limit=1)
    sessions = []
    make = factory(tmp_path, [submit("Hello")], sessions)
    with TestClient(create_app(api_keys=KEYS, session_factory=make)) as c:
        token = c.post("/v1/demo/access", json={"user": "ceo"}).json()["token"]
        headers = {"Authorization": "Bearer " + token}
        assert c.post("/v1/chat", headers=headers, json={"message": "hello"}).status_code == 200
        assert c.post("/v1/chat", headers=headers, json={"message": "again"}).status_code == 429
    with TestClient(create_app(api_keys=KEYS, session_factory=make)) as c:
        token = c.post("/v1/demo/access", json={"user": "mark"}).json()["token"]
        denied = c.post("/v1/chat", headers={"Authorization": "Bearer " + token}, json={"message": "hi"})
        assert denied.status_code == 429
        assert "daily limit" in denied.json()["detail"]
    assert len(sessions) == 1  # no session constructor/BQ startup after quota exhaustion


def test_daily_budget_is_atomic_across_threads(tmp_path):
    gate = DemoAccess(load_users(), tmp_path / "demo.sqlite", daily_limit=3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: gate.charge(), range(12)))
    assert sum(results) == 3


def test_expired_visitor_cannot_resume(tmp_path, monkeypatch):
    gate = DemoAccess(load_users(), tmp_path / "demo.sqlite")
    token, _ = gate.issue("anna")
    now = __import__("time").monotonic()
    monkeypatch.setattr("retail_analyst.demo.time.monotonic", lambda: now + 3601)
    assert gate.resolve(token) is None


def test_demo_budget_requires_positive_limit(tmp_path):
    with pytest.raises(ValueError):
        DemoAccess(load_users(), tmp_path / "demo.sqlite", daily_limit=0)
