"""HTTP auth, owner isolation, real graph interrupt/resume and capacity. No Google calls."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from retail_analyst.api import HttpSession, _configured_keys, create_app
from retail_analyst.config import load_users
from retail_analyst.tracing import Tracer
from tests.fakes import call, make_settings, submit
from tests.test_graph_scenarios import Harness

KEYS = {"anna": "a" * 32, "mark": "m" * 32}


def auth(user="anna"):
    return {"Authorization": f"Bearer {KEYS[user]}"}


def factory(tmp_path, scripts, sessions):
    def make(user, settings):
        h = Harness(tmp_path / str(len(sessions)), list(scripts), user=user)
        s = HttpSession.__new__(HttpSession)
        s.user, s.settings = user, settings
        s.session_id = f"http-{len(sessions)}"
        s.tracer = Tracer(tmp_path / "traces", s.session_id, user.user_id)
        s.langfuse = SimpleNamespace(run_config=lambda *args: {}, flush=lambda: None)
        s.graph = h.graph
        s.thread, s.started, s.awaiting = s.session_id, False, None
        sessions.append((s, h))
        return s
    return make


def test_auth_and_validation_do_not_start_paid_session(tmp_path):
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [], sessions))
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.post("/v1/chat", json={"message": "hi"}).status_code == 401
        assert c.post("/v1/chat", headers=auth(), json={"message": " "}).status_code == 422
        assert c.post("/v1/chat", headers=auth(), json={"message": "x" * 8001}).status_code == 422
        assert c.post("/v1/chat", headers=auth(), json={"message": "x", "session_id": "../x"}).status_code == 422
        assert sessions == []


def test_http_turn_and_owner_isolation(tmp_path):
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [submit("Hello")], sessions))
    with TestClient(app) as c:
        r = c.post("/v1/chat", headers=auth(), json={"message": "hi"})
        assert r.status_code == 200
        result = r.json()
        assert result["answer"] == "Hello" and result["trace_id"]
        assert result["confirmation"] is None
        r = c.post("/v1/chat", headers=auth("mark"), json={"message": "yes", "session_id": result["session_id"]})
        assert r.status_code == 404
        assert len(sessions) == 1


@pytest.mark.parametrize("reply,deleted", [("yes", True), ("no", False), ("yes please", False)])
def test_real_graph_confirmation_over_http(tmp_path, reply, deleted):
    sessions = []
    script = [submit("Ready"), call("delete_reports", text="weekly"), submit("Finished")]
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, script, sessions))
    with TestClient(app) as c:
        sid = c.post("/v1/chat", headers=auth(), json={"message": "hi"}).json()["session_id"]
        s, h = sessions[0]
        report = h.save("Weekly bob@example.com", owner="anna")
        paused = c.post("/v1/chat", headers=auth(), json={"message": "delete weekly", "session_id": sid}).json()
        assert paused["answer"] is None
        assert paused["confirmation"]["preview"][0]["report_id"] == report
        assert "bob@example.com" not in str(paused)
        assert h.active_ids("anna") == {report}
        assert c.post("/v1/chat", headers=auth("mark"), json={"message": "yes", "session_id": sid}).status_code == 404
        result = c.post("/v1/chat", headers=auth(), json={"message": reply, "session_id": sid})
        assert result.status_code == 200
        assert result.json()["answer"] == "Finished"
        assert h.active_ids("anna") == (set() if deleted else {report})


def test_busy_api_rejects_parallel_turns_and_health_stays_available(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow(user, settings):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("simulated startup failure")

    app = create_app(api_keys=KEYS, session_factory=slow)
    with TestClient(app, raise_server_exceptions=False) as c:
        t = threading.Thread(target=lambda: c.post("/v1/chat", headers=auth(), json={"message": "hi"}))
        t.start()
        try:
            assert entered.wait(timeout=3)
            assert c.get("/healthz").status_code == 200
            assert c.post("/v1/chat", headers=auth(), json={"message": "hi"}).status_code == 429
        finally:
            release.set()
            t.join(timeout=5)


def test_expired_session_cannot_resume(tmp_path, monkeypatch):
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [submit("Hello")], sessions))
    with TestClient(app) as c:
        sid = c.post("/v1/chat", headers=auth(), json={"message": "hi"}).json()["session_id"]
        monkeypatch.setattr("retail_analyst.api.SESSION_TTL_S", -1)
        assert c.post("/v1/chat", headers=auth(), json={"message": "yes", "session_id": sid}).status_code == 404


@pytest.mark.parametrize("mapping", ["{}", "[]", "invalid", '{"unknown":"' + "x" * 32 + '"}',
                                      '{"anna":"short"}', '{"anna":"' + "x" * 32 + '","mark":"' + "x" * 32 + '"}'])
def test_startup_rejects_missing_weak_unknown_or_duplicate_keys(monkeypatch, mapping):
    monkeypatch.setattr("retail_analyst.api.get_settings", lambda: SimpleNamespace(id_salt="private"))
    monkeypatch.setenv("ANALYST_API_KEYS", mapping)
    with pytest.raises(RuntimeError):
        _configured_keys(load_users())


def test_dotenv_is_loaded_before_api_keys_are_read(monkeypatch):
    def settings():
        monkeypatch.setenv("ANALYST_API_KEYS", json_mapping)
        return SimpleNamespace(id_salt="private")
    json_mapping = '{"anna":"' + "x" * 32 + '"}'
    monkeypatch.delenv("ANALYST_API_KEYS", raising=False)
    monkeypatch.setattr("retail_analyst.api.get_settings", settings)
    assert _configured_keys(load_users()) == {"anna": "x" * 32}


def test_startup_rejects_development_salt(monkeypatch):
    monkeypatch.setattr("retail_analyst.api.get_settings", lambda: SimpleNamespace(id_salt="dev-only-salt"))
    monkeypatch.setenv("ANALYST_API_KEYS", '{"anna":"' + "x" * 32 + '"}')
    with pytest.raises(RuntimeError, match="ID_SALT"):
        _configured_keys(load_users())


def test_personal_gemini_key_is_session_scoped_and_does_not_replace_server_default(tmp_path, monkeypatch):
    sessions = []
    default_settings = make_settings()
    default_settings.google_api_key = "server-gemini-secret"
    monkeypatch.setattr("retail_analyst.api.get_settings", lambda: default_settings)
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [submit("Hello"), submit("Again")], sessions))
    with TestClient(app) as c:
        headers = {**auth(), "X-Gemini-Api-Key": "reviewer-gemini-secret"}
        r = c.post("/v1/chat", headers=headers, json={"message": "hi"})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert sessions[0][0].settings.google_api_key == "reviewer-gemini-secret"
        # No header on continuation: retain this session's key rather than falling back to ours.
        follow_up = c.post("/v1/chat", headers=auth(), json={"message": "again", "session_id": sid})
        assert follow_up.status_code == 200
        assert sessions[0][0].settings.google_api_key == "reviewer-gemini-secret"
        assert "reviewer-gemini-secret" not in r.text + follow_up.text
        assert c.post("/v1/chat", headers=auth(), json={"message": "hi"}).status_code == 200
        assert sessions[1][0].settings.google_api_key == "server-gemini-secret"
        assert default_settings.google_api_key == "server-gemini-secret"
        blocked = c.post("/v1/chat", headers=headers, json={"message": "again", "session_id": sid})
        assert blocked.status_code == 400
        assert sessions[0][0].settings.google_api_key == "reviewer-gemini-secret"


@pytest.mark.parametrize("key", [" ", "x" * 513])
def test_bad_personal_key_rejected_without_echoing_or_starting_session(tmp_path, key):
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [], sessions))
    with TestClient(app) as c:
        r = c.post("/v1/chat", headers={**auth(), "X-Gemini-Api-Key": key}, json={"message": "hi"})
        assert r.status_code == 400
        assert sessions == []
        if key.strip():
            assert key not in r.text


def test_remote_reports_are_owner_scoped_and_titles_redacted(tmp_path, monkeypatch):
    from retail_analyst.reports.store import ReportStore

    settings = make_settings(db_path=str(tmp_path / "reports.sqlite"))
    monkeypatch.setattr("retail_analyst.api.get_settings", lambda: settings)
    store = ReportStore(tmp_path / "reports.sqlite")
    store.save("anna", "Anna bob@example.com", "body", "q", [], "s")
    store.save("mark", "Private Mark", "body", "q", [], "s")
    with TestClient(create_app(api_keys=KEYS)) as c:
        assert c.get("/v1/reports").status_code == 401
        data = c.get("/v1/reports", headers=auth()).json()
        assert len(data) == 1
        assert data[0]["title"] == "Anna [redacted]"
        assert "body" not in data[0]


def test_remote_trace_is_owner_scoped_and_expires(tmp_path, monkeypatch):
    sessions = []
    app = create_app(api_keys=KEYS, session_factory=factory(tmp_path, [submit("Hello")], sessions))
    with TestClient(app) as c:
        sid = c.post("/v1/chat", headers=auth(), json={"message": "hi"}).json()["session_id"]
        url = f"/v1/sessions/{sid}/trace"
        assert c.get(url).status_code == 401
        assert c.get(url, headers=auth("mark")).status_code == 404
        result = c.get(url, headers=auth())
        assert result.status_code == 200 and result.json()[0]["type"] == "turn_start"
        monkeypatch.setattr("retail_analyst.api.SESSION_TTL_S", -1)
        assert c.get(url, headers=auth()).status_code == 404
