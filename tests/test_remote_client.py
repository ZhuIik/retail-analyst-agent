"""Hosted CLI transport, key isolation and explicit deletion confirmation. No network."""

import io
import json
import urllib.error
from unittest.mock import Mock

import pytest

from client import AnalystClient, ClientError, chat_loop, connection_settings, load_access, terminal_text


def response(**fields):
    return io.BytesIO(json.dumps(fields).encode())


@pytest.mark.parametrize("url", ["http://example.com", "https://user:pass@example.com", "https://example.com?token=x"])
def test_rejects_unsafe_urls(url):
    with pytest.raises(ClientError):
        AnalystClient(url, "access")


def test_personal_key_only_sent_when_session_is_created():
    client = AnalystClient("https://example.com", "access", "personal-gemini")
    client.opener.open = Mock(side_effect=[response(session_id="s1"), response(session_id="s1")])
    client.ask("hello")
    client.ask("continue")
    first, second = [c.args[0] for c in client.opener.open.call_args_list]
    assert first.headers["X-gemini-api-key"] == "personal-gemini"
    assert "X-gemini-api-key" not in second.headers
    assert json.loads(first.data) == {"message": "hello"}
    assert json.loads(second.data) == {"message": "continue", "session_id": "s1"}


def test_network_failure_is_not_retried_or_repeated_as_confirmation():
    client = AnalystClient("https://example.com", "access")
    client.opener.open = Mock(side_effect=urllib.error.URLError("timeout"))
    with pytest.raises(ClientError, match="may have completed"):
        client.ask("yes")
    assert client.opener.open.call_count == 1


def test_cli_does_not_auto_confirm_deletion(monkeypatch, capsys):
    client = AnalystClient("https://example.com", "access")
    client.ask = Mock(side_effect=[
        {"session_id": "s1", "trace_id": "t1", "confirmation": {
            "preview": [{"report_id": 7, "title": "Revenue", "created_at": "2026-11-17"}],
            "required_answer": "yes",
        }},
        {"session_id": "s1", "trace_id": "t1", "answer": "Cancelled", "confirmation": None},
    ])
    answers = iter(["delete revenue", "no", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    chat_loop(client)
    assert [c.args[0] for c in client.ask.call_args_list] == ["delete revenue", "no"]
    shown = capsys.readouterr().out
    assert "#7  Revenue" in shown and "Deletion is permanent" in shown and "Cancelled" in shown


def test_terminal_text_cannot_emit_control_sequences():
    assert "\x1b" not in terminal_text("\x1b[2Jhello\x00\x07")


def test_private_access_file_is_found_without_using_working_directory(tmp_path, monkeypatch):
    path = tmp_path / "demo-access.json"
    path.write_text(json.dumps({"server": "https://example.com", "token": "demo-secret"}))
    monkeypatch.delenv("ANALYST_TOKEN", raising=False)
    monkeypatch.delenv("ANALYST_SERVER_URL", raising=False)
    assert connection_settings(None, load_access(path)) == ("https://example.com", "demo-secret")


def test_server_override_does_not_leak_bundled_key(monkeypatch):
    monkeypatch.delenv("ANALYST_TOKEN", raising=False)
    access = {"server": "https://example.com", "token": "demo-secret"}
    assert connection_settings("https://different.example.com", access) == ("https://different.example.com", "")


@pytest.mark.parametrize("content", ['{"token":"private-value"}', '[]', 'not json'])
def test_invalid_access_file_error_does_not_echo_its_contents(tmp_path, content):
    path = tmp_path / "demo-access.json"
    path.write_text(content)
    with pytest.raises(ClientError) as exc:
        load_access(path)
    assert "private-value" not in str(exc.value)


def test_cli_selects_user_without_secret_prompt(monkeypatch, capsys):
    client = AnalystClient("https://example.com")
    client.opener.open = Mock(side_effect=[
        io.BytesIO(json.dumps([
            {"id": "anna", "display_name": "Anna", "access": {"values": ["Calvin Klein"]}},
            {"id": "ceo", "display_name": "CEO", "access": {"values": []}},
        ]).encode()),
        response(token="temporary-session-token", display_name="CEO", expires_in=3600),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    client.select_demo_user()
    assert client.token == "temporary-session-token"
    request = client.opener.open.call_args_list[1].args[0]
    assert json.loads(request.data) == {"user": "ceo"}
    assert "Authorization" not in request.headers
    assert "temporary-session-token" not in capsys.readouterr().out
