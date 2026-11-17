"""Hosted analyst CLI. Python 3.11+, standard library only; no Google credentials needed."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SERVER = "https://194.59.244.59"
HELP = """Ask about sales, customers and products in your brand scope.
Examples:
  Compare revenue for my brands over the last 3 full months.
  Save that as a report.
  Delete my report about monthly revenue.
Commands: /reports  /trace  /new  /help  /quit
Deletion always shows a preview and requires your explicit reply.
"""


class ClientError(Exception):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward bearer/Gemini credentials to a redirected destination.
        return None


class AnalystClient:
    def __init__(self, server: str, token: str = "", gemini_key: str = "") -> None:
        url = urllib.parse.urlsplit(server)
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ClientError("Use a server URL without credentials, query parameters or fragments.")
        local = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if url.scheme != "https" and not (url.scheme == "http" and local):
            raise ClientError("HTTPS is required, except for a localhost SSH tunnel.")
        self.server, self.token, self.gemini_key = server.rstrip("/"), token.strip(), gemini_key.strip()
        self.session_id: str | None = None
        self.opener = urllib.request.build_opener(NoRedirects())

    def request(self, path: str, payload: dict | None = None, *, first_turn: bool = False) -> Any:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode()
        if first_turn and self.gemini_key:
            headers["X-Gemini-Api-Key"] = self.gemini_key
        req = urllib.request.Request(self.server + path, data=data, headers=headers)
        try:
            with self.opener.open(req, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            messages = {
                401: "Access expired or invalid. Restart the client and select a demo user again.",
                404: "Session expired or the server restarted. Use /new to start a new conversation.",
                503: "The service is unavailable. Use /new if the conversation cannot be continued.",
            }
            if exc.code == 429:
                # The API's bounded detail distinguishes a busy server from the daily demo quota.
                try:
                    detail = json.loads(exc.read(4096)).get("detail", "The server is busy; retry later.")
                except (ValueError, OSError):
                    detail = "The server is busy; retry later."
                raise ClientError(str(detail)) from None
            raise ClientError(messages.get(exc.code, f"Server returned HTTP {exc.code}.")) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ClientError(
                "Connection failed. Check the server address/network. The request may have completed; "
                "do not automatically repeat a deletion confirmation."
            ) from None
        except (ValueError, UnicodeError):
            raise ClientError("The server returned an invalid response.") from None

    def ask(self, message: str) -> dict:
        payload: dict[str, Any] = {"message": message}
        if self.session_id:
            payload["session_id"] = self.session_id
        result = self.request("/v1/chat", payload, first_turn=self.session_id is None)
        self.session_id = result["session_id"]
        return result

    def select_demo_user(self, selected: str | None = None) -> None:
        choices = self.request("/v1/demo/users")
        if not choices:
            raise ClientError("No demo users are available.")
        if selected is None:
            print("Choose a demo user (simulated login):")
            for index, user in enumerate(choices, 1):
                scope = user["access"]["values"]
                label = ", ".join(scope) if scope else "All brands"
                print(terminal_text(f"  {index}. {user['display_name']} | {label}"))
            while True:
                choice = input("User [1]: ").strip() or "1"
                if choice.isdigit() and 1 <= int(choice) <= len(choices):
                    selected = choices[int(choice) - 1]["id"]
                    break
                if choice.lower() in {u["id"] for u in choices}:
                    selected = choice.lower()
                    break
                print("Enter a listed number or user name.")
        if selected not in {u["id"] for u in choices}:
            raise ClientError("Unknown demo user. Choose anna, mark or ceo.")
        result = self.request("/v1/demo/access", {"user": selected})
        self.token = result["token"]
        print(terminal_text("Demo user: " + result["display_name"]))


def terminal_text(value: Any) -> str:
    # Model/data text must not control the reviewer's terminal.
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(value))


def load_access(path: Path) -> dict[str, str]:
    """The private review bundle carries a demo token beside this script."""
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 65536:
            raise ValueError
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or any(
            not isinstance(data.get(key), str) or not data[key].strip() for key in ("server", "token")
        ):
            raise ValueError
        return {key: data[key].strip() for key in ("server", "token")}
    except (OSError, ValueError):
        raise ClientError("Cannot read the demo access file. Ask the author for a fresh review bundle.") from None


def connection_settings(server: str | None, access: dict[str, str]) -> tuple[str, str]:
    target = server or os.getenv("ANALYST_SERVER_URL") or access.get("server") or DEFAULT_SERVER
    token = os.getenv("ANALYST_TOKEN", "")
    # A URL override must not send a bundled credential to a different server.
    if not token and access.get("server", "").rstrip("/") == target.rstrip("/"):
        token = access.get("token", "")
    return target, token


def chat_loop(client: AnalystClient) -> None:
    pending = False
    print("Retail Analyst | hosted CLI\n" + HELP)
    while True:
        try:
            text = input("confirm> " if pending else "you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"/quit", "/exit"}:
            return
        if text == "/help":
            print(HELP)
            continue
        if text == "/new":
            client.session_id, pending = None, False
            print("Started a new conversation.")
            continue
        if not text and not pending:
            continue
        try:
            if text == "/reports":
                reports = client.request("/v1/reports")
                for report in reports:
                    print(terminal_text(f"#{report['report_id']}  {report['title']}  {report['created_at']}"))
                if not reports:
                    print("No saved reports yet.")
                continue
            if text == "/trace":
                if not client.session_id:
                    print("No trace yet.")
                    continue
                events = client.request(f"/v1/sessions/{client.session_id}/trace")
                print(terminal_text(json.dumps(events, ensure_ascii=False, indent=2)))
                continue
            print("analyst> thinking...", flush=True)
            result = client.ask(text)
            confirmation = result.get("confirmation")
            pending = bool(confirmation)
            if confirmation:
                print("Reports to delete:")
                for report in confirmation["preview"]:
                    print(terminal_text(f"#{report['report_id']}  {report['title']}  {report['created_at']}"))
                required = confirmation["required_answer"]
                print(f"Deletion is permanent. Type {required} to confirm; any other reply cancels.")
            elif result.get("answer"):
                print("analyst> " + terminal_text(result["answer"]))
            print("trace " + terminal_text(result.get("trace_id") or "unavailable"))
        except ClientError as exc:
            print(terminal_text(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the hosted Retail Analyst; no deployment needed.")
    parser.add_argument("--server", help="override the hosted server URL")
    parser.add_argument("--user", choices=["anna", "mark", "ceo"], help="select a demo user without the menu")
    parser.add_argument("--access-file", type=Path, default=Path(__file__).with_name("demo-access.json"))
    parser.add_argument(
        "--own-gemini-key", action="store_true", help="use your Gemini key instead of the server default"
    )
    args = parser.parse_args()
    try:
        server, token = connection_settings(args.server, load_access(args.access_file))
        client = AnalystClient(server, token if not args.user else "")
        if not client.token:
            client.select_demo_user(args.user)
        key = ""
        if args.own_gemini_key:
            key = os.getenv("ANALYST_GEMINI_KEY") or getpass.getpass("Your Gemini API key: ")
            if not key.strip():
                raise ClientError("A personal Gemini key is required when --own-gemini-key is selected.")
        client.gemini_key = key.strip()
        chat_loop(client)
    except (ClientError, KeyboardInterrupt, EOFError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
