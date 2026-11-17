"""Opt-in public demo identities, isolated visitors and a persistent shared daily quota."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from retail_analyst.contracts import UserContext

DEMO_USERS = ("anna", "mark", "ceo")


class DemoAccess:
    def __init__(self, users: dict[str, UserContext], path: Path, daily_limit: int = 50) -> None:
        if daily_limit < 1:
            raise ValueError("Demo daily limit must be positive")
        self.users = users
        self.path, self.daily_limit = path, daily_limit
        self._tokens: dict[str, tuple[UserContext, float]] = {}
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS daily_budget (day TEXT PRIMARY KEY, used INTEGER NOT NULL)")

    def choices(self) -> list[dict]:
        return [
            {"id": uid, "display_name": self.users[uid].display_name, "access": self.users[uid].access.model_dump()}
            for uid in DEMO_USERS if uid in self.users
        ]

    def issue(self, uid: str) -> tuple[str, UserContext]:
        if uid not in DEMO_USERS or uid not in self.users:
            raise ValueError("Unknown demo user")
        now = time.monotonic()
        with self._lock:
            self._tokens = {key: item for key, item in self._tokens.items() if item[1] > now}
            if len(self._tokens) >= 128:
                raise OverflowError("Demo visitor capacity reached; try again later")
            user = self.users[uid].model_copy(update={"user_id": f"demo-{uuid4().hex}"})
            token = secrets.token_urlsafe(32)
            self._tokens[token] = (user, now + 3600)
        return token, user

    def resolve(self, token: str) -> UserContext | None:
        with self._lock:
            item = self._tokens.get(token)
            if item is None:
                return None
            if item[1] <= time.monotonic():
                del self._tokens[token]
                return None
            return item[0]

    def charge(self) -> bool:
        """Atomically reserve a turn before any model/BQ work; survives process restarts."""
        day = datetime.now(UTC).date().isoformat()
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "INSERT INTO daily_budget(day, used) VALUES (?, 1) "
                "ON CONFLICT(day) DO UPDATE SET used = used + 1 WHERE used < ? RETURNING used",
                (day, self.daily_limit),
            ).fetchone()
        return row is not None
