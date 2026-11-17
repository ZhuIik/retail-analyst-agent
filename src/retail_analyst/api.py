"""Single-process HTTP adapter. Keys map to server-owned users; sessions expire in memory."""

import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr

from retail_analyst.cli import Session
from retail_analyst.config import Settings, get_settings, load_users
from retail_analyst.contracts import UserContext
from retail_analyst.demo import DemoAccess
from retail_analyst.reports.store import ReportStore
from retail_analyst.safety.pii import redact_pii

log = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
SESSION_TTL_S = 3600
MAX_SESSIONS = 128


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    trace_id: str | None
    answer: str | None
    confirmation: dict[str, Any] | None


class DemoUserRequest(BaseModel):
    user: str = Field(min_length=1, max_length=32)


class HttpSession(Session):
    """Use the same turn/reset/resume logic as the CLI, without terminal output."""

    def _run(self, payload: object) -> None:
        for chunk in self.graph.stream(payload, self.config, stream_mode="updates"):
            if pauses := chunk.get("__interrupt__"):
                self.awaiting = pauses[0].value

    def _fail(self, message: str, detail: str) -> None:
        self.tracer.event("error", source="api", detail=detail[:500])
        raise RuntimeError(message)


@dataclass
class Conversation:
    owner: str
    session: Any
    touched: float


def _configured_keys(users: dict[str, UserContext]) -> dict[str, str]:
    settings = get_settings()  # loads .env before reading the API key mapping
    try:
        keys = json.loads(os.environ.get("ANALYST_API_KEYS", "{}"))
    except ValueError as exc:
        raise RuntimeError("ANALYST_API_KEYS must be a JSON object mapping user ids to keys") from exc
    if (
        not isinstance(keys, dict)
        or not keys
        or any(uid not in users or not isinstance(key, str) or len(key) < 32 for uid, key in keys.items())
        or len(set(keys.values())) != len(keys)
    ):
        raise RuntimeError("Configure unique API keys of at least 32 characters for known users")
    if settings.id_salt == "dev-only-salt":
        raise RuntimeError("Set a private ID_SALT before serving the HTTP API")
    return keys


def create_app(
    *,
    api_keys: dict[str, str] | None = None,
    session_factory: Callable[[UserContext, Settings], Any] | None = None,
) -> FastAPI:
    users = load_users()
    conversations: dict[UUID, Conversation] = {}
    # Serialize turns across users to bound demo quota and prevent checkpoint/resume races.
    # FastAPI runs these sync endpoints in a thread pool, keeping /healthz responsive.
    busy = threading.Lock()
    keys: dict[str, str] = {}
    demo: DemoAccess | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal demo
        keys.update(api_keys if api_keys is not None else _configured_keys(users))
        if os.getenv("ANALYST_DEMO_ENABLED", "").lower() == "true":
            settings = get_settings()
            demo = DemoAccess(
                users, settings.resolve(settings.reports.db_path).with_name("demo-budget.sqlite"),
                daily_limit=int(os.getenv("ANALYST_DEMO_DAILY_TURNS", "50")),
            )
        try:
            yield
        finally:
            for item in conversations.values():
                item.session.langfuse.flush()

    app = FastAPI(title="Retail Analyst API", version="0.1.0", lifespan=lifespan)

    def identity(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> UserContext:
        token = credentials.credentials if credentials else ""
        if demo is not None and (visitor := demo.resolve(token)) is not None:
            return visitor
        for uid, key in keys.items():
            if secrets.compare_digest(token.encode(), key.encode()):
                return users[uid]
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/healthz")
    def health() -> dict[str, str]:
        # Process/configuration liveness, deliberately no paid Google API calls.
        return {"status": "ok"}

    @app.get("/v1/demo/users")
    def demo_users() -> list[dict]:
        if demo is None:
            raise HTTPException(404, "Public demo is disabled")
        return demo.choices()

    @app.post("/v1/demo/access")
    def demo_access(body: DemoUserRequest) -> dict:
        if demo is None:
            raise HTTPException(404, "Public demo is disabled")
        try:
            token, user = demo.issue(body.user)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"token": token, "display_name": user.display_name, "expires_in": 3600}

    @app.get("/v1/reports")
    def reports(user: Annotated[UserContext, Depends(identity)]) -> list[dict]:
        settings = get_settings()
        store = ReportStore(settings.resolve(settings.reports.db_path))
        return [{**r.model_dump(), "title": redact_pii(r.title)[0]} for r in store.list(user.user_id)]

    @app.get("/v1/sessions/{session_id}/trace")
    def trace(session_id: UUID, user: Annotated[UserContext, Depends(identity)]) -> list[dict]:
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "Another turn is running; retry later", headers={"Retry-After": "5"})
        try:
            item = conversations.get(session_id)
            if item is None or item.owner != user.user_id or time.monotonic() - item.touched > SESSION_TTL_S:
                raise HTTPException(404, "Session not found or expired")
            return item.session.tracer.last_turn()
        finally:
            busy.release()

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        user: Annotated[UserContext, Depends(identity)],
        gemini_api_key: Annotated[SecretStr | None, Header(
            alias="X-Gemini-Api-Key",
            description="Optional personal Gemini key; send only when creating a new session. "
                        "Omit to use the server key. This is separate from the bearer login key.",
        )] = None,
    ) -> ChatResponse:
        if not body.message.strip():
            raise HTTPException(422, "Message cannot be blank")
        supplied_key = gemini_api_key.get_secret_value().strip() if gemini_api_key is not None else None
        if supplied_key is not None and (not supplied_key or len(supplied_key) > 512):
            raise HTTPException(400, "Personal Gemini key must be nonempty and at most 512 characters")
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "Another turn is running; retry later", headers={"Retry-After": "5"})
        try:
            now = time.monotonic()
            for sid, item in list(conversations.items()):
                if now - item.touched > SESSION_TTL_S:
                    item.session.langfuse.flush()
                    del conversations[sid]
            if demo is not None and user.user_id.startswith("demo-") and not demo.charge():
                raise HTTPException(429, "The shared public demo daily limit is reached; try again tomorrow (UTC)")
            sid = body.session_id
            if sid is not None:
                item = conversations.get(sid)
                if item is None or item.owner != user.user_id:
                    raise HTTPException(404, "Session not found or expired")
                if supplied_key is not None:
                    raise HTTPException(400, "Set a personal Gemini key only when creating a new session")
            else:
                if len(conversations) >= MAX_SESSIONS:
                    raise HTTPException(503, "Session capacity reached; retry after sessions expire")
                sid = uuid4()
                # Copy cached settings: a reviewer's key must never replace the server default.
                settings = get_settings().model_copy(deep=True)
                if supplied_key is not None:
                    settings.google_api_key = supplied_key
                from retail_analyst.runtime import ProcessSession

                factory = session_factory or ProcessSession
                try:
                    item = Conversation(user.user_id, factory(user, settings), now)
                except Exception as exc:
                    log.warning("API session startup failed (%s)", type(exc).__name__)
                    raise HTTPException(503, "Cannot start conversation; check server credentials") from exc
                conversations[sid] = item
            try:
                item.session.ask(body.message)
            except Exception as exc:
                # A failed checkpoint must never leave a confirmation available to retry.
                del conversations[sid]
                log.warning("API turn failed (%s)", type(exc).__name__)
                raise HTTPException(503, "Conversation failed; start a new session") from exc
            item.touched = time.monotonic()
            state = item.session.graph.get_state(item.session.config).values
            confirmation = item.session.awaiting
            if confirmation:
                confirmation = {
                    **confirmation,
                    "preview": [{**r, "title": redact_pii(r["title"])[0]} for r in confirmation["preview"]],
                }
            return ChatResponse(
                session_id=sid,
                trace_id=item.session.tracer.trace_id,
                answer=None if confirmation else state.get("final_text"),
                confirmation=confirmation,
            )
        finally:
            busy.release()

    return app


app = create_app()
