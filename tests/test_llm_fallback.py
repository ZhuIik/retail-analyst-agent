"""Model factory: tools on every model, fallback only on 429/5xx, error classification (D17)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai.chat_models import GoogleAuthenticationError, GoogleRateLimitError
from pydantic import BaseModel

from retail_analyst import llm
from retail_analyst.config import (
    AgentSettings,
    ModelSettings,
    ReportSettings,
    Settings,
    SqlSettings,
    TracingSettings,
)
from retail_analyst.contracts import AgentError, ErrorKind
from retail_analyst.faults import FaultInjector


class SubmitAnswer(BaseModel):
    """Final answer tool."""

    message: str


class FakeChat(BaseChatModel):
    """Answers with a fixed message, or raises `error`. Records tool binding and calls."""

    model_name: str
    error: Any = None
    calls: int = 0
    bound: dict = {}

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound = {"tools": tools, "tool_choice": tool_choice}
        return self.bind(**kwargs)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        message = AIMessage(
            content=f"answer from {self.model_name}",
            usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_settings() -> Settings:
    return Settings(
        dataset="d",
        models=ModelSettings(analyst="primary", fallbacks=["backup"], requests_per_minute=600),
        agent=AgentSettings(),
        sql=SqlSettings(),
        reports=ReportSettings(),
        tracing=TracingSettings(),
        google_api_key="fake-key",
    )


def gemini_client_error(code: int, message: str, status: str, details: list | None = None) -> ClientError:
    return ClientError(code, {"error": {"code": code, "message": message, "status": status, "details": details or []}})


def rate_limit(quota_id: str) -> GoogleRateLimitError:
    """Shaped like langchain-google-genai 4.x: GoogleRateLimitError raised from the SDK ClientError."""
    violations = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    cause = gemini_client_error(429, "You exceeded your current quota.", "RESOURCE_EXHAUSTED", violations)
    error = GoogleRateLimitError(f"Error calling model 'primary' (RESOURCE_EXHAUSTED): {cause}")
    error.__cause__ = cause
    return error


LOCATION = gemini_client_error(400, "User location is not supported for the API use.", "FAILED_PRECONDITION")


@pytest.fixture
def models(monkeypatch) -> dict[str, FakeChat]:
    fakes = {"primary": FakeChat(model_name="primary"), "backup": FakeChat(model_name="backup")}
    limiters = []

    def factory(settings, name, limiter):
        limiters.append(limiter)
        return fakes[name]

    monkeypatch.setattr(llm, "_make_chat_model", factory)
    fakes["limiters"] = limiters  # type: ignore[assignment]
    return fakes


def build(faults: FaultInjector | None = None) -> tuple[Any, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    model = llm.build_analyst_model(make_settings(), [SubmitAnswer], faults, lambda t, p: events.append((t, p)))
    return model, events


def test_tools_bound_on_every_model_with_mode_any(models) -> None:
    build()
    for name in ("primary", "backup"):
        assert models[name].bound == {"tools": [SubmitAnswer], "tool_choice": "any"}
    first, second = models["limiters"]
    assert first is second  # one shared rate limiter


def test_primary_answers_and_reports_llm_call(models) -> None:
    model, events = build()
    assert model.invoke([HumanMessage("hi")]).content == "answer from primary"
    assert events == [
        (
            "llm_call",
            {"model": "primary", "latency_ms": events[0][1]["latency_ms"], "input_tokens": 10, "output_tokens": 3},
        )
    ]


def test_simulated_429_falls_back(models) -> None:
    model, events = build(FaultInjector({"llm_429"}))
    assert model.invoke([HumanMessage("hi")]).content == "answer from backup"
    assert events[0] == ("fallback", {"from_model": "primary", "to_model": "backup", "error_kind": "transient"})
    assert events[1][0] == "llm_call" and events[1][1]["model"] == "backup"
    assert models["primary"].calls == 0  # the fault fires before the request


def test_daily_quota_falls_back(models) -> None:
    models["primary"].error = rate_limit("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    model, events = build()
    assert model.invoke([HumanMessage("hi")]).content == "answer from backup"
    assert events[0][1]["error_kind"] == "quota_exhausted"


def test_auth_error_does_not_fall_back(models) -> None:
    cause = gemini_client_error(401, "Request had invalid authentication credentials.", "UNAUTHENTICATED")
    error = GoogleAuthenticationError(f"Error calling model 'primary': {cause}")
    error.__cause__ = cause
    models["primary"].error = error
    model, events = build()
    with pytest.raises(AgentError) as info:
        model.invoke([HumanMessage("hi")])
    assert info.value.kind is ErrorKind.AUTH
    assert models["backup"].calls == 0 and events == []


def test_location_block_does_not_fall_back(models) -> None:
    models["primary"].error = LOCATION
    model, _ = build()
    with pytest.raises(AgentError) as info:
        model.invoke([HumanMessage("hi")])
    assert info.value.kind is ErrorKind.LOCATION_BLOCKED
    assert models["backup"].calls == 0


def test_all_models_down_gives_honest_transient_error(models) -> None:
    model, events = build(FaultInjector({"llm_down"}))
    with pytest.raises(AgentError) as info:
        model.invoke([HumanMessage("hi")])
    assert info.value.kind is ErrorKind.TRANSIENT
    assert llm.classify_llm_error(info.value) is ErrorKind.TRANSIENT
    assert [t for t, _ in events] == ["fallback"]


def test_unknown_errors_propagate_unchanged(models) -> None:
    models["primary"].error = ValueError("bug in message conversion")
    model, _ = build()
    with pytest.raises(ValueError):
        model.invoke([HumanMessage("hi")])


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (rate_limit("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), ErrorKind.TRANSIENT),
        (rate_limit("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), ErrorKind.QUOTA_EXHAUSTED),
        (
            ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}),
            ErrorKind.TRANSIENT,
        ),
        (httpx.ReadTimeout("timed out"), ErrorKind.TRANSIENT),
        (TimeoutError(), ErrorKind.TRANSIENT),
        (LOCATION, ErrorKind.LOCATION_BLOCKED),
        (
            gemini_client_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"),
            ErrorKind.AUTH,
        ),
        (gemini_client_error(403, "Permission denied", "PERMISSION_DENIED"), ErrorKind.AUTH),
        (gemini_client_error(404, "models/x is not found", "NOT_FOUND"), ErrorKind.UNKNOWN),
        (ValueError("bug"), ErrorKind.UNKNOWN),
    ],
)
def test_classify_llm_error(exc: BaseException, kind: ErrorKind) -> None:
    assert llm.classify_llm_error(exc) is kind


def test_real_chat_model_settings() -> None:
    """The real factory wires SDK retries, timeout and the limiter (constructed only, no request)."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(requests_per_second=1)
    chat = llm._make_chat_model(make_settings(), "gemini-x", limiter)
    assert chat.max_retries == 1 and chat.timeout == 60 and chat.rate_limiter is limiter


def test_quota_reset_hint() -> None:
    assert "Pacific" in llm.quota_reset_hint()
