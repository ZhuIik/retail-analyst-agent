"""Gemini model factory with fallbacks and rate limiting (D17, H15).

    def build_analyst_model(settings, tools, faults=None, on_event=None) -> Runnable
    def classify_llm_error(exc) -> ErrorKind
    def quota_reset_hint() -> str

Built from LangChain's own pieces (D17): `ChatGoogleGenerativeAI`, a shared
`InMemoryRateLimiter`, `bind_tools` and `with_fallbacks`. What we add is a thin wrapper per
model that injects faults, reports `llm_call` / `fallback` events and turns errors into a
classified AgentError — which is also what decides whether the next model is tried.

Facts about the installed langchain-google-genai 4.x that this module relies on:
- 4xx surface as `GoogleRateLimitError` / `GoogleAuthenticationError` / ... raised *from* the
  google-genai `ClientError` (code, status, details JSON); 5xx as `GoogleAPIError`, itself a
  google-genai `ServerError`. Network timeouts pass through as raw `httpx` exceptions.
- `max_retries` is the SDK's *total attempts*: 1 means no SDK retry, so a 429 reaches the
  fallback immediately instead of backing off on the exhausted model (trap, D17).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
from google.genai import errors as genai_errors
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from retail_analyst.bq.client import LOCATION_BLOCKED_MESSAGE
from retail_analyst.config import Settings
from retail_analyst.contracts import AgentError, ErrorKind
from retail_analyst.faults import FaultInjector

OnEvent = Callable[[str, dict], None]

# Another model has its own quota and availability (D17); a bad key or a region block fails the same way.
_FALLBACK_KINDS = {ErrorKind.TRANSIENT, ErrorKind.QUOTA_EXHAUSTED}
_LOCATION_MARKER = "User location is not supported"  # Gemini answers 400 FAILED_PRECONDITION (section 6, 9a)
_INVALID_KEY_MARKERS = ("API_KEY_INVALID", "API key not valid")  # Gemini answers 400, not 401, for a bad key


class FallbackableLLMError(AgentError):
    """A model failed in a way another model may not (429 / 5xx / timeout): `with_fallbacks` handles it."""


def quota_reset_hint() -> str:
    return "Daily quota resets at midnight Pacific time."


def build_analyst_model(
    settings: Settings,
    tools: list,
    faults: FaultInjector | None = None,
    on_event: OnEvent | None = None,
) -> Runnable:
    names = [settings.models.analyst, *settings.models.fallbacks]
    rpm = settings.models.requests_per_minute
    # One limiter for all models: the free-tier limits are per project (section 6).
    limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=1)
    limiter.available_tokens = 1.0  # the limiter starts empty; don't make the first question wait 60/rpm s
    wrapped = []
    for i, name in enumerate(names):
        # bind_tools on EVERY model before with_fallbacks, or the fallback runs without tools (trap, D17).
        bound = _make_chat_model(settings, name, limiter).bind_tools(tools, tool_choice="any")
        next_name = names[i + 1] if i + 1 < len(names) else None
        wrapped.append(_observed(bound, name, next_name, is_primary=i == 0, faults=faults, on_event=on_event))
    if len(wrapped) == 1:
        return wrapped[0]
    return wrapped[0].with_fallbacks(wrapped[1:], exceptions_to_handle=(FallbackableLLMError,))


def _make_chat_model(settings: Settings, name: str, limiter: InMemoryRateLimiter) -> BaseChatModel:
    m = settings.models
    optional = {"temperature": m.temperature} if m.temperature is not None else {}
    return ChatGoogleGenerativeAI(
        model=name,
        api_key=settings.google_api_key or None,
        timeout=m.request_timeout_s,
        max_retries=m.max_sdk_retries,
        rate_limiter=limiter,
        **optional,
    )


def _observed(
    bound: Runnable,
    name: str,
    next_name: str | None,
    *,
    is_primary: bool,
    faults: FaultInjector | None,
    on_event: OnEvent | None,
) -> Runnable:
    def emit(event_type: str, payload: dict) -> None:
        if on_event:
            on_event(event_type, payload)

    def call(messages: Any, config: RunnableConfig) -> Any:
        started = time.monotonic()
        try:
            if faults:
                faults.before_llm_call(name, is_primary=is_primary)
            message = bound.invoke(messages, config)
        except AgentError:
            raise
        except Exception as exc:
            kind = classify_llm_error(exc)
            if kind is ErrorKind.UNKNOWN:
                raise  # likely a bug or a bad request: keep the original traceback
            detail = f"{name}: {type(exc).__name__}: {str(exc)[:1000]}"
            if kind in _FALLBACK_KINDS and next_name:
                # The cause (429 per-minute vs 503 overload) matters when debugging why fallbacks happen.
                emit(
                    "fallback",
                    {"from_model": name, "to_model": next_name, "error_kind": kind.value, "detail": detail[:300]},
                )
                raise FallbackableLLMError(kind, _user_message(kind), detail) from exc
            raise AgentError(kind, _user_message(kind), detail) from exc
        usage = getattr(message, "usage_metadata", None) or {}
        emit(
            "llm_call",
            {
                "model": name,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )
        return message

    return RunnableLambda(call, name=name)


def _user_message(kind: ErrorKind) -> str:
    return {
        ErrorKind.QUOTA_EXHAUSTED: f"The Gemini daily quota is exhausted. {quota_reset_hint()}",
        ErrorKind.TRANSIENT: "Gemini is temporarily unavailable or rate-limited. Please try again in a minute.",
        ErrorKind.AUTH: "Gemini rejected the API key. Check GOOGLE_API_KEY in .env (run `analyst-doctor`).",
        ErrorKind.LOCATION_BLOCKED: LOCATION_BLOCKED_MESSAGE,
    }.get(kind, "The language model failed unexpectedly.")


def classify_llm_error(exc: BaseException) -> ErrorKind:
    if isinstance(exc, AgentError):
        return exc.kind
    chain = list(_cause_chain(exc))
    text = " ".join(str(e) for e in chain)  # the genai error text includes the response JSON
    if _LOCATION_MARKER in text:
        return ErrorKind.LOCATION_BLOCKED
    if any(marker in text for marker in _INVALID_KEY_MARKERS):
        return ErrorKind.AUTH
    code = next((e.code for e in chain if isinstance(e, genai_errors.APIError)), None)
    if code == 429 or any(isinstance(e, ModelRateLimitError) for e in chain):
        # Per-minute limits pass in seconds; a per-day quota id (e.g. ...PerDayPerProject...) does not.
        return ErrorKind.QUOTA_EXHAUSTED if "PerDay" in text else ErrorKind.TRANSIENT
    if code in (401, 403) or any(isinstance(e, ModelAuthenticationError | ModelPermissionDeniedError) for e in chain):
        return ErrorKind.AUTH
    transient_types = (TimeoutError, httpx.TransportError, ModelAPIError, ModelTimeoutError, ModelConnectionError)
    if (code is not None and (code >= 500 or code == 408)) or any(isinstance(e, transient_types) for e in chain):
        return ErrorKind.TRANSIENT
    return ErrorKind.UNKNOWN


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__
