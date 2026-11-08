"""Fault injection for demos and tests (`--simulate`, D17 / Б1).

Each mode raises the exception type the real library raises for that failure, so the real
classification, retry and fallback code paths run — nothing in the agent knows it is a drill.

    bq_timeout  first 2 BigQuery attempts time out, then succeed   -> shows code-level retries
    bq_quota    BigQuery answers 403 quotaExceeded                 -> fatal, honest message
    llm_429     the primary model answers 429 on every call         -> the fallback model answers
    llm_down    every model answers 503                             -> honest "LLM unavailable"
    empty       BigQuery returns zero rows                          -> empty-result diagnosis (D13)
"""

from __future__ import annotations

import concurrent.futures

from google.api_core import exceptions as api_exceptions

FAULT_MODES = {"bq_timeout", "bq_quota", "llm_429", "llm_down", "empty"}

_BQ_TIMEOUT_ATTEMPTS = 2  # below settings.sql.max_transient_retries, so the query still succeeds


class FaultInjector:
    def __init__(self, modes: set[str]) -> None:
        unknown = set(modes) - FAULT_MODES
        if unknown:
            raise ValueError(
                f"Unknown fault mode(s): {', '.join(sorted(unknown))}. Valid modes: {', '.join(sorted(FAULT_MODES))}"
            )
        self.modes = frozenset(modes)
        self._bq_calls = 0

    def active(self, mode: str) -> bool:
        return mode in self.modes

    def before_bq_call(self) -> None:
        """Called before every BigQuery execution attempt."""
        self._bq_calls += 1
        if self.active("bq_timeout") and self._bq_calls <= _BQ_TIMEOUT_ATTEMPTS:
            # QueryJob.result(timeout=...) raises exactly this when the job does not finish in time.
            raise concurrent.futures.TimeoutError(
                f"[simulated] query did not finish in time (attempt {self._bq_calls})"
            )
        if self.active("bq_quota"):
            # What QueryJob raises for a job failing with reason quotaExceeded (mapped to HTTP 403).
            message = "[simulated] Quota exceeded: Your project exceeded quota for free query bytes scanned."
            raise api_exceptions.Forbidden(message, errors=[{"reason": "quotaExceeded", "message": message}])

    def before_llm_call(self, model: str, *, is_primary: bool = True) -> None:
        """Called before every chat-model call. `is_primary` is False for fallback models."""
        if self.active("llm_down"):
            raise _gemini_server_error(model)
        if self.active("llm_429") and is_primary:
            raise _gemini_rate_limit_error(model)

    def empty_result(self) -> bool:
        return self.active("empty")


# Imported lazily: the BigQuery path must not pay for importing the LLM stack.


def _gemini_rate_limit_error(model: str) -> Exception:
    """What langchain-google-genai 4.x raises on a per-minute 429: GoogleRateLimitError from a ClientError."""
    from google.genai.errors import ClientError
    from langchain_google_genai.chat_models import GoogleRateLimitError

    body = {
        "error": {
            "code": 429,
            "message": "[simulated] You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
                }
            ],
        }
    }
    cause = ClientError(429, body)
    error = GoogleRateLimitError(f"Error calling model '{model}' ({cause.status}): {cause}")
    error.__cause__ = cause
    return error


def _gemini_server_error(model: str) -> Exception:
    """What langchain-google-genai 4.x raises on a 5xx: GoogleAPIError (a google.genai ServerError)."""
    from langchain_google_genai.chat_models import GoogleAPIError

    body = {"error": {"code": 503, "message": f"[simulated] {model} is overloaded.", "status": "UNAVAILABLE"}}
    return GoogleAPIError(code=503, response_json=body)
