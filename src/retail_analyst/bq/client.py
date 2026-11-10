"""BigQuery execution layer, extending the assignment's starter BigQueryRunner.

    class BigQueryClient:
        def __init__(self, settings: Settings, faults: FaultInjector | None = None) -> None
        def dry_run(self, sql: str) -> int                      # bytes that would be processed
        def execute(self, sql: str, labels: dict[str, str]) -> ExecResult
        def table_schema(self, table: str) -> list[dict]        # used by `doctor` only

    def classify_bq_error(exc: BaseException) -> ErrorKind

Every failure leaves this module as a contracts.AgentError with a classified ErrorKind.
Only TRANSIENT errors are retried, here in code; the library's own job-level retries
are switched off so that there is exactly one, bounded, retry policy.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import random
import re
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypeVar

import google.auth.exceptions
import requests
from google.api_core import exceptions as api_exceptions
from google.cloud import bigquery

from retail_analyst.config import Settings
from retail_analyst.contracts import AgentError, ErrorKind, ExecResult, QueryMeta, TableData
from retail_analyst.faults import FaultInjector

T = TypeVar("T")

LOCATION_BLOCKED_MESSAGE = (
    "Google APIs are not available from your current region/network — use a VPN that routes all traffic."
)
# Region blocks reach BigQuery as a Google front-end HTML page, not a JSON API error (section 6, check 9a).
_LOCATION_BLOCK_MARKER = "Your client does not have permission to get URL"

# Classified by the error *reason*, not the HTTP code: bytesBilledLimitExceeded arrives as
# HTTP 500 and would otherwise be retried as a server error (section 6, check 2).
_REASON_KIND = {
    "bytesBilledLimitExceeded": ErrorKind.TOO_EXPENSIVE,
    "responseTooLarge": ErrorKind.TOO_EXPENSIVE,
    "invalidQuery": ErrorKind.INVALID_SQL,
    "invalid": ErrorKind.INVALID_SQL,
    "notFound": ErrorKind.INVALID_SQL,
    "rateLimitExceeded": ErrorKind.TRANSIENT,
    "backendError": ErrorKind.TRANSIENT,
    "internalError": ErrorKind.TRANSIENT,
    "quotaExceeded": ErrorKind.QUOTA_EXHAUSTED,
    "accessDenied": ErrorKind.AUTH,
}
_TRANSIENT_TYPES = (
    TimeoutError,  # concurrent.futures.TimeoutError is an alias since Python 3.11
    ConnectionError,
    api_exceptions.DeadlineExceeded,
    api_exceptions.ServiceUnavailable,
    api_exceptions.BadGateway,
    api_exceptions.GatewayTimeout,
    api_exceptions.TooManyRequests,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    google.auth.exceptions.TransportError,
)
_AUTH_TYPES = (google.auth.exceptions.DefaultCredentialsError, google.auth.exceptions.RefreshError)

# Library-level retry of single HTTP calls (e.g. a blip while polling), with a short deadline instead
# of the default 10 minutes. Job-level retries are ours (job_retry=None everywhere).
_API_RETRY = bigquery.DEFAULT_RETRY.with_deadline(15.0)
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 8.0
_MAX_TOTAL_WAIT_S = 30.0
_MAX_MESSAGE_CHARS = 500


def classify_bq_error(exc: BaseException) -> ErrorKind:
    if isinstance(exc, AgentError):
        return exc.kind
    if _LOCATION_BLOCK_MARKER in str(exc):
        return ErrorKind.LOCATION_BLOCKED
    if isinstance(exc, _AUTH_TYPES):
        return ErrorKind.AUTH
    if isinstance(exc, api_exceptions.GoogleAPICallError):
        for reason in _reasons(exc):
            if reason in _REASON_KIND:
                return _REASON_KIND[reason]
        if exc.code in (401, 403):
            return ErrorKind.AUTH
        # A 500 with an unknown reason is NOT assumed transient: retrying what we do not
        # understand is exactly how bytesBilledLimitExceeded would have been mishandled.
        if exc.code == 500 and not _reasons(exc):
            return ErrorKind.TRANSIENT
    if isinstance(exc, _TRANSIENT_TYPES):
        return ErrorKind.TRANSIENT
    return ErrorKind.UNKNOWN


def _reasons(exc: api_exceptions.GoogleAPICallError) -> list[str]:
    return [e["reason"] for e in exc.errors if isinstance(e, dict) and e.get("reason")]


class BigQueryClient:
    def __init__(self, settings: Settings, faults: FaultInjector | None = None) -> None:
        self._settings = settings
        self._faults = faults
        try:
            self._client = bigquery.Client(project=settings.gcp_project_id or None)
        except Exception as exc:  # typically DefaultCredentialsError: no ADC on this machine
            raise self._to_agent_error(exc, classify_bq_error(exc), 1, "") from exc
        self._sleep: Callable[[float], None] = time.sleep  # replaced in tests

    def dry_run(self, sql: str) -> int:
        """Validate syntax/types and return the bytes the query would process (free)."""
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job, _ = self._with_retries(
            lambda: self._client.query(sql, job_config=config, retry=_API_RETRY, job_retry=None), sql
        )
        return int(job.total_bytes_processed or 0)

    def execute(self, sql: str, labels: dict[str, str]) -> ExecResult:
        sql_settings = self._settings.sql
        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=sql_settings.max_bytes_billed,  # BigQuery itself enforces the cost cap
            labels=sanitize_labels(labels),
        )

        def attempt() -> tuple[bigquery.QueryJob, Any]:
            if self._faults:
                self._faults.before_bq_call()
            job = self._client.query(sql, job_config=config, retry=_API_RETRY, job_retry=None)
            try:
                rows = job.result(
                    timeout=sql_settings.query_timeout_s,
                    max_results=sql_settings.stored_rows,
                    retry=_API_RETRY,
                    job_retry=None,
                )
            except TimeoutError:
                _cancel_quietly(job)  # the job keeps running (and billing) server-side otherwise
                raise
            return job, rows

        started = time.monotonic()
        (job, rows), attempts = self._with_retries(attempt, sql)
        columns = [field.name for field in rows.schema]
        data = [[_json_safe(value) for value in row.values()] for row in rows]
        rows_total = rows.total_rows if rows.total_rows is not None else len(data)
        if self._faults and self._faults.empty_result():
            data, rows_total = [], 0
        meta = QueryMeta(
            job_id=job.job_id,
            bytes_processed=job.total_bytes_processed or 0,
            bytes_billed=job.total_bytes_billed or 0,
            rows_total=rows_total,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
        )
        return ExecResult(data=TableData(columns=columns, rows=data), meta=meta)

    def table_schema(self, table: str) -> list[dict]:
        ref = f"{self._settings.dataset}.{table}"
        tbl, _ = self._with_retries(lambda: self._client.get_table(ref, retry=_API_RETRY), "")
        return [
            {"name": f.name, "type": f.field_type, "mode": f.mode, "description": f.description or ""}
            for f in tbl.schema
        ]

    # ----------------------------------------------------------------------------------

    def _with_retries(self, call: Callable[[], T], sql: str) -> tuple[T, int]:
        """Run `call`, retrying only TRANSIENT errors with exponential backoff + jitter.

        Returns (result, attempts). Anything else — or a transient error once the retry or
        total-wait budget is spent — leaves as AgentError.
        """
        attempts, waited = 0, 0.0
        while True:
            attempts += 1
            try:
                return call(), attempts
            except Exception as exc:
                kind = classify_bq_error(exc)
                delay = min(_BACKOFF_BASE_S * 2 ** (attempts - 1), _BACKOFF_MAX_S) * random.uniform(0.5, 1.0)
                retry = (
                    kind is ErrorKind.TRANSIENT
                    and attempts <= self._settings.sql.max_transient_retries
                    and waited + delay <= _MAX_TOTAL_WAIT_S
                )
                if not retry:
                    raise self._to_agent_error(exc, kind, attempts, sql) from exc
                self._sleep(delay)
                waited += delay

    def _to_agent_error(self, exc: BaseException, kind: ErrorKind, attempts: int, sql: str) -> AgentError:
        if isinstance(exc, AgentError):
            return exc
        reason_message = self._clean(_bq_message(exc), sql)
        mb_cap = self._settings.sql.max_bytes_billed / 1e6
        messages = {
            ErrorKind.TOO_EXPENSIVE: (
                f"Query too expensive: it would bill more than the {mb_cap:,.0f} MB per-query cap. "
                "Add filters (dates, categories) or aggregate before joining."
            ),
            ErrorKind.INVALID_SQL: f"BigQuery rejected the query: {reason_message}",
            ErrorKind.TRANSIENT: f"BigQuery is temporarily unavailable (gave up after {attempts} attempts).",
            ErrorKind.QUOTA_EXHAUSTED: "BigQuery quota for this project is exhausted; try again later.",
            ErrorKind.AUTH: (
                "BigQuery credentials are missing or lack access. Run "
                "`gcloud auth application-default login` and set GCP_PROJECT_ID in .env."
            ),
            ErrorKind.LOCATION_BLOCKED: LOCATION_BLOCKED_MESSAGE,
        }
        user_message = messages.get(kind, f"BigQuery error: {reason_message}")
        detail = self._clean(f"{type(exc).__name__}: {exc} (attempts={attempts})", sql)
        return AgentError(kind, user_message, detail)

    def _clean(self, text: str, sql: str) -> str:
        """Never leak the salt or the scoped SQL (the model must not see the rewrite)."""
        if sql:
            text = text.replace(sql, "[query]")
        if self._settings.id_salt:
            text = text.replace(self._settings.id_salt, "[salt]")
        return text[:_MAX_MESSAGE_CHARS]


def _bq_message(exc: BaseException) -> str:
    """The most specific human message: the job error's own message, without HTTP noise."""
    if isinstance(exc, api_exceptions.GoogleAPICallError):
        for error in exc.errors:
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
        return str(exc.message)
    return str(exc) or type(exc).__name__


def _cancel_quietly(job: Any) -> None:
    with contextlib.suppress(Exception):  # best effort: the timeout is what we report
        job.cancel()


_LABEL_BAD_CHARS = re.compile(r"[^a-z0-9_-]")


def sanitize_labels(labels: dict[str, str]) -> dict[str, str]:
    """BigQuery labels: lowercase [a-z0-9_-], <= 63 chars; keys must start with a letter."""
    clean: dict[str, str] = {}
    for key, value in labels.items():
        k = _LABEL_BAD_CHARS.sub("_", str(key).lower())[:63]
        if not k or not k[0].isalpha():
            k = ("l_" + k)[:63]
        clean[k] = _LABEL_BAD_CHARS.sub("_", str(value).lower())[:63]
    return clean


def _json_safe(value: Any) -> Any:
    """Rows go into graph state (checkpointer), so they must be plain JSON values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return str(value)
