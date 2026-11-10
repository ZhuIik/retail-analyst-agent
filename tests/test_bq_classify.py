"""BigQuery error classification, retries and result conversion — no network (fake client)."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
from decimal import Decimal
from types import SimpleNamespace

import google.auth.exceptions
import pytest
import requests
from google.api_core import exceptions as ex

from retail_analyst.bq import client as bq_module
from retail_analyst.bq.client import LOCATION_BLOCKED_MESSAGE, BigQueryClient, classify_bq_error, sanitize_labels
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

SALT = "s3cr3t-salt-value"
SQL = "WITH scoped_orders AS (SELECT FARM_FINGERPRINT(CONCAT('s3cr3t-salt-value', user_id)) AS c FROM o) SELECT 1"


def make_settings() -> Settings:
    return Settings(
        dataset="bigquery-public-data.thelook_ecommerce",
        models=ModelSettings(analyst="primary"),
        agent=AgentSettings(),
        sql=SqlSettings(max_transient_retries=3),
        reports=ReportSettings(),
        tracing=TracingSettings(),
        id_salt=SALT,
    )


def html_403() -> ex.GoogleAPICallError:
    """What a region block looks like: Google's front end answers with an HTML page (section 6, 9a)."""
    response = requests.Response()
    response.status_code = 403
    response._content = (
        b"<!DOCTYPE html><html><title>Error 403 (Forbidden)!!1</title><p><b>403.</b> That\xe2\x80\x99s an error."
        b"<p>Your client does not have permission to get URL <code>/bigquery/v2/projects/p/jobs</code> "
        b"from this server.</html>"
    )
    response.request = requests.Request("POST", "https://bigquery.googleapis.com/bigquery/v2/projects/p/jobs").prepare()
    return ex.from_http_response(response)


def api_error(cls: type[ex.GoogleAPICallError], reason: str, message: str = "boom") -> ex.GoogleAPICallError:
    return cls(message, errors=[{"reason": reason, "message": message}])


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        # Verified live: the bytes cap comes back as HTTP 500 — must not be treated as transient.
        (api_error(ex.InternalServerError, "bytesBilledLimitExceeded"), ErrorKind.TOO_EXPENSIVE),
        (api_error(ex.Forbidden, "responseTooLarge"), ErrorKind.TOO_EXPENSIVE),
        (api_error(ex.BadRequest, "invalidQuery"), ErrorKind.INVALID_SQL),
        (api_error(ex.BadRequest, "invalid"), ErrorKind.INVALID_SQL),
        (api_error(ex.NotFound, "notFound"), ErrorKind.INVALID_SQL),
        (api_error(ex.TooManyRequests, "rateLimitExceeded"), ErrorKind.TRANSIENT),
        (api_error(ex.InternalServerError, "backendError"), ErrorKind.TRANSIENT),
        (api_error(ex.InternalServerError, "internalError"), ErrorKind.TRANSIENT),
        (ex.InternalServerError("no reason given"), ErrorKind.TRANSIENT),
        (ex.ServiceUnavailable("down"), ErrorKind.TRANSIENT),
        (ex.BadGateway("bad gateway"), ErrorKind.TRANSIENT),
        (ex.GatewayTimeout("gateway timeout"), ErrorKind.TRANSIENT),
        (ex.DeadlineExceeded("deadline"), ErrorKind.TRANSIENT),
        (concurrent.futures.TimeoutError(), ErrorKind.TRANSIENT),
        (requests.exceptions.ConnectionError("reset"), ErrorKind.TRANSIENT),
        (api_error(ex.Forbidden, "quotaExceeded"), ErrorKind.QUOTA_EXHAUSTED),
        (api_error(ex.Forbidden, "accessDenied"), ErrorKind.AUTH),
        (ex.Unauthorized("bad token"), ErrorKind.AUTH),
        (google.auth.exceptions.DefaultCredentialsError("no ADC"), ErrorKind.AUTH),
        (html_403(), ErrorKind.LOCATION_BLOCKED),
        (api_error(ex.InternalServerError, "someNewReason"), ErrorKind.UNKNOWN),
        (ValueError("bug"), ErrorKind.UNKNOWN),
        (AgentError(ErrorKind.AUTH, "x"), ErrorKind.AUTH),
    ],
)
def test_classify_bq_error(exc: BaseException, kind: ErrorKind) -> None:
    assert classify_bq_error(exc) is kind


# --- fake BigQuery client ---------------------------------------------------------------


class FakeRows:
    def __init__(self, columns: list[str], rows: list[tuple], total: int | None = None) -> None:
        self.schema = [SimpleNamespace(name=c) for c in columns]
        self._rows = rows
        self.total_rows = len(rows) if total is None else total

    def __iter__(self):
        return iter(SimpleNamespace(values=lambda r=r: r) for r in self._rows)


class FakeJob:
    def __init__(self, rows: FakeRows | None = None, result_error: BaseException | None = None) -> None:
        self.job_id = "job_1"
        self.total_bytes_processed = 1234
        self.total_bytes_billed = 10_485_760
        self._rows = rows or FakeRows(["n"], [(1,)])
        self._result_error = result_error
        self.cancelled = False
        self.result_kwargs: dict = {}

    def result(self, **kwargs):
        self.result_kwargs = kwargs
        if self._result_error:
            raise self._result_error
        return self._rows

    def cancel(self) -> None:
        self.cancelled = True


class FakeBQ:
    """Each query() call consumes the next scripted outcome: a job or an exception."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def query(self, sql, job_config=None, **kwargs):
        self.calls.append({"sql": sql, "config": job_config, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def get_table(self, ref, **kwargs):
        field = SimpleNamespace(name="order_id", field_type="INTEGER", mode="NULLABLE", description=None)
        return SimpleNamespace(schema=[field])


def make_client(
    monkeypatch, outcomes: list, faults: FaultInjector | None = None
) -> tuple[BigQueryClient, FakeBQ, list]:
    fake = FakeBQ(outcomes)
    monkeypatch.setattr(bq_module.bigquery, "Client", lambda project=None: fake)
    client = BigQueryClient(make_settings(), faults)
    sleeps: list[float] = []
    client._sleep = sleeps.append
    return client, fake, sleeps


# --- retries ------------------------------------------------------------------------------


def test_transient_error_is_retried_then_succeeds(monkeypatch) -> None:
    client, fake, sleeps = make_client(monkeypatch, [ex.ServiceUnavailable("down"), FakeJob()])
    result = client.execute("SELECT 1", {"user": "u1"})
    assert result.meta.attempts == 2
    assert len(fake.calls) == 2 and len(sleeps) == 1
    assert result.data.rows == [[1]]


def test_timeout_is_retried_and_timed_out_job_cancelled(monkeypatch) -> None:
    stuck = FakeJob(result_error=concurrent.futures.TimeoutError())
    client, fake, _ = make_client(monkeypatch, [stuck, FakeJob()])
    assert client.execute("SELECT 1", {}).meta.attempts == 2
    assert stuck.cancelled


def test_bytes_billed_limit_is_not_retried(monkeypatch) -> None:
    job = FakeJob(result_error=api_error(ex.InternalServerError, "bytesBilledLimitExceeded", "Query exceeded limit"))
    client, fake, sleeps = make_client(monkeypatch, [job])
    with pytest.raises(AgentError) as info:
        client.execute("SELECT 1", {})
    assert info.value.kind is ErrorKind.TOO_EXPENSIVE
    assert "too expensive" in info.value.user_message.lower()
    assert len(fake.calls) == 1 and sleeps == []


def test_invalid_sql_is_not_retried(monkeypatch) -> None:
    client, fake, sleeps = make_client(monkeypatch, [api_error(ex.BadRequest, "invalidQuery", "Unrecognized name: x")])
    with pytest.raises(AgentError) as info:
        client.dry_run("SELECT x")
    assert info.value.kind is ErrorKind.INVALID_SQL
    assert "Unrecognized name: x" in info.value.user_message
    assert len(fake.calls) == 1 and sleeps == []


def test_transient_retries_are_bounded(monkeypatch) -> None:
    client, fake, sleeps = make_client(monkeypatch, [ex.ServiceUnavailable("down")] * 10)
    with pytest.raises(AgentError) as info:
        client.execute("SELECT 1", {})
    assert info.value.kind is ErrorKind.TRANSIENT
    assert len(fake.calls) == 4  # 1 attempt + max_transient_retries
    assert sum(sleeps) <= 30


def test_salt_and_sql_never_leak_into_messages(monkeypatch) -> None:
    error = api_error(ex.BadRequest, "invalidQuery", f"Bad thing near '{SALT}' in query: {SQL}")
    client, _, _ = make_client(monkeypatch, [error])
    with pytest.raises(AgentError) as info:
        client.execute(SQL, {})
    for text in (info.value.user_message, info.value.detail, str(info.value)):
        assert SALT not in text
        assert SQL not in text


# --- results, labels, faults --------------------------------------------------------------------


def test_execute_returns_json_safe_rows_and_meta(monkeypatch) -> None:
    rows = FakeRows(
        ["day", "ts", "revenue", "blob", "n"],
        [(dt.date(2026, 1, 2), dt.datetime(2026, 1, 2, 3, 4, 5), Decimal("12.50"), b"\x00\x01", None)],
        total=900,
    )
    client, fake, _ = make_client(monkeypatch, [FakeJob(rows)])
    result = client.execute("SELECT 1", {"User": "Brand.Manager@X", "trace_id": "AbC123"})
    assert result.data.columns == ["day", "ts", "revenue", "blob", "n"]
    assert result.data.rows == [["2026-01-02", "2026-01-02T03:04:05", 12.5, "AAE=", None]]
    assert result.meta.rows_total == 900
    assert result.meta.bytes_billed == 10_485_760 and result.meta.job_id == "job_1"
    config = fake.calls[0]["config"]
    assert config.maximum_bytes_billed == 1_000_000_000
    assert config.labels == {"user": "brand_manager_x", "trace_id": "abc123"}
    assert fake.calls[0]["job_retry"] is None  # job-level retries are ours


def test_dry_run_returns_bytes(monkeypatch) -> None:
    client, fake, _ = make_client(monkeypatch, [FakeJob()])
    assert client.dry_run("SELECT 1") == 1234
    assert fake.calls[0]["config"].dry_run is True


def test_table_schema(monkeypatch) -> None:
    client, _, _ = make_client(monkeypatch, [])
    assert client.table_schema("orders") == [
        {"name": "order_id", "type": "INTEGER", "mode": "NULLABLE", "description": ""}
    ]


def test_sanitize_labels() -> None:
    labels = sanitize_labels({"1st Key": "X" * 100, "ok_key": "Valid-value_1"})
    assert labels == {"l_1st_key": "x" * 63, "ok_key": "valid-value_1"}


def test_fault_bq_timeout_shows_retries(monkeypatch) -> None:
    client, _, sleeps = make_client(monkeypatch, [FakeJob()], FaultInjector({"bq_timeout"}))
    assert client.execute("SELECT 1", {}).meta.attempts == 3
    assert len(sleeps) == 2


def test_fault_bq_quota_is_fatal(monkeypatch) -> None:
    client, fake, sleeps = make_client(monkeypatch, [FakeJob()], FaultInjector({"bq_quota"}))
    with pytest.raises(AgentError) as info:
        client.execute("SELECT 1", {})
    assert info.value.kind is ErrorKind.QUOTA_EXHAUSTED and info.value.kind.is_fatal
    assert sleeps == [] and fake.calls == []


def test_fault_empty_returns_zero_rows(monkeypatch) -> None:
    client, _, _ = make_client(monkeypatch, [FakeJob()], FaultInjector({"empty"}))
    result = client.execute("SELECT 1", {})
    assert result.data.rows == [] and result.meta.rows_total == 0
    assert result.data.columns == ["n"]


def test_location_blocked_message(monkeypatch) -> None:
    client, _, _ = make_client(monkeypatch, [html_403()])
    with pytest.raises(AgentError) as info:
        client.dry_run("SELECT 1")
    assert info.value.kind is ErrorKind.LOCATION_BLOCKED
    assert info.value.user_message == LOCATION_BLOCKED_MESSAGE


def test_missing_credentials_become_auth_error(monkeypatch) -> None:
    def no_adc(project=None):
        raise google.auth.exceptions.DefaultCredentialsError("Could not automatically determine credentials")

    monkeypatch.setattr(bq_module.bigquery, "Client", no_adc)
    with pytest.raises(AgentError) as info:
        BigQueryClient(make_settings())
    assert info.value.kind is ErrorKind.AUTH
