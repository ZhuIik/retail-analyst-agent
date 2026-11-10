"""run_sql post-processing: small demographic groups are hidden (k-anonymity). No network."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from retail_analyst.config import load_users
from retail_analyst.contracts import ErrorKind, ExecResult, QueryMeta, TableData
from retail_analyst.sql.demographics import drop_small_groups
from retail_analyst.tools.data_tools import DescribeDataArgs, RunSqlArgs, describe_data, run_sql
from retail_analyst.tools.registry import ToolContext
from tests.fakes import FakeTracer, make_settings

SETTINGS = make_settings()
K = SETTINGS.sql.min_group_size
BY_STATE = "SELECT u.state, COUNT(DISTINCT u.id) AS customers FROM users u GROUP BY 1"


class Tracer(FakeTracer):
    trace_id = "t1"


class FakeBigQuery:
    """Returns a fixed result for any query and records what it was asked to run."""

    def __init__(self, columns: list[str], rows: list[list[Any]]) -> None:
        self.data = TableData(columns=columns, rows=rows)
        self.executed: list[str] = []

    def dry_run(self, sql: str) -> int:
        return 1_000

    def execute(self, sql: str, labels: dict[str, str]) -> ExecResult:
        self.executed.append(sql)
        return ExecResult(data=self.data, meta=QueryMeta(job_id="job-1", rows_total=len(self.data.rows)))


def make_ctx(bq: FakeBigQuery) -> ToolContext:
    return ToolContext(
        user=load_users()["ceo"],
        settings=SETTINGS,
        session_id="s1",
        trace=Tracer(),
        state={},
        services={"bq": bq, "query_counter": itertools.count(1)},
    )


def run(sql: str, columns: list[str], rows: list[list[Any]]):
    bq = FakeBigQuery(columns, rows)
    ctx = make_ctx(bq)
    outcome = run_sql(RunSqlArgs(sql=sql, purpose="test"), ctx)
    return outcome, ctx, bq


def test_small_groups_are_removed_from_preview_and_stored_state():
    rows = [["Texas", 120], ["Ohio", K], ["Wyoming", K - 1], ["Vermont", 1], [None, None]]
    outcome, ctx, _ = run(BY_STATE, ["state", "customers"], rows)

    result = outcome.result
    assert result.status == "ok"
    assert result.data.rows == [["Texas", 120], ["Ohio", K]]
    assert result.rows_total == 2
    stored = outcome.state_update["queries"]["q1"]
    assert stored["rows"] == [["Texas", 120], ["Ohio", K]]
    assert stored["rows_total"] == 2
    assert "3 groups with fewer than 5 customers were hidden for privacy" in result.message
    rendered = result.render_for_model()
    assert "Wyoming" not in rendered and "Vermont" not in rendered
    assert ctx.trace.of("bq_job")[0]["hidden_groups"] == 3


def test_one_hidden_group_message_is_singular():
    outcome, _, _ = run(BY_STATE, ["state", "customers"], [["Texas", 120], ["Wyoming", 2]])
    assert "1 group with fewer than 5 customers was hidden for privacy" in outcome.result.message


def test_all_groups_hidden_is_privacy_suppression_not_missing_data():
    outcome, _, _ = run(BY_STATE, ["state", "customers"], [["Wyoming", 2], ["Vermont", 1]])

    result = outcome.result
    assert result.status == "empty"
    assert result.error_kind is None  # not EMPTY_RESULT: no filter diagnosis, no fix attempt
    assert result.data is None
    assert "privacy" in result.message and "not missing data" in result.message
    assert outcome.state_update["queries"]["q1"]["rows"] == []


def test_queries_without_demographics_are_untouched():
    sql = "SELECT oi.user_id, SUM(oi.sale_price) AS spend FROM order_items oi GROUP BY 1"
    rows = [["C-1", 10.0], ["C-2", 1.0]]
    outcome, ctx, _ = run(sql, ["user_id", "spend"], rows)

    assert outcome.result.status == "ok"
    assert outcome.result.data.rows == rows
    assert outcome.result.message == ""
    assert outcome.state_update["queries"]["q1"]["rows"] == rows
    assert ctx.trace.of("bq_job")[0]["hidden_groups"] == 0


def test_rejected_demographic_query_never_reaches_bigquery():
    outcome, _, bq = run("SELECT state, COUNT(*) AS n FROM users GROUP BY 1", ["state", "n"], [["Texas", 3]])
    assert outcome.result.status == "error"
    assert outcome.result.error_kind == ErrorKind.RULE_VIOLATION
    assert bq.executed == []


def test_describe_data_on_customer_columns_counts_customers_and_hides_rare_values():
    bq = FakeBigQuery(["value", "customers"], [["Texas", 900], ["Wyoming", 3]])
    outcome = describe_data(DescribeDataArgs(column="state"), make_ctx(bq))
    assert outcome.result.status == "ok"
    assert outcome.result.data.rows == [["Texas", 900]]
    assert "COUNT(DISTINCT" in bq.executed[0]


def test_describe_data_on_products_is_unchanged():
    bq = FakeBigQuery(["value", "n"], [["Levi's", 3]])
    outcome = describe_data(DescribeDataArgs(column="brand"), make_ctx(bq))
    assert outcome.result.data.rows == [["Levi's", 3]]


@pytest.mark.parametrize(
    ("columns", "rows", "kept"),
    [
        (["state", "CUSTOMERS"], [["a", 5], ["b", 4.0], ["c", "7"], ["d", True]], [["a", 5]]),
        (["state", "n"], [["a", 100]], []),  # column missing: fail closed
    ],
)
def test_drop_small_groups_fails_closed(columns, rows, kept):
    result, hidden = drop_small_groups(columns, rows, "customers", 5)
    assert result == kept and hidden == len(rows) - len(kept)


def test_wide_results_are_rendered_as_named_records_with_rounded_floats():
    from retail_analyst.contracts import TableData, ToolResult

    cols = ["month", "revenue", "orders", "buyers", "return_rate", "revenue_growth_mom"]
    data = TableData(columns=cols, rows=[["2026-10-01", 24555.0299949646, 285, 284, 0.10703363914373089, 0.2179]])
    text = ToolResult(status="ok", query_id="q1", data=data, rows_total=1).render_for_model()
    assert "row 1: month=2026-10-01, revenue=24555.03, orders=285, buyers=284, return_rate=0.107, " in text
    assert "revenue_growth_mom=0.2179" in text


def test_narrow_results_stay_a_table():
    from retail_analyst.contracts import TableData, ToolResult

    data = TableData(columns=["state", "customers"], rows=[["Texas", 12], ["Ohio", None]])
    text = ToolResult(status="ok", data=data, rows_total=2).render_for_model()
    assert "state | customers\nTexas | 12\nOhio | " in text
