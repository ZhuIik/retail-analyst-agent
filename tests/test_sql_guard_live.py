"""Live checks of the scoped SQL against BigQuery: scoped queries dry-run, and joins through
hashed customer ids give the same counts as joins on raw ids.

Run with real credentials only; skipped cleanly otherwise. Uses google-cloud-bigquery directly
so it does not depend on the retail_analyst.bq layer.
"""

from __future__ import annotations

import pytest

from retail_analyst.config import get_settings, load_schema, load_users
from retail_analyst.sql.guard import guard_and_scope


def _client():
    try:
        import google.auth
        from google.cloud import bigquery

        project = get_settings().gcp_project_id
        if not project:
            return None
        credentials, _ = google.auth.default()
        return bigquery.Client(project=project, credentials=credentials)
    except Exception:
        return None


CLIENT = _client()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(CLIENT is None, reason="no BigQuery credentials / GCP_PROJECT_ID"),
]

SETTINGS = get_settings()
SCHEMA = load_schema()
USERS = load_users()


def _scoped(sql: str, user: str) -> str:
    result = guard_and_scope(sql, USERS[user], SETTINGS, SCHEMA)
    assert result.ok, result.message
    return result.scoped_sql


@pytest.mark.parametrize("user", ["anna", "mark", "ceo"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT status, COUNT(*) AS n FROM orders GROUP BY status",
        "SELECT p.brand, SUM(oi.sale_price) AS revenue FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
        "SELECT u.country, COUNT(DISTINCT o.user_id) AS customers FROM orders o "
        "JOIN users u ON u.id = o.user_id GROUP BY 1",
    ],
)
def test_scoped_sql_dry_runs(sql, user):
    from google.cloud import bigquery

    job = CLIENT.query(_scoped(sql, user), job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    assert job.total_bytes_processed is not None


def test_join_through_hashed_ids_matches_raw_ids():
    d = SETTINGS.dataset
    raw = f"SELECT COUNT(*) FROM `{d}.orders` o JOIN `{d}.users` u ON u.id = o.user_id"
    scoped = _scoped("SELECT COUNT(*) AS n FROM orders o JOIN users u ON u.id = o.user_id", "ceo")
    raw_count = list(CLIENT.query(raw).result())[0][0]
    scoped_count = list(CLIENT.query(scoped).result())[0][0]
    assert raw_count == scoped_count
