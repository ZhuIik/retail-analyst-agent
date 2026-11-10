"""Live check: every golden trio's SQL, scoped for the CEO, dry-runs in BigQuery.

A dry run validates names, types and syntax at no cost, so the examples shown to the model are
known to be valid queries. Skipped cleanly without credentials (same pattern as
test_sql_guard_live.py).
"""

from __future__ import annotations

import pytest

from retail_analyst.config import get_settings, load_schema, load_users
from retail_analyst.golden import load_trios
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
TRIOS = [t for t in load_trios(SETTINGS.resolve(SETTINGS.golden.path)).trios if t.sql]


@pytest.mark.parametrize("trio", TRIOS, ids=lambda t: t.id)
def test_golden_sql_dry_runs_for_ceo(trio):
    from google.cloud import bigquery

    result = guard_and_scope(trio.sql, load_users()["ceo"], SETTINGS, load_schema())
    assert result.ok, result.message
    job = CLIENT.query(result.scoped_sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    assert job.total_bytes_processed is not None
