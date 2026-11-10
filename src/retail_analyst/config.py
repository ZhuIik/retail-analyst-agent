"""Configuration: YAML files in config/ plus secrets and overrides from .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from retail_analyst.contracts import AccessScope, UserContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ModelSettings(BaseModel):
    analyst: str
    fallbacks: list[str] = []
    temperature: float | None = None
    request_timeout_s: int = 60
    max_sdk_retries: int = 1
    requests_per_minute: int = 14


class AgentSettings(BaseModel):
    max_llm_calls_per_turn: int = 6
    max_sql_fix_attempts: int = 3
    history_messages: int = 12


class SqlSettings(BaseModel):
    default_limit: int = 1000
    max_bytes_billed: int = 1_000_000_000
    dry_run_max_bytes: int = 500_000_000
    query_timeout_s: int = 60
    max_transient_retries: int = 3
    preview_rows: int = 20
    stored_rows: int = 500
    forbidden_function_prefixes: list[str] = []
    min_group_size: int = 5


class GoldenSettings(BaseModel):
    path: str = "config/golden_trios.yaml"
    top_k: int = 2
    min_score: float = 0.15


class ReportSettings(BaseModel):
    db_path: str = "data/reports.sqlite"
    bulk_delete_threshold: int = 3


class TracingSettings(BaseModel):
    dir: str = "traces"


class Settings(BaseModel):
    dataset: str
    models: ModelSettings
    agent: AgentSettings
    sql: SqlSettings
    reports: ReportSettings
    tracing: TracingSettings
    golden: GoldenSettings = GoldenSettings()
    # From .env
    google_api_key: str = ""
    gcp_project_id: str = ""
    id_salt: str = ""

    def resolve(self, relative: str) -> Path:
        """Resolve a path from settings relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    raw = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    settings = Settings(**raw)
    if model := os.getenv("GEMINI_MODEL", "").strip():
        settings.models.analyst = model
    if fallback := os.getenv("GEMINI_FALLBACK_MODEL", "").strip():
        settings.models.fallbacks = [fallback]
    settings.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    settings.gcp_project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    # Salt for customer-id pseudonyms. A fixed dev default keeps pseudonyms stable
    # between runs; production would read it from Secret Manager.
    settings.id_salt = os.getenv("ID_SALT", "").strip() or "dev-only-salt"
    return settings


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """config/schema.yaml: tables -> columns with type, description, hashed, exposed flags."""
    return yaml.safe_load((CONFIG_DIR / "schema.yaml").read_text(encoding="utf-8"))["tables"]


def load_users() -> dict[str, UserContext]:
    raw = yaml.safe_load((CONFIG_DIR / "users.yaml").read_text(encoding="utf-8"))
    return {
        uid: UserContext(
            user_id=uid,
            display_name=u["display_name"],
            role=u["role"],
            access=AccessScope(**u["access"]),
        )
        for uid, u in raw["users"].items()
    }
