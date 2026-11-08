"""Doctor: region blocks are recognised, failures give exit code 1 — no network."""

from __future__ import annotations

import pytest
from google.genai.errors import ClientError

from retail_analyst import doctor
from retail_analyst.bq.client import LOCATION_BLOCKED_MESSAGE
from retail_analyst.config import AgentSettings, ModelSettings, ReportSettings, Settings, SqlSettings, TracingSettings


def make_settings() -> Settings:
    return Settings(
        dataset="d",
        models=ModelSettings(analyst="gemini-a", fallbacks=["gemini-b"]),
        agent=AgentSettings(),
        sql=SqlSettings(),
        reports=ReportSettings(),
        tracing=TracingSettings(),
        google_api_key="k",
        gcp_project_id="p",
    )


class FakeModels:
    def __init__(self, names=None, error=None) -> None:
        self.names, self.error = names or [], error

    def list(self):
        if self.error:
            raise self.error
        return [type("M", (), {"name": f"models/{n}"})() for n in self.names]


def patch_genai(monkeypatch, models: FakeModels) -> None:
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: type("C", (), {"models": models})())


def test_gemini_location_block_gets_vpn_hint(monkeypatch) -> None:
    error = ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "User location is not supported for the API use.",
                "status": "FAILED_PRECONDITION",
            }
        },
    )
    patch_genai(monkeypatch, FakeModels(error=error))
    with pytest.raises(doctor.CheckFailed) as info:
        doctor._check_gemini(make_settings())
    assert info.value.hint == LOCATION_BLOCKED_MESSAGE


def test_gemini_missing_model_id(monkeypatch) -> None:
    patch_genai(monkeypatch, FakeModels(names=["gemini-a"]))
    with pytest.raises(doctor.CheckFailed) as info:
        doctor._check_gemini(make_settings())
    assert "gemini-b" in str(info.value)


def test_gemini_ok(monkeypatch) -> None:
    patch_genai(monkeypatch, FakeModels(names=["gemini-a", "gemini-b", "other"]))
    assert "gemini-a" in doctor._check_gemini(make_settings())


def test_exit_code_1_when_a_check_fails(monkeypatch) -> None:
    def fail(*_):
        raise doctor.CheckFailed("nope", "fix it")

    monkeypatch.setattr(doctor, "get_settings", make_settings)
    monkeypatch.setattr(doctor, "_check_env", lambda s: "")
    monkeypatch.setattr(doctor, "_check_gemini", fail)
    monkeypatch.setattr(doctor, "_check_adc", fail)  # BigQuery checks are skipped without ADC
    with pytest.raises(SystemExit) as info:
        doctor.main()
    assert info.value.code == 1
