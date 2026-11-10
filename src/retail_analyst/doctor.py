"""`analyst-doctor`: check the environment before the first question.

Prints OK / FAIL with a fix hint for each check and exits with code 1 if any check fails.
Region blocks are recognised for both Gemini and BigQuery (section 6, check 9a).
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from rich.console import Console

from retail_analyst.bq.client import LOCATION_BLOCKED_MESSAGE, BigQueryClient
from retail_analyst.config import PROJECT_ROOT, Settings, get_settings
from retail_analyst.contracts import AgentError, ErrorKind
from retail_analyst.llm import classify_llm_error

TABLES = ("orders", "order_items", "products", "users")
console = Console()


class CheckFailed(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def _check(name: str, fn: Callable[[], str]) -> bool:
    try:
        info = fn()
    except CheckFailed as exc:
        console.print(f"[red]FAIL[/red] {name}: {exc}")
        if exc.hint:
            console.print(f"     [yellow]fix:[/yellow] {exc.hint}")
        return False
    console.print(f"[green]OK[/green]   {name}" + (f" — {info}" if info else ""))
    return True


def _check_env(settings: Settings) -> str:
    if not (PROJECT_ROOT / ".env").exists():
        raise CheckFailed(".env not found", "create .env with GOOGLE_API_KEY and GCP_PROJECT_ID (see README)")
    missing = [
        k
        for k, v in (("GOOGLE_API_KEY", settings.google_api_key), ("GCP_PROJECT_ID", settings.gcp_project_id))
        if not v
    ]
    if missing:
        raise CheckFailed(f"missing {', '.join(missing)}", "set them in .env")
    return "dev-only ID_SALT (fine for the prototype)" if settings.id_salt == "dev-only-salt" else ""


def _check_gemini(settings: Settings) -> str:
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=settings.google_api_key, http_options=types.HttpOptions(timeout=20_000))
        available = {m.name.removeprefix("models/") for m in client.models.list() if m.name}
    except Exception as exc:
        kind = classify_llm_error(exc)
        hints = {
            ErrorKind.LOCATION_BLOCKED: LOCATION_BLOCKED_MESSAGE,
            ErrorKind.AUTH: "create a key at https://aistudio.google.com/apikey and set GOOGLE_API_KEY in .env",
        }
        raise CheckFailed(f"{kind.value}: {str(exc)[:200]}", hints.get(kind, "check network / VPN")) from exc
    configured = [settings.models.analyst, *settings.models.fallbacks]
    missing = [m for m in configured if m not in available]
    if missing:
        raise CheckFailed(
            f"model id(s) not available to this key: {', '.join(missing)}",
            "fix models.analyst / models.fallbacks in config/settings.yaml or GEMINI_MODEL in .env",
        )
    return f"key valid; models found: {', '.join(configured)}"


def _check_adc(settings: Settings) -> str:
    import google.auth

    try:
        _, project = google.auth.default()
    except Exception as exc:
        raise CheckFailed(
            f"no Application Default Credentials ({type(exc).__name__})", "run `gcloud auth application-default login`"
        ) from exc
    return f"ADC found (default project: {project or 'none'}; billing project: {settings.gcp_project_id})"


def _bq_failure(exc: AgentError) -> CheckFailed:
    hint = {
        ErrorKind.LOCATION_BLOCKED: LOCATION_BLOCKED_MESSAGE,
        ErrorKind.AUTH: "run `gcloud auth application-default login`; check GCP_PROJECT_ID and BigQuery API access",
    }.get(exc.kind, "see the message above")
    return CheckFailed(f"{exc.kind.value}: {exc.user_message}", hint)


def _check_dry_run(settings: Settings) -> str:
    try:
        n = BigQueryClient(settings).dry_run(f"SELECT order_id, status FROM `{settings.dataset}.orders` LIMIT 1")
    except AgentError as exc:
        raise _bq_failure(exc) from exc
    return f"would scan {n / 1e6:.1f} MB"


def _check_tables(settings: Settings) -> str:
    try:
        bq = BigQueryClient(settings)
        counts = {t: len(bq.table_schema(t)) for t in TABLES}
    except AgentError as exc:
        raise _bq_failure(exc) from exc
    return ", ".join(f"{t} ({n} cols)" for t, n in counts.items())


def main() -> None:
    settings = get_settings()
    console.print("[bold]Retail analyst — environment check[/bold]")
    results = [
        _check(".env values", lambda: _check_env(settings)),
        _check("Gemini key and model ids", lambda: _check_gemini(settings)),
    ]
    adc_ok = _check("BigQuery credentials (ADC)", lambda: _check_adc(settings))
    results.append(adc_ok)
    if adc_ok:  # without credentials the BigQuery checks would only repeat the same failure
        results.append(_check(f"dry-run on {settings.dataset}.orders", lambda: _check_dry_run(settings)))
        results.append(_check("dataset tables reachable", lambda: _check_tables(settings)))
    ok = all(results)
    console.print("[green]All checks passed.[/green]" if ok else "[red]Some checks failed.[/red]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
