"""Command-line chat (prototype adapter of the interface-agnostic agent, П3).

    uv run analyst --user anna
    uv run analyst --user ceo --simulate llm_429,bq_timeout

Commands that work without the LLM (D17): /reports, /trace, /help, /quit.
While a deletion waits for confirmation, the next input is the confirmation answer (D16):
it is passed to the graph as Command(resume=...), and anything but the exact answer cancels.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import uuid
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from retail_analyst.agent.graph import build_graph
from retail_analyst.agent.prompts import PROMPT_VERSION, PromptContext, make_prompt_fn
from retail_analyst.bq.client import BigQueryClient
from retail_analyst.config import Settings, get_settings, load_users
from retail_analyst.contracts import AgentError, UserContext
from retail_analyst.faults import FAULT_MODES, FaultInjector
from retail_analyst.llm import build_analyst_model
from retail_analyst.reports.store import ReportStore
from retail_analyst.tools.data_tools import register_data_tools
from retail_analyst.tools.registry import ToolRegistry
from retail_analyst.tools.report_tools import register_report_tools
from retail_analyst.tracing import Tracer

console = Console()

HELP = """\
Ask questions about sales, customers and products, e.g.
  - What was our revenue by month this year?
  - Which categories have the highest return rate?
  - Write a short report on last month's performance with recommendations.
  - Delete my reports about Jeans.

Commands: /reports (your saved reports)  /trace (what happened in the last answer)  /help  /quit"""

_PROGRESS = {
    "run_sql": lambda a: f"querying: {a.get('purpose') or 'data'}",
    "describe_data": lambda a: f"looking up {a.get('column', 'column')} values",
    "find_reports": lambda a: "searching your reports",
    "save_report": lambda a: f"saving report '{a.get('title', '')}'",
    "delete_reports": lambda a: "finding reports to delete",
    "submit_answer": lambda a: "writing the answer",
}


class Session:
    def __init__(self, user: UserContext, settings: Settings, faults: FaultInjector | None) -> None:
        self.user, self.settings = user, settings
        self.session_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self.tracer = Tracer(settings.resolve(settings.tracing.dir), self.session_id, user.user_id)
        self.reports = ReportStore(settings.resolve(settings.reports.db_path))
        bq = BigQueryClient(settings, faults)
        # A separate client without fault injection: the simulated faults are kept for the demo questions.
        prompt_ctx = _prompt_context(BigQueryClient(settings), settings.dataset)
        services = {
            "bq": bq,
            "reports": self.reports,
            "tracer": self.tracer,
            "query_counter": itertools.count(1),
            "prompt_ctx": prompt_ctx,
        }
        registry = ToolRegistry()
        register_data_tools(registry)
        register_report_tools(registry)
        self.graph = build_graph(
            registry,
            lambda tools: build_analyst_model(settings, tools, faults, on_event=self._llm_event),
            services,
            settings,
            make_prompt_fn(prompt_ctx),
        )
        self.thread = self.session_id
        self.started = False
        self.awaiting: dict | None = None  # interrupt payload while a deletion waits for confirmation

    def _llm_event(self, event_type: str, payload: dict) -> None:
        self.tracer.event(event_type, **payload)  # llm.py reports (type, payload); the tracer takes kwargs

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread}}

    def ask(self, text: str) -> None:
        if self.awaiting:
            payload: object = Command(resume=text)
            self.awaiting = None
        else:
            trace_id = self.tracer.new_turn()
            self.tracer.event(
                "turn_start", question=text, prompt_version=PROMPT_VERSION, model=self.settings.models.analyst
            )
            payload = {
                "messages": [HumanMessage(content=text)],
                "question": text,
                "trace_id": trace_id,
                "llm_calls": 0,
                "sql_fix_attempts": 0,
                "final_text": None,
            }
            if not self.started:
                payload |= {"user": self.user.model_dump(), "session_id": self.session_id}
        started = time.monotonic()
        try:
            self._run(payload)
        except AgentError as e:  # infrastructure failure outside the graph's own handling
            self._fail(e.user_message, e.detail)
        except Exception as e:  # never crash the CLI; start a clean conversation thread
            self._fail("Something went wrong, so I started a fresh conversation.", repr(e))
            self.thread, self.started = f"{self.session_id}-{uuid.uuid4().hex[:4]}", False
            return
        self.started = True
        state = self.graph.get_state(self.config).values
        if not self.awaiting:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.tracer.event("turn_end", llm_calls=state.get("llm_calls", 0), ms=elapsed_ms)

    def _run(self, payload: object) -> None:
        with console.status("thinking…", spinner="dots") as status:
            for chunk in self.graph.stream(payload, self.config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        self.awaiting = update[0].value
                        continue
                    for message in (update or {}).get("messages", []):
                        self._progress(message, status)
        if self.awaiting:
            self._show_confirmation(self.awaiting)
            return
        final = self.graph.get_state(self.config).values.get("final_text")
        if final:
            console.print(Markdown(final))
        console.print(f"[dim]trace {self.tracer.trace_id}[/]")

    def _progress(self, message: object, status) -> None:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                describe = _PROGRESS.get(call["name"])
                line = describe(call.get("args") or {}) if describe else call["name"]
                console.print(f"[dim]· {line}[/]")
                status.update(line + "…")
        elif isinstance(message, ToolMessage) and message.status == "error":
            console.print(f"[dim yellow]· {message.content.splitlines()[0][:120]}[/]")

    def _show_confirmation(self, payload: dict) -> None:
        table = Table(title="Reports to delete", show_header=True)
        for col in ("id", "title", "created"):
            table.add_column(col)
        for r in payload["preview"]:
            table.add_row(str(r["report_id"]), r["title"], r["created_at"][:16])
        console.print(table)
        answer = payload["required_answer"]
        hint = "yes" if answer == "yes" else f"the number of reports ({answer})"
        console.print(f"[bold]Type {hint} to delete. Anything else cancels.[/]")

    def _fail(self, message: str, detail: str) -> None:
        self.tracer.event("error", source="cli", detail=detail[:500])
        console.print(f"[red]{message}[/] [dim](trace {self.tracer.trace_id})[/]")

    # ------------------------------------------------------------- commands without the LLM
    def show_reports(self) -> None:
        rows = self.reports.list(self.user.user_id)
        if not rows:
            console.print("No saved reports yet.")
            return
        table = Table(title="Your reports")
        for col in ("id", "title", "created"):
            table.add_column(col)
        for r in rows:
            table.add_row(str(r.report_id), r.title, r.created_at[:16])
        console.print(table)

    def show_trace(self) -> None:
        events = self.tracer.last_turn()
        if not events:
            console.print("No trace yet.")
            return
        table = Table(title=f"trace {events[0].get('trace_id')}")
        table.add_column("time")
        table.add_column("event")
        table.add_column("details", overflow="fold")
        for e in events:
            payload = ", ".join(f"{k}={v}" for k, v in (e.get("payload") or {}).items() if v not in (None, "", []))
            table.add_row(e["ts"][11:19], e["type"], payload[:400])
        console.print(table)


def _prompt_context(bq: BigQueryClient, dataset: str) -> PromptContext:
    """Data range for the prompt; dates are not sensitive, so this one query is not scoped."""
    try:
        res = bq.execute(
            "SELECT CAST(MIN(DATE(created_at)) AS STRING), CAST(MAX(DATE(created_at)) AS STRING) "
            f"FROM `{dataset}.orders`",
            labels={"purpose": "startup"},
        )
        lo, hi = res.data.rows[0]
        return PromptContext(today=date.today(), data_from=lo, data_to=hi)
    except Exception:  # BigQuery down: the agent still starts and can answer questions about the data model
        return PromptContext(today=date.today())


def main() -> None:
    users = load_users()
    parser = argparse.ArgumentParser(prog="analyst", description="Retail data analysis agent (CLI).")
    parser.add_argument("--user", required=True, choices=sorted(users), help="simulated login (config/users.yaml)")
    parser.add_argument(
        "--simulate", default="", help=f"comma-separated faults to inject: {', '.join(sorted(FAULT_MODES))}"
    )
    args = parser.parse_args()

    modes = {m.strip() for m in args.simulate.split(",") if m.strip()}
    try:
        faults = FaultInjector(modes) if modes else None
    except ValueError as e:
        parser.error(str(e))
    user = users[args.user]
    session = Session(user, get_settings(), faults)

    console.print(f"[bold]Retail Analyst[/] — signed in as {user.display_name}")
    if not user.access.is_unrestricted:
        console.print(f"[dim]Data scope: {user.access.column} in {', '.join(user.access.values)}[/]")
    if modes:
        console.print(f"[yellow]Simulating faults: {', '.join(sorted(modes))}[/]")
    console.print("[dim]Type /help for examples and commands.[/]\n")

    while True:
        try:
            text = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        command = text.lower()
        if command in ("/quit", "/exit"):
            break
        if command == "/help":
            console.print(HELP)
        elif command == "/reports":
            session.show_reports()
        elif command == "/trace":
            session.show_trace()
        else:
            session.ask(text)
        console.print()
    sys.exit(0)


if __name__ == "__main__":
    main()
