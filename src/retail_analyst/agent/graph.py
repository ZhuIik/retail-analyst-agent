"""The agent graph (D22): agent -> tools -> (confirm) -> agent ... -> finalize -> END.

One analyst model with tools; every protection is code around it (D1):
- `agent`    calls the model with the system prompt and a trimmed history (H12), under a hard
             per-turn call limit; LLM failures end the turn with an honest message.
- `tools`    runs the model's tool calls through the registry (role filter, argument validation,
             tracing, SQL-fix budget). A tool that needs confirmation sets `pending_deletion`.
- `confirm`  (D16) pauses on `interrupt()`; only the user's exact answer deletes, and only the
             snapshot of ids that was previewed, filtered by owner again in the store.
- `finalize` turns `submit_answer` (D15) into the user-facing text: formatting, PII net, number check.

Caller contract, once per user message (a "turn"):

    graph.invoke({"messages": [HumanMessage(text)], "question": text, "trace_id": tracer.new_turn(),
                  "llm_calls": 0, "sql_fix_attempts": 0, "final_text": None}, config)

The per-turn counters are reset by the caller in that state update, never by the graph, so a turn
cannot extend its own budget. The first turn also sets `user` (UserContext.model_dump()) and
`session_id`. If the result has `__interrupt__` (type `confirm_deletion`), the next user input must
be sent as `graph.invoke(Command(resume=text), config)` — D16 CLI rule — not as a new turn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.utils.json_schema import dereference_refs
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from retail_analyst import llm
from retail_analyst.agent.answer import check_numbers, format_answer
from retail_analyst.agent.state import AgentState
from retail_analyst.config import Settings
from retail_analyst.contracts import (
    AgentError,
    AnswerSubmission,
    ErrorKind,
    PendingDeletion,
    ToolResult,
    UserContext,
)
from retail_analyst.safety.pii import redact_pii
from retail_analyst.tools.registry import ToolContext, ToolOutcome, ToolRegistry, ToolSpec

SUBMIT_ANSWER = "submit_answer"
_SUBMIT_DESCRIPTION = (
    "Give your response to the user: an analysis, a clarifying question, a refusal or information. "
    "Call it alone, after you have the results of all other tool calls. The text is assembled from these fields."
)
_SUBMIT_ALONE = "Call submit_answer alone, after you have the results of your other tool calls."
_STOP_FIXING = (
    "STOP: {n} consecutive query errors. Do not retry the query again; answer now with submit_answer "
    "using what is available, and say what could not be computed."
)
_LLM_FAILURE = {
    ErrorKind.QUOTA_EXHAUSTED: "The AI model's daily quota is used up, so I cannot answer right now. "
    "Please try again later.",
    ErrorKind.AUTH: "The AI model rejected the credentials (check GOOGLE_API_KEY), so I cannot answer.",
    ErrorKind.LOCATION_BLOCKED: "The AI model is not available from this location, so I cannot answer.",
}
_LLM_UNAVAILABLE = (
    "The AI model is temporarily unavailable (retries and the fallback model failed). Please try again in a minute."
)


# --------------------------------------------------------------------------------------
# Tool definitions for bind_tools
# --------------------------------------------------------------------------------------


def tool_definition(name: str, description: str, args_model: type[BaseModel]) -> dict[str, Any]:
    """A `{name, description, parameters}` dict that langchain-google-genai's bind_tools accepts.

    Nested models (AnswerSubmission -> Fact) are inlined here: Gemini function declarations do
    not support `$ref`/`$defs`. The description comes from the registry, not the model docstring.
    """
    schema = dereference_refs(args_model.model_json_schema())
    for key in ("$defs", "title", "description"):
        schema.pop(key, None)
    return {"name": name, "description": description, "parameters": schema}


# --------------------------------------------------------------------------------------
# History trimming (H12)
# --------------------------------------------------------------------------------------


def _turn_start(messages: list[AnyMessage]) -> int:
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)


def _result_label(message: ToolMessage) -> str:
    artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
    return artifact.get("query_id") or message.name or "tool call"


def trim_history(messages: list[AnyMessage], keep: int) -> list[AnyMessage]:
    """The last ~`keep` messages, starting at a user message; older tool results become stubs.

    Starting the window at a HumanMessage never separates an AIMessage with tool_calls from its
    ToolMessages, and Gemini expects the history to begin with a user turn. The current turn is
    always kept whole (it is bounded by the per-turn call limit). Tool results of earlier turns
    are replaced by a one-line stub: their numbers live on in the answers the model submitted.
    """
    turn_start = _turn_start(messages)
    cut = max(0, len(messages) - keep)
    start = min((i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and i >= cut), default=turn_start)
    start = min(start, turn_start)
    trimmed: list[AnyMessage] = []
    for i, message in enumerate(messages[start:], start):
        if i < turn_start and isinstance(message, ToolMessage):
            message = message.model_copy(update={"content": f"[result of {_result_label(message)} omitted]"})
        trimmed.append(message)
    return trimmed


def _turn_query_ids(messages: list[AnyMessage]) -> list[str]:
    """Ids of the queries that returned results in the current turn (ToolMessage artifacts)."""
    return [
        m.artifact["query_id"]
        for m in messages[_turn_start(messages) :]
        if isinstance(m, ToolMessage) and isinstance(m.artifact, Mapping) and m.artifact.get("query_id")
    ]


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------


def _last_ai(state: AgentState) -> AIMessage:
    message = state["messages"][-1]
    assert isinstance(message, AIMessage), "expected the model's message last"
    return message


def _user(state: AgentState) -> UserContext:
    return UserContext.model_validate(state["user"])


def _is_valid_submission(args: Mapping[str, Any]) -> bool:
    try:
        AnswerSubmission.model_validate(args)
    except ValidationError:
        return False
    return True


def _error(message: str, kind: ErrorKind | None = None) -> ToolOutcome:
    return ToolOutcome(ToolResult(status="error", message=message, error_kind=kind))


def build_graph(
    registry: ToolRegistry,
    model_factory: Callable[[list], Runnable],
    services: dict,
    settings: Settings,
    system_prompt: Callable[[AgentState], str],
    checkpointer=None,
) -> CompiledStateGraph:
    tracer = services["tracer"]
    models: dict[str, Runnable] = {}

    def model_for(role: str) -> Runnable:
        # The role decides which tools the model even sees; bind once per role and reuse.
        if role not in models:
            tools = [tool_definition(t.name, t.description, t.args_model) for t in registry.for_role(role)]
            tools.append(tool_definition(SUBMIT_ANSWER, _SUBMIT_DESCRIPTION, AnswerSubmission))
            models[role] = model_factory(tools)
        return models[role]

    # ---------------------------------------------------------------- agent
    def agent(state: AgentState) -> dict[str, Any]:
        calls = state.get("llm_calls", 0)
        limit = settings.agent.max_llm_calls_per_turn
        if calls >= limit:
            # Hard budget: no model call; an honest partial answer goes straight to finalize.
            tracer.event("error", kind="llm_call_limit", llm_calls=calls)
            done = _turn_query_ids(state["messages"])
            text = f"I reached the limit of {limit} model calls for one question, so I stopped instead of guessing."
            text += (
                f" Queries completed so far: {', '.join(done)}. Ask me to summarise them, or narrow the question."
                if done
                else " Please try a narrower question (one metric, one period)."
            )
            return {"messages": [AIMessage(content=text)]}

        prompt = [
            SystemMessage(content=system_prompt(state)),
            *trim_history(state["messages"], settings.agent.history_messages),
        ]
        try:
            response = model_for(_user(state).role).invoke(prompt)
        except Exception as exc:  # any LLM failure ends the turn honestly (retries/fallbacks live in llm.py)
            kind = exc.kind if isinstance(exc, AgentError) else llm.classify_llm_error(exc)
            tracer.event("error", source="llm", kind=kind.value, detail=str(exc)[:500])
            text = exc.user_message if isinstance(exc, AgentError) else _LLM_FAILURE.get(kind, _LLM_UNAVAILABLE)
            return {"llm_calls": calls + 1, "messages": [AIMessage(content=text)]}
        return {"llm_calls": calls + 1, "messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        calls = _last_ai(state).tool_calls
        if not calls:
            return "finalize"  # a code-written stop message, or plain text from the model
        if len(calls) == 1 and calls[0]["name"] == SUBMIT_ANSWER and _is_valid_submission(calls[0]["args"]):
            return "finalize"
        return "tools"  # includes an invalid or mixed submit_answer: the tools node explains the error

    # ---------------------------------------------------------------- tools
    def run_call(call: Mapping[str, Any], ctx: ToolContext, visible: dict[str, ToolSpec]) -> ToolOutcome:
        name, args = call["name"], call.get("args") or {}
        if name == SUBMIT_ANSWER:  # only reaches this node when invalid or mixed with other calls
            try:
                AnswerSubmission.model_validate(args)
            except ValidationError as exc:
                return _error(f"Invalid submit_answer arguments: {exc.errors(include_url=False)}")
            return _error(_SUBMIT_ALONE)
        spec = visible.get(name)
        if spec is None:  # unknown, or not allowed for this role: same answer, no hint
            return _error(f"Unknown tool {name!r}. Available tools: {', '.join(sorted(visible))}.")
        try:
            parsed = spec.args_model.model_validate(args)
        except ValidationError as exc:
            return _error(f"Invalid arguments for {name}: {exc.errors(include_url=False)}")
        try:
            return spec.handler(parsed, ctx)
        except AgentError as exc:
            tracer.event("error", source=name, kind=exc.kind.value, detail=exc.detail)
            if exc.kind.is_fatal:
                raise  # quota / auth / region: the model cannot fix it, the turn ends (D12)
            return _error(exc.user_message, exc.kind)
        except Exception as exc:  # a tool bug must not kill the turn; the model can work around it
            tracer.event("error", source=name, kind=ErrorKind.UNKNOWN.value, detail=repr(exc)[:500])
            return _error(f"{name} failed with an internal error.", ErrorKind.UNKNOWN)

    def tools(state: AgentState) -> dict[str, Any]:
        calls = _last_ai(state).tool_calls
        user = _user(state)
        visible = {t.name: t for t in registry.for_role(user.role)}
        view: dict[str, Any] = dict(state)  # later calls in the batch see earlier calls' updates
        update: dict[str, Any] = {}
        messages: list[ToolMessage] = []
        fix_attempts = state.get("sql_fix_attempts", 0)
        fix_limit_hit = False
        fatal: AgentError | None = None

        for call in calls:
            if fatal:  # every tool call must get an answer, or the next turn's history is invalid
                messages.append(_tool_message(call, _error("Not executed: the turn was stopped.").result))
                continue
            ctx = ToolContext(
                user=user,
                settings=settings,
                session_id=state["session_id"],
                trace=tracer,
                state=MappingProxyType(view),
                services=services,
            )
            try:
                outcome = run_call(call, ctx, visible)
            except AgentError as exc:
                fatal = exc
                outcome = _error(exc.user_message, exc.kind)
            result, changes = outcome.result, dict(outcome.state_update)
            tracer.event(
                "tool_call",
                tool=call["name"],
                args=call.get("args"),
                status=result.status,
                error_kind=result.error_kind.value if result.error_kind else None,
                query_id=result.query_id,
            )

            awaits_confirmation = False
            if pending := changes.pop("pending_deletion", None):
                if update.get("pending_deletion"):
                    messages.append(_tool_message(call, _error("One deletion request at a time.").result))
                    continue
                # Bind the confirmation to this exact call; the confirm node answers it (no ToolMessage here).
                update["pending_deletion"] = view["pending_deletion"] = {**pending, "tool_call_id": call["id"]}
                awaits_confirmation = True
                tracer.event("interrupt", report_ids=pending["report_ids"], required_answer=pending["required_answer"])

            for key, value in changes.items():
                if key == "queries":  # merged by the state reducer; mirror that in the view
                    update["queries"] = {**update.get("queries", {}), **value}
                    view["queries"] = {**(view.get("queries") or {}), **value}
                else:
                    update[key] = view[key] = value

            if result.error_kind and result.error_kind.fixable_by_model:
                fix_attempts += 1
                fix_limit_hit |= fix_attempts >= settings.agent.max_sql_fix_attempts
            elif result.status == "ok" and result.query_id:
                fix_attempts = 0
            if not awaits_confirmation:
                messages.append(_tool_message(call, result))

        if fix_limit_hit and messages:
            last = messages[-1]
            messages[-1] = last.model_copy(update={"content": f"{last.content}\n{_STOP_FIXING.format(n=fix_attempts)}"})
        if fatal:
            # A code-written stop message; finalize shows it as an "info" answer.
            update.pop("pending_deletion", None)
            messages.append(AIMessage(content=fatal.user_message))
        return {**update, "messages": messages, "sql_fix_attempts": fix_attempts}

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_deletion"):
            return "confirm"
        last = state["messages"][-1]
        return "finalize" if isinstance(last, AIMessage) else "agent"  # fatal error: stop the turn

    # ---------------------------------------------------------------- confirm (D16)
    def confirm(state: AgentState) -> dict[str, Any]:
        pending = PendingDeletion.model_validate(state["pending_deletion"])
        # The node re-runs from the top on resume: nothing before interrupt() may have side effects.
        reply = interrupt(
            {
                "type": "confirm_deletion",
                "preview": [r.model_dump() for r in pending.preview],
                "required_answer": pending.required_answer,
            }
        )
        # Code decides, not the model: only the exact answer confirms; anything else cancels.
        confirmed = isinstance(reply, str) and reply.strip().casefold() == pending.required_answer.casefold()
        deleted = 0
        if confirmed:
            owner = _user(state).user_id
            deleted = services["reports"].soft_delete(owner, pending.report_ids, state.get("trace_id", ""))
            titles = "; ".join(f"#{r.report_id} '{r.title}'" for r in pending.preview)
            outcome = f"Deleted {deleted} of {len(pending.report_ids)} report(s): {titles}."
        else:
            outcome = "User cancelled; nothing was deleted."
        tracer.event(
            "confirmation",
            confirmed=confirmed,
            required_answer=pending.required_answer,
            report_ids=pending.report_ids,
            deleted=deleted,
        )
        return {
            "messages": [ToolMessage(content=outcome, tool_call_id=pending.tool_call_id)],
            "pending_deletion": None,
        }

    # ---------------------------------------------------------------- finalize (D15)
    def finalize(state: AgentState) -> dict[str, Any]:
        ai = _last_ai(state)
        call = ai.tool_calls[0] if ai.tool_calls else None  # routing guarantees a lone, valid submit_answer
        if call:
            answer = AnswerSubmission.model_validate(call["args"])
        else:  # a code-written stop message (call limit, LLM outage) or plain model text
            answer = AnswerSubmission(kind="info", message=ai.text)
        text, hits = redact_pii(format_answer(answer))
        if hits:
            tracer.event("pii_redaction", where="answer", hits=hits)
        if unverified := check_numbers(answer, state.get("queries") or {}):
            tracer.event("number_check", unverified=unverified, facts=len(answer.facts))
        update: dict[str, Any] = {"final_text": text}
        if call:
            # Only real answers become "the last answer" that save_report stores (Г1); a stop message
            # (call limit, LLM outage) must not replace it.
            update["last_answer"] = {
                **answer.model_dump(),
                "question": state.get("question", ""),
                "query_ids": _turn_query_ids(state["messages"]),
            }
            update["last_answer_text"] = text
            # Answer the submit_answer call so the history stays valid for the next turn.
            update["messages"] = [ToolMessage(content="Answer shown to the user.", tool_call_id=call["id"])]
        return update

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_node("confirm", confirm)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "finalize"])
    builder.add_conditional_edges("tools", route_after_tools, ["confirm", "agent", "finalize"])
    builder.add_edge("confirm", "agent")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def _tool_message(call: Mapping[str, Any], result: ToolResult) -> ToolMessage:
    return ToolMessage(
        content=result.render_for_model(),
        tool_call_id=call["id"],
        name=call["name"],
        status="error" if result.status == "error" else "success",
        # Not sent to the model: lets trimming label stubs and save_report find the turn's queries.
        artifact={"query_id": result.query_id} if result.query_id else None,
    )
