"""Tool registry: every capability is one entry with a risk class.

The graph never special-cases a tool by name except `submit_answer` (which ends the turn).
INTERNAL_WRITE handlers are preparation functions: they MUST have no side effects and return
pending_deletion for the confirmation node. The only supported executor is owner-scoped report
deletion in that node. Risk metadata does not sandbox Python code. External tools are rejected
until their preview, confirmation and execution policy has been implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from retail_analyst.config import Settings
from retail_analyst.contracts import ToolResult, ToolRisk, UserContext


@dataclass
class ToolContext:
    """Everything a tool handler may use. Handlers get no other global state."""

    user: UserContext
    settings: Settings
    session_id: str
    trace: Any  # retail_analyst.tracing.Tracer (kept as Any to avoid an import cycle)
    state: Mapping[str, Any]  # read-only view of the graph state (e.g. stored query results)
    services: Mapping[str, Any] = field(default_factory=dict)  # bq client, report store, ...


@dataclass
class ToolOutcome:
    result: ToolResult
    state_update: dict[str, Any] = field(default_factory=dict)  # merged into graph state


Handler = Callable[[BaseModel, ToolContext], ToolOutcome]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # shown to the model
    args_model: type[BaseModel]  # the tool's JSON schema for function calling
    handler: Handler
    risk: ToolRisk = ToolRisk.READ
    roles: frozenset[str] | None = None  # None = visible to every role


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} registered twice")
        if spec.risk in {ToolRisk.EXTERNAL_WRITE, ToolRisk.EXTERNAL_READ}:
            raise ValueError("External tools require an implemented execution policy before registration")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def for_role(self, role: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.roles is None or role in t.roles]
