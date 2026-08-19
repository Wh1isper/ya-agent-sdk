"""YA Claw host capabilities that are not portable SDK feature grants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset

from ya_claw.context import ClawAgentContext
from ya_claw.toolsets.agency import (
    GetSourceRunTraceTool,
    ListAgencyRunsTool,
    ListSourceSessionTurnsTool,
    SubmitToSessionTool,
)
from ya_claw.toolsets.schedule import (
    CreateOnceScheduleTool,
    CreateOnceWorkflowScheduleTool,
    CreateScheduleTool,
    CreateWorkflowScheduleTool,
    DeleteScheduleTool,
    ListSchedulesTool,
    TriggerScheduleTool,
    UpdateScheduleTool,
)
from ya_claw.toolsets.session import GetRunTraceTool, ListSessionTurnsTool
from ya_claw.toolsets.workflow import (
    ArchiveWorkflowTool,
    CancelWorkflowRunTool,
    CreateWorkflowTool,
    GetWorkflowRunTool,
    GetWorkflowTool,
    ListAgentPresetsTool,
    ListWorkflowRunsTool,
    ListWorkflowsTool,
    StartWorkflowTool,
    SteerWorkflowNodeTool,
    UpdateWorkflowTool,
)

_HOST_TOOL_REGISTRY: dict[str, tuple[type[BaseTool], ...]] = {
    "session": (ListSessionTurnsTool, GetRunTraceTool),
    "agency": (ListSourceSessionTurnsTool, GetSourceRunTraceTool, ListAgencyRunsTool, SubmitToSessionTool),
    "schedule": (
        ListSchedulesTool,
        CreateScheduleTool,
        CreateOnceScheduleTool,
        CreateWorkflowScheduleTool,
        CreateOnceWorkflowScheduleTool,
        UpdateScheduleTool,
        DeleteScheduleTool,
        TriggerScheduleTool,
    ),
    "workflow": (
        ListWorkflowsTool,
        GetWorkflowTool,
        CreateWorkflowTool,
        UpdateWorkflowTool,
        ArchiveWorkflowTool,
        StartWorkflowTool,
        ListWorkflowRunsTool,
        GetWorkflowRunTool,
        SteerWorkflowNodeTool,
        CancelWorkflowRunTool,
        ListAgentPresetsTool,
    ),
}


@dataclass(kw_only=True)
class ClawToolsCapability(AbstractCapability[AgentContext]):
    """Own Claw control-plane tools and enforce source/recursion boundaries."""

    groups: tuple[str, ...]
    allowlist: frozenset[str] | None = None
    id: str | None = "claw_tools"

    _safe_at_runtime: ClassVar[bool] = False

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    def _tool_types(self) -> tuple[type[BaseTool], ...]:
        selected: list[type[BaseTool]] = []
        seen: set[str] = set()
        for group in self.groups:
            for tool_type in _HOST_TOOL_REGISTRY.get(group, ()):
                if tool_type.name in seen:
                    continue
                if self.allowlist is not None and tool_type.name not in self.allowlist:
                    continue
                seen.add(tool_type.name)
                selected.append(tool_type)
        return tuple(selected)

    def get_toolset(self) -> AbstractToolset[AgentContext] | None:
        tools = list(self._tool_types())
        if not tools:
            return None
        return Toolset(tools=tools, toolset_id=self.id or "claw_tools")

    async def prepare_tools(
        self,
        ctx: RunContext[AgentContext],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        types_by_name = {tool_type.name: tool_type for tool_type in self._tool_types()}
        selected: list[ToolDefinition] = []
        source_kind = ctx.deps.source_kind if isinstance(ctx.deps, ClawAgentContext) else None
        for tool_def in tool_defs:
            tool_type = types_by_name.get(tool_def.name)
            if tool_type is None:
                selected.append(tool_def)
                continue
            if source_kind != "agency" and tool_type is SubmitToSessionTool:
                continue
            selected.append(tool_def)
        return selected
