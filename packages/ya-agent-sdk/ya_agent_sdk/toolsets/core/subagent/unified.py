"""Unified subagent tool that combines multiple subagents into a single tool.

This module provides factory functions to create a single "delegate" tool that
can call any of multiple subagents by name, instead of creating separate tools
for each subagent.

Key differences from individual subagent tools:
- Single tool entry point instead of N tools
- subagent_name parameter to select which subagent to call
- Dynamic instruction that lists only available subagents
- Literal type for subagent_name based on configured subagents

Usage::

    from ya_agent_sdk.subagents import SubagentConfig
    from ya_agent_sdk.toolsets.core.subagent.unified import create_unified_subagent_tool

    configs = [
        SubagentConfig(name="debugger", description="...", system_prompt="..."),
        SubagentConfig(name="explorer", description="...", system_prompt="..."),
    ]

    DelegateTool = create_unified_subagent_tool(
        configs,
        parent_toolset,
        model="anthropic:claude-sonnet-4",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.subagents.builder import _build_subagent_agent
from ya_agent_sdk.subagents.config import SubagentConfig
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.core.subagent.factory import (
    SubagentCallFunc,
    create_self_fork_call_func,
    create_subagent_call_func,
)

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model

logger = get_logger(__name__)

SELF_SUBAGENT_NAME = "self"
SELF_SUBAGENT_INSTRUCTION = """Fork the current agent with the current message history, system prompt, model, capabilities, and ordinary tools. Delegation tools are hidden from self forks, so use them for focused work that benefits from the parent context."""


@runtime_checkable
class UnifiedSubagentToolClass(Protocol):
    """Protocol for classes created by create_unified_subagent_tool."""

    _available_subagents: tuple[str, ...]

    @staticmethod
    def _get_roster_instruction(ctx: RunContext[AgentContext]) -> str | None: ...

    @staticmethod
    def _can_delegate(ctx: RunContext[AgentContext]) -> bool: ...


@dataclass
class SubagentEntry:
    """Internal registry entry for a subagent."""

    config: SubagentConfig
    agent: Agent[AgentContext, str]
    call_func: SubagentCallFunc
    required_tools: list[str] | None


def _build_subagent_entry(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    sdk_capabilities: list[AbstractCapability[Any]] | None = None,
) -> SubagentEntry:
    """Build a SubagentEntry from config."""
    agent, resolved_model_cfg = _build_subagent_agent(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
        sdk_capabilities=sdk_capabilities,
    )
    call_func = create_subagent_call_func(agent, model_cfg=resolved_model_cfg)

    return SubagentEntry(
        config=config,
        agent=agent,
        call_func=call_func,
        required_tools=config.tools,
    )


def _is_subagent_available(
    entry: SubagentEntry,
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> bool:
    """Check if a subagent is available based on its required tools."""
    if entry.required_tools is None:
        return True
    return all(parent_toolset.is_tool_available(name, ctx) for name in entry.required_tools)


def _is_self_fork_available(ctx: RunContext[AgentContext]) -> bool:
    """Return whether the current runtime has a self fork agent configured."""
    return ctx.deps.self_fork_agent is not None


def _available_subagent_entries(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> list[tuple[str, SubagentEntry]]:
    """Return configured subagents currently available to the parent."""
    return [(name, entry) for name, entry in entries.items() if _is_subagent_available(entry, parent_toolset, ctx)]


def _has_available_target(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> bool:
    """Return whether self fork or any configured subagent is callable."""
    return _is_self_fork_available(ctx) or bool(_available_subagent_entries(entries, parent_toolset, ctx))


def _generate_subagent_roster_instruction(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> str | None:
    """Generate a concise roster of currently available subagents."""
    available_entries = _available_subagent_entries(entries, parent_toolset, ctx)
    self_available = _is_self_fork_available(ctx)

    if not available_entries and not self_available:
        return None

    lines = ["Available subagents:"]
    if self_available:
        lines.append(f'<subagent name="{SELF_SUBAGENT_NAME}">')
        lines.append(SELF_SUBAGENT_INSTRUCTION)
        lines.append("</subagent>\n")

    for name, entry in available_entries:
        lines.append(f'<subagent name="{name}">')
        lines.append((entry.config.instruction or entry.config.description).strip())
        lines.append("</subagent>\n")

    return "\n".join(lines).rstrip()


def _generate_instruction(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> str | None:
    """Generate dynamic instruction listing available subagents."""
    roster = _generate_subagent_roster_instruction(entries, parent_toolset, ctx)
    available_entries = _available_subagent_entries(entries, parent_toolset, ctx)
    self_available = _is_self_fork_available(ctx)

    if roster is None:
        return None

    lines = ["Use the delegate tool for bounded subtasks that can return compact results.\n"]
    lines.append("<delegation-best-practices>")
    lines.append("Plan first, then call multiple delegates in the same response for independent work.")
    if self_available:
        lines.append(
            "Use self forks for full-context plan steps, mid-task repository exploration, "
            "assumption checks, approach comparisons, and implementation spikes."
        )
    if available_entries:
        lines.append("Use named specialist subagents when a listed role matches the task.")
    lines.append("Ask each delegate to return concise findings, changed files, tests run, and risks.")
    lines.append("</delegation-best-practices>\n")

    lines.append(roster)
    lines.append("")

    lines.append("<execution-model>")
    lines.append("Delegate calls are blocking: the parent waits for each delegated result before proceeding.")
    lines.append("Multiple delegate calls in the same model response run concurrently.")
    lines.append("The parent resumes after all delegate calls in that response complete.")
    lines.append("Sequential delegate calls across turns run serially.")
    lines.append("</execution-model>")

    return "\n".join(lines)


def _build_registry(
    configs: Sequence[SubagentConfig],
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    sdk_capabilities: list[AbstractCapability[Any]] | None = None,
) -> dict[str, SubagentEntry]:
    """Build registry of subagent entries from configs."""
    registry: dict[str, SubagentEntry] = {}
    for config in configs:
        if config.name == SELF_SUBAGENT_NAME:
            msg = f"{SELF_SUBAGENT_NAME!r} is reserved for the built-in self fork subagent"
            raise ValueError(msg)
        entry = _build_subagent_entry(
            config,
            parent_toolset,
            model=model,
            model_settings=model_settings,
            model_cfg=model_cfg,
            inherit_hooks=inherit_hooks,
            pre_capabilities=pre_capabilities,
            capabilities=capabilities,
            sdk_capabilities=sdk_capabilities,
        )
        registry[config.name] = entry
    return registry


def create_unified_subagent_tool(
    configs: Sequence[SubagentConfig],
    parent_toolset: Toolset[Any],
    *,
    name: str = "delegate",
    description: str = "Delegate task to a specialized subagent",
    hidden: bool = False,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a unified subagent tool from multiple SubagentConfigs.

    This creates a single tool that can delegate to any of the configured subagents
    by specifying the subagent_name parameter. This is an alternative to creating
    individual tools for each subagent.

    Args:
        configs: List of SubagentConfig objects defining the subagents.
        parent_toolset: The parent toolset to derive tools from.
        name: Tool name (default: "delegate").
        description: Tool description shown to the model.
        hidden: If True, keep the tool callable by code but hide it from model-visible tools and instructions.
        model: Fallback model for subagents with model="inherit".
        model_settings: Fallback model settings for subagents.
        model_cfg: Fallback ModelConfig for subagents.
        inherit_hooks: Whether to inherit hooks from parent toolset.
        pre_capabilities: Parent pre-capabilities to inherit (if config doesn't override).
        capabilities: Parent user capabilities to inherit (if config doesn't override).

    Returns:
        A BaseTool subclass that delegates to subagents by name.
    """
    return _create_unified_subagent_tool(
        configs,
        parent_toolset,
        name=name,
        description=description,
        hidden=hidden,
        model=model,
        model_settings=model_settings,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
    )


def _ensure_configs(configs: Sequence[SubagentConfig]) -> None:
    """Validate that unified subagent tool creation has configured subagents."""
    if not configs:
        msg = "At least one SubagentConfig is required"
        raise ValueError(msg)


def _is_unified_tool_available(
    *,
    hidden: bool,
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> bool:
    """Return model-visible availability for the unified subagent tool."""
    if hidden:
        return False
    return _has_available_target(entries, parent_toolset, ctx)


async def _get_unified_tool_instruction(
    *,
    hidden: bool,
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> str | None:
    """Return model-visible instructions for the unified subagent tool."""
    if hidden:
        return None
    return _generate_instruction(entries, parent_toolset, ctx)


def _get_unified_tool_roster_instruction(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> str | None:
    """Return subagent roster instructions for wrappers around a hidden backend."""
    return _generate_subagent_roster_instruction(entries, parent_toolset, ctx)


def _can_unified_tool_delegate(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> bool:
    """Return whether the unified backend has at least one callable target."""
    return _has_available_target(entries, parent_toolset, ctx)


async def _call_unified_subagent(
    tool: BaseTool,
    ctx: RunContext[AgentContext],
    *,
    subagent_name: str,
    prompt: str,
    agent_id: str | None,
    entries: dict[str, SubagentEntry],
    subagent_names: tuple[str, ...],
    parent_toolset: Toolset[Any],
    self_call_func: SubagentCallFunc,
) -> str:
    """Call the selected unified subagent target."""
    if subagent_name == SELF_SUBAGENT_NAME:
        return await self_call_func(tool, ctx, prompt, agent_id)

    if subagent_name not in entries:
        available = ", ".join((SELF_SUBAGENT_NAME, *subagent_names))
        return f"Error: Unknown subagent '{subagent_name}'. Available: {available}"

    entry = entries[subagent_name]
    if not _is_subagent_available(entry, parent_toolset, ctx):
        missing = [
            tool_name
            for tool_name in entry.required_tools or []
            if not parent_toolset.is_tool_available(tool_name, ctx)
        ]
        return f"Error: Subagent '{subagent_name}' is not available. Missing required tools: {missing}"

    return await entry.call_func(tool, ctx, prompt, agent_id)


def _create_unified_subagent_tool(
    configs: Sequence[SubagentConfig],
    parent_toolset: Toolset[Any],
    *,
    name: str = "delegate",
    description: str = "Delegate task to a specialized subagent",
    hidden: bool = False,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    sdk_capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a unified subagent tool, including SDK internal capabilities."""
    _ensure_configs(configs)

    # Build registry of subagent entries
    registry = _build_registry(
        configs,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
        sdk_capabilities=sdk_capabilities,
    )

    # Store references for closure
    subagent_names = tuple(registry.keys())
    _registry = registry
    _parent_toolset = parent_toolset
    _self_call_func = create_self_fork_call_func(model_cfg=model_cfg)

    class UnifiedSubagentTool(BaseTool):
        """Dynamically created unified subagent tool."""

        # These will be overwritten
        name = ""
        description = ""

        # Store names for introspection and wrappers.
        _available_subagents: tuple[str, ...] = subagent_names

        tags = frozenset({"delegation"})

        def is_available(self, ctx: RunContext[AgentContext]) -> bool:
            """Tool is available if self fork or at least one configured subagent is available."""
            return _is_unified_tool_available(
                hidden=hidden,
                entries=_registry,
                parent_toolset=_parent_toolset,
                ctx=ctx,
            )

        async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            """Generate instruction listing available subagents."""
            return await _get_unified_tool_instruction(
                hidden=hidden,
                entries=_registry,
                parent_toolset=_parent_toolset,
                ctx=ctx,
            )

        async def call(
            self,
            ctx: RunContext[AgentContext],
            subagent_name: Annotated[str, Field(description="Name of the subagent to delegate to")],
            prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
            agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
        ) -> str:
            """Delegate task to the specified subagent."""
            return await _call_unified_subagent(
                self,
                ctx,
                subagent_name=subagent_name,
                prompt=prompt,
                agent_id=agent_id,
                entries=_registry,
                subagent_names=subagent_names,
                parent_toolset=_parent_toolset,
                self_call_func=_self_call_func,
            )

    def _get_roster_instruction(ctx: RunContext[AgentContext]) -> str | None:
        return _get_unified_tool_roster_instruction(_registry, _parent_toolset, ctx)

    def _can_delegate(ctx: RunContext[AgentContext]) -> bool:
        return _can_unified_tool_delegate(_registry, _parent_toolset, ctx)

    # Set class attributes
    UnifiedSubagentTool.name = name
    UnifiedSubagentTool.description = description
    UnifiedSubagentTool._get_roster_instruction = staticmethod(_get_roster_instruction)
    UnifiedSubagentTool._can_delegate = staticmethod(_can_delegate)
    UnifiedSubagentTool.__name__ = f"{_to_pascal_case(name)}Tool"
    UnifiedSubagentTool.__qualname__ = UnifiedSubagentTool.__name__

    return UnifiedSubagentTool


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts)


def get_available_subagent_names(tool_cls: type[BaseTool]) -> tuple[str, ...]:
    """Get the available subagent names from a unified subagent tool class.

    This reads the _available_subagents class attribute set during tool creation.

    Args:
        tool_cls: A tool class created by create_unified_subagent_tool.

    Returns:
        Tuple of subagent names.

    Raises:
        TypeError: If the tool is not a unified subagent tool.
    """
    if not isinstance(tool_cls, UnifiedSubagentToolClass):
        msg = "Tool class does not appear to be a unified subagent tool"
        raise TypeError(msg)
    return tool_cls._available_subagents
