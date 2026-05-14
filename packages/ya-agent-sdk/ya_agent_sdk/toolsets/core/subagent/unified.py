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
    create_subagent_call_func,
)

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model

logger = get_logger(__name__)


@runtime_checkable
class UnifiedSubagentToolClass(Protocol):
    """Protocol for classes created by create_unified_subagent_tool."""

    _available_subagents: tuple[str, ...]


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


def _generate_instruction(
    entries: dict[str, SubagentEntry],
    parent_toolset: Toolset[Any],
    ctx: RunContext[AgentContext],
) -> str | None:
    """Generate dynamic instruction listing available subagents."""
    available_entries = [
        (name, entry) for name, entry in entries.items() if _is_subagent_available(entry, parent_toolset, ctx)
    ]

    if not available_entries:
        return None

    lines = ["Use the delegate tool to call specialized subagents:\n"]

    for name, entry in available_entries:
        instruction = entry.config.instruction
        lines.append(f'<subagent name="{name}">')
        if instruction:
            lines.append(instruction.strip())
        else:
            lines.append(entry.config.description)
        lines.append("</subagent>\n")

    lines.append("<execution-model>")
    lines.append("Delegate calls are BLOCKING -- the agent waits for each subagent to finish before proceeding.")
    lines.append("Multiple delegate calls in the SAME model response run concurrently (parallel), but the agent")
    lines.append("is still blocked until ALL of them complete. This is parallel-but-blocking, not async.")
    lines.append("For independent tasks, call multiple delegates in a single response to run them in parallel.")
    lines.append("Sequential delegate calls (one per turn) run serially.")
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
        model=model,
        model_settings=model_settings,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
    )


def _create_unified_subagent_tool(
    configs: Sequence[SubagentConfig],
    parent_toolset: Toolset[Any],
    *,
    name: str = "delegate",
    description: str = "Delegate task to a specialized subagent",
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    sdk_capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a unified subagent tool, including SDK internal capabilities."""
    if not configs:
        msg = "At least one SubagentConfig is required"
        raise ValueError(msg)

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
    _subagent_names = subagent_names

    class UnifiedSubagentTool(BaseTool):
        """Dynamically created unified subagent tool."""

        # These will be overwritten
        name = ""
        description = ""

        # Store names for introspection and parameter description
        _available_subagents: tuple[str, ...] = subagent_names

        def is_available(self, ctx: RunContext[AgentContext]) -> bool:
            """Tool is available if at least one subagent is available."""
            return any(_is_subagent_available(entry, _parent_toolset, ctx) for entry in _registry.values())

        async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            """Generate instruction listing available subagents."""
            return _generate_instruction(_registry, _parent_toolset, ctx)

        async def call(
            self,
            ctx: RunContext[AgentContext],
            subagent_name: Annotated[str, Field(description="Name of the subagent to delegate to")],
            prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
            agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
        ) -> str:
            """Delegate task to the specified subagent."""
            # Validate subagent exists
            if subagent_name not in _registry:
                available = ", ".join(_registry.keys())
                return f"Error: Unknown subagent '{subagent_name}'. Available: {available}"

            entry = _registry[subagent_name]

            # Check availability
            if not _is_subagent_available(entry, _parent_toolset, ctx):
                missing = []
                if entry.required_tools:
                    for tool_name in entry.required_tools:
                        if not _parent_toolset.is_tool_available(tool_name, ctx):
                            missing.append(tool_name)
                return f"Error: Subagent '{subagent_name}' is not available. Missing required tools: {missing}"

            # Delegate to subagent
            return await entry.call_func(self, ctx, prompt, agent_id)

    # Set class attributes
    UnifiedSubagentTool.name = name
    UnifiedSubagentTool.description = description
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
