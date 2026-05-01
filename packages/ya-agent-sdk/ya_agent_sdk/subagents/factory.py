"""Subagent tool creation from configuration.

This module provides functions to create subagent tools from SubagentConfig objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.capabilities import AbstractCapability

from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.subagents.builder import _build_subagent_agent
from ya_agent_sdk.subagents.config import (
    SubagentConfig,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
)
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.core.subagent.factory import (
    create_subagent_call_func,
    create_subagent_tool,
)

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model


def create_subagent_tool_from_config(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from a SubagentConfig.

    Args:
        config: The parsed subagent configuration.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.
        history_processors: Deprecated history processors to use for the subagent.
        model_cfg: Fallback ModelConfig. Used if config.model_cfg is None.
        inherit_hooks: Whether to inherit hooks from parent toolset.
        pre_capabilities: Parent pre-capabilities to inherit (if config doesn't override).
        capabilities: Parent user capabilities to inherit (if config doesn't override).

    Returns:
        A BaseTool subclass that wraps the subagent.
    """
    return _create_subagent_tool(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
    )


def _create_subagent_tool(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
    sdk_capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from resolved configuration."""
    agent, resolved_model_cfg = _build_subagent_agent(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
        sdk_capabilities=sdk_capabilities,
    )
    return _create_tool(config, parent_toolset, agent, resolved_model_cfg)


def _create_tool(
    config: SubagentConfig, parent_toolset: Toolset[Any], agent: Any, model_cfg: ModelConfig | None
) -> type[BaseTool]:
    required_tools = config.tools

    def check_tools_available(ctx: RunContext[AgentContext]) -> bool:
        if required_tools is None:
            return True
        return all(parent_toolset.is_tool_available(name, ctx) for name in required_tools)

    return create_subagent_tool(
        name=config.name,
        description=config.description,
        call_func=create_subagent_call_func(agent, model_cfg=model_cfg),
        instruction=config.instruction,
        availability_check=check_tools_available,
    )


def create_subagent_tool_from_markdown(
    content: str | Path,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from markdown content or file path.

    This is the main convenience function for creating subagent tools.

    Args:
        content: Markdown string or path to markdown file.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.
        history_processors: Deprecated history processors to use for the subagent.
        model_cfg: Fallback ModelConfig. Used if config.model_cfg is None.
        pre_capabilities: Parent pre-capabilities to inherit (if config doesn't override).
        capabilities: Parent user capabilities to inherit (if config doesn't override).

    Returns:
        A BaseTool subclass that wraps the subagent.

    Example::

        # From file
        DebuggerTool = create_subagent_tool_from_markdown(
            "ya_agent_sdk/subagents/debugger.md",
            parent_toolset=main_toolset,
            model="anthropic:claude-sonnet-4",
        )

        # From string
        SearchTool = create_subagent_tool_from_markdown(
            '''
            ---
            name: search
            description: Search for information
            model_settings: anthropic_low
            ---
            You are a search agent...
            ''',
            parent_toolset=main_toolset,
        )
    """
    if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
        config = load_subagent_from_file(content)
    else:
        config = parse_subagent_markdown(content)

    return create_subagent_tool_from_config(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
    )


def load_subagent_tools_from_dir(
    dir_path: Path | str,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
) -> list[type[BaseTool]]:
    """Load all subagent tools from a directory.

    Scans the directory for .md files and creates a subagent tool for each.

    Args:
        dir_path: Path to the directory containing markdown files.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model for all subagents.
        model_settings: Fallback model settings for all subagents.
        history_processors: Deprecated history processors to use for all subagents.
        model_cfg: Fallback ModelConfig for all subagents.
        pre_capabilities: Parent pre-capabilities to inherit unless a subagent config overrides them.
        capabilities: Parent user capabilities to inherit unless a subagent config overrides them.

    Returns:
        List of BaseTool subclasses.

    Example::

        subagent_tools = load_subagent_tools_from_dir(
            "ya_agent_sdk/subagents",
            parent_toolset=main_toolset,
            model="anthropic:claude-sonnet-4",
            model_settings="anthropic_medium",
        )
    """
    configs = load_subagents_from_dir(dir_path)
    tools: list[type[BaseTool]] = []

    for config in configs.values():
        tool = create_subagent_tool_from_config(
            config,
            parent_toolset,
            model=model,
            model_settings=model_settings,
            history_processors=history_processors,
            model_cfg=model_cfg,
            inherit_hooks=inherit_hooks,
            pre_capabilities=pre_capabilities,
            capabilities=capabilities,
        )
        tools.append(tool)

    return tools
