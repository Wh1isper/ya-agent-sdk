"""Shared subagent Agent builder.

This module provides the shared logic for constructing a pydantic-ai Agent from
a SubagentConfig. It is used by both individual subagent tools (factory.py) and
unified subagent tools (unified.py).

Separated into its own module to avoid circular imports between
ya_agent_sdk.subagents.factory and ya_agent_sdk.toolsets.core.subagent.unified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory

from ya_agent_sdk.agents.guards import attach_message_bus_guard
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.filters.system_prompt import create_system_prompt_filter
from ya_agent_sdk.presets import INHERIT, resolve_model_cfg, resolve_model_settings
from ya_agent_sdk.subagents.config import SubagentConfig
from ya_agent_sdk.toolsets.core.base import Toolset

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model


def _resolve_model(config: SubagentConfig, model: str | Model | None) -> str | Model:
    """Resolve effective model from config and fallback."""
    if config.model is not None and config.model != INHERIT:
        return config.model
    if model is not None:
        return model
    return "test"  # Placeholder, actual model passed at runtime


def _resolve_model_settings(
    config: SubagentConfig, model_settings: ModelSettings | dict[str, Any] | str | None
) -> dict[str, Any] | None:
    """Resolve effective model settings from config and fallback."""
    if config.model_settings is not None and config.model_settings != INHERIT:
        return resolve_model_settings(config.model_settings)
    if model_settings is not None:
        return resolve_model_settings(model_settings)
    return None


def _resolve_model_cfg(config: SubagentConfig, model_cfg: ModelConfig | None) -> ModelConfig | None:
    """Resolve effective ModelConfig from config and fallback.

    Resolution order:
    1. config.model_cfg is not None and != 'inherit' -> resolve to ModelConfig
    2. Otherwise use model_cfg fallback (inherit from parent)
    """
    resolved = resolve_model_cfg(config.model_cfg)
    if resolved is not None:
        return ModelConfig(**resolved)
    return model_cfg


def _collect_tools(config: SubagentConfig) -> list[str] | None:
    """Collect all tools (required + optional) from config."""
    if config.tools is None and config.optional_tools is None:
        return None
    all_tools: list[str] = []
    if config.tools:
        all_tools.extend(config.tools)
    if config.optional_tools:
        all_tools.extend(config.optional_tools)
    return all_tools


def _build_toolsets(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    inherit_hooks: bool = False,
) -> list[Any]:
    """Build the toolset list for a subagent based on its configuration.

    Handles the 4 combinations from the independent toolsets matrix:

    | config.toolsets | config.tools | Result                                     |
    |-----------------|-------------|---------------------------------------------|
    | None            | None        | All parent tools                            |
    | None            | set         | Parent subset + auto_inherit                |
    | set             | None        | Own toolsets + auto_inherit from parent      |
    | set             | set         | Own toolsets + parent subset + auto_inherit  |
    """
    inherited_tools = _collect_tools(config)

    if config.toolsets is not None:
        # When own toolsets exist and tools=None, only get auto_inherit (not all parent tools)
        if inherited_tools is None:
            parent_subset = parent_toolset.subset([], include_auto_inherit=True, inherit_hooks=inherit_hooks)
        else:
            parent_subset = parent_toolset.subset(
                inherited_tools, include_auto_inherit=True, inherit_hooks=inherit_hooks
            )
        return [*config.toolsets, parent_subset]
    else:
        # Current behavior: None means all parent tools
        parent_subset = parent_toolset.subset(inherited_tools, include_auto_inherit=True, inherit_hooks=inherit_hooks)
        return [parent_subset]


def _resolve_capabilities(
    config: SubagentConfig,
    parent_pre_capabilities: list[AbstractCapability[Any]] | None,
    parent_capabilities: list[AbstractCapability[Any]] | None,
    sdk_capabilities: list[AbstractCapability[Any]] | None,
) -> list[AbstractCapability[Any]]:
    """Resolve effective capabilities for a subagent."""
    user_pre_capabilities = config.pre_capabilities if config.pre_capabilities is not None else parent_pre_capabilities
    user_capabilities = config.capabilities if config.capabilities is not None else parent_capabilities
    return [
        *(user_pre_capabilities or []),
        *(sdk_capabilities or []),
        *(user_capabilities or []),
        ProcessHistory(create_system_prompt_filter(config.system_prompt)),
    ]


def build_subagent_agent(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    model_cfg: ModelConfig | None = None,
    inherit_hooks: bool = False,
    pre_capabilities: list[AbstractCapability[Any]] | None = None,
    capabilities: list[AbstractCapability[Any]] | None = None,
) -> tuple[Agent[AgentContext, str], ModelConfig | None]:
    """Build a pydantic-ai Agent from a SubagentConfig.

    Args:
        config: The parsed subagent configuration.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings.
        model_cfg: Fallback ModelConfig.
        inherit_hooks: Whether to inherit hooks from parent toolset.
        pre_capabilities: Parent pre-capabilities to inherit (if config doesn't override).
        capabilities: Parent user capabilities to inherit (if config doesn't override).

    Returns:
        Tuple of (Agent, resolved_model_cfg).
    """
    return _build_subagent_agent(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        model_cfg=model_cfg,
        inherit_hooks=inherit_hooks,
        pre_capabilities=pre_capabilities,
        capabilities=capabilities,
    )


def _build_subagent_agent(
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
) -> tuple[Agent[AgentContext, str], ModelConfig | None]:
    """Build a subagent Agent, including SDK internal capabilities."""
    effective_model = _resolve_model(config, model)
    resolved_settings = _resolve_model_settings(config, model_settings)
    resolved_model_cfg = _resolve_model_cfg(config, model_cfg)
    resolved_capabilities = _resolve_capabilities(config, pre_capabilities, capabilities, sdk_capabilities)
    toolsets = _build_toolsets(config, parent_toolset, inherit_hooks=inherit_hooks)

    agent: Agent[AgentContext, str] = Agent(
        model=infer_model(effective_model),
        system_prompt=config.system_prompt,
        toolsets=toolsets,
        model_settings=resolved_settings,  # type: ignore[arg-type]
        deps_type=AgentContext,
        capabilities=resolved_capabilities,
        name=config.name,
    )

    # Attach message bus guard for pending message handling
    attach_message_bus_guard(agent)

    return agent, resolved_model_cfg
