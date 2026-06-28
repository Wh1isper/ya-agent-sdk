"""Agent runtime factory for yaacli.

This module provides factory functions to create AgentRuntime configured
for TUI usage. It wraps the SDK's create_agent() with TUI-specific
configuration and integrates MCP toolsets.

Example:
    from yaacli.runtime import create_tui_runtime
    from yaacli.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load_config()
    mcp_config = config_manager.load_mcp_config()

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
    )
    async with runtime:
        # Use runtime.agent, runtime.ctx, runtime.env
        pass
"""

from __future__ import annotations

import tempfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import DeferredToolRequests, ModelSettings
from pydantic_ai.output import OutputSpec
from ya_agent_sdk.agents.lifecycle import BaseLifecycleExtension, ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.agents.main import AgentRuntime, create_agent
from ya_agent_sdk.context import SecurityConfig, ShellReviewConfig, ToolConfig
from ya_agent_sdk.mcp import build_mcp_servers, extract_mcp_descriptions, extract_optional_mcps
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.subagents import SubagentConfig, load_subagents_from_dir
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.content import tools as content_tools
from ya_agent_sdk.toolsets.core.context import tools as context_tools
from ya_agent_sdk.toolsets.core.document import tools as document_tools
from ya_agent_sdk.toolsets.core.enhance import tools as enhance_tools
from ya_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from ya_agent_sdk.toolsets.core.multimodal import tools as multimodal_tools
from ya_agent_sdk.toolsets.core.shell import tools as shell_tools
from ya_agent_sdk.toolsets.core.subagent import tools as subagent_tools
from ya_agent_sdk.toolsets.core.web import tools as web_tools
from ya_agent_sdk.toolsets.skills.toolset import SHARED_SKILLS_DIR_NAME, SkillToolset
from ya_agent_sdk.toolsets.tool_proxy.toolset import ToolProxyToolset
from ya_agent_sdk.toolsets.tool_search import create_best_strategy

from yaacli.background import DELEGATE_BACKEND_TOOL_NAME
from yaacli.config import ConfigManager, MCPConfig, SubagentsConfig, YaacliConfig
from yaacli.environment import TUIEnvironment
from yaacli.guards import attach_goal_guard
from yaacli.logging import get_logger
from yaacli.model_profiles import ResolvedModelProfile, get_startup_model_profile, resolve_profile_model_cfg
from yaacli.session import TUIContext
from yaacli.toolsets.background import AsyncDelegateTool, MonitoredShellTool, SpawnDelegateTool, SteerSubagentTool

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = get_logger(__name__)


class GoalContextHandoffExtension(BaseLifecycleExtension[TUIContext, TUIEnvironment]):
    """Bridge SDK context handoff lifecycle to YAACLI goal-mode state.

    SDK compaction and summarize-tool handoff are context lifecycle concerns.
    YAACLI owns goal completion semantics, so this extension only records that
    an active goal crossed a restored-context boundary. The goal guard then
    requires one fresh completion audit before accepting a completion marker.
    """

    name = "yaacli_goal_context_handoff"

    async def on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[TUIContext]) -> None:
        deps = ctx.deps
        if not isinstance(deps, TUIContext):
            return

        if isinstance(ctx.source, ContextHandoffSource):
            source = ctx.source.value
        else:
            source = str(ctx.source)
        deps.mark_goal_context_restored(source)


def _load_system_prompt(config: YaacliConfig) -> str:
    """Load system prompt from config or built-in default.

    Priority:
    1. Custom file path from config.general.system_prompt_file
    2. Built-in default from templates/system_prompt.md
    """
    if config.general.system_prompt_file:
        prompt_path = Path(config.general.system_prompt_file).expanduser()
        if prompt_path.exists():
            logger.debug("Loading system prompt from: %s", prompt_path)
            return prompt_path.read_text(encoding="utf-8")
        logger.warning("System prompt file not found: %s, using default", prompt_path)

    # Load built-in default
    template_files = resources.files("yaacli").joinpath("templates")
    default_prompt = template_files.joinpath("system_prompt.md").read_text(encoding="utf-8")
    logger.debug("Using built-in system prompt")
    return default_prompt


def _load_subagent_configs(
    subagents_config: SubagentsConfig,
    config_dir: Path | None = None,
) -> list[SubagentConfig]:
    """Load subagent configs from user config directory.

    Subagents are loaded from ~/.yaacli/subagents/
    and filtered based on the disabled list in config.

    Args:
        subagents_config: Subagents configuration with disabled list and overrides.
        config_dir: Config directory (defaults to ConfigManager.DEFAULT_CONFIG_DIR).

    Returns:
        List of SubagentConfig objects.
    """
    if config_dir is None:
        config_dir = ConfigManager.DEFAULT_CONFIG_DIR

    subagents_dir = config_dir / "subagents"
    if not subagents_dir.exists():
        logger.debug("Subagents directory not found: %s", subagents_dir)
        return []

    # Load all subagents from directory
    all_configs = load_subagents_from_dir(subagents_dir)
    logger.debug("Found %d subagents in %s", len(all_configs), subagents_dir)

    # Filter out disabled subagents
    disabled_set = set(subagents_config.disabled)
    enabled_configs: list[SubagentConfig] = []

    for name, cfg in all_configs.items():
        if name in disabled_set:
            logger.debug("Subagent disabled: %s", name)
            continue

        # Apply overrides if any
        if name in subagents_config.overrides:
            override = subagents_config.overrides[name]
            if override.model is not None:
                cfg = cfg.model_copy(update={"model": override.model})
            if override.model_settings is not None:
                cfg = cfg.model_copy(update={"model_settings": override.model_settings})
            logger.debug("Applied overrides for subagent: %s", name)

        enabled_configs.append(cfg)

    logger.info("Loaded %d subagents (disabled: %d)", len(enabled_configs), len(disabled_set))
    return enabled_configs


def create_tui_runtime(
    config: YaacliConfig,
    mcp_config: MCPConfig | None = None,
    *,
    working_dir: Path | None = None,
    system_prompt: str | None = None,
    config_dir: Path | None = None,
    model_profile: ResolvedModelProfile | None = None,
    enable_async_subagents: bool = True,
    enable_delegate_subagents: bool = False,
) -> AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment]:
    """Create AgentRuntime configured for TUI.

    This function wraps the SDK's create_agent() with TUI-specific
    configuration, integrating:
    - TUIEnvironment with Shell ABC background process management
    - TUIContext with SteeringManager
    - MCP servers from configuration
    - Steering guard for output validation

    Args:
        config: YAACLI CLI configuration.
        mcp_config: MCP server configuration. If None, no MCP servers are added.
        working_dir: Working directory for the environment. Defaults to cwd.
        system_prompt: Custom system prompt. If None, uses default.
        config_dir: Global config directory used for subagents and allowed paths.
        model_profile: Optional resolved model profile override.
        enable_async_subagents: Include background subagent tools. When synchronous delegation is disabled, the async tool is exposed as delegate.
        enable_delegate_subagents: Include the synchronous unified delegate subagent tool. Defaults to False for TUI async-only delegation.

    Returns:
        AgentRuntime configured for TUI usage. Use as async context manager.

    Example:
        runtime = create_tui_runtime(config, mcp_config)
        async with runtime:
            async with stream_agent(runtime, "Hello") as stream:
                async for event in stream:
                    print(event)
    """
    # Collect toolsets
    # Order matters: ToolProxyToolset must be LAST so its two fixed proxy
    # tools (search_tools, call_tool) occupy stable positions at the end
    # of the model's tool list, maximizing prompt cache hit rates.
    toolsets: list[AbstractToolset[Any]] = [
        SkillToolset(toolset_id="skills", extra_dir_names=[SHARED_SKILLS_DIR_NAME]),
    ]

    # Add MCP servers wrapped in ToolProxyToolset for on-demand invocation.
    # This is intentionally last: proxy tools occupy stable positions at
    # the end of the tool list, keeping all existing tool positions stable.
    if mcp_config:
        mcp_servers = build_mcp_servers(mcp_config, need_approval_mcps=config.tools.need_approval_mcps)
        if mcp_servers:
            mcp_descriptions = extract_mcp_descriptions(mcp_config)
            optional_mcps = extract_optional_mcps(mcp_config)
            mcp_proxy = ToolProxyToolset(
                toolsets=mcp_servers,
                namespace_descriptions=mcp_descriptions if mcp_descriptions else None,
                search_strategy=create_best_strategy(),
                optional_namespaces=optional_mcps if optional_mcps else None,
                prefix="mcp",
            )
            toolsets.append(mcp_proxy)
            logger.info(
                "Added %d MCP servers via ToolProxyToolset (descriptions: %d, prefix=mcp)",
                len(mcp_servers),
                len(mcp_descriptions),
            )

    # Environment configuration
    # Include system temp dir so TUI agents can read/write user-specified temp files.
    # Include global config dir in allowed_paths so agent can modify configs directly.
    # Include ~/.agents for shared skills following the Agent Skills open standard.
    # Order matters for skill priority (later = higher priority):
    #   system temp < ~/.yaacli < ~/.agents < project dir < project .yaacli
    global_config_dir = config_dir or ConfigManager.DEFAULT_CONFIG_DIR
    active_model_profile = model_profile or get_startup_model_profile(config, global_config_dir)
    active_model = active_model_profile.model if active_model_profile else config.general.model
    active_model_settings = (
        active_model_profile.model_settings if active_model_profile else config.general.model_settings
    )
    active_model_cfg = active_model_profile.model_cfg if active_model_profile else config.general.model_cfg

    # Ensure .gitignore exists in config dir to keep session data out of file tree context
    ConfigManager(config_dir=global_config_dir).ensure_config_dir()
    shared_agents_dir = Path.home() / ".agents"
    env_kwargs: dict[str, Any] = {}
    if working_dir:
        workspace_dir = working_dir
    else:
        workspace_dir = Path.cwd()
    project_config_dir = workspace_dir / ConfigManager.PROJECT_CONFIG_DIR
    system_tmp_dir = Path(tempfile.gettempdir())
    env_kwargs["default_path"] = workspace_dir
    env_kwargs["allowed_paths"] = [
        system_tmp_dir,
        global_config_dir,
        shared_agents_dir,
        workspace_dir,
        project_config_dir,
    ]
    env_kwargs["instructions_paths"] = [workspace_dir]

    # Shell environment isolation: configurable via config.toml
    env_kwargs["include_os_env"] = config.include_os_env

    # Model configuration - resolve from preset name or dict
    model_cfg = resolve_profile_model_cfg(active_model_cfg)
    if active_model_cfg:
        logger.debug(f"Using model_cfg: {active_model_cfg}")

    # Resolve model settings from preset name or dict
    model_settings = resolve_model_settings(active_model_settings)
    if model_settings:
        logger.debug(f"Using model settings: {active_model_settings} -> {model_settings}")

    # Tool configuration - setup media hooks if S3 is configured
    tool_config_kwargs: dict[str, Any] = {}

    if config.media.s3.enabled and config.media.s3.bucket:
        try:
            from ya_agent_sdk.media import S3MediaConfig, create_s3_media_hook

            s3_config = S3MediaConfig(
                bucket=config.media.s3.bucket,
                region=config.media.s3.region,
                access_key_id=config.media.s3.access_key_id,
                secret_access_key=config.media.s3.secret_access_key,
                endpoint_url=config.media.s3.endpoint_url,
                prefix=config.media.s3.prefix,
                url_mode=config.media.s3.url_mode,
                cdn_base_url=config.media.s3.cdn_base_url,
                presign_expires_seconds=config.media.s3.presign_expires_seconds,
                force_path_style=config.media.s3.force_path_style,
            )
            hook = create_s3_media_hook(s3_config)

            # Bedrock does not support image url, and our image processing is better than vendor
            # TODO: Maybe we can upload after processing in history_processor?
            # tool_config_kwargs["image_to_url_hook"] = hook

            # Use url hook for video, prevent too large requests
            tool_config_kwargs["video_to_url_hook"] = hook
            logger.info(
                "S3 media upload enabled: bucket=%s, url_mode=%s",
                config.media.s3.bucket,
                config.media.s3.url_mode,
            )
        except ImportError:
            logger.warning("S3 media upload requires boto3. Install with: pip install ya-agent-sdk[s3]")

    tool_config = ToolConfig(**tool_config_kwargs)
    if config.tools.need_approval:
        logger.debug("Tools requiring approval: %s", config.tools.need_approval)
    if config.tools.need_approval_mcps:
        logger.debug("MCP servers requiring approval: %s", config.tools.need_approval_mcps)

    # Load subagent configs whenever either visible synchronous delegation or
    # the async delegate backend is needed.
    subagent_configs = (
        _load_subagent_configs(config.subagents, config_dir=global_config_dir)
        if enable_delegate_subagents or enable_async_subagents
        else []
    )

    # Core tools
    core_tools: list[type[BaseTool]] = [
        *content_tools,
        *context_tools,
        *document_tools,
        *enhance_tools,
        *filesystem_tools,
        *multimodal_tools,
        *shell_tools,
        *web_tools,
    ]

    selected_background_tools: list[type[BaseTool]] = [MonitoredShellTool]
    if enable_async_subagents:
        if enable_delegate_subagents:
            selected_background_tools.extend([SpawnDelegateTool, SteerSubagentTool])
        else:
            selected_background_tools.extend([AsyncDelegateTool, SteerSubagentTool])

    # Build final tools list
    all_tools: list[type[BaseTool]] = [
        *core_tools,
        *subagent_tools,  # SubagentInfoTool for introspection
        *selected_background_tools,
    ]

    # Load system prompt
    effective_system_prompt = system_prompt or _load_system_prompt(config)

    # Output type - SDK's message_bus_guard automatically handles pending messages
    output_type: OutputSpec[str | DeferredToolRequests] = [str, DeferredToolRequests]
    # Create runtime using SDK factory
    # Use unified_subagents=True to create single 'delegate' tool for all subagents
    # Output retry budget must be large enough to support goal iterations + normal retries
    max_goal = config.general.max_goal_iterations
    output_retries = max_goal + 5

    # Pass config [shell_env] for shell command execution
    extra_ctx_kwargs: dict[str, Any] | None = None
    if config.shell_env:
        extra_ctx_kwargs = {"shell_env": dict(config.shell_env)}
    if config.security.shell_review.enabled:
        extra_ctx_kwargs = dict(extra_ctx_kwargs or {})
        extra_ctx_kwargs["security"] = SecurityConfig(
            shell_review=ShellReviewConfig.model_validate(config.security.shell_review.model_dump())
        )

    runtime = create_agent(
        model=active_model or None,
        model_settings=cast(ModelSettings, model_settings),
        output_type=output_type,
        env=TUIEnvironment,
        env_kwargs=env_kwargs,
        context_type=TUIContext,
        model_cfg=model_cfg,
        tools=all_tools,
        tool_config=tool_config,
        toolsets=toolsets if toolsets else None,
        system_prompt=effective_system_prompt,
        need_user_approve_tools=config.tools.need_approval or None,
        need_user_approve_mcps=config.tools.need_approval_mcps or None,
        subagent_configs=subagent_configs if subagent_configs else None,
        unified_subagents=True,
        unified_subagent_tool_name="delegate" if enable_delegate_subagents else DELEGATE_BACKEND_TOOL_NAME,
        hide_unified_subagent_tool=not enable_delegate_subagents,
        retries={"output": output_retries},
        extra_context_kwargs=extra_ctx_kwargs,
        lifecycle_extensions=[GoalContextHandoffExtension()],
    )

    # Attach goal guard for /goal command support
    attach_goal_guard(runtime.agent)

    logger.info(
        "Created TUI runtime: model=%s, toolsets=%d, output_retries=%d, async_subagents=%s, delegate_subagents=%s",
        active_model,
        len(toolsets),
        output_retries,
        enable_async_subagents,
        enable_delegate_subagents,
    )

    return runtime
