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
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, cast

from pydantic_ai import AgentSpec, DeferredToolRequests, InstructionPart, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities import Toolset as NativeToolsetCapability
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset
from ya_agent_sdk.agents.lifecycle import BaseLifecycleExtension, ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.agents.main import AgentRuntime, create_agent
from ya_agent_sdk.capabilities import (
    CapabilityCatalog,
    CodeActCapability,
    DocumentConversionCapability,
    FilesystemCapability,
    MediaReadCapability,
    NoteCapability,
    ResolvedCapabilityPlugins,
    RuntimeFoundationCapability,
    ShellCapability,
    SkillsCapability,
    TaskCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolProxyCapability,
    ToolTimeoutCapability,
    UserInteractionCapability,
    WebContentCapability,
    WebSearchCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.context import ModelConfig, SecurityConfig, ShellReviewConfig, ToolConfig
from ya_agent_sdk.events import NamespaceStatus, NamespaceStatusEvent
from ya_agent_sdk.mcp import MCPConfig, build_mcp_servers, extract_mcp_descriptions, extract_optional_mcps
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.subagents import (
    DelegationCapability,
    SelfForkPolicy,
    SubagentDeferredResolver,
    SubagentExecutionMode,
    SubagentExecutionService,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)
from ya_agent_sdk.toolsets.core.base import Toolset
from ya_agent_sdk.toolsets.search import create_best_strategy
from ya_agent_sdk.toolsets.skills.toolset import SHARED_SKILLS_DIR_NAME, SkillToolset

from yaacli.config import ConfigManager, SubagentsConfig, YaacliConfig
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.models import ChildPlanManifest
from yaacli.durable.subagents import (
    DurableSubagentCompletionDelivery,
    DurableSubagentInboxCapability,
    FileRetainedSubagentPlanProvider,
    FileSubagentExecutionStore,
    LocalProcessorSubagentExecutionHost,
    LocalSubagentDriver,
)
from yaacli.environment import TUIEnvironment
from yaacli.guards import GoalGuardCapability
from yaacli.logging import get_logger
from yaacli.model_profiles import (
    ResolvedModelProfile,
    get_startup_model_profile,
    resolve_profile_model_cfg,
)
from yaacli.session import TUIContext
from yaacli.subagent_config import (
    YAACLI_INHERIT_MODEL_CFG_METADATA_KEY,
    YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY,
    YAACLI_MODEL_CFG_METADATA_KEY,
    load_subagent_specs,
    materialize_subagent_model_configuration,
)
from yaacli.toolsets.monitored_shell import MonitoredShellTool

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeSourceSnapshot:
    """Immutable file-derived inputs shared by runtime construction."""

    system_prompt: str
    subagent_specs: tuple[SubagentSpec, ...]


def runtime_child_plan_manifest(runtime: AgentRuntime[Any, Any, Any]) -> ChildPlanManifest:
    """Read the exact active child plans from an assembled runtime."""
    for capability in runtime.capabilities:
        if not isinstance(capability, DelegationCapability):
            continue
        active_plans = capability.registry.list()
        return ChildPlanManifest(
            active_routes={plan.spec.route: plan.descriptor_id for plan in active_plans},
            descriptors=tuple(plan.to_descriptor() for plan in active_plans),
        )
    return ChildPlanManifest()


def compile_runtime_sources(
    config: YaacliConfig,
    *,
    config_dir: Path,
    include_subagents: bool,
) -> RuntimeSourceSnapshot:
    """Read mutable runtime source files exactly once before worker launch."""
    return RuntimeSourceSnapshot(
        system_prompt=_load_system_prompt(config),
        subagent_specs=(
            _compile_subagent_specs(
                config.subagents,
                config_dir=config_dir,
                enable_codeact=config.tools.enable_codeact,
            )
            if include_subagents
            else ()
        ),
    )


def build_runtime_agent_spec(
    config: YaacliConfig,
    *,
    profile: ResolvedModelProfile | None,
    capability_plugins: ResolvedCapabilityPlugins | None = None,
    agent_name: str = "yaacli_main_v2",
) -> dict[str, Any]:
    """Return the normalized main-agent definition for a process-local runtime."""
    model = profile.model if profile is not None else config.general.model
    settings = profile.model_settings if profile is not None else config.general.model_settings
    model_cfg = profile.model_cfg if profile is not None else config.general.model_cfg
    profile_instructions = profile.instructions if profile is not None else config.general.instructions
    capabilities = (
        capability_plugins.root_agent_spec.model_dump(mode="json")["capabilities"]
        if capability_plugins is not None
        else []
    )
    return {
        "name": agent_name,
        "model": model,
        "model_settings": dict(resolve_model_settings(settings) or {}),
        "instructions": [profile_instructions]
        if isinstance(profile_instructions, str) and profile_instructions.strip()
        else [],
        "metadata": {YAACLI_MODEL_CFG_METADATA_KEY: resolve_profile_model_cfg(model_cfg).model_dump(mode="json")},
        "capabilities": capabilities,
    }


@dataclass
class _OptionalMCPToolset(WrapperToolset[Any]):
    """Keep an optional direct MCP server from blocking the runtime."""

    server_name: str
    _entry_success: list[bool] = field(default_factory=list, init=False)
    _available: bool = field(default=True, init=False)
    _status: NamespaceStatus = field(default=NamespaceStatus.connected, init=False)

    @property
    def id(self) -> str:
        return self.server_name

    async def __aenter__(self) -> _OptionalMCPToolset:
        self._available = True
        self._status = NamespaceStatus.connected
        try:
            await self.wrapped.__aenter__()
        except Exception:
            self._available = False
            self._status = NamespaceStatus.skipped
            self._entry_success.append(False)
            logger.warning(
                "Optional MCP server %r failed to initialize, skipping",
                self.server_name,
                exc_info=True,
            )
        else:
            self._entry_success.append(True)
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if not self._entry_success:
            logger.warning("Optional MCP server %r exited without a matching entry", self.server_name)
            return None
        if not self._entry_success.pop():
            return None
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        if not self._available:
            await self._emit_status(ctx)
            return {}
        try:
            tools = await self.wrapped.get_tools(ctx)
        except Exception:
            self._available = False
            self._status = NamespaceStatus.error
            logger.warning(
                "Optional MCP server %r failed while listing tools, skipping",
                self.server_name,
                exc_info=True,
            )
            await self._emit_status(ctx)
            return {}
        await self._emit_status(ctx)
        return tools

    async def _emit_status(self, ctx: RunContext[Any]) -> None:
        await ctx.deps.emit_event(
            NamespaceStatusEvent(
                event_id=f"optional-mcp-{self.server_name}-{ctx.deps.run_id[:8]}",
                namespace_status={self.server_name: self._status},
            )
        )

    async def get_instructions(
        self,
        ctx: RunContext[Any],
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        if not self._available:
            return None
        try:
            return await self.wrapped.get_instructions(ctx)
        except Exception:
            self._available = False
            self._status = NamespaceStatus.error
            logger.warning(
                "Optional MCP server %r failed while loading instructions, skipping",
                self.server_name,
                exc_info=True,
            )
            return None


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


def _compile_subagent_specs(
    subagents_config: SubagentsConfig,
    *,
    config_dir: Path,
    enable_codeact: bool,
) -> tuple[SubagentSpec, ...]:
    """Load portable specs from supported files and apply explicit overrides."""
    loaded = load_subagent_specs(
        config_dir / "subagents",
        markdown_capabilities=_standard_child_capability_specs(enable_codeact=enable_codeact),
    )
    disabled = set(subagents_config.disabled)
    compiled: list[SubagentSpec] = []
    for route, spec in sorted(loaded.items()):
        if route in disabled:
            continue
        override = subagents_config.overrides.get(route)
        if override is None:
            compiled.append(spec)
            continue

        agent_payload = spec.agent.model_dump(mode="json", by_alias=True)
        if override.model is not None:
            agent_payload["model"] = override.model
        if override.model_settings is not None:
            agent_payload["model_settings"] = dict(resolve_model_settings(override.model_settings) or {})
            metadata = dict(agent_payload.get("metadata") or {})
            metadata.pop(YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY, None)
            agent_payload["metadata"] = metadata or None
        if override.model_cfg is not None:
            metadata = dict(agent_payload.get("metadata") or {})
            metadata.pop(YAACLI_INHERIT_MODEL_CFG_METADATA_KEY, None)
            metadata[YAACLI_MODEL_CFG_METADATA_KEY] = resolve_profile_model_cfg(override.model_cfg).model_dump(
                mode="json"
            )
            agent_payload["metadata"] = metadata
        compiled.append(spec.model_copy(update={"agent": AgentSpec.model_validate(agent_payload)}))
    logger.info("Loaded %d subagent specs", len(compiled))
    return tuple(compiled)


def _build_delegation_capability(
    manifest: ChildPlanManifest,
    *,
    default_model: str | None,
    default_mode: SubagentExecutionMode,
    host_capabilities: Sequence[AbstractCapability[Any]],
    durable_database_path: Path,
    durable_binding_ref: str,
    request_limit: int,
    default_model_cfg: ModelConfig,
    deferred_resolver: SubagentDeferredResolver | None,
    capability_catalog: CapabilityCatalog | None = None,
) -> DelegationCapability | None:
    if not manifest.descriptors:
        return None
    catalog = capability_catalog if capability_catalog is not None else build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model=default_model,
        host_capabilities=host_capabilities,
        restart_durable=False,
    )
    descriptors_by_id = {descriptor.descriptor_id: descriptor for descriptor in manifest.descriptors}
    active_plans = tuple(
        resolver.restore(descriptors_by_id[descriptor_id])
        for _route, descriptor_id in sorted(manifest.active_routes.items())
    )
    store = FileSubagentExecutionStore(durable_database_path)
    try:
        registry = SubagentRegistry(active_plans)
        for plan in active_plans:
            store.put_descriptor(plan)
        driver = LocalSubagentDriver(
            store=store,
            request_limit=request_limit,
            default_model_cfg=default_model_cfg,
            custom_capability_types=catalog.custom_capability_types,
            runtime_capabilities=(DurableSubagentInboxCapability(store=store),),
        )
        service = SubagentExecutionService(
            registry,
            store,
            driver,
            completion_delivery=DurableSubagentCompletionDelivery(durable_binding_ref),
            deferred_resolver=deferred_resolver,
            retained_plan_provider=FileRetainedSubagentPlanProvider(store, resolver),
            execution_host=LocalProcessorSubagentExecutionHost(),
        )
        return DelegationCapability(
            registry=registry,
            service=service,
            default_mode=default_mode,
        )
    except BaseException:
        store.close_sync()
        raise


def _standard_child_capability_specs(*, enable_codeact: bool) -> list[dict[str, Any]]:
    """Materialize YAACLI's standard safe child feature grants."""
    capability_specs: list[dict[str, Any]] = [
        {"RuntimeFoundationCapability": {}},
        {"MediaReadCapability": {}},
        {"DocumentConversionCapability": {}},
        {"FilesystemCapability": {}},
        {"ShellCapability": {}},
        {"WebSearchCapability": {}},
        {"WebContentCapability": {}},
        {"TaskCapability": {}},
        {"NoteCapability": {}},
    ]
    if enable_codeact:
        capability_specs.append({"CodeActCapability": {}})
    return capability_specs


def compile_child_plan_manifest(
    config: YaacliConfig,
    *,
    profile: ResolvedModelProfile | None,
    sources: RuntimeSourceSnapshot,
    capability_catalog: CapabilityCatalog | None = None,
) -> ChildPlanManifest:
    """Compile the exact active child plans required by a new root runtime."""
    active_model = profile.model if profile is not None else config.general.model
    active_settings = profile.model_settings if profile is not None else config.general.model_settings
    active_model_cfg = profile.model_cfg if profile is not None else config.general.model_cfg
    active_instructions = profile.instructions if profile is not None else config.general.instructions
    model_settings = cast(ModelSettings | None, resolve_model_settings(active_settings))
    model_cfg = resolve_profile_model_cfg(active_model_cfg)
    self_fork_policy = SelfForkPolicy(
        agent=AgentSpec.from_dict({
            "name": "self",
            "description": "Fork the current agent for a bounded parallel task",
            "model": active_model or None,
            "model_settings": dict(model_settings or {}),
            "instructions": "\n\n".join(
                part.strip() for part in (sources.system_prompt, active_instructions or "") if part.strip()
            ),
            "metadata": {YAACLI_MODEL_CFG_METADATA_KEY: model_cfg.model_dump(mode="json")},
            "capabilities": _standard_child_capability_specs(
                enable_codeact=config.tools.enable_codeact,
            ),
        }),
        execution_modes=(
            SubagentExecutionMode.foreground,
            SubagentExecutionMode.background,
        ),
    )
    host_capabilities: tuple[AbstractCapability[Any], ...] = (
        ToolApprovalCapability(
            tools=frozenset(config.tools.need_approval),
            toolset_ids=frozenset(config.tools.need_approval_mcps),
        ),
        ToolObservationCapability(),
        ToolTimeoutCapability(),
    )
    catalog = capability_catalog if capability_catalog is not None else build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model=active_model or None,
        host_capabilities=host_capabilities,
        restart_durable=False,
    )
    child_specs = tuple(
        materialize_subagent_model_configuration(
            spec,
            inherited_model_settings=dict(model_settings) if model_settings is not None else None,
            inherited_model_cfg=model_cfg,
        )
        for spec in sources.subagent_specs
    )
    active_plans = [resolver.resolve(spec) for spec in child_specs]
    if not any(plan.spec.route == "self" for plan in active_plans):
        active_plans.append(resolver.resolve_self(self_fork_policy))
    return ChildPlanManifest(
        active_routes={plan.spec.route: plan.descriptor_id for plan in active_plans},
        descriptors=tuple(plan.to_descriptor() for plan in active_plans),
    )


def create_tui_runtime(
    config: YaacliConfig,
    mcp_config: MCPConfig | None = None,
    *,
    working_dir: Path | None = None,
    system_prompt: str | None = None,
    child_plan_manifest: ChildPlanManifest | None = None,
    config_dir: Path | None = None,
    model_profile: ResolvedModelProfile | None = None,
    subagent_default_mode: SubagentExecutionMode | None = None,
    enable_user_input: bool = False,
    skill_toolset: SkillToolset | None = None,
    durable_binding_ref: str | None = None,
    durable_database_path: Path | None = None,
    subagent_deferred_resolver: SubagentDeferredResolver | None = None,
    agent_spec: AgentSpec | None = None,
    capability_catalog: CapabilityCatalog | None = None,
    agent_name: str = "yaacli_main_v2",
) -> AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment]:
    """Compile YAACLI configuration into one capability-first runtime plan."""
    global_config_dir = config_dir or ConfigManager.DEFAULT_CONFIG_DIR
    ConfigManager(config_dir=global_config_dir).ensure_config_dir()

    active_profile = model_profile or get_startup_model_profile(config, global_config_dir)
    active_model = active_profile.model if active_profile else config.general.model
    active_settings_config = active_profile.model_settings if active_profile else config.general.model_settings
    active_model_cfg = active_profile.model_cfg if active_profile else config.general.model_cfg
    active_instructions = active_profile.instructions if active_profile else config.general.instructions
    model_settings = cast(ModelSettings | None, resolve_model_settings(active_settings_config))
    model_cfg = resolve_profile_model_cfg(active_model_cfg)
    effective_system_prompt = system_prompt if system_prompt is not None else _load_system_prompt(config)
    effective_catalog = capability_catalog if capability_catalog is not None else build_default_capability_catalog()
    effective_agent_spec = agent_spec or AgentSpec.model_validate(
        build_runtime_agent_spec(config, profile=active_profile, agent_name=agent_name)
    )

    workspace_dir = working_dir or Path.cwd()
    project_config_dir = workspace_dir / ConfigManager.PROJECT_CONFIG_DIR
    env_kwargs: dict[str, Any] = {
        "default_path": workspace_dir,
        "allowed_paths": [
            Path(tempfile.gettempdir()),
            global_config_dir,
            Path.home() / ".agents",
            workspace_dir,
            project_config_dir,
        ],
        "instructions_paths": [workspace_dir],
        "include_os_env": config.include_os_env,
    }

    tool_config_kwargs: dict[str, Any] = {}
    if config.media.s3.enabled and config.media.s3.bucket:
        try:
            from ya_agent_sdk.media import S3MediaConfig, create_s3_media_hook

            s3 = config.media.s3
            media_config = S3MediaConfig(
                bucket=s3.bucket,
                region=s3.region,
                access_key_id=s3.access_key_id,
                secret_access_key=s3.secret_access_key,
                endpoint_url=s3.endpoint_url,
                prefix=s3.prefix,
                url_mode=s3.url_mode,
                cdn_base_url=s3.cdn_base_url,
                presign_expires_seconds=s3.presign_expires_seconds,
                force_path_style=s3.force_path_style,
            )
            tool_config_kwargs["video_to_url_hook"] = create_s3_media_hook(media_config)
        except ImportError:
            logger.warning("S3 media upload requires the ya-agent-sdk s3 extra")

    context_kwargs: dict[str, Any] = {
        "tool_config": ToolConfig(**tool_config_kwargs),
        "durable_binding_ref": durable_binding_ref,
        "model_profile_instructions": active_instructions,
        "need_user_approve_tools": list(config.tools.need_approval),
        "need_user_approve_mcps": list(config.tools.need_approval_mcps),
    }
    if config.shell_env:
        context_kwargs["shell_env"] = dict(config.shell_env)
    if config.security.shell_review.enabled:
        context_kwargs["security"] = SecurityConfig(
            shell_review=ShellReviewConfig.model_validate(config.security.shell_review.model_dump())
        )

    capabilities: list[AbstractCapability[Any]] = [
        RuntimeFoundationCapability(),
        MediaReadCapability(),
        DocumentConversionCapability(),
        FilesystemCapability(),
        ShellCapability(),
        WebSearchCapability(),
        WebContentCapability(),
        TaskCapability(),
        NoteCapability(),
    ]
    if durable_binding_ref is not None:
        capabilities.insert(1, DurableInboxPumpCapability())
    if skill_toolset is None:
        capabilities.append(SkillsCapability(extra_dir_names=(SHARED_SKILLS_DIR_NAME,)))
    else:
        capabilities.append(NativeToolsetCapability(skill_toolset, id="skills"))
    if enable_user_input:
        capabilities.append(UserInteractionCapability())
    capabilities.append(
        NativeToolsetCapability(
            Toolset(
                tools=[MonitoredShellTool],
                toolset_id="yaacli_background_shell",
            ),
            id="yaacli_background_shell",
        )
    )
    if config.tools.enable_codeact:
        capabilities.append(CodeActCapability())

    if subagent_default_mode is not None:
        if durable_database_path is None or durable_binding_ref is None or child_plan_manifest is None:
            raise ValueError(
                "YAACLI subagents require an exact child manifest, product database, and execution binding"
            )
        delegation = _build_delegation_capability(
            child_plan_manifest,
            default_model=active_model or None,
            default_mode=subagent_default_mode,
            host_capabilities=(
                ToolApprovalCapability(
                    tools=frozenset(config.tools.need_approval),
                    toolset_ids=frozenset(config.tools.need_approval_mcps),
                ),
                ToolObservationCapability(),
                ToolTimeoutCapability(),
            ),
            durable_database_path=durable_database_path,
            durable_binding_ref=durable_binding_ref,
            request_limit=config.general.max_requests,
            default_model_cfg=model_cfg,
            deferred_resolver=subagent_deferred_resolver,
            capability_catalog=effective_catalog,
        )
        if delegation is not None:
            capabilities.append(delegation)

    if mcp_config is not None:
        mcp_servers = build_mcp_servers(
            mcp_config,
            need_approval_mcps=config.tools.need_approval_mcps,
        )
        if mcp_servers and config.tools.mcp_mode == "proxy":
            capabilities.append(
                ToolProxyCapability(
                    toolsets=tuple(mcp_servers),
                    namespace_descriptions=extract_mcp_descriptions(mcp_config),
                    search_strategy=create_best_strategy(),
                    optional_namespaces=frozenset(extract_optional_mcps(mcp_config)),
                    prefix="mcp",
                )
            )
        else:
            optional_mcps = extract_optional_mcps(mcp_config)
            for server in mcp_servers:
                if server.id is None:
                    logger.warning("Skipping direct MCP toolset without a stable id")
                    continue
                direct: AbstractToolset[Any] = server
                if server.id in optional_mcps:
                    direct = _OptionalMCPToolset(direct, server_name=server.id)
                server_config = mcp_config.servers.get(server.id)
                prefix = server.id if server_config is None or server_config.prefix is None else server_config.prefix
                if prefix:
                    direct = direct.prefixed(prefix)
                capabilities.append(NativeToolsetCapability(direct, id=f"mcp:{server.id}"))

    capabilities.extend([
        ToolApprovalCapability(
            tools=frozenset(config.tools.need_approval),
            toolset_ids=frozenset(config.tools.need_approval_mcps),
        ),
        ToolObservationCapability(),
        ToolTimeoutCapability(),
        GoalGuardCapability(),
    ])

    runtime = create_agent(
        model=active_model or None,
        spec=effective_agent_spec,
        custom_capability_types=effective_catalog.custom_capability_types,
        capabilities=cast(Sequence[AbstractCapability[TUIContext]], capabilities),
        model_settings=model_settings,
        output_type=[str, DeferredToolRequests],
        env=TUIEnvironment,
        env_kwargs=env_kwargs,
        context_type=TUIContext,
        model_cfg=model_cfg,
        context_kwargs=context_kwargs,
        system_prompt=effective_system_prompt,
        lifecycle_extensions=[GoalContextHandoffExtension()],
        agent_name=agent_name,
    )
    logger.info(
        "Created capability-first TUI runtime: model=%s capabilities=%d",
        active_model,
        len(capabilities),
    )
    return runtime
