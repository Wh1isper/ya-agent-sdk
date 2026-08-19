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

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, cast

from pydantic_ai import AgentSpec, DeferredToolRequests, InstructionPart, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities import Toolset as NativeToolsetCapability
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset
from ya_agent_sdk.agents.lifecycle import BaseLifecycleExtension, ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.agents.main import AgentRuntime, create_agent
from ya_agent_sdk.capabilities import (
    CodeActCapability,
    DocumentConversionCapability,
    FilesystemCapability,
    MediaReadCapability,
    NoteCapability,
    RuntimeFoundationCapability,
    ShellCapability,
    SkillsCapability,
    TaskCapability,
    ThinkingCapability,
    TodoCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolProxyCapability,
    ToolRetryCapability,
    ToolTimeoutCapability,
    UserInteractionCapability,
    WebContentCapability,
    WebSearchCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.context import ModelConfig, SecurityConfig, ShellReviewConfig, ToolConfig
from ya_agent_sdk.events import NamespaceStatus, NamespaceStatusEvent
from ya_agent_sdk.mcp import build_mcp_servers, extract_mcp_descriptions, extract_optional_mcps
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.subagents import (
    DelegationCapability,
    SelfForkPolicy,
    SubagentDeferredResolver,
    SubagentExecutionMode,
    SubagentExecutionService,
    SubagentPlanDescriptor,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)
from ya_agent_sdk.toolsets.core.base import Toolset
from ya_agent_sdk.toolsets.search import create_best_strategy
from ya_agent_sdk.toolsets.skills.toolset import SHARED_SKILLS_DIR_NAME, SkillToolset

from yaacli.config import ConfigManager, MCPConfig, SubagentsConfig, YaacliConfig
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.models import (
    ChildPlanManifest,
    MainRuntimeManifest,
    RuntimeSecretRequirement,
)
from yaacli.durable.subagents import (
    DurableSubagentCompletionDelivery,
    DurableSubagentInboxCapability,
    LocalSubagentDriver,
    SQLiteSubagentExecutionStore,
)
from yaacli.environment import TUIEnvironment
from yaacli.guards import GoalGuardCapability
from yaacli.logging import get_logger
from yaacli.model_profiles import (
    ResolvedModelProfile,
    get_model_profile,
    get_startup_model_profile,
    resolve_profile_model_cfg,
)
from yaacli.session import TUIContext
from yaacli.subagent_config import (
    YAACLI_MODEL_CFG_METADATA_KEY,
    load_subagent_specs,
)
from yaacli.toolsets.monitored_shell import MonitoredShellTool

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeSourceSnapshot:
    """Immutable file-derived inputs shared by runtime construction and descriptors."""

    system_prompt: str
    subagent_specs: tuple[SubagentSpec, ...]

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "subagent_specs": [spec.model_dump(mode="json", by_alias=True) for spec in self.subagent_specs],
        }


_RUNTIME_CONFIG_KEYS = (
    "general",
    "subagents",
    "media",
    "env",
    "shell_env",
    "include_os_env",
    "security",
    "tools",
)
_SENSITIVE_CONFIG_PATHS = {
    ("media", "s3", "access_key_id"),
    ("media", "s3", "secret_access_key"),
}
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "cookie", "credential", "password", "secret", "token")
SecretSource = Literal["config", "mcp", "profile", "environment"]


def _value_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_value(value: Any, path: tuple[str | int, ...]) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list):
                raise KeyError(path)
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise KeyError(path)
            current = current[part]
    return current


def _set_path_value(value: Any, path: tuple[str | int, ...], replacement: Any) -> None:
    if not path:
        raise ValueError("Runtime secret target path cannot be empty")
    parent = _path_value(value, path[:-1]) if len(path) > 1 else value
    final = next(reversed(path))
    if isinstance(final, int):
        if not isinstance(parent, list):
            raise KeyError(path)
        parent[final] = replacement
    else:
        if not isinstance(parent, dict):
            raise KeyError(path)
        parent[final] = replacement


def _key_is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_snapshot(
    value: Any,
    *,
    source: SecretSource,
    target_root: str,
    force_reference: Callable[[tuple[str | int, ...], Any], bool] | None = None,
    path: tuple[str | int, ...] = (),
) -> tuple[Any, list[RuntimeSecretRequirement]]:
    requirements: list[RuntimeSecretRequirement] = []
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            item_path = (*path, key)
            should_reference = item is not None and (
                _key_is_sensitive(key) or (force_reference is not None and force_reference(item_path, item))
            )
            if should_reference:
                redacted[key] = None
                requirements.append(
                    RuntimeSecretRequirement(
                        source=source,
                        source_path=item_path,
                        target_path=(target_root, *item_path),
                        sha256=_value_digest(item),
                    )
                )
                continue
            redacted_item, nested = _redact_snapshot(
                item,
                source=source,
                target_root=target_root,
                force_reference=force_reference,
                path=item_path,
            )
            redacted[key] = redacted_item
            requirements.extend(nested)
        return redacted, requirements
    if isinstance(value, list):
        redacted_items: list[Any] = []
        for index, item in enumerate(value):
            item_path = (*path, index)
            if force_reference is not None and force_reference(item_path, item):
                redacted_items.append(None)
                requirements.append(
                    RuntimeSecretRequirement(
                        source=source,
                        source_path=item_path,
                        target_path=(target_root, *item_path),
                        sha256=_value_digest(item),
                    )
                )
                continue
            redacted_item, nested = _redact_snapshot(
                item,
                source=source,
                target_root=target_root,
                force_reference=force_reference,
                path=item_path,
            )
            redacted_items.append(redacted_item)
            requirements.extend(nested)
        return redacted_items, requirements
    return value, requirements


def build_main_runtime_manifest(
    config: YaacliConfig,
    mcp_config: MCPConfig | None,
    *,
    profile: ResolvedModelProfile | None,
    sources: RuntimeSourceSnapshot,
    working_dir: Path,
    config_dir: Path,
    subagent_default_mode: SubagentExecutionMode | None,
    enable_user_input: bool,
    frontend: Literal["tui", "headless"],
    hitl_policy: Literal["wait", "deny"],
) -> MainRuntimeManifest:
    """Snapshot one exact YAACLI host plan while retaining secrets only by digest."""
    full_config = config.model_dump(mode="json")
    runtime_config = {key: copy.deepcopy(full_config[key]) for key in _RUNTIME_CONFIG_KEYS}

    config_snapshot, requirements = _redact_snapshot(
        runtime_config,
        source="config",
        target_root="config",
        force_reference=lambda path, item: (
            path in _SENSITIVE_CONFIG_PATHS or (len(path) >= 2 and path[0] in {"env", "shell_env"})
        ),
    )
    for requirement in tuple(requirements):
        if requirement.source_path and requirement.source_path[0] == "env":
            requirements.remove(requirement)
            env_name = requirement.source_path[1]
            if not isinstance(env_name, str):
                raise TypeError("Environment variable names must be strings")
            requirements.append(
                requirement.model_copy(
                    update={
                        "source": "environment",
                        "source_path": (env_name,),
                    }
                )
            )

    mcp_snapshot: dict[str, Any] | None = None
    if mcp_config is not None:
        raw_mcp = mcp_config.model_dump(mode="json")

        def reference_mcp_value(path: tuple[str | int, ...], item: Any) -> bool:
            return len(path) >= 3 and path[0] == "servers" and path[2] in {"args", "env", "url", "headers"}

        mcp_snapshot, mcp_requirements = _redact_snapshot(
            raw_mcp,
            source="mcp",
            target_root="mcp",
            force_reference=reference_mcp_value,
        )
        requirements.extend(mcp_requirements)

    raw_profile = profile.model_dump(mode="json") if profile is not None else None
    profile_snapshot: dict[str, Any] | None = None
    if raw_profile is not None:
        profile_snapshot, profile_requirements = _redact_snapshot(
            raw_profile,
            source="profile",
            target_root="profile",
        )
        requirements.extend(profile_requirements)

    return MainRuntimeManifest(
        kind="yaacli",
        config_snapshot=config_snapshot,
        mcp_snapshot=mcp_snapshot,
        profile_snapshot=profile_snapshot,
        system_prompt=sources.system_prompt,
        subagent_specs=tuple(spec.model_dump(mode="json", by_alias=True) for spec in sources.subagent_specs),
        workspace_ref=str(working_dir.expanduser().resolve()),
        config_dir_ref=str(config_dir.expanduser().resolve()),
        subagent_default_mode=(subagent_default_mode.value if subagent_default_mode is not None else None),
        enable_user_input=enable_user_input,
        frontend=frontend,
        hitl_policy=hitl_policy,
        request_limit=config.general.max_requests,
        secret_requirements=tuple(
            sorted(requirements, key=lambda item: (item.target_path, item.source, item.source_path))
        ),
    )


def restore_main_runtime_manifest(
    manifest: MainRuntimeManifest,
    *,
    current_config: YaacliConfig,
    current_mcp_config: MCPConfig | None,
) -> tuple[
    YaacliConfig,
    MCPConfig | None,
    ResolvedModelProfile | None,
    RuntimeSourceSnapshot,
    Path,
    Path,
]:
    """Resolve secret references and validate every authority needed by a retained plan."""
    if manifest.kind != "yaacli":
        raise ValueError("Registered runtime manifests cannot be reconstructed after worker restart")

    snapshots: dict[str, Any] = {
        "config": copy.deepcopy(manifest.config_snapshot),
        "mcp": copy.deepcopy(manifest.mcp_snapshot),
        "profile": copy.deepcopy(manifest.profile_snapshot),
    }
    persisted_profile = manifest.profile_snapshot
    current_profile = None
    if isinstance(persisted_profile, dict):
        profile_id = persisted_profile.get("id")
        if isinstance(profile_id, str):
            current_profile = get_model_profile(current_config, profile_id)
    current_sources: dict[str, Any] = {
        "config": current_config.model_dump(mode="json"),
        "mcp": current_mcp_config.model_dump(mode="json") if current_mcp_config is not None else None,
        "profile": current_profile.model_dump(mode="json") if current_profile is not None else None,
        "environment": dict(os.environ),
    }
    for requirement in manifest.secret_requirements:
        source = current_sources[requirement.source]
        try:
            resolved = _path_value(source, requirement.source_path)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Required runtime secret source {requirement.source}:{requirement.source_path!r} is unavailable"
            ) from exc
        if _value_digest(resolved) != requirement.sha256:
            raise ValueError(f"Required runtime secret source {requirement.source}:{requirement.source_path!r} changed")
        target_root = requirement.target_path[0]
        if not isinstance(target_root, str):
            raise TypeError("Runtime secret target roots must be strings")
        _set_path_value(snapshots[target_root], requirement.target_path[1:], resolved)

    config = YaacliConfig.model_validate(snapshots["config"])
    mcp = MCPConfig.model_validate(snapshots["mcp"]) if snapshots["mcp"] is not None else None
    profile = ResolvedModelProfile.model_validate(snapshots["profile"]) if snapshots["profile"] is not None else None
    sources = RuntimeSourceSnapshot(
        system_prompt=manifest.system_prompt,
        subagent_specs=tuple(SubagentSpec.model_validate(item) for item in manifest.subagent_specs),
    )
    workspace = Path(cast(str, manifest.workspace_ref)).expanduser().resolve()
    config_dir = Path(cast(str, manifest.config_dir_ref)).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Runtime workspace authority {workspace} is unavailable")
    if not config_dir.is_dir():
        raise ValueError(f"Runtime config authority {config_dir} is unavailable")
    return config, mcp, profile, sources, workspace, config_dir


def runtime_child_plan_manifest(runtime: AgentRuntime[Any, Any, Any]) -> ChildPlanManifest:
    """Read the exact active and retained child plans from an assembled runtime."""
    for capability in runtime.capabilities:
        if not isinstance(capability, DelegationCapability):
            continue
        active_routes = {plan.spec.route: plan.descriptor_id for plan in capability.registry.list()}
        descriptors = tuple(plan.to_descriptor() for plan in capability.registry.list_registered())
        return ChildPlanManifest(
            active_routes=active_routes,
            descriptors=descriptors,
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
        subagent_specs=(_compile_subagent_specs(config.subagents, config_dir=config_dir) if include_subagents else ()),
    )


def build_runtime_agent_spec(
    config: YaacliConfig,
    *,
    profile: ResolvedModelProfile | None,
    sources: RuntimeSourceSnapshot,
    agent_name: str = "yaacli_main_v2",
) -> dict[str, Any]:
    """Return the normalized main-agent definition used by a durable descriptor."""
    model = profile.model if profile is not None else config.general.model
    settings = profile.model_settings if profile is not None else config.general.model_settings
    model_cfg = profile.model_cfg if profile is not None else config.general.model_cfg
    profile_instructions = profile.instructions if profile is not None else config.general.instructions
    instructions = [sources.system_prompt]
    if isinstance(profile_instructions, str) and profile_instructions.strip():
        instructions.append(profile_instructions)
    return {
        "name": agent_name,
        "model": model,
        "model_settings": dict(resolve_model_settings(settings) or {}),
        "instructions": instructions,
        "metadata": {YAACLI_MODEL_CFG_METADATA_KEY: resolve_profile_model_cfg(model_cfg).model_dump(mode="json")},
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
) -> tuple[SubagentSpec, ...]:
    """Load native portable specs and apply explicit config-file overrides."""
    loaded = load_subagent_specs(config_dir / "subagents")
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
        if override.model_cfg is not None:
            metadata = dict(agent_payload.get("metadata") or {})
            metadata[YAACLI_MODEL_CFG_METADATA_KEY] = resolve_profile_model_cfg(override.model_cfg).model_dump(
                mode="json"
            )
            agent_payload["metadata"] = metadata
        compiled.append(spec.model_copy(update={"agent": AgentSpec.model_validate(agent_payload)}))
    logger.info("Loaded %d native subagent specs", len(compiled))
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
) -> DelegationCapability | None:
    if not manifest.descriptors:
        return None
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model=default_model,
        host_capabilities=host_capabilities,
        restart_durable=False,
    )
    plans_by_id = {descriptor.descriptor_id: resolver.restore(descriptor) for descriptor in manifest.descriptors}
    active_plans = tuple(plans_by_id[descriptor_id] for _route, descriptor_id in sorted(manifest.active_routes.items()))
    store = SQLiteSubagentExecutionStore(durable_database_path)
    try:
        registry = SubagentRegistry(active_plans)
        active_ids = set(manifest.active_routes.values())
        for descriptor in manifest.descriptors:
            if descriptor.descriptor_id not in active_ids:
                registry.register_retained(plans_by_id[descriptor.descriptor_id])
        for plan in registry.list_registered():
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
        )
        return DelegationCapability(
            registry=registry,
            service=service,
            default_mode=default_mode,
        )
    except BaseException:
        store.close_sync()
        raise


def _self_fork_capability_specs(*, enable_codeact: bool) -> list[dict[str, Any]]:
    """Return self-fork native feature grants without host-injected policy duplicates."""
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
        {"ThinkingCapability": {}},
        {"TodoCapability": {}},
    ]
    if enable_codeact:
        capability_specs.append({"CodeActCapability": {}})
    return capability_specs


def compile_child_plan_manifest(
    config: YaacliConfig,
    *,
    profile: ResolvedModelProfile | None,
    sources: RuntimeSourceSnapshot,
    retained_descriptors: Sequence[SubagentPlanDescriptor] = (),
) -> ChildPlanManifest:
    """Compile exact active child plans and every retained referenced plan."""
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
            "capabilities": _self_fork_capability_specs(
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
        ToolRetryCapability(),
        ToolTimeoutCapability(),
    )
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model=active_model or None,
        host_capabilities=host_capabilities,
        restart_durable=False,
    )
    active_plans = [resolver.resolve(spec) for spec in sources.subagent_specs]
    if not any(plan.spec.route == "self" for plan in active_plans):
        active_plans.append(resolver.resolve_self(self_fork_policy))
    active_routes = {plan.spec.route: plan.descriptor_id for plan in active_plans}
    descriptors_by_id = {plan.descriptor_id: plan.to_descriptor() for plan in active_plans}
    for descriptor in retained_descriptors:
        descriptors_by_id.setdefault(descriptor.descriptor_id, descriptor)
    active_ids = {plan.descriptor_id for plan in active_plans}
    ordered_descriptors = (
        *(plan.to_descriptor() for plan in active_plans),
        *(descriptors_by_id[descriptor_id] for descriptor_id in sorted(descriptors_by_id.keys() - active_ids)),
    )
    return ChildPlanManifest(
        active_routes=active_routes,
        descriptors=ordered_descriptors,
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
        ThinkingCapability(),
        TodoCapability(),
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
                ToolRetryCapability(),
                ToolTimeoutCapability(),
            ),
            durable_database_path=durable_database_path,
            durable_binding_ref=durable_binding_ref,
            request_limit=config.general.max_requests,
            default_model_cfg=model_cfg,
            deferred_resolver=subagent_deferred_resolver,
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
        ToolRetryCapability(),
        ToolTimeoutCapability(),
        GoalGuardCapability(),
    ])

    runtime = create_agent(
        model=active_model or None,
        capabilities=cast(Sequence[AbstractCapability[TUIContext]], capabilities),
        model_settings=model_settings,
        output_type=[str, DeferredToolRequests],
        env=TUIEnvironment,
        env_kwargs=env_kwargs,
        context_type=TUIContext,
        model_cfg=model_cfg,
        context_kwargs=context_kwargs,
        system_prompt=effective_system_prompt,
        instructions=active_instructions,
        lifecycle_extensions=[GoalContextHandoffExtension()],
        agent_name=agent_name,
    )
    logger.info(
        "Created capability-first TUI runtime: model=%s capabilities=%d",
        active_model,
        len(capabilities),
    )
    return runtime
