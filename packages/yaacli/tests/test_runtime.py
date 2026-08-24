"""Capability-first runtime assembly tests for YAACLI 2.0."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.lifecycle import (
    ContextHandoffCompleteContext,
    ContextHandoffSource,
)
from ya_agent_sdk.capabilities import (
    CodeActCapability,
    TaskCapability,
    ThinkingCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolTimeoutCapability,
    UserInteractionCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.mcp import MCPConfig, MCPServerConfig
from ya_agent_sdk.subagents import (
    DelegationCapability,
    SelfForkPolicy,
    SubagentDurability,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentPlanResolver,
    SubagentSpec,
)
from yaacli.config import (
    ConfigManager,
    GeneralConfig,
    ModelProfileConfig,
    SubagentOverride,
    SubagentsConfig,
    ToolsConfig,
    YaacliConfig,
)
from yaacli.durable.models import ChildPlanManifest
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.durable.subagents import FileSubagentExecutionStore, LocalProcessorSubagentExecutionHost
from yaacli.model_profiles import save_selected_model_profile_id
from yaacli.runtime import (
    GoalContextHandoffExtension,
    _build_delegation_capability,
    _compile_subagent_specs,
    _OptionalMCPToolset,
    _standard_child_capability_specs,
    build_runtime_agent_spec,
    compile_child_plan_manifest,
    compile_runtime_sources,
    create_tui_runtime,
    runtime_child_plan_manifest,
)
from yaacli.session import TUIContext
from yaacli.subagent_config import model_cfg_from_agent_spec


def _config() -> YaacliConfig:
    return YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4"))


@dataclass
class HostPluginCapability(AbstractCapability[Any]):
    label: str = "default"

    @classmethod
    def get_serialization_name(cls) -> str:
        return "test.host_plugin"


class HostPluginEntryPoint:
    name = "test.host_plugin"
    value = "tests.host_plugin:HostPluginCapability"
    dist = SimpleNamespace(name="test-host-plugin", version="1.0")

    def load(self) -> object:
        return HostPluginCapability


def _write_subagent_spec(
    directory: Path,
    *,
    route: str = "helper",
    capabilities: list[object] | None = None,
    durability: str = "restart",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "route": route,
        "agent": {
            "name": route,
            "description": "Helper",
            "instructions": "Help.",
            "capabilities": capabilities or [],
        },
        "execution_modes": ["foreground", "background"],
        "durability": durability,
    }
    (directory / f"{route}.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_runtime_is_an_unentered_plan_then_builds_agent_after_authorities(
    tmp_path: Path,
) -> None:
    runtime = create_tui_runtime(config=_config(), working_dir=tmp_path)

    with pytest.raises(RuntimeError, match="before runtime entry"):
        _ = runtime.agent

    async with runtime:
        assert runtime.agent is not None
        assert runtime.ctx.resources is runtime.env.resources
        assert any(isinstance(extension, GoalContextHandoffExtension) for extension in runtime.lifecycle_extensions)
        assert runtime.capability_sources


@pytest.mark.asyncio
async def test_runtime_executes_native_pydantic_ai_agent_without_stream_wrapper(
    tmp_path: Path,
) -> None:
    captured_instructions: list[str | None] = []

    def respond(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(
                model="openai-chat:gpt-4",
                instructions="Use the selected profile.",
            )
        ),
        working_dir=tmp_path,
    )
    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            result = await runtime.agent.run("test", deps=runtime.ctx)

    assert result.output == "ok"
    assert len(captured_instructions) == 1
    assert "Use the selected profile." in (captured_instructions[0] or "")


@pytest.mark.asyncio
async def test_runtime_allowed_paths_keep_workspace_and_config_authorities(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    workspace = tmp_path / "workspace"
    runtime = create_tui_runtime(
        config=_config(),
        working_dir=workspace,
        config_dir=config_dir,
    )

    async with runtime:
        allowed = runtime.env.file_operator._allowed_paths
        assert allowed[:5] == [
            Path(tempfile.gettempdir()).resolve(),
            config_dir.resolve(),
            (Path.home() / ".agents").resolve(),
            workspace.resolve(),
            (workspace / ".yaacli").resolve(),
        ]
        assert runtime.env.tmp_dir is not None
        assert runtime.env.tmp_dir.resolve() in allowed


def test_runtime_uses_persisted_model_profile(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
        model_profiles={
            "long": ModelProfileConfig(
                label="Long",
                model="openai-chat:gpt-4",
                model_cfg="gemini_1m",
            )
        },
    )
    save_selected_model_profile_id(config_dir, "long")

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
    )

    assert runtime.ctx.model_cfg.context_window == 1_000_000


@pytest.mark.asyncio
async def test_runtime_codeact_and_user_input_are_explicit_capabilities(
    tmp_path: Path,
) -> None:
    runtime = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(model="openai-chat:gpt-4"),
            tools=ToolsConfig(enable_codeact=True),
        ),
        working_dir=tmp_path,
        enable_user_input=True,
    )

    async with runtime:
        assert any(isinstance(item, CodeActCapability) for item in runtime.capabilities)
        assert any(isinstance(item, TaskCapability) for item in runtime.capabilities)
        assert any(isinstance(item, UserInteractionCapability) for item in runtime.capabilities)
        assert not any(isinstance(item, ThinkingCapability) for item in runtime.capabilities)


@pytest.mark.asyncio
async def test_runtime_uses_one_plugin_catalog_for_root_and_explicit_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ConfigManager.PLUGIN_MANIFEST_NAME).write_text(
        """\
schema_version = 1
entry_points = ["test.host_plugin"]

[[capabilities]]
name = "test.host_plugin"
arguments = { label = "root" }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [HostPluginEntryPoint()],
    )
    config = _config()
    plugins = ConfigManager(config_dir=config_dir, project_dir=tmp_path).load_capability_plugin_config()
    _write_subagent_spec(
        config_dir / "subagents",
        capabilities=[{"name": "test.host_plugin", "arguments": {"label": "child"}}],
        durability="process",
    )
    sources = compile_runtime_sources(config, config_dir=config_dir, include_subagents=True)
    child_manifest = compile_child_plan_manifest(
        config,
        profile=None,
        sources=sources,
        capability_catalog=plugins.catalog,
    )
    descriptors_by_route = {descriptor.spec.route: descriptor for descriptor in child_manifest.descriptors}
    child_capability_names = [item.name for item in descriptors_by_route["helper"].normalized_agent_spec.capabilities]
    self_capability_names = [item.name for item in descriptors_by_route["self"].normalized_agent_spec.capabilities]
    agent_spec = AgentSpec.model_validate(build_runtime_agent_spec(config, profile=None, capability_plugins=plugins))
    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
        agent_spec=agent_spec,
        capability_catalog=plugins.catalog,
    )

    async with runtime:
        root_capability = runtime.agent.root_capability

    assert isinstance(root_capability, CombinedCapability)
    root_plugins = [item for item in root_capability.capabilities if isinstance(item, HostPluginCapability)]
    assert len(root_plugins) == 1
    assert root_plugins[0].label == "root"
    assert "test.host_plugin" in child_capability_names
    assert "test.host_plugin" not in self_capability_names


def test_markdown_subagent_inherits_active_model_settings_before_plan_snapshot(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "helper.md").write_text(
        """---
name: helper
description: Markdown helper
model: inherit
model_settings: inherit
model_cfg: inherit
---

Return a bounded result.
""",
        encoding="utf-8",
    )
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_settings={"temperature": 0.4, "max_tokens": 4096},
            model_cfg={"context_window": 100_000},
        )
    )

    sources = compile_runtime_sources(config, config_dir=config_dir, include_subagents=True)
    manifest = compile_child_plan_manifest(config, profile=None, sources=sources)
    helper = next(descriptor for descriptor in manifest.descriptors if descriptor.spec.route == "helper")

    assert helper.normalized_agent_spec.model == "openai-chat:gpt-4"
    assert helper.normalized_agent_spec.model_settings == {"temperature": 0.4, "max_tokens": 4096}
    assert helper.normalized_agent_spec.metadata is not None
    assert helper.normalized_agent_spec.metadata["yaacli_model_cfg"]["context_window"] == 100_000
    assert "yaacli_inherit_model_settings" not in helper.normalized_agent_spec.metadata
    assert "yaacli_inherit_model_cfg" not in helper.normalized_agent_spec.metadata


def test_markdown_subagent_route_overrides_replace_inherited_model_configuration(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "helper.md").write_text(
        """---
name: helper
description: Markdown helper
model_settings: inherit
model_cfg: inherit
---

Return a bounded result.
""",
        encoding="utf-8",
    )
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_settings={"temperature": 0.9},
            model_cfg={"context_window": 100_000},
        ),
        subagents=SubagentsConfig(
            overrides={
                "helper": SubagentOverride(
                    model_settings={"temperature": 0.1},
                    model_cfg={"context_window": 300_000},
                )
            }
        ),
    )

    sources = compile_runtime_sources(config, config_dir=config_dir, include_subagents=True)
    manifest = compile_child_plan_manifest(config, profile=None, sources=sources)
    helper = next(descriptor for descriptor in manifest.descriptors if descriptor.spec.route == "helper")

    assert helper.normalized_agent_spec.model_settings == {"temperature": 0.1}
    assert helper.normalized_agent_spec.metadata is not None
    assert helper.normalized_agent_spec.metadata["yaacli_model_cfg"]["context_window"] == 300_000
    assert "yaacli_inherit_model_settings" not in helper.normalized_agent_spec.metadata
    assert "yaacli_inherit_model_cfg" not in helper.normalized_agent_spec.metadata


def test_markdown_inherited_model_cfg_participates_in_descriptor_identity(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "helper.md").write_text(
        """---
name: helper
description: Markdown helper
model_cfg: inherit
---

Return a bounded result.
""",
        encoding="utf-8",
    )
    first_config = YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4", model_cfg={"context_window": 100_000}))
    second_config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4", model_cfg={"context_window": 200_000})
    )
    sources = compile_runtime_sources(first_config, config_dir=config_dir, include_subagents=True)

    first_manifest = compile_child_plan_manifest(first_config, profile=None, sources=sources)
    second_manifest = compile_child_plan_manifest(second_config, profile=None, sources=sources)
    first = next(descriptor for descriptor in first_manifest.descriptors if descriptor.spec.route == "helper")
    second = next(descriptor for descriptor in second_manifest.descriptors if descriptor.spec.route == "helper")

    assert first.normalized_agent_spec.metadata is not None
    assert second.normalized_agent_spec.metadata is not None
    assert first.normalized_agent_spec.metadata["yaacli_model_cfg"]["context_window"] == 100_000
    assert second.normalized_agent_spec.metadata["yaacli_model_cfg"]["context_window"] == 200_000
    assert first.descriptor_id != second.descriptor_id


def test_packaged_subagent_presets_support_yaacli_process_local_driver() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    )
    presets = resources.files("ya_agent_sdk.subagents.presets")

    loaded_names: set[str] = set()
    for item in presets.iterdir():
        if not item.name.endswith((".yaml", ".yml", ".json")):
            continue
        spec = SubagentSpec.model_validate(yaml.safe_load(item.read_text(encoding="utf-8")))
        plan = resolver.resolve(spec)
        loaded_names.add(item.name)
        assert spec.durability is SubagentDurability.process
        assert plan.restart_durable is False

    assert loaded_names


async def test_delegation_builder_isolates_unavailable_retained_descriptor_until_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    )
    retained = resolver.resolve(
        SubagentSpec(
            route="helper",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="helper", instructions="retained plan"),
        )
    )
    active = resolver.resolve(
        SubagentSpec(
            route="helper",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="helper", instructions="active plan"),
        )
    )
    manifest = ChildPlanManifest(
        active_routes={"helper": active.descriptor_id},
        descriptors=(retained.to_descriptor(), active.to_descriptor()),
    )
    database_path = tmp_path / "descriptors.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    retained_store = FileSubagentExecutionStore(database_path)
    retained_store.put_descriptor(retained)
    record = SubagentExecutionRecord(
        root_execution_id="legacy-execution",
        execution_id="legacy-execution",
        owner_scope_id="session",
        idempotency_key="legacy-execution",
        descriptor_id=retained.descriptor_id,
        plan_fingerprint=retained.fingerprint,
        route=retained.spec.route,
        mode=SubagentExecutionMode.foreground,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="resume legacy work",
    )
    await retained_store.create(record)
    retained_store.close_sync()

    original_restore = SubagentPlanResolver.restore

    def restore_available_plan(self: SubagentPlanResolver, descriptor: Any) -> Any:
        if descriptor.descriptor_id == retained.descriptor_id:
            raise ValueError("historical capability unavailable")
        return original_restore(self, descriptor)

    monkeypatch.setattr(SubagentPlanResolver, "restore", restore_available_plan)
    capability = _build_delegation_capability(
        manifest,
        default_model="test",
        default_mode=SubagentExecutionMode.foreground,
        host_capabilities=(),
        durable_database_path=database_path,
        durable_binding_ref="test-binding",
        request_limit=2,
        default_model_cfg=TUIContext().model_cfg,
        deferred_resolver=None,
    )
    assert capability is not None
    store = capability.service.store
    assert isinstance(store, FileSubagentExecutionStore)
    assert isinstance(capability.service.execution_host, LocalProcessorSubagentExecutionHost)
    assert capability.allow_mode_override is False
    try:
        assert store.get_descriptor(active.descriptor_id) == active.to_descriptor()
        assert store.get_descriptor(retained.descriptor_id) == retained.to_descriptor()
        with pytest.raises(KeyError, match="Unknown subagent descriptor"):
            capability.registry.get_descriptor(retained.descriptor_id)

        provider = capability.service.retained_plan_provider
        assert provider is not None
        with pytest.raises(ValueError, match="historical capability unavailable"):
            await provider.load_retained_plan(record)

        monkeypatch.setattr(SubagentPlanResolver, "restore", original_restore)
        restored = await provider.load_retained_plan(record)
        assert restored is not None
        capability.registry.register_retained(restored)
        exported = runtime_child_plan_manifest(MagicMock(capabilities=(capability,)))
        assert exported.active_routes == {"helper": active.descriptor_id}
        assert exported.descriptors == (active.to_descriptor(),)
    finally:
        await capability.service.close()
        product_store.close()


def test_native_subagent_model_cfg_override_is_explicit(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    _write_subagent_spec(subagents_dir)

    specs = _compile_subagent_specs(
        SubagentsConfig(overrides={"helper": SubagentOverride(model_cfg={"context_window": 1_000_000})}),
        config_dir=config_dir,
        enable_codeact=True,
    )

    assert len(specs) == 1
    child_model_cfg = model_cfg_from_agent_spec(specs[0].agent)
    assert child_model_cfg is not None
    assert child_model_cfg.context_window == 1_000_000


def test_self_fork_native_grants_compose_with_host_policy_capabilities() -> None:
    capability_specs = _standard_child_capability_specs(enable_codeact=True)
    capability_names = {next(iter(item)) for item in capability_specs}
    assert "TaskCapability" in capability_names
    assert capability_names.isdisjoint({
        "ThinkingCapability",
        "TodoCapability",
        "ToolApprovalCapability",
        "ToolObservationCapability",
        "ToolTimeoutCapability",
    })

    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        host_capabilities=(
            ToolApprovalCapability(),
            ToolObservationCapability(),
            ToolTimeoutCapability(),
        ),
    )
    plan = resolver.resolve_self(
        SelfForkPolicy(
            agent=AgentSpec.from_dict({
                "name": "self",
                "description": "Bounded self fork",
                "capabilities": capability_specs,
            })
        )
    )

    assert plan.spec.route == "self"
    assert len(plan.injected_policy_ids) == 3


def test_runtime_defaults_to_no_process_local_subagents(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    _write_subagent_spec(subagents_dir)

    runtime = create_tui_runtime(
        config=_config(),
        working_dir=tmp_path,
        config_dir=config_dir,
    )

    assert not any(isinstance(capability, DelegationCapability) for capability in runtime.explicit_capabilities)


def test_runtime_rejects_subagents_without_durable_worker(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    _write_subagent_spec(subagents_dir)

    with pytest.raises(ValueError, match="require an exact child manifest"):
        create_tui_runtime(
            config=_config(),
            working_dir=tmp_path,
            config_dir=config_dir,
            subagent_default_mode=SubagentExecutionMode.background,
        )


def test_runtime_direct_and_proxy_mcp_assembly(tmp_path: Path) -> None:
    direct = create_tui_runtime(
        config=_config(),
        mcp_config=MCPConfig(servers={"docs": MCPServerConfig(command="unused")}),
        working_dir=tmp_path,
    )
    proxy = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(model="openai-chat:gpt-4"),
            tools=ToolsConfig(mcp_mode="proxy"),
        ),
        mcp_config=MCPConfig(servers={"docs": MCPServerConfig(command="unused")}),
        working_dir=tmp_path,
    )

    assert direct.explicit_capabilities
    assert proxy.explicit_capabilities
    assert direct is not proxy


@pytest.mark.asyncio
async def test_runtime_namespaces_duplicate_direct_mcp_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collide() -> str:
        return "ok"

    servers = [
        FunctionToolset([collide], id="one"),
        FunctionToolset([collide], id="two"),
    ]
    monkeypatch.setattr(
        "yaacli.runtime.build_mcp_servers",
        lambda *_args, **_kwargs: servers,
    )
    visible: set[str] = set()

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=_config(),
        mcp_config=MCPConfig(
            servers={
                "one": MCPServerConfig(command="unused"),
                "two": MCPServerConfig(command="unused"),
            }
        ),
        working_dir=tmp_path,
    )
    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            await runtime.agent.run("test", deps=runtime.ctx)

    assert {"one_collide", "two_collide"} <= visible


@pytest.mark.asyncio
async def test_optional_mcp_toolset_skips_failed_entry() -> None:
    class OfflineToolset(FunctionToolset):
        async def __aenter__(self):
            raise ConnectionError("offline")

    wrapper = _OptionalMCPToolset(
        OfflineToolset([], id="offline"),
        server_name="offline",
    )
    ctx = MagicMock()
    ctx.deps.emit_event = AsyncMock()

    async with wrapper:
        assert await wrapper.get_tools(ctx) == {}

    ctx.deps.emit_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_goal_handoff_extension_marks_only_active_goal() -> None:
    extension = GoalContextHandoffExtension()
    active = TUIContext()
    active.goal_task = "ship"
    inactive = TUIContext()

    await extension.on_context_handoff_complete(
        ContextHandoffCompleteContext(
            event_id="active-handoff",
            deps=active,
            source=ContextHandoffSource.COMPACT,
            original_messages=[],
            trimmed_messages=[],
            handoff_messages=[],
            summary_markdown="summary",
        )
    )
    await extension.on_context_handoff_complete(
        ContextHandoffCompleteContext(
            event_id="inactive-handoff",
            deps=inactive,
            source=ContextHandoffSource.COMPACT,
            original_messages=[],
            trimmed_messages=[],
            handoff_messages=[],
            summary_markdown="summary",
        )
    )

    assert active.goal_needs_post_restore_audit is True
    assert active.goal_last_context_handoff_source == "compact"
    assert inactive.goal_needs_post_restore_audit is False


def test_runtime_source_snapshot_freezes_prompt_and_subagent_definitions(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt one", encoding="utf-8")
    _write_subagent_spec(tmp_path / "subagents")
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            system_prompt_file=str(prompt_path),
        )
    )

    first = compile_runtime_sources(
        config,
        config_dir=tmp_path,
        include_subagents=True,
    )
    first_agent_spec = build_runtime_agent_spec(
        config,
        profile=None,
    )

    prompt_path.write_text("prompt two", encoding="utf-8")
    _write_subagent_spec(tmp_path / "subagents", route="second")
    second = compile_runtime_sources(
        config,
        config_dir=tmp_path,
        include_subagents=True,
    )

    assert first.system_prompt == "prompt one"
    assert [spec.route for spec in first.subagent_specs] == ["helper"]
    assert first_agent_spec["instructions"] == []
    assert first != second
