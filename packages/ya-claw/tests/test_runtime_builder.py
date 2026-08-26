from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import AgentSpec, DeferredToolRequests
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from sqlalchemy import create_engine as create_sync_engine
from ya_agent_sdk.agents.main import stream_agent
from ya_agent_sdk.capabilities import (
    CapabilityCatalog,
    FilesystemCapability,
    ShellCapability,
    SkillsCapability,
    ToolSupersessionCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.environment import VirtualMount
from ya_agent_sdk.subagents import (
    DelegationCapability,
    SubagentDurability,
    SubagentExecutionMode,
    SubagentSpec,
)
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.capabilities import ClawToolsCapability
from ya_claw.execution.profile import ClawShellReviewConfig, ResolvedProfile
from ya_claw.execution.runtime import ClawRuntimeBuilder
from ya_claw.orm.base import Base
from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY
from ya_claw.workspace import MappedLocalEnvironment, WorkspaceBinding
from ya_claw.workspace.models import WorkspaceMountBinding


@dataclass
class _RuntimePluginCapability(AbstractCapability[Any]):
    label: str = "default"

    @classmethod
    def get_serialization_name(cls) -> str:
        return "test.runtime_plugin"


def _settings(tmp_path: Path) -> ClawSettings:
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )


def _binding(host_path: Path, *, metadata: dict[str, object] | None = None) -> WorkspaceBinding:
    host_path.mkdir(parents=True, exist_ok=True)
    mount = WorkspaceMountBinding(
        id="workspace",
        host_path=host_path,
        virtual_path=Path("/workspace"),
        mode="rw",
    )
    return WorkspaceBinding(
        host_path=host_path,
        virtual_path=Path("/workspace"),
        cwd=Path("/workspace"),
        readable_paths=[Path("/workspace")],
        writable_paths=[Path("/workspace")],
        mounts=[mount],
        fingerprint="sha256:test",
        metadata=dict(metadata or {}),
        backend_hint="local",
    )


class _StubSubagentClient:
    async def setup(self) -> None:
        return None

    def get_capabilities(self) -> tuple[Any, ...]:
        return ()

    async def close(self) -> None:
        return None

    async def spawn_delegate(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(kwargs)

    async def get_async_subagent(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(kwargs)

    async def steer_async_subagent(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(kwargs)

    async def cancel_async_subagent(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(kwargs)


def _environment(host_path: Path) -> MappedLocalEnvironment:
    environment = MappedLocalEnvironment(
        mounts=[VirtualMount(host_path=host_path, virtual_path=Path("/workspace"))],
        host_cwd=host_path,
    )
    environment.resources.set(CLAW_SELF_CLIENT_KEY, _StubSubagentClient())
    return environment


def _profile(
    *,
    capabilities: list[object] | None = None,
    host_tool_groups: tuple[str, ...] = (),
    subagent_specs: tuple[SubagentSpec, ...] = (),
    approval_tools: frozenset[str] = frozenset(),
    shell_review: ClawShellReviewConfig | None = None,
    instructions: str = "Be concise.",
) -> ResolvedProfile:
    return ResolvedProfile(
        name="default",
        agent_spec=AgentSpec.model_validate({
            "model": "test",
            "name": "default",
            "instructions": instructions,
            "capabilities": capabilities or [],
        }),
        host_tool_groups=host_tool_groups,
        subagent_specs=subagent_specs,
        approval_tools=approval_tools,
        shell_review=shell_review,
        workspace_backend_hint="local",
    )


def _build_runtime(
    tmp_path: Path,
    profile: ResolvedProfile,
    *,
    source_kind: str = "api",
    source_metadata: dict[str, object] | None = None,
    capability_catalog: CapabilityCatalog | None = None,
):
    host_path = tmp_path / "workspace"
    binding = _binding(host_path)
    database_path = tmp_path / "runtime-builder.sqlite3"
    sync_engine = create_sync_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    session_factory = create_session_factory(create_engine(f"sqlite+aiosqlite:///{database_path}"))
    runtime = ClawRuntimeBuilder(
        settings=_settings(tmp_path),
        session_factory=session_factory,
        capability_catalog=capability_catalog,
    ).build(
        profile=profile,
        binding=binding,
        environment=_environment(host_path),
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind=source_kind,
        source_metadata=dict(source_metadata or {}),
        claw_metadata={},
    )
    return runtime, binding


async def test_runtime_builder_streams_with_test_model_and_native_agent_spec(tmp_path: Path) -> None:
    runtime, _ = _build_runtime(tmp_path, _profile(capabilities=["FilesystemCapability"]))

    seen_events: list[object] = []
    async with runtime:
        with runtime.agent.override(model=TestModel(call_tools=[])):
            async with stream_agent(runtime, "say hello") as streamer:
                async for event in streamer:
                    seen_events.append(event)
        assert streamer.run is not None
        assert streamer.run.result is not None
        output = streamer.run.result.output
        exported_state = runtime.ctx.export_state()
        resolved_capability_types = {type(capability) for capability in runtime.capabilities}
        root_capability_types = {type(capability) for capability in runtime.agent.root_capability.capabilities}

    assert output == "success (no tool calls)"
    assert seen_events
    assert FilesystemCapability in root_capability_types
    assert SkillsCapability in resolved_capability_types
    assert ToolSupersessionCapability in resolved_capability_types
    assert runtime.ctx.session_id == "session-1"
    assert runtime.ctx.claw_run_id == "run-1"
    assert exported_state is not None


async def test_runtime_builder_instantiates_configured_plugin_once(tmp_path: Path) -> None:
    catalog = build_default_capability_catalog(explicit_types=[_RuntimePluginCapability])
    runtime, _ = _build_runtime(
        tmp_path,
        _profile(
            capabilities=[
                {
                    "name": "test.runtime_plugin",
                    "arguments": {"label": "persisted-root"},
                }
            ]
        ),
        capability_catalog=catalog,
    )

    async with runtime:
        configured = [
            capability
            for capability in runtime.agent.root_capability.capabilities
            if isinstance(capability, _RuntimePluginCapability)
        ]

    assert len(configured) == 1
    assert configured[0].label == "persisted-root"


async def test_runtime_builder_preserves_native_output_and_execution_contract(tmp_path: Path) -> None:
    profile = ResolvedProfile(
        name="structured",
        agent_spec=AgentSpec.model_validate({
            "model": "test",
            "name": "structured",
            "instructions": "Return the structured answer.",
            "output_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "retries": {"tools": 2, "output": 3},
            "end_strategy": "exhaustive",
            "tool_timeout": 12.5,
            "metadata": {"contract": "native"},
        }),
    )
    runtime, _ = _build_runtime(tmp_path, profile)
    model = TestModel(call_tools=[], custom_output_args={"answer": "ok"})

    async with runtime:
        with runtime.agent.override(model=model):
            async with stream_agent(runtime, "answer") as streamer:
                async for _ in streamer:
                    pass
        assert streamer.run is not None
        assert streamer.run.result is not None
        output = streamer.run.result.output
        schema = runtime.agent.output_json_schema()
        end_strategy = runtime.agent.end_strategy
        tool_timeout = runtime.agent._tool_timeout

    request_parameters = model.last_model_request_parameters
    assert output == {"answer": "ok"}
    assert "answer" in str(schema)
    assert end_strategy == "exhaustive"
    assert tool_timeout == 12.5
    assert request_parameters is not None
    assert any("Return the structured answer." in str(part.content) for part in request_parameters.instruction_parts)


async def test_runtime_builder_uses_native_deferred_approval(tmp_path: Path) -> None:
    profile = _profile(
        capabilities=["FilesystemCapability"],
        approval_tools=frozenset({"write"}),
    )
    runtime, _ = _build_runtime(tmp_path, profile)

    async with runtime:
        with runtime.agent.override(model=TestModel(call_tools=["write"])):
            async with stream_agent(runtime, "write a file") as streamer:
                async for _ in streamer:
                    pass
        assert streamer.run is not None
        assert streamer.run.result is not None
        output = streamer.run.result.output

    assert isinstance(output, DeferredToolRequests)
    assert [request.tool_name for request in output.approvals] == ["write"]


async def test_runtime_builder_resolves_feature_and_claw_capabilities(tmp_path: Path) -> None:
    profile = _profile(
        capabilities=["FilesystemCapability", "ShellCapability"],
        host_tool_groups=("session", "schedule"),
    )
    runtime, _ = _build_runtime(tmp_path, profile)

    async with runtime:
        capability_types = {type(capability) for capability in runtime.capabilities}
        root_capability_types = {type(capability) for capability in runtime.agent.root_capability.capabilities}

    assert FilesystemCapability in root_capability_types
    assert ShellCapability in root_capability_types
    assert ClawToolsCapability in capability_types


async def test_runtime_builder_resolves_portable_registry_to_durable_delegation(tmp_path: Path) -> None:
    child = SubagentSpec(
        route="explorer",
        agent=AgentSpec(
            model="test",
            name="explorer",
            description="Explore a codebase",
            instructions="Explore and report evidence.",
            capabilities=["FilesystemCapability"],
        ),
        execution_modes=(SubagentExecutionMode.foreground, SubagentExecutionMode.background),
        durability=SubagentDurability.restart,
    )
    runtime, _ = _build_runtime(tmp_path, _profile(subagent_specs=(child,)))

    async with runtime:
        delegation = next(
            capability for capability in runtime.capabilities if isinstance(capability, DelegationCapability)
        )

    assert [plan.spec.route for plan in delegation.registry.list()] == ["explorer"]
    assert delegation.registry.get("explorer").restart_durable is True


async def test_async_child_omits_skills_and_recursive_delegation(tmp_path: Path) -> None:
    child = SubagentSpec(
        route="explorer",
        agent=AgentSpec(model="test", name="explorer", capabilities=["FilesystemCapability"]),
        execution_modes=(SubagentExecutionMode.foreground, SubagentExecutionMode.background),
        durability=SubagentDurability.restart,
    )
    profile = _profile(subagent_specs=(child,))
    runtime, _ = _build_runtime(
        tmp_path,
        profile,
        source_kind="async_task",
        source_metadata={"async_task": {"task_id": "task-1"}},
    )

    async with runtime:
        capability_types = {type(capability) for capability in runtime.capabilities}

    assert runtime.ctx.is_async_subagent is True
    assert SkillsCapability not in capability_types
    assert DelegationCapability not in capability_types


async def test_unattended_run_denies_shell_review_deferral(tmp_path: Path) -> None:
    review = ClawShellReviewConfig(
        enabled=True,
        model="test",
        on_needs_approval="defer",
        risk_threshold="extra_high",
    )
    runtime, _ = _build_runtime(
        tmp_path,
        _profile(capabilities=["ShellCapability"], shell_review=review),
        source_kind="schedule",
    )

    assert runtime.ctx.security.shell_review is not None
    assert runtime.ctx.security.shell_review.on_needs_approval == "deny"


async def test_api_run_preserves_shell_review_deferral(tmp_path: Path) -> None:
    review = ClawShellReviewConfig(
        enabled=True,
        model="test",
        on_needs_approval="defer",
        risk_threshold="extra_high",
    )
    runtime, _ = _build_runtime(tmp_path, _profile(capabilities=["ShellCapability"], shell_review=review))

    assert runtime.ctx.security.shell_review is not None
    assert runtime.ctx.security.shell_review.on_needs_approval == "defer"


def test_workspace_prompt_adds_guidance_without_duplicating_native_instructions(tmp_path: Path) -> None:
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True)
    (host_path / "AGENTS.md").write_text("# Workspace\nUse pytest.\n", encoding="utf-8")
    binding = _binding(host_path)
    builder = ClawRuntimeBuilder(settings=_settings(tmp_path))

    prompt = builder._build_system_prompt(
        profile=_profile(instructions="Native profile instructions."),
        binding=binding,
        source_kind="api",
        source_metadata={},
    )

    assert "Native profile instructions." not in prompt
    assert "Use pytest." in prompt
    assert "Default working directory: /workspace" in prompt


def test_heartbeat_prompt_includes_heartbeat_guidance_only_for_heartbeat(tmp_path: Path) -> None:
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True)
    (host_path / "HEARTBEAT.md").write_text("Check runtime health.", encoding="utf-8")
    binding = _binding(host_path)
    builder = ClawRuntimeBuilder(settings=_settings(tmp_path))
    profile = _profile()

    heartbeat = builder._build_system_prompt(
        profile=profile,
        binding=binding,
        source_kind="heartbeat",
        source_metadata={"heartbeat_fire_id": "fire-1"},
    )
    schedule = builder._build_system_prompt(
        profile=profile,
        binding=binding,
        source_kind="schedule",
        source_metadata={"schedule_id": "schedule-1"},
    )

    assert "Check runtime health." in heartbeat
    assert "Heartbeat fire ID: fire-1" in heartbeat
    assert "Check runtime health." not in schedule
