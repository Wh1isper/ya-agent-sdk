from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.agents.main import stream_agent
from ya_agent_sdk.context import ShellReviewAction, ShellReviewConfig
from ya_agent_sdk.environment import SandboxEnvironment, VirtualMount
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.skills.toolset import SkillToolset
from ya_agent_sdk.toolsets.tool_proxy.toolset import ToolProxyToolset
from ya_claw.config import ClawSettings
from ya_claw.execution.profile import ClawShellReviewConfig, ResolvedProfile
from ya_claw.execution.runtime import ClawRuntimeBuilder
from ya_claw.workspace import MappedLocalEnvironment, WorkspaceBinding
from ya_claw.workspace.models import WorkspaceMountBinding


def _build_workspace_binding(
    host_path: Path,
    *,
    backend_hint: str = "local",
    metadata: dict[str, object] | None = None,
) -> WorkspaceBinding:
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
        backend_hint=backend_hint,
    )


def test_runtime_builder_propagates_container_id_from_workspace_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_API_KEY", "test-gateway-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://gateway.example.test")
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True, exist_ok=True)
    binding = _build_workspace_binding(
        host_path,
        backend_hint="docker",
        metadata={
            "provider": "docker",
            "sandbox": {
                "container_id": "container-xyz",
                "container_ref": "ya-claw-workspace-ref",
            },
        },
    )
    environment = SandboxEnvironment(
        mounts=[VirtualMount(host_path=host_path, virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        container_id="container-xyz",
    )
    profile = ResolvedProfile(
        name="default",
        model="gateway@openai-responses:gpt-5.5",
        model_settings=None,
        model_config=None,
        workspace_backend_hint="docker",
    )

    runtime = builder.build(
        profile=profile,
        binding=binding,
        environment=environment,
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="api",
        source_metadata={},
        claw_metadata={},
    )

    assert runtime.ctx.container_id == "container-xyz"
    assert runtime.ctx.workspace_binding is not None
    assert runtime.ctx.workspace_binding.metadata["sandbox"]["container_id"] == "container-xyz"


def test_runtime_builder_resolves_core_builtin_toolset(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)

    resolved_tool_names = [getattr(tool, "name", tool.__name__) for tool in builder._resolve_builtin_tools(["core"])]

    assert "view" in resolved_tool_names
    assert "shell_exec" in resolved_tool_names
    assert "spawn_delegate" in resolved_tool_names
    assert "list_session_turns" in resolved_tool_names
    assert "get_run_trace" in resolved_tool_names


def test_runtime_builder_resolves_runtime_mcp_toolsets_from_profile(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    binding = _build_workspace_binding(tmp_path / "workspace")
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        mcp_servers={
            "github": {
                "transport": "streamable_http",
                "url": "https://mcp.github.example/mcp",
            },
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
                "description": "Library docs",
                "required": False,
            },
        },
        enabled_mcps=["context7", "github"],
        disabled_mcps=["github"],
        need_user_approve_mcps=["context7"],
    )

    toolsets = builder._resolve_runtime_toolsets(profile=profile, binding=binding)

    assert len(toolsets) == 2
    assert isinstance(toolsets[0], SkillToolset)
    assert isinstance(toolsets[1], ToolProxyToolset)
    assert [toolset.tool_prefix for toolset in toolsets[1]._toolsets] == ["context7"]
    assert toolsets[1]._optional_namespaces == {"context7"}


def test_runtime_builder_rejects_profile_stdio_mcp(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    binding = _build_workspace_binding(tmp_path / "workspace")
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        mcp_servers={"github": {"transport": "stdio", "command": "npx"}},
    )

    with pytest.raises(ValueError, match="unsupported transport"):
        builder._resolve_runtime_toolsets(profile=profile, binding=binding)


async def test_runtime_builder_streams_with_pydantic_ai_test_model_and_exports_state(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True, exist_ok=True)
    binding = _build_workspace_binding(host_path)
    environment = MappedLocalEnvironment(
        mounts=[VirtualMount(host_path=host_path, virtual_path=Path("/workspace"))],
        host_cwd=host_path,
    )
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        builtin_toolsets=[],
        workspace_backend_hint="local",
    )

    runtime = builder.build(
        profile=profile,
        binding=binding,
        environment=environment,
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="api",
        source_metadata={},
        claw_metadata={},
    )

    seen_events: list[object] = []
    async with runtime:
        async with stream_agent(runtime, "say hello") as streamer:
            async for event in streamer:
                seen_events.append(event)
        assert streamer.run is not None
        assert streamer.run.result is not None
        result_output = streamer.run.result.output
        exported_state = runtime.ctx.export_state()

    assert result_output == "success (no tool calls)"
    assert seen_events
    assert runtime.ctx.session_id == "session-1"
    assert runtime.ctx.claw_run_id == "run-1"
    assert runtime.ctx.workspace_binding is not None
    assert runtime.ctx.workspace_binding.cwd == "/workspace"
    assert exported_state is not None


def test_runtime_builder_system_prompt_loads_workspace_guidance(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True, exist_ok=True)
    (host_path / "AGENTS.md").write_text("# Workspace\nUse pytest.\n", encoding="utf-8")
    binding = _build_workspace_binding(host_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
    )

    system_prompt = builder._build_system_prompt(profile=profile, binding=binding)

    assert '<workspace-guidance path="/workspace/AGENTS.md">' in system_prompt
    assert "# Workspace\nUse pytest." in system_prompt
    assert "Workspace virtual root: /workspace" in system_prompt
    assert "Default working directory: /workspace" in system_prompt
    assert "Workspace skills are discovered from /workspace/.agents/skills/." in system_prompt


def test_runtime_builder_system_prompt_skips_memory_for_heartbeat_and_loads_automated_context(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        memory_enabled=True,
        memory_inject_enabled=True,
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    memory_path = host_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "MEMORY.md").write_text("# Memory\n\n- User prefers concise updates.\n", encoding="utf-8")
    (host_path / "HEARTBEAT.md").write_text("# Heartbeat\nCheck runtime health.\n", encoding="utf-8")
    binding = _build_workspace_binding(host_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
    )

    system_prompt = builder._build_system_prompt(
        profile=profile,
        binding=binding,
        source_kind="heartbeat",
        source_metadata={"heartbeat_fire_id": "heartbeat-fire-1"},
    )

    assert '<heartbeat-context source="heartbeat">' in system_prompt
    assert "Heartbeat fire ID: heartbeat-fire-1" in system_prompt
    assert '<heartbeat-guidance path="/workspace/HEARTBEAT.md">' in system_prompt
    assert '<memory-md-context path="/workspace/memory/MEMORY.md">' not in system_prompt
    assert '<memory-file-index path="/workspace/memory">' not in system_prompt


def test_runtime_builder_system_prompt_skips_memory_for_schedule_and_loads_schedule_context(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        memory_enabled=True,
        memory_inject_enabled=True,
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    memory_path = host_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "MEMORY.md").write_text("# Memory\n\n- User prefers concise updates.\n", encoding="utf-8")
    binding = _build_workspace_binding(host_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
    )

    system_prompt = builder._build_system_prompt(
        profile=profile,
        binding=binding,
        source_kind="schedule",
        source_metadata={
            "schedule_id": "schedule-1",
            "schedule_fire_id": "fire-1",
            "execution_mode": "fork_session",
        },
    )

    assert '<schedule-context source="schedule">' in system_prompt
    assert "Schedule ID: schedule-1" in system_prompt
    assert "Schedule fire ID: fire-1" in system_prompt
    assert "Execution mode: fork_session" in system_prompt
    assert '<memory-md-context path="/workspace/memory/MEMORY.md">' not in system_prompt
    assert '<memory-file-index path="/workspace/memory">' not in system_prompt


def test_runtime_builder_memory_system_prompt_loads_source_identity(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    host_path.mkdir(parents=True, exist_ok=True)
    binding = _build_workspace_binding(host_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
    )

    system_prompt = builder._build_system_prompt(
        profile=profile,
        binding=binding,
        source_kind="memory",
        source_metadata={
            "memory": {
                "kind": "extract",
                "source_session_id": "session-1",
                "source_identity": {
                    "bridge": {
                        "latest_message": {
                            "adapter": "lark",
                            "tenant_key": "tenant-1",
                            "chat_id": "chat-1",
                            "sender_id": "user-1",
                        }
                    }
                },
            }
        },
    )

    assert "Source identity:" in system_prompt
    assert '"chat_id": "chat-1"' in system_prompt
    assert '"sender_id": "user-1"' in system_prompt


def test_runtime_builder_system_prompt_loads_memory_context(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        memory_enabled=True,
        memory_inject_enabled=True,
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    host_path = tmp_path / "workspace"
    memory_path = host_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "MEMORY.md").write_text("# Memory\n\n- User prefers concise updates.\n", encoding="utf-8")
    (memory_path / "20260501-event.md").write_text(
        "---\nname: Project Facts\ndescription: Stable project facts\n---\n\nBody\n",
        encoding="utf-8",
    )
    binding = _build_workspace_binding(host_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
    )

    system_prompt = builder._build_system_prompt(profile=profile, binding=binding)

    assert '<memory-md-context path="/workspace/memory/MEMORY.md">' in system_prompt
    assert '"untrusted": true' in system_prompt
    assert '"content": "# Memory\\n\\n- User prefers concise updates."' in system_prompt
    assert '<memory-file-index path="/workspace/memory">' in system_prompt
    assert (
        '<memory-file path="/workspace/memory/20260501-event.md" name="Project Facts" description="Stable project facts" />'
        in system_prompt
    )


def test_runtime_builder_preserves_claw_shell_review_defer_mode_for_api_runs(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    binding = _build_workspace_binding(tmp_path / "workspace")
    environment = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        shell_review=ShellReviewConfig(
            enabled=True,
            model="test:model",
            model_settings={"openai_reasoning_effort": "low"},
            on_needs_approval="defer",
            risk_threshold="extra_high",
        ),
    )

    runtime = builder.build(
        profile=profile,
        binding=binding,
        environment=environment,
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="api",
        source_metadata={},
        claw_metadata={},
    )

    assert runtime.ctx.security.shell_review is not None
    assert runtime.ctx.security.shell_review.on_needs_approval == ShellReviewAction.DEFER
    assert runtime.ctx.security.shell_review.risk_threshold == "extra_high"
    assert runtime.ctx.security.shell_review.model_settings == {"openai_reasoning_effort": "low"}
    assert runtime.agent.output_type == [str, DeferredToolRequests]
    assert runtime.agent._output_schema.allows_deferred_tools is True


def test_runtime_builder_uses_deny_policy_for_unattended_shell_review(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    builder = ClawRuntimeBuilder(settings=settings)
    binding = _build_workspace_binding(tmp_path / "workspace")
    environment = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path)
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        need_user_approve_tools=["file_write"],
        need_user_approve_mcps=["context7"],
        shell_review=ClawShellReviewConfig(
            enabled=True,
            model="test:model",
            on_needs_approval="defer",
            risk_threshold="extra_high",
            unattended_risk_threshold="high",
        ),
    )

    runtime = builder.build(
        profile=profile,
        binding=binding,
        environment=environment,
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="schedule",
        source_metadata={"schedule_id": "schedule-1"},
        claw_metadata={},
    )

    assert runtime.ctx.security.shell_review is not None
    assert runtime.ctx.security.shell_review.on_needs_approval == ShellReviewAction.DENY
    assert runtime.ctx.security.shell_review.risk_threshold == "high"
    assert runtime.ctx.need_user_approve_tools == []
    assert runtime.ctx.need_user_approve_mcps == []

    heartbeat_runtime = builder.build(
        profile=profile,
        binding=binding,
        environment=environment,
        restore_state=None,
        session_id="session-1",
        run_id="run-2",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="heartbeat",
        source_metadata={"heartbeat_fire_id": "heartbeat-1"},
        claw_metadata={},
    )
    assert heartbeat_runtime.ctx.security.shell_review is not None
    assert heartbeat_runtime.ctx.security.shell_review.on_needs_approval == ShellReviewAction.DENY
    assert heartbeat_runtime.ctx.security.shell_review.risk_threshold == "high"
