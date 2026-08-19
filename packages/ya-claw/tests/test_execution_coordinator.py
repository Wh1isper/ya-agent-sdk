from __future__ import annotations

import asyncio
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.usage import RunUsage
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_environment import Environment
from ya_agent_sdk.context import StreamEvent
from ya_agent_sdk.events import UsageSnapshotEvent
from ya_agent_sdk.subagents import SubagentDurability, SubagentSpec
from ya_agent_sdk.usage import CostEstimate, UsageSnapshot
from ya_claw.config import ClawSettings
from ya_claw.controller.async_task import AsyncTaskController
from ya_claw.controller.models import SessionSubmitRequest, TextPart
from ya_claw.controller.run import RunController
from ya_claw.controller.session import SessionController
from ya_claw.controller.store import extract_usage_snapshot_from_metadata
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution import coordinator as coordinator_module
from ya_claw.execution.coordinator import (
    ExecutionBuffers,
    ExecutionSupervisor,
    RunCoordinator,
    _index_run_usage_snapshot,
    _run_restores_state,
    _runtime_source_metadata,
    _with_cumulative_usage_snapshot,
)
from ya_claw.execution.profile import ResolvedProfile
from ya_claw.execution.state_machine import interrupt_run, mark_run_running
from ya_claw.execution.store import RunStore
from ya_claw.execution.subagents import resolve_claw_subagent_plan
from ya_claw.orm.tables import ProfileRecord, RunRecord, SessionAsyncTaskRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState, create_runtime_state
from ya_claw.workspace import WorkspaceBinding, WorkspaceProvider
from ya_claw.workspace.models import WorkspaceMountBinding


class StubWorkspaceProvider(WorkspaceProvider):
    def __init__(self, workspace_dir: Path) -> None:
        self._workspace_dir = workspace_dir

    def resolve(self, metadata: dict[str, object] | None = None) -> WorkspaceBinding:
        host_path = self._workspace_dir
        host_path.mkdir(parents=True, exist_ok=True)
        virtual_path = Path("/workspace")
        mount = WorkspaceMountBinding(
            id="workspace",
            host_path=host_path,
            virtual_path=virtual_path,
            mode="rw",
        )
        return WorkspaceBinding(
            host_path=host_path,
            virtual_path=virtual_path,
            cwd=virtual_path,
            readable_paths=[virtual_path],
            writable_paths=[virtual_path],
            mounts=[mount],
            fingerprint="sha256:test",
            metadata=dict(metadata or {}),
            backend_hint="local",
        )


class StubProfileResolver:
    async def resolve(self, profile_name: str | None) -> ResolvedProfile:
        name = profile_name or "general"
        return ResolvedProfile(
            name=name,
            agent_spec=AgentSpec(model="test", name=name),
        )


class StubEnvironment(Environment):
    async def _setup(self) -> None:
        return None

    async def _teardown(self) -> None:
        return None


class StubEnvironmentFactory:
    def build(self, binding: WorkspaceBinding, *, profile: ResolvedProfile | None = None) -> Environment:
        return StubEnvironment()


class StubRuntimeBuilder:
    def build(self, **_: object) -> object:
        return object()


class StubRunCoordinator(RunCoordinator):
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory,
        runtime_state: InMemoryRuntimeState,
        workspace_provider: WorkspaceProvider,
        failure: Exception | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            session_factory=session_factory,
            runtime_state=runtime_state,
            workspace_provider=workspace_provider,
            environment_factory=StubEnvironmentFactory(),
            profile_resolver=StubProfileResolver(),
            runtime_builder=StubRuntimeBuilder(),
        )
        self.failure = failure
        self.restore_run_ids: list[str | None] = []

    async def _resolve_run_profile(self, db_session, session_record, run_record):
        if session_record.session_type == "async_task":
            return await super()._resolve_run_profile(db_session, session_record, run_record)
        return await self._profile_resolver.resolve(run_record.profile_name)

    async def _execute_agent_run(
        self,
        *,
        run_id: str,
        session_id: str,
        dispatch_mode: str,
        workspace_binding: WorkspaceBinding,
        restore_point,
        input_parts,
        profile,
        profile_name: str | None,
        trigger_type: str,
        run_metadata: dict[str, Any],
        buffers: ExecutionBuffers,
    ) -> None:
        self.restore_run_ids.append(restore_point.run_id if restore_point is not None else None)
        await self._runtime_state.append_run_event(
            run_id,
            {
                "type": "agent.stream",
                "run_id": run_id,
                "session_id": session_id,
                "dispatch_mode": dispatch_mode,
                "event_type": "StubEvent",
                "event": {"input_parts": [part.model_dump(mode="json") for part in input_parts]},
            },
        )
        assert run_metadata is not None
        context_state = {
            "schema_version": 2,
            "notes": {},
            "tasks": {},
            "usage_snapshot_entries": {},
            "tool_search_loaded_tools": [],
            "tool_search_loaded_namespaces": [],
            "user_prompts": None,
            "handoff_message": None,
            "deferred_tool_metadata": {},
            "need_user_approve_tools": [],
            "need_user_approve_mcps": [],
            "auto_load_files": [],
        }
        buffers.latest_message_payload = {
            "events": [{"role": "assistant", "content": f"completed {run_id}"}],
            "message_history": [{"role": "assistant", "content": f"completed {run_id}"}],
            "messages": [{"role": "assistant", "content": f"completed {run_id}"}],
            "message_count": 1,
        }
        buffers.latest_state_payload = {
            "container_id": run_metadata.get("container_id"),
            "context_state": {
                **context_state,
                "container_id": run_metadata.get("container_id"),
            },
            "resumable_state": {
                **context_state,
                "container_id": run_metadata.get("container_id"),
            },
            "message_history": list(buffers.latest_message_payload["message_history"]),
            "message_count": 1,
            "profile_name": profile_name,
            "workspace": {
                "virtual_path": str(workspace_binding.virtual_path),
                "cwd": str(workspace_binding.cwd),
            },
            "version": 4,
        }
        buffers.output_text = f"completed {run_id}"
        buffers.output_json = {"answer": f"completed {run_id}"}
        if self.failure is not None:
            raise self.failure


class BlockingCommitRunCoordinator(StubRunCoordinator):
    def __init__(self, *args: Any, entered_gate: asyncio.Event, release_gate: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered_gate = entered_gate
        self.release_gate = release_gate

    async def _commit_successful_run(self, **kwargs: Any) -> None:
        self.entered_gate.set()
        await self.release_gate.wait()
        await super()._commit_successful_run(**kwargs)


class InterruptingFailureRunCoordinator(StubRunCoordinator):
    async def _execute_agent_run(
        self,
        *,
        run_id: str,
        session_id: str,
        dispatch_mode: str,
        workspace_binding: WorkspaceBinding,
        restore_point,
        input_parts,
        profile,
        profile_name: str | None,
        trigger_type: str,
        run_metadata: dict[str, Any],
        buffers: ExecutionBuffers,
    ) -> None:
        await super()._execute_agent_run(
            run_id=run_id,
            session_id=session_id,
            dispatch_mode=dispatch_mode,
            workspace_binding=workspace_binding,
            restore_point=restore_point,
            input_parts=input_parts,
            profile=profile,
            profile_name=profile_name,
            trigger_type=trigger_type,
            run_metadata=run_metadata,
            buffers=buffers,
        )
        async with self._session_factory() as db_session:
            session_record = await db_session.get(SessionRecord, session_id)
            run_record = await db_session.get(RunRecord, run_id)
            assert isinstance(session_record, SessionRecord)
            assert isinstance(run_record, RunRecord)
            await self._runtime_state.request_stop(run_id, "interrupt")
            interrupt_run(session_record, run_record)
            await db_session.commit()
        raise RuntimeError("boom")


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[[str], None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'coordinator.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
        session.add(
            ProfileRecord(
                name="general",
                agent_spec={"model": "test"},
                host_config={},
                subagent_specs=[],
                enabled=True,
                source_type="test",
                source_version="1",
            )
        )
        await session.commit()
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=data_dir,
        workspace_dir=workspace_dir,
    )


@pytest.fixture
def runtime_state() -> InMemoryRuntimeState:
    return create_runtime_state()


def test_extract_message_history_restores_tool_return_binary_content(tmp_path: Path, db_engine: AsyncEngine) -> None:
    from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart
    from ya_claw.execution.restore import ResolvedRestorePoint

    coordinator = StubRunCoordinator(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            data_dir=tmp_path / "runtime-data",
            workspace_dir=tmp_path / "workspace",
        ),
        session_factory=create_session_factory(db_engine),
        runtime_state=create_runtime_state(),
        workspace_provider=StubWorkspaceProvider(tmp_path / "workspace"),
    )
    image_data = b"\x89PNG\r\n\x1a\nexample"
    serialized_messages = ModelMessagesTypeAdapter.dump_python(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="fetch",
                        content=[BinaryContent(data=image_data, media_type="image/png")],
                        tool_call_id="call_1",
                    )
                ]
            )
        ],
        mode="json",
    )
    restore_point = ResolvedRestorePoint(
        run_id="run-1",
        session_id="session-1",
        status="completed",
        state={"message_history": serialized_messages},
        message=None,
    )

    history = coordinator._extract_message_history(restore_point)

    assert history is not None
    request = history[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert isinstance(part.content, list)
    restored_content = part.content[0]
    assert isinstance(restored_content, BinaryContent)
    assert restored_content.data == image_data
    assert restored_content.media_type == "image/png"


def test_build_user_prompt_returns_only_mapped_user_input(tmp_path: Path, db_engine: AsyncEngine) -> None:
    from ya_claw.controller.models import CommandPart, ModePart, TextPart
    from ya_claw.execution.input import InputMappingResult

    coordinator = StubRunCoordinator(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            data_dir=tmp_path / "runtime-data",
            workspace_dir=tmp_path / "workspace",
        ),
        session_factory=create_session_factory(db_engine),
        runtime_state=create_runtime_state(),
        workspace_provider=StubWorkspaceProvider(tmp_path / "workspace"),
    )
    mapping = InputMappingResult(
        user_prompt=["hello"],
        mode_parts=[ModePart(type="mode", mode="plan")],
        command_parts=[CommandPart(type="command", name="summarize")],
        content_parts=[TextPart(type="text", text="hello")],
        input_preview="hello",
    )

    assert coordinator._build_user_prompt(mapping) == "hello"

    mapping.user_prompt = ["hello", "world"]
    assert coordinator._build_user_prompt(mapping) == ["hello", "world"]


async def test_async_task_profile_restores_only_server_owned_descriptor(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    plan = resolve_claw_subagent_plan(
        SubagentSpec(
            route="explorer",
            agent=AgentSpec(
                model="test",
                name="explorer",
                instructions="Immutable child instructions.",
                capabilities=["FilesystemCapability"],
                metadata={
                    "claw": {
                        "model_config_override": {"request_limit": 5},
                        "tool_groups": ["session"],
                        "need_user_approve_tools": ["shell_exec"],
                        "need_user_approve_mcps": ["context7"],
                        "enabled_mcps": ["context7"],
                        "disabled_mcps": [],
                        "mcp_servers": {},
                        "workspace_backend_hint": "docker",
                    }
                },
            ),
            durability=SubagentDurability.restart,
        )
    )
    parent_workspace = {
        "mounts": [
            {
                "id": "parent",
                "host_path": str(settings.resolved_workspace_dir / "parent"),
                "virtual_path": "/parent-workspace",
                "mode": "rw",
            }
        ],
        "default_mount_id": "parent",
        "cwd": "/parent-workspace",
    }
    child_workspace = {
        "mounts": [
            {
                "id": "child",
                "host_path": str(settings.resolved_workspace_dir / "child"),
                "virtual_path": "/child-workspace",
                "mode": "rw",
            }
        ],
        "default_mount_id": "child",
        "cwd": "/child-workspace",
    }
    parent = SessionRecord(
        id="parent",
        profile_name="mutable-profile",
        session_metadata={"workspace": parent_workspace},
    )
    child = SessionRecord(
        id="child",
        parent_session_id=parent.id,
        profile_name="mutable-profile",
        session_type="async_task",
        session_metadata={
            "async_task": {"task_id": "task-1"},
            "workspace": child_workspace,
        },
        active_run_id="child-run",
    )
    run = RunRecord(
        id="child-run",
        session_id=child.id,
        sequence_no=1,
        status="running",
        trigger_type="async_task",
        profile_name="mutable-profile",
        input_parts=[{"type": "text", "text": "work"}],
        run_metadata={"async_task": {"task_id": "task-1"}},
    )
    task = SessionAsyncTaskRecord(
        id="task-1",
        parent_session_id=parent.id,
        task_session_id=child.id,
        task_run_id=run.id,
        subagent_name="explorer",
        name="explorer",
        status="running",
        plan_fingerprint=plan.fingerprint,
        plan_descriptor_ref=plan.descriptor_id,
        plan_descriptor=plan.to_descriptor().model_dump(mode="json"),
    )
    db_session.add_all([parent, child, run, task])
    await db_session.commit()

    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )
    resolved = await coordinator._resolve_run_profile(db_session, child, run)

    assert resolved.name == "explorer"
    assert resolved.agent_spec.instructions == "Immutable child instructions."
    assert resolved.host_tool_groups == ("session",)
    assert resolved.host_tool_allowlist is None
    assert resolved.approval_tools == frozenset({"shell_exec"})
    assert resolved.approval_mcps == frozenset({"context7"})
    assert resolved.enabled_mcps == frozenset({"context7"})
    assert resolved.mcp_servers == {}
    assert resolved.workspace_backend_hint == "docker"
    assert resolved.metadata["plan_fingerprint"] == plan.fingerprint

    binding = await coordinator._resolve_workspace_binding(
        db_session,
        run,
        child,
        resolved,
    )
    binding_workspace = binding.metadata["workspace"]
    assert binding_workspace["mounts"][0]["id"] == "parent"
    assert binding_workspace["cwd"] == "/parent-workspace"
    assert binding.backend_hint == "docker"


async def test_run_dispatcher_submits_with_profile_model_only(
    tmp_path: Path,
    db_engine: AsyncEngine,
    runtime_state: InMemoryRuntimeState,
) -> None:
    from ya_claw.execution.dispatcher import RunDispatcher

    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text("profiles:\n- name: default\n  model: test\n", encoding="utf-8")
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
        auto_seed_profiles=True,
    )
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )

    result = RunDispatcher(supervisor).dispatch("run-profile-model", "async")

    assert result.submitted is True
    assert result.reason is None
    task = runtime_state.get_background_task("run-profile-model")
    assert task is not None
    task.cancel()
    await runtime_state.aclose()


async def test_execution_supervisor_shutdown_waits_for_active_tasks(
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )
    release = asyncio.Event()
    completed = False

    async def active_run() -> None:
        nonlocal completed
        await release.wait()
        completed = True

    task = asyncio.create_task(active_run())
    runtime_state.register_background_task("run-active", task)
    shutdown_task = asyncio.create_task(supervisor.shutdown())

    await asyncio.sleep(0)
    assert shutdown_task.done() is False
    assert supervisor.submit_run("run-new") is False

    release.set()
    await shutdown_task

    assert completed is True
    assert task.done() is True


async def test_execution_supervisor_shutdown_cancels_tasks_after_timeout(
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    settings.shutdown_timeout_seconds = 1
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )
    cancelled = False

    async def hanging_run() -> None:
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    task = asyncio.create_task(hanging_run())
    runtime_state.register_background_task("run-hanging", task)

    await supervisor.shutdown()

    assert cancelled is True
    assert task.cancelled() is True
    assert runtime_state.get_background_task("run-hanging") is None


async def test_execution_supervisor_skips_claim_when_shutdown_races_with_db_load(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(
        id="session-1",
        profile_name="general",
        session_metadata={},
        head_run_id="run-1",
        active_run_id="run-1",
    )
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    await db_session.commit()

    original_load_run_scope = coordinator_module._load_run_scope
    scope_loaded = asyncio.Event()
    release_scope = asyncio.Event()

    async def blocking_load_run_scope(db_session_arg: AsyncSession, run_id: str) -> tuple[SessionRecord, RunRecord]:
        result = await original_load_run_scope(db_session_arg, run_id)
        scope_loaded.set()
        await release_scope.wait()
        return result

    monkeypatch.setattr(coordinator_module, "_load_run_scope", blocking_load_run_scope)

    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )
    claim_task = asyncio.create_task(supervisor._claim_run("run-1"))
    await asyncio.wait_for(scope_loaded.wait(), timeout=1)

    await supervisor.shutdown()
    release_scope.set()
    claimed = await asyncio.wait_for(claim_task, timeout=1)

    refreshed_run = await db_session.get(RunRecord, "run-1")
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert claimed is False
    assert isinstance(refreshed_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    await db_session.refresh(refreshed_run)
    await db_session.refresh(refreshed_session)
    assert refreshed_run.status == "queued"
    assert refreshed_run.claimed_by is None
    assert refreshed_run.claimed_at is None
    assert refreshed_run.started_at is None
    assert refreshed_session.head_run_id == "run-1"
    assert refreshed_session.active_run_id == "run-1"
    assert runtime_state.get_run_handle("run-1") is None


async def test_execution_supervisor_does_not_resurrect_run_cancelled_during_claim(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(
        id="session-claim-cancel",
        profile_name="general",
        session_metadata={},
        head_run_id="run-claim-cancel",
    )
    run_record = RunRecord(
        id="run-claim-cancel",
        session_id=session_record.id,
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    await db_session.commit()

    original_load_run_scope = coordinator_module._load_run_scope
    scope_loaded = asyncio.Event()
    release_scope = asyncio.Event()

    async def blocking_load_run_scope(
        db_session_arg: AsyncSession,
        run_id: str,
    ) -> tuple[SessionRecord, RunRecord]:
        result = await original_load_run_scope(db_session_arg, run_id)
        scope_loaded.set()
        await release_scope.wait()
        return result

    monkeypatch.setattr(coordinator_module, "_load_run_scope", blocking_load_run_scope)
    session_factory = create_session_factory(db_engine)
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )

    claim_task = asyncio.create_task(supervisor._claim_run(run_record.id))
    await asyncio.wait_for(scope_loaded.wait(), timeout=1)
    async with session_factory() as cancel_session:
        await RunController().cancel(
            cancel_session,
            settings,
            runtime_state,
            run_record.id,
        )
    release_scope.set()

    assert await asyncio.wait_for(claim_task, timeout=1) is False
    await db_session.refresh(run_record)
    assert run_record.status == "cancelled"
    assert run_record.started_at is None
    assert run_record.claimed_by is None
    assert runtime_state.get_run_handle(run_record.id) is None


async def test_claim_then_cancel_clears_active_run_on_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(
        id="session-claim-then-cancel",
        profile_name="general",
        session_metadata={},
        head_run_id="run-claim-then-cancel",
    )
    run_record = RunRecord(
        id="run-claim-then-cancel",
        session_id=session_record.id,
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add_all([session_record, run_record])
    await db_session.commit()

    original_lock_run_scope = coordinator_module._lock_run_scope
    claim_locked = asyncio.Event()
    release_claim = asyncio.Event()

    async def blocking_lock_run_scope(
        db_session_arg: AsyncSession,
        *,
        session_id: str,
        run_id: str,
    ) -> tuple[SessionRecord, RunRecord]:
        result = await original_lock_run_scope(
            db_session_arg,
            session_id=session_id,
            run_id=run_id,
        )
        claim_locked.set()
        await release_claim.wait()
        return result

    monkeypatch.setattr(coordinator_module, "_lock_run_scope", blocking_lock_run_scope)
    session_factory = create_session_factory(db_engine)
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )

    claim_task = asyncio.create_task(supervisor._claim_run(run_record.id))
    await asyncio.wait_for(claim_locked.wait(), timeout=1)

    async def cancel() -> None:
        async with session_factory() as cancel_session:
            await RunController().cancel(
                cancel_session,
                settings,
                runtime_state,
                run_record.id,
            )

    cancel_task = asyncio.create_task(cancel())
    await asyncio.sleep(0.05)
    release_claim.set()
    assert await asyncio.wait_for(claim_task, timeout=1) is True
    await asyncio.wait_for(cancel_task, timeout=1)

    await db_session.refresh(run_record)
    await db_session.refresh(session_record)
    assert run_record.status == "cancelled"
    assert session_record.active_run_id is None


async def test_execution_supervisor_claims_queued_run(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    await db_session.commit()

    runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )

    claimed = await supervisor._claim_run("run-1")

    refreshed_run = await db_session.get(RunRecord, "run-1")
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert claimed is True
    assert isinstance(refreshed_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    await db_session.refresh(refreshed_run)
    await db_session.refresh(refreshed_session)

    handle = runtime_state.get_run_handle("run-1")
    assert handle is not None
    assert refreshed_run.status == "running"
    assert refreshed_session.active_run_id == "run-1"
    assert handle.events[0].payload["type"] == "RUN_STARTED"
    assert handle.events[0].payload["runId"] == "run-1"


async def test_run_coordinator_completes_run_and_commits_artifacts(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    mark_run_running(session_record, run_record)
    await db_session.commit()

    runtime_state.register_run("session-1", "run-1")
    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )

    await coordinator.execute("run-1")

    refreshed_run = await db_session.get(RunRecord, "run-1")
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert isinstance(refreshed_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    await db_session.refresh(refreshed_run)
    await db_session.refresh(refreshed_session)

    run_store = RunStore(settings)
    state_payload = run_store.read_state("run-1")
    message_payload = run_store.read_message("run-1")
    assert refreshed_run.status == "completed"
    assert refreshed_run.output_text == "completed run-1"
    assert refreshed_run.output_json == {"answer": "completed run-1"}
    assert refreshed_session.head_success_run_id == "run-1"
    assert refreshed_session.active_run_id is None
    assert state_payload is not None
    assert state_payload["container_id"] is None
    assert state_payload["context_state"]["notes"] == {}
    assert state_payload["message_history"][0]["content"] == "completed run-1"
    assert message_payload is not None
    assert message_payload[0]["content"] == "completed run-1"

    assert runtime_state.get_run_handle("run-1") is None


async def test_post_commit_hook_failure_cannot_rewrite_completed_run(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_record = SessionRecord(
        id="session-post-commit",
        profile_name="general",
        session_metadata={},
    )
    run_record = RunRecord(
        id="run-post-commit",
        session_id=session_record.id,
        sequence_no=1,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add_all([session_record, run_record])
    mark_run_running(session_record, run_record)
    await db_session.commit()

    async def fail_terminal_hook(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise RuntimeError("post-commit projection failed")

    monkeypatch.setattr(
        AsyncTaskController,
        "on_run_terminal",
        fail_terminal_hook,
    )
    runtime_state.register_run(session_record.id, run_record.id)
    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )

    await coordinator.execute(run_record.id)

    await db_session.refresh(run_record)
    await db_session.refresh(session_record)
    assert run_record.status == "completed"
    assert run_record.termination_reason == "completed"
    assert run_record.error_message is None
    assert session_record.head_success_run_id == run_record.id


async def test_run_coordinator_loads_restore_point_from_previous_run(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(
        id="session-1",
        profile_name="general",
        session_metadata={},
        head_run_id="run-1",
        head_success_run_id="run-1",
    )
    base_run = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="completed",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "base"}],
        run_metadata={},
    )
    rerun = RunRecord(
        id="run-2",
        session_id="session-1",
        sequence_no=2,
        restore_from_run_id="run-1",
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "rerun"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(base_run)
    db_session.add(rerun)
    mark_run_running(session_record, rerun)
    await db_session.commit()

    run_store = RunStore(settings)
    run_store.write_state(
        "run-1",
        {
            "resumable_state": {
                "schema_version": 2,
                "notes": {},
                "tasks": {},
                "usage_snapshot_entries": {},
                "tool_search_loaded_tools": [],
                "tool_search_loaded_namespaces": [],
                "user_prompts": None,
                "handoff_message": None,
                "deferred_tool_metadata": {},
                "need_user_approve_tools": [],
                "need_user_approve_mcps": [],
                "auto_load_files": [],
            }
        },
    )
    run_store.write_message("run-1", [])

    runtime_state.register_run("session-1", "run-2")
    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )

    await coordinator.execute("run-2")

    assert coordinator.restore_run_ids == ["run-1"]
    refreshed_run = await db_session.get(RunRecord, "run-2")
    assert isinstance(refreshed_run, RunRecord)
    await db_session.refresh(refreshed_run)
    assert refreshed_run.status == "completed"


async def test_run_coordinator_terminal_gate_blocks_submit_until_completion(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    mark_run_running(session_record, run_record)
    await db_session.commit()

    runtime_state.register_run("session-1", "run-1")
    entered_gate = asyncio.Event()
    release_gate = asyncio.Event()
    coordinator = BlockingCommitRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        entered_gate=entered_gate,
        release_gate=release_gate,
    )

    execute_task = asyncio.create_task(coordinator.execute("run-1"))
    await asyncio.wait_for(entered_gate.wait(), timeout=2)

    submit_done = asyncio.Event()
    submit_result: dict[str, object] = {}

    async def submit_after_terminal_gate() -> None:
        session_factory = create_session_factory(db_engine)
        async with session_factory() as submit_session:
            response = await SessionController().submit_input(
                submit_session,
                settings,
                runtime_state,
                "session-1",
                SessionSubmitRequest(input_parts=[TextPart(type="text", text="after gate")]),
            )
            submit_result["delivery"] = response.delivery
            submit_result["run_id"] = response.run_id
        submit_done.set()

    submit_task = asyncio.create_task(submit_after_terminal_gate())
    await asyncio.sleep(0.05)
    assert not submit_done.is_set()

    release_gate.set()
    await execute_task
    await asyncio.wait_for(submit_task, timeout=2)

    assert submit_result["delivery"] == "submitted"
    assert submit_result["run_id"] != "run-1"
    refreshed_run = await db_session.get(RunRecord, "run-1")
    assert isinstance(refreshed_run, RunRecord)
    await db_session.refresh(refreshed_run)
    assert refreshed_run.status == "completed"


async def test_cancelled_success_commit_indexes_completed_usage(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        status="cancelled",
        trigger_type="api",
        profile_name="general",
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([session_record, run_record])
    await db_session.commit()

    usage = RunUsage(requests=1, input_tokens=10, output_tokens=2)
    snapshot = UsageSnapshot(
        run_id="run-1",
        total_usage=usage,
        total_cost_estimate=CostEstimate(total_amount=Decimal("0.007"), priced_requests=1),
    )
    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )

    await coordinator._commit_successful_run(
        run_id="run-1",
        session_id="session-1",
        dispatch_mode="async",
        buffers=ExecutionBuffers(latest_usage_snapshot=snapshot),
    )

    await db_session.refresh(run_record)
    assert run_record.status == "cancelled"
    assert extract_usage_snapshot_from_metadata(run_record.run_metadata) == snapshot


def test_run_restores_state_honors_restore_state_metadata() -> None:
    run_record = RunRecord(
        id="run-reset",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[],
        run_metadata={"restore_state": False},
    )

    assert _run_restores_state(run_record) is False
    run_record.run_metadata = {}
    assert _run_restores_state(run_record) is True


def test_runtime_source_metadata_includes_trigger_type_and_run_metadata() -> None:
    assert _runtime_source_metadata(
        trigger_type="schedule",
        run_metadata={"source": "schedule", "schedule_id": "schedule-1", "schedule_fire_id": "fire-1"},
        memory_metadata=None,
    ) == {
        "trigger_type": "schedule",
        "source": "schedule",
        "schedule_id": "schedule-1",
        "schedule_fire_id": "fire-1",
    }
    assert _runtime_source_metadata(
        trigger_type="memory",
        run_metadata={"memory": {"kind": "extract"}},
        memory_metadata={"kind": "extract"},
    ) == {"trigger_type": "memory", "memory": {"kind": "extract"}}


async def test_run_coordinator_marks_run_failed_on_exception(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    mark_run_running(session_record, run_record)
    await db_session.commit()

    runtime_state.register_run("session-1", "run-1")
    coordinator = StubRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        failure=RuntimeError("boom"),
    )

    await coordinator.execute("run-1")

    refreshed_run = await db_session.get(RunRecord, "run-1")
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert isinstance(refreshed_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    await db_session.refresh(refreshed_run)
    await db_session.refresh(refreshed_session)

    assert refreshed_run.status == "failed"
    assert refreshed_run.error_message == "boom"
    assert refreshed_session.head_success_run_id is None
    assert runtime_state.get_run_handle("run-1") is None


async def test_run_coordinator_preserves_interrupt_when_failure_races_with_stop(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    mark_run_running(session_record, run_record)
    await db_session.commit()

    runtime_state.register_run("session-1", "run-1")
    coordinator = InterruptingFailureRunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
    )

    await coordinator.execute("run-1")

    refreshed_run = await db_session.get(RunRecord, "run-1")
    assert isinstance(refreshed_run, RunRecord)
    await db_session.refresh(refreshed_run)
    assert refreshed_run.status == "cancelled"
    assert refreshed_run.termination_reason == "interrupt"

    assert runtime_state.get_run_handle("run-1") is None


def test_run_usage_snapshot_is_indexed_in_internal_metadata() -> None:
    usage = RunUsage(requests=1, input_tokens=10, output_tokens=2)
    snapshot = UsageSnapshot(
        run_id="run-1",
        total_usage=usage,
        total_cost_estimate=CostEstimate(total_amount=Decimal("0.007"), priced_requests=1),
    )
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        status="completed",
        trigger_type="api",
        input_parts=[],
        run_metadata={"source": "test"},
    )

    _index_run_usage_snapshot(run_record, ExecutionBuffers(latest_usage_snapshot=snapshot))

    restored = extract_usage_snapshot_from_metadata(run_record.run_metadata)
    assert restored == snapshot
    assert run_record.run_metadata["source"] == "test"


def test_cumulative_usage_snapshot_spans_deferred_execution_segments() -> None:
    def stream_event(sdk_run_id: str, requests: int, amount: str) -> StreamEvent:
        usage = RunUsage(requests=requests, input_tokens=requests * 10, output_tokens=requests * 2)
        snapshot = UsageSnapshot(
            run_id=sdk_run_id,
            total_usage=usage,
            total_cost_estimate=CostEstimate(
                total_amount=Decimal(amount),
                priced_requests=requests,
            ),
            model_usages={"model-a": usage},
            model_cost_estimates={
                "model-a": CostEstimate(
                    total_amount=Decimal(amount),
                    priced_requests=requests,
                )
            },
        )
        return StreamEvent(
            agent_id="main",
            agent_name="main",
            event=UsageSnapshotEvent(event_id=f"usage-{sdk_run_id}", snapshot=snapshot),
        )

    first_event, first = _with_cumulative_usage_snapshot(
        stream_event("sdk-segment-1", 1, "0.003"),
        segment_base=None,
        run_id="claw-run",
    )
    assert first is not None
    assert isinstance(first_event.event, UsageSnapshotEvent)
    assert first_event.event.snapshot is not None
    assert first_event.event.snapshot.run_id == "claw-run"

    second_event, second = _with_cumulative_usage_snapshot(
        stream_event("sdk-segment-2", 2, "0.007"),
        segment_base=first,
        run_id="claw-run",
    )

    assert second is not None
    assert isinstance(second_event.event, UsageSnapshotEvent)
    assert second_event.event.snapshot is not None
    assert second_event.event.snapshot.run_id == "claw-run"
    assert second.total_usage.requests == 3
    assert second.total_cost_estimate is not None
    assert second.total_cost_estimate.total_amount == Decimal("0.010")
    assert second.total_cost_estimate.priced_requests == 3


async def test_execution_supervisor_periodically_retries_pending_deliveries(
    monkeypatch: pytest.MonkeyPatch,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    supervisor = ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )
    recovered = asyncio.Event()
    attempts = 0

    async def startup_recover() -> dict[str, list[str]]:
        return {
            "cancelled_running": [],
            "recovered_async_tasks": [],
            "recovered_async_deliveries": [],
            "submitted_queued": [],
        }

    async def recover_pending_deliveries() -> list[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient delivery failure")
        recovered.set()
        return ["delivery-run"]

    monkeypatch.setattr(
        coordinator_module,
        "_ASYNC_DELIVERY_RECOVERY_INTERVAL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(supervisor, "startup_recover", startup_recover)
    monkeypatch.setattr(
        supervisor,
        "recover_pending_async_deliveries",
        recover_pending_deliveries,
    )

    await supervisor.startup()
    await asyncio.wait_for(recovered.wait(), timeout=1)
    await supervisor.shutdown()

    assert attempts >= 2
    assert supervisor._delivery_recovery_task is None
