from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_environment import Environment
from ya_claw.config import ClawSettings
from ya_claw.controller.models import SessionSubmitRequest, TextPart
from ya_claw.controller.session import SessionController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution import coordinator as coordinator_module
from ya_claw.execution.coordinator import (
    ExecutionBuffers,
    ExecutionSupervisor,
    RunCoordinator,
    _run_restores_state,
    _runtime_source_metadata,
)
from ya_claw.execution.profile import ResolvedProfile
from ya_claw.execution.state_machine import interrupt_run, mark_run_running
from ya_claw.execution.store import RunStore
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, SessionRecord
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
        return ResolvedProfile(
            name=profile_name or "general",
            model="stub-model",
            model_settings=None,
            model_config=None,
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


class TerminalGateRuntimeContext:
    def __init__(self) -> None:
        self.file_operator = None
        self.container_id = None
        self.claw_metadata: dict[str, Any] = {}
        self.session_id = "session-1"
        self.claw_run_id = "run-1"
        self.profile_name = "general"
        self.restore_from_run_id = None
        self.dispatch_mode = "async"
        self.source_kind = "api"
        self.source_metadata: dict[str, Any] = {}
        self.workspace_binding = None

    def export_state(self) -> Any:
        class ExportedState:
            def model_dump(self, *, mode: str) -> dict[str, Any]:
                return {
                    "notes": {},
                    "tasks": [],
                    "tool_search_loaded_tools": [],
                    "tool_search_loaded_namespaces": [],
                    "subagent_history": {},
                    "extra_usages": [],
                    "user_prompts": None,
                    "steering_messages": [],
                    "handoff_message": None,
                    "deferred_tool_metadata": {},
                    "agent_registry": {},
                    "need_user_approve_tools": [],
                    "need_user_approve_mcps": [],
                    "auto_load_files": [],
                }

        return ExportedState()

    def send_message(self, message: object) -> None:
        return None


class TerminalGateRuntime:
    core_toolset = None

    def __init__(self) -> None:
        self.ctx = TerminalGateRuntimeContext()

    async def __aenter__(self) -> TerminalGateRuntime:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class TerminalGateRuntimeBuilder:
    def build(self, **_: object) -> TerminalGateRuntime:
        return TerminalGateRuntime()


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
            "notes": {},
            "tasks": [],
            "tool_search_loaded_tools": [],
            "tool_search_loaded_namespaces": [],
            "subagent_history": {},
            "extra_usages": [],
            "user_prompts": None,
            "steering_messages": [],
            "handoff_message": None,
            "deferred_tool_metadata": {},
            "agent_registry": {},
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


class DummyRuntimeContext:
    file_operator = None


class DummyRuntime:
    def __init__(self) -> None:
        self.ctx = DummyRuntimeContext()


class DummyRun:
    result = None

    def all_messages(self) -> list[Any]:
        return []


class DummyStreamer:
    run = DummyRun()

    def __aiter__(self) -> DummyStreamer:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    def raise_if_exception(self) -> None:
        return None

    def recoverable_messages(self) -> list[Any]:
        return []


@asynccontextmanager
async def terminal_gate_stream_agent(*_: Any, **__: Any):
    yield DummyStreamer()


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
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'coordinator.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
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
    assert refreshed_session.head_success_run_id == "run-1"
    assert refreshed_session.active_run_id is None
    assert state_payload is not None
    assert state_payload["container_id"] is None
    assert state_payload["context_state"]["notes"] == {}
    assert state_payload["message_history"][0]["content"] == "completed run-1"
    assert message_payload is not None
    assert message_payload[0]["content"] == "completed run-1"

    assert runtime_state.get_run_handle("run-1") is None


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
                "notes": {},
                "tasks": [],
                "tool_search_loaded_tools": [],
                "tool_search_loaded_namespaces": [],
                "subagent_history": {},
                "extra_usages": [],
                "user_prompts": None,
                "steering_messages": [],
                "handoff_message": None,
                "deferred_tool_metadata": {},
                "agent_registry": {},
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


async def test_execute_agent_run_terminal_gate_continues_late_steering(
    monkeypatch: pytest.MonkeyPatch,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    runtime_state.register_run("session-1", "run-1")
    coordinator = RunCoordinator(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=StubWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=TerminalGateRuntimeBuilder(),
    )
    buffers = ExecutionBuffers()
    commit_calls = 0
    prompts: list[str | list[Any] | None] = []

    async def fake_stream_agent(runtime: object, *args: Any, **kwargs: Any):
        prompts.append(kwargs.get("user_prompt"))
        if len(prompts) == 1:
            await runtime_state.record_steering(
                "run-1",
                [TextPart(type="text", text="late steer").model_dump(mode="json")],
            )
        async with terminal_gate_stream_agent(runtime, *args, **kwargs) as streamer:
            yield streamer

    async def fake_commit_successful_run(**_: Any) -> None:
        nonlocal commit_calls
        commit_calls += 1
        buffers.success_committed = True
        buffers.terminal_event_emitted = True
        buffers.clear_runtime_handle = True

    monkeypatch.setattr(coordinator_module, "stream_agent", asynccontextmanager(fake_stream_agent))
    monkeypatch.setattr(coordinator, "_commit_successful_run", fake_commit_successful_run)

    await coordinator._execute_agent_run(
        run_id="run-1",
        session_id="session-1",
        dispatch_mode="async",
        workspace_binding=StubWorkspaceProvider(settings.resolved_workspace_dir).resolve(),
        restore_point=None,
        input_parts=[TextPart(type="text", text="hello")],
        profile=ResolvedProfile(name="general", model="stub-model", model_settings=None, model_config=None),
        profile_name="general",
        trigger_type="api",
        run_metadata={},
        buffers=buffers,
    )

    assert len(prompts) == 2
    assert prompts[0] is None
    assert prompts[1] == "late steer"
    assert commit_calls == 1
    assert runtime_state.consume_steering_inputs("run-1") == []


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
