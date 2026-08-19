from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
import yaacli.durable.sqlite as durable_sqlite
from pydantic_ai import DeferredToolRequests, Tool
from pydantic_ai.capabilities import Toolset as NativeToolsetCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.main import create_agent
from yaacli.durable.application import SessionApplicationService, build_runtime_descriptor
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.executor import (
    LocalExecutionCoordinator,
    LocalExecutionWorker,
    RuntimePlanRegistry,
    RuntimePlanUnavailableError,
)
from yaacli.durable.models import (
    ExecutionCheckpointRecord,
    InputState,
    LogicalRunStatus,
    MainRuntimeManifest,
    RevisionPayload,
    RuntimeDescriptor,
    StartRunRequest,
)
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.environment import TUIEnvironment
from yaacli.session import TUIContext


def _manifest(
    *,
    hitl_policy: Literal["wait", "deny"] = "deny",
    request_limit: int = 1000,
) -> MainRuntimeManifest:
    return MainRuntimeManifest(hitl_policy=hitl_policy, request_limit=request_limit)


def _runtime_factory(
    tmp_path: Path,
    model_for: Callable[[RuntimeDescriptor], Any],
    *,
    capabilities_for: Callable[[RuntimeDescriptor], list[Any]] | None = None,
):
    def factory(descriptor: RuntimeDescriptor, binding_ref: str):
        capabilities = [DurableInboxPumpCapability()]
        if capabilities_for is not None:
            capabilities.extend(capabilities_for(descriptor))
        return create_agent(
            model_for(descriptor),
            capabilities=capabilities,
            context_type=TUIContext,
            context_kwargs={"durable_binding_ref": binding_ref},
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    return factory


async def test_turn_runs_through_local_coordinator_and_commits_revision(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
    )
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=_runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="local answer"),
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="session-test")

        revision = await service.run_turn(
            session.session_id,
            ["hello"],
            descriptor=descriptor,
            idempotency_key="turn-test",
        )

        assert revision.terminal == {"output": "local answer", "status": "completed"}
        run = store.get_run(revision.logical_run_id)
        assert run is not None
        assert run.status is LogicalRunStatus.completed
        assert store.list_inputs(run.logical_run_id)[0].state is InputState.applied
        checkpoint = store.get_execution_checkpoint(run.execution_id)
        assert checkpoint is not None
        assert checkpoint.segment_status == "completed"
        assert store.read_events(session.session_id)[-1].event_type == "RUN_FINISHED"
        persisted = store.get_session(session.session_id)
        assert persisted is not None
        assert persisted.head_revision_id == revision.revision_id
    finally:
        await worker.close()
        store.close()


async def test_run_turn_retries_transient_initial_dispatch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    current_time = [datetime(2026, 8, 18, tzinfo=UTC)]
    monkeypatch.setattr(durable_sqlite, "utc_now", lambda: current_time[0])
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
    )
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=_runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="retried answer"),
        ),
    )
    original_start = worker.coordinator._start_execution
    attempts = 0

    def transient_start(execution_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            asyncio.get_running_loop().call_soon(
                current_time.__setitem__,
                0,
                current_time[0] + timedelta(seconds=3),
            )
            raise RuntimeError("transient dispatch failure")
        original_start(execution_id)

    monkeypatch.setattr(worker.coordinator, "_start_execution", transient_start)
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="retry-session")
        revision = await asyncio.wait_for(
            service.run_turn(
                session.session_id,
                ["retry me"],
                descriptor=descriptor,
                idempotency_key="retry-turn",
            ),
            timeout=5,
        )

        run = store.get_run(revision.logical_run_id)
        assert attempts == 2
        assert run is not None
        assert run.status is LogicalRunStatus.completed
        assert revision.terminal["output"] == "retried answer"
    finally:
        await worker.close()
        store.close()


async def test_cancel_before_dispatch_commits_terminal_revision(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
    )
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=_runtime_factory(tmp_path, lambda _descriptor: TestModel()),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="cancel-before-dispatch")
        run = service.accept_turn(
            session.session_id,
            ["cancel me"],
            descriptor=descriptor,
            idempotency_key="cancelled-turn",
        )
        service.accept_cancel(run.logical_run_id, reason="cancelled before dispatch")

        await service.dispatch_pending()

        cancelled = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        persisted_session = store.get_session(session.session_id)
        assert cancelled is not None
        assert cancelled.status is LogicalRunStatus.cancelled
        assert revision is not None
        assert revision.terminal == {
            "reason": "cancelled before dispatch",
            "status": "cancelled",
        }
        assert persisted_session is not None
        assert persisted_session.active_execution_id is None
    finally:
        await worker.close()
        store.close()


async def test_outbox_failure_does_not_strand_sibling_delivery_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "product.sqlite3"
    store = SQLiteSessionStore(database_path)
    current_time = [datetime(2026, 8, 18, tzinfo=UTC)]
    monkeypatch.setattr(durable_sqlite, "utc_now", lambda: current_time[0])
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
        executable_version="test-executable",
    )
    runs = []
    for index in range(2):
        session = store.create_session(str(tmp_path), session_id=f"session-{index}")
        runs.append(
            store.start_run(
                StartRunRequest(
                    session_id=session.session_id,
                    idempotency_key=f"turn-{index}",
                    descriptor=descriptor,
                    initial_content=[f"turn {index}"],
                    plan_fingerprint=descriptor.plan_fingerprint,
                    executable_version=descriptor.executable_version,
                )
            )
        )
    coordinator = LocalExecutionCoordinator(
        store=store,
        runtime_registry=RuntimePlanRegistry(descriptor.descriptor_id),
    )
    attempts: list[str] = []

    def fail_first(execution_id: str) -> None:
        attempts.append(execution_id)
        if len(attempts) == 1:
            raise RuntimeError("transient dispatch failure")

    monkeypatch.setattr(coordinator, "_start_execution", fail_first)
    with pytest.raises(RuntimeError, match="transient dispatch failure"):
        await coordinator.dispatch_outbox()

    assert await coordinator.dispatch_outbox() == 1
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT aggregate_id, state FROM outbox_commands ORDER BY created_at, rowid"
        ).fetchall()
    assert set(attempts) == {run.execution_id for run in runs}
    assert len(attempts) == 2
    assert dict(rows) == {
        attempts[0]: "pending",
        attempts[1]: "delivered",
    }
    store.close()


async def test_unavailable_runtime_descriptor_fails_closed_and_commits_failure(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    available = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        host_envelope={"behavior": "available"},
        main_plan_manifest=_manifest(),
    )
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=available,
        runtime_factory=_runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="must not run"),
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="descriptor-session")
        unavailable = build_runtime_descriptor(
            agent_spec={"name": "yaacli_main_v2", "model": "test"},
            host_envelope={"behavior": "different"},
            main_plan_manifest=_manifest(),
        )
        run = await service.start_turn(session.session_id, ["do not execute"], descriptor=unavailable)

        with pytest.raises(RuntimePlanUnavailableError, match="not registered"):
            await service.wait(run.logical_run_id)

        failed = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert failed is not None
        assert failed.status is LogicalRunStatus.failed
        assert revision is not None
        assert revision.terminal["error_type"] == "RuntimePlanUnavailableError"
        assert store.read_events(session.session_id)[-1].event_type == "RUN_ERROR"
    finally:
        await worker.close()
        store.close()


async def test_worker_dispatches_persisted_and_new_runs_to_exact_runtime_descriptors(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    executable_version = "test-executable"
    old_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test:old"},
        host_envelope={"answer": "old runtime"},
        main_plan_manifest=_manifest(),
        executable_version=executable_version,
    )
    new_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test:new"},
        host_envelope={"answer": "new runtime"},
        main_plan_manifest=_manifest(),
        executable_version=executable_version,
    )
    session = store.create_session(str(tmp_path), session_id="historical-runtime")
    persisted = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="old-turn",
            descriptor=old_descriptor,
            initial_content=["execute old plan"],
            plan_fingerprint=old_descriptor.plan_fingerprint,
            executable_version=old_descriptor.executable_version,
        )
    )

    def model_for(descriptor: RuntimeDescriptor) -> TestModel:
        answer = descriptor.host_envelope["answer"]
        assert isinstance(answer, str)
        return TestModel(custom_output_text=answer)

    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=new_descriptor,
        runtime_factory=_runtime_factory(tmp_path, model_for),
        executable_version=executable_version,
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        old_run = await service.wait(persisted.logical_run_id)
        old_revision = store.get_revision_for_run(old_run.logical_run_id)
        assert old_revision is not None
        assert old_revision.terminal["output"] == "old runtime"

        new_revision = await service.run_turn(
            session.session_id,
            ["execute new plan"],
            descriptor=worker.descriptor,
            idempotency_key="new-turn",
        )
        assert new_revision.terminal["output"] == "new runtime"
        assert worker.descriptor.descriptor_id == new_descriptor.descriptor_id
        assert {plan.descriptor.descriptor_id for plan in worker.runtime_registry.list()} == {
            old_descriptor.descriptor_id,
            new_descriptor.descriptor_id,
        }
    finally:
        await worker.close()
        store.close()


async def test_worker_fails_closed_when_historical_runtime_cannot_be_rebuilt(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    executable_version = "test-executable"
    old_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test:old"},
        host_envelope={"answer": "old"},
        main_plan_manifest=_manifest(),
        executable_version=executable_version,
    )
    new_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test:new"},
        host_envelope={"answer": "new"},
        main_plan_manifest=_manifest(),
        executable_version=executable_version,
    )
    session = store.create_session(str(tmp_path), session_id="missing-runtime")
    store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="old-turn",
            descriptor=old_descriptor,
            initial_content=["must not fall back"],
            plan_fingerprint=old_descriptor.plan_fingerprint,
            executable_version=old_descriptor.executable_version,
        )
    )

    def runtime_factory(descriptor: RuntimeDescriptor, binding_ref: str):
        if descriptor.descriptor_id == old_descriptor.descriptor_id:
            raise ValueError("historical manifest unavailable")
        return _runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="new"),
        )(descriptor, binding_ref)

    with pytest.raises(RuntimePlanUnavailableError, match="historical manifest unavailable"):
        await LocalExecutionWorker.create(
            store=store,
            state_path=tmp_path / "coordinator.state",
            active_descriptor=new_descriptor,
            runtime_factory=runtime_factory,
            executable_version=executable_version,
        )
    persisted_session = store.get_session(session.session_id)
    assert persisted_session is not None
    assert persisted_session.active_execution_id is not None
    execution = store.get_execution(persisted_session.active_execution_id)
    assert execution is not None
    run = store.get_run(execution.logical_run_id)
    assert run is not None
    assert run.status is LogicalRunStatus.pending
    assert store.list_nonterminal_descriptors() == (old_descriptor,)
    store.close()


async def test_cumulative_request_limit_spans_deferred_segments(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    effects: list[str] = []

    async def guarded_effect(value: str) -> str:
        effects.append(value)
        return value

    approval_toolset = FunctionToolset[TUIContext](
        [Tool(guarded_effect, requires_approval=True)],
        id="approval",
    )
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(request_limit=1),
    )

    def runtime_factory(_descriptor: RuntimeDescriptor, binding_ref: str):
        return create_agent(
            TestModel(call_tools=["guarded_effect"], custom_output_text="unreachable"),
            capabilities=[
                NativeToolsetCapability(approval_toolset, id="approval"),
                DurableInboxPumpCapability(),
            ],
            output_type=[str, DeferredToolRequests],
            context_type=TUIContext,
            context_kwargs={"durable_binding_ref": binding_ref},
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=runtime_factory,
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="session-budget")
        run = await service.start_turn(
            session.session_id,
            ["request the guarded effect"],
            descriptor=descriptor,
            idempotency_key="budget-turn",
        )

        with pytest.raises(RuntimeError, match="cumulative model request limit of 1"):
            await service.wait(run.logical_run_id)

        persisted_run = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert persisted_run is not None
        assert persisted_run.status is LogicalRunStatus.failed
        assert revision is not None
        assert revision.terminal == {
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "Execution exhausted the cumulative model request limit of 1.",
        }
        assert revision.usage["requests"] == 1
        assert effects == []
    finally:
        await worker.close()
        store.close()


async def test_suspended_run_does_not_block_another_session(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    approval_toolset = FunctionToolset[TUIContext](id="approval")

    async def guarded_effect(value: str) -> str:
        return value

    approval_toolset.add_tool(Tool(guarded_effect, requires_approval=True))

    async def stream_response(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        has_tool_result = any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, ToolReturnPart | RetryPromptPart) for part in message.parts)
            for message in messages
        )
        if "suspend this run" in str(messages) and not has_tool_result:
            yield {
                0: DeltaToolCall(
                    name="guarded_effect",
                    json_args='{"value":"one"}',
                    tool_call_id="suspended-call",
                )
            }
        else:
            yield "completed independently"

    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "function"},
        main_plan_manifest=_manifest(hitl_policy="wait"),
    )

    def runtime_factory(_descriptor: RuntimeDescriptor, binding_ref: str):
        return create_agent(
            FunctionModel(stream_function=stream_response),
            capabilities=[
                NativeToolsetCapability(approval_toolset, id="approval"),
                DurableInboxPumpCapability(),
            ],
            output_type=[str, DeferredToolRequests],
            context_type=TUIContext,
            context_kwargs={"durable_binding_ref": binding_ref},
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=runtime_factory,
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        first_session = service.create_session(str(tmp_path), session_id="suspended-session")
        second_session = service.create_session(str(tmp_path), session_id="independent-session")
        first_run = await service.start_turn(first_session.session_id, ["suspend this run"], descriptor=descriptor)
        for _ in range(100):
            persisted = store.get_run(first_run.logical_run_id)
            if persisted is not None and persisted.status is LogicalRunStatus.suspended:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("first run did not suspend")

        second_revision = await asyncio.wait_for(
            service.run_turn(
                second_session.session_id,
                ["complete independently"],
                descriptor=descriptor,
            ),
            timeout=5,
        )
        assert second_revision.terminal["output"] == "completed independently"

        persisted = store.get_run(first_run.logical_run_id)
        assert persisted is not None
        assert persisted.pending_action_batch_id is not None
        batch = store.get_action_batch(persisted.pending_action_batch_id)
        assert batch is not None
        await service.decide_action(
            batch.items[0].action_item_id,
            {"approved": False, "message": "test completed"},
        )
        completed = await service.wait(first_run.logical_run_id)
        assert completed.status is LogicalRunStatus.completed
        revision = store.get_revision_for_run(first_run.logical_run_id)
        assert revision is not None
        assert revision.usage["requests"] == 2
    finally:
        await worker.close()
        store.close()


async def test_startup_marks_active_execution_interrupted_from_latest_checkpoint(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
    )
    session = store.create_session(str(tmp_path), session_id="interrupted-session")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="interrupted-turn",
            descriptor=descriptor,
            initial_content=["start"],
            plan_fingerprint=descriptor.plan_fingerprint,
            executable_version=descriptor.executable_version,
        )
    )
    store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
    now = datetime.now(UTC)
    store.put_execution_checkpoint(
        ExecutionCheckpointRecord(
            execution_id=run.execution_id,
            logical_run_id=run.logical_run_id,
            segment_index=0,
            segment_status="suspended",
            payload=RevisionPayload(
                message_history=[{"kind": "checkpoint"}],
                usage={"requests": 1},
            ),
            deferred_requests={"approvals": [], "calls": [], "metadata": {}},
            created_at=now,
            updated_at=now,
        )
    )

    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=descriptor,
        runtime_factory=_runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="must not replay"),
        ),
    )
    try:
        recovered = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert recovered is not None
        assert recovered.status is LogicalRunStatus.interrupted
        assert revision is not None
        assert revision.message_history == [{"kind": "checkpoint"}]
        assert revision.usage == {"requests": 1}
        assert revision.terminal["status"] == "interrupted"
        assert "process restart" in str(revision.terminal["reason"])
        assert store.list_nonterminal_descriptors() == ()
    finally:
        await worker.close()
        store.close()


async def test_start_command_crash_window_is_recovered_as_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=_manifest(),
    )
    session = store.create_session(str(tmp_path), session_id="dispatch-crash-session")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="dispatch-crash-turn",
            descriptor=descriptor,
            initial_content=["start"],
            plan_fingerprint=descriptor.plan_fingerprint,
            executable_version=descriptor.executable_version,
        )
    )
    coordinator = LocalExecutionCoordinator(
        store=store,
        runtime_registry=RuntimePlanRegistry(descriptor.descriptor_id),
    )

    def crash_before_scheduling(coroutine: Any, *, name: str) -> Any:
        del name
        coroutine.close()
        raise RuntimeError("process crashed before task scheduling")

    monkeypatch.setattr(asyncio, "create_task", crash_before_scheduling)
    with pytest.raises(RuntimeError, match="before task scheduling"):
        await coordinator.dispatch_outbox()
    persisted = store.get_run(run.logical_run_id)
    assert persisted is not None
    assert persisted.status is LogicalRunStatus.running

    recovered = await coordinator.recover_orphaned_executions()
    assert recovered == (run.logical_run_id,)
    persisted = store.get_run(run.logical_run_id)
    assert persisted is not None
    assert persisted.status is LogicalRunStatus.interrupted
    store.close()


async def test_startup_recovers_running_old_executable_before_runtime_reconstruction(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    old_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "old"},
        main_plan_manifest=_manifest(),
        executable_version="old-build",
    )
    current_descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "current"},
        main_plan_manifest=_manifest(),
        executable_version="current-build",
    )
    session = store.create_session(str(tmp_path), session_id="upgrade-recovery-session")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="old-running-turn",
            descriptor=old_descriptor,
            initial_content=["do not replay"],
            plan_fingerprint=old_descriptor.plan_fingerprint,
            executable_version=old_descriptor.executable_version,
        )
    )
    store.set_run_status(run.logical_run_id, LogicalRunStatus.running)

    reconstructed: list[str] = []

    def runtime_factory(descriptor: RuntimeDescriptor, binding_ref: str):
        reconstructed.append(descriptor.descriptor_id)
        return _runtime_factory(
            tmp_path,
            lambda _descriptor: TestModel(custom_output_text="current"),
        )(descriptor, binding_ref)

    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_descriptor=current_descriptor,
        runtime_factory=runtime_factory,
        executable_version="current-build",
    )
    try:
        recovered = store.get_run(run.logical_run_id)
        assert recovered is not None
        assert recovered.status is LogicalRunStatus.interrupted
        assert reconstructed == [current_descriptor.descriptor_id]
    finally:
        await worker.close()
        store.close()


def test_runtime_descriptor_fingerprints_executable_and_host_plan() -> None:
    base = build_runtime_descriptor(
        agent_spec={"name": "main", "model": "test"},
        executable_version="build-a",
        host_envelope={"behavior": "a"},
    )
    changed_build = build_runtime_descriptor(
        agent_spec={"name": "main", "model": "test"},
        executable_version="build-b",
        host_envelope={"behavior": "a"},
    )
    changed_plan = build_runtime_descriptor(
        agent_spec={"name": "main", "model": "test"},
        executable_version="build-a",
        host_envelope={"behavior": "b"},
    )

    assert base.descriptor_id != changed_build.descriptor_id
    assert base.plan_fingerprint != changed_build.plan_fingerprint
    assert base.descriptor_id != changed_plan.descriptor_id
    assert base.executable_version == "build-a"


def test_runtime_descriptor_rejects_nested_mutation_after_fingerprinting(tmp_path: Path) -> None:
    source_spec: dict[str, Any] = {"name": "main", "nested": {"model": "test"}}
    descriptor = build_runtime_descriptor(agent_spec=source_spec)
    source_spec["nested"] = {"model": "changed externally"}
    descriptor.assert_integrity()
    assert descriptor.agent_spec["nested"] == {"model": "test"}

    nested = descriptor.agent_spec["nested"]
    assert isinstance(nested, dict)
    nested["model"] = "mutated descriptor"
    with pytest.raises(ValueError, match="does not match its fingerprint"):
        descriptor.assert_integrity()
    with SQLiteSessionStore(tmp_path / "integrity.sqlite3") as store:
        with pytest.raises(ValueError, match="does not match its fingerprint"):
            store.put_descriptor(descriptor)
