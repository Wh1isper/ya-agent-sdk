from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic_ai import AgentSpec
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition, LogicalRunInputRouter, RunInputLedger
from ya_agent_sdk.subagents import (
    AsyncioSubagentExecutionHost,
    SubagentDeliveryState,
    SubagentDurability,
    SubagentExecutionMode,
    SubagentExecutionService,
    SubagentExecutionState,
    SubagentInputState,
    SubagentLinkagePolicy,
    SubagentRegistry,
    SubagentSpec,
)
from ya_claw.config import ClawSettings
from ya_claw.controller.async_task import AsyncTaskController
from ya_claw.controller.models import (
    AsyncTaskCancelRequest,
    AsyncTaskSpawnRequest,
    AsyncTaskSteerRequest,
    RunCreateRequest,
    TextPart,
)
from ya_claw.controller.run import RunController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.state_machine import complete_run, fail_run, mark_run_running
from ya_claw.execution.subagents import (
    ClawSubagentCompletionDelivery,
    ClawSubagentDriver,
    ClawSubagentExecutionStore,
    resolve_claw_subagent_plan,
)
from ya_claw.orm.tables import (
    ProfileRecord,
    RunInputInboxRecord,
    RunRecord,
    SessionAsyncTaskRecord,
    SessionRecord,
)
from ya_claw.runtime_state import InMemoryRuntimeState, create_runtime_state
from ya_claw.toolsets.session import ClawSelfApiError


class _Profile:
    name = "general"
    subagent_specs = (
        SubagentSpec(
            route="worker",
            agent=AgentSpec(
                model="test",
                name="worker",
                description="Test worker",
                instructions="Complete the requested work.",
                capabilities=[],
            ),
            execution_modes=(
                SubagentExecutionMode.foreground,
                SubagentExecutionMode.background,
            ),
            durability=SubagentDurability.restart,
        ),
    )


class _ProfileResolver:
    def __init__(self, profile: Any | None = None) -> None:
        self.profile = profile or _Profile()

    async def resolve(self, profile_name: str | None) -> Any:
        assert profile_name == "general"
        return self.profile


class _ControllerClient:
    """Exercise the same controller boundary used by ClawSelfClient HTTP calls."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        profile_resolver: _ProfileResolver | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.runtime_state = runtime_state
        self.profile_resolver = profile_resolver or _ProfileResolver()
        self.controller = AsyncTaskController()

    async def spawn_delegate(
        self,
        *,
        subagent_name: str,
        prompt: str,
        name: str | None,
        context: dict[str, Any] | None,
        sdk_owner_scope_id: str,
        sdk_idempotency_key: str,
        sdk_intent_fingerprint: str,
    ) -> dict[str, Any]:
        async with self.session_factory() as db_session:
            response = await self.controller.spawn_delegate(
                db_session,
                self.settings,
                self.runtime_state,
                parent_session_id="parent-session",
                parent_run_id=None,
                request=AsyncTaskSpawnRequest(
                    subagent_name=subagent_name,
                    prompt=prompt,
                    name=name,
                    context=context or {},
                    sdk_owner_scope_id=sdk_owner_scope_id,
                    sdk_idempotency_key=sdk_idempotency_key,
                    sdk_intent_fingerprint=sdk_intent_fingerprint,
                ),
                profile_resolver=self.profile_resolver,
            )
            return response.model_dump(mode="json")

    async def get_async_subagent(
        self,
        *,
        name_or_task_id: str,
    ) -> dict[str, Any]:
        async with self.session_factory() as db_session:
            response = await self.controller.get_task(
                db_session,
                self.settings,
                parent_session_id="parent-session",
                task_id_or_name=name_or_task_id,
            )
            return response.model_dump(mode="json")

    async def steer_async_subagent(
        self,
        *,
        name_or_task_id: str,
        prompt: str | None,
        input_parts: list[dict[str, Any]] | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        async with self.session_factory() as db_session:
            response = await self.controller.steer_task(
                db_session,
                self.settings,
                self.runtime_state,
                parent_session_id="parent-session",
                task_id_or_name=name_or_task_id,
                request=AsyncTaskSteerRequest(
                    prompt=prompt,
                    input_parts=input_parts or [],
                    idempotency_key=idempotency_key,
                ),
            )
            return response.model_dump(mode="json")

    async def cancel_async_subagent(
        self,
        *,
        name_or_task_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        async with self.session_factory() as db_session:
            response = await self.controller.cancel_task(
                db_session,
                self.settings,
                self.runtime_state,
                parent_session_id="parent-session",
                task_id_or_name=name_or_task_id,
                request=AsyncTaskCancelRequest(reason=reason),
            )
            return response.model_dump(mode="json")

    async def task(self, execution_id: str) -> SessionAsyncTaskRecord:
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord).where(
                    SessionAsyncTaskRecord.parent_session_id == "parent-session",
                    SessionAsyncTaskRecord.name == execution_id,
                )
            )
            task = result.scalar_one()
            db_session.expunge(task)
            return task

    async def mark_running(self, execution_id: str) -> SessionAsyncTaskRecord:
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.name == execution_id)
            )
            task = result.scalar_one()
            child_session = await db_session.get(SessionRecord, task.task_session_id)
            child_run = await db_session.get(RunRecord, task.task_run_id)
            assert isinstance(child_session, SessionRecord)
            assert isinstance(child_run, RunRecord)
            mark_run_running(child_session, child_run, claimed_by="adapter-test")
            await db_session.commit()
            await db_session.refresh(task)
            db_session.expunge(task)
            return task

    async def complete(self, execution_id: str, output: str) -> SessionAsyncTaskRecord:
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.name == execution_id)
            )
            task = result.scalar_one()
            child_session = await db_session.get(SessionRecord, task.task_session_id)
            child_run = await db_session.get(RunRecord, task.task_run_id)
            assert isinstance(child_session, SessionRecord)
            assert isinstance(child_run, RunRecord)
            child_run.output_text = output
            complete_run(child_session, child_run, committed_at=datetime.now(UTC))
            await self.controller.on_run_terminal(
                db_session,
                self.settings,
                self.runtime_state,
                run_record=child_run,
            )
            await db_session.refresh(task)
            db_session.expunge(task)
            return task

    async def fail(self, execution_id: str, error: str) -> SessionAsyncTaskRecord:
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.name == execution_id)
            )
            task = result.scalar_one()
            child_session = await db_session.get(SessionRecord, task.task_session_id)
            child_run = await db_session.get(RunRecord, task.task_run_id)
            assert isinstance(child_session, SessionRecord)
            assert isinstance(child_run, RunRecord)
            child_run.error_message = error
            fail_run(
                child_session,
                child_run,
                finished_at=datetime.now(UTC),
            )
            await self.controller.on_run_terminal(
                db_session,
                self.settings,
                self.runtime_state,
                run_record=child_run,
            )
            await db_session.refresh(task)
            db_session.expunge(task)
            return task


@pytest.fixture
async def db_engine(
    tmp_path: Path,
    initialize_sqlite_database: Callable[[str], None],
) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'sdk-subagent-adapter.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


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


async def _create_parent(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as db_session:
        db_session.add(
            ProfileRecord(
                name="general",
                agent_spec={"model": "test", "name": "general"},
                host_config={},
                subagent_specs=[],
            )
        )
        db_session.add(
            SessionRecord(
                id="parent-session",
                profile_name="general",
                session_metadata={},
                session_type="conversation",
            )
        )
        await db_session.commit()


def _service(
    session_factory: async_sessionmaker[AsyncSession],
    client: _ControllerClient,
    settings: ClawSettings,
    *,
    active_specs: tuple[SubagentSpec, ...] = _Profile.subagent_specs,
) -> SubagentExecutionService:
    plans = tuple(resolve_claw_subagent_plan(spec) for spec in active_specs)
    store = ClawSubagentExecutionStore(
        session_factory=session_factory,
        parent_session_id="parent-session",
        client=client,
    )
    return SubagentExecutionService(
        SubagentRegistry(plans),
        store,
        ClawSubagentDriver(
            client=client,
            data_settings=settings,
            poll_interval_seconds=0.001,
        ),
        completion_delivery=ClawSubagentCompletionDelivery(
            session_factory=session_factory,
            parent_session_id="parent-session",
        ),
        retained_plan_provider=store,
        execution_host=AsyncioSubagentExecutionHost(),
    )


def _parent_context() -> AgentContext:
    return AgentContext(
        agent_id="main",
        delegation_scope_id="parent-session",
        run_input_ledger=RunInputLedger(logical_run_id="parent-logical-run"),
    )


async def test_sdk_service_preserves_public_handle_case_and_punctuation(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    spec = SubagentSpec(
        route="Worker/a",
        agent=AgentSpec(model="test", name="Worker/a", capabilities=[]),
        execution_modes=(SubagentExecutionMode.foreground,),
        durability=SubagentDurability.restart,
    )
    profile = _Profile()
    profile.subagent_specs = (spec,)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
        profile_resolver=_ProfileResolver(profile),
    )
    service = _service(session_factory, client, settings, active_specs=(spec,))

    handle = await service.spawn(
        "Worker/a",
        "preserve identity",
        _parent_context(),
        idempotency_key="preserve-handle",
    )
    task = await client.task(handle.execution_id)

    assert handle.execution_id.startswith("Worker/a-")
    assert task.name == handle.execution_id
    await service.close()


async def test_sdk_steering_reports_terminal_closure_as_rejected(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    try:
        handle = await service.spawn(
            "worker",
            "finish before steering",
            _parent_context(),
            idempotency_key="terminal-child",
        )
        await client.mark_running(handle.execution_id)
        await client.complete(handle.execution_id, "done")

        receipt = await service.steer(
            handle.execution_id,
            "too late",
            caller_scope_id="parent-session",
            idempotency_key="terminal-steer",
        )

        assert receipt.disposition is InputDisposition.rejected
        assert receipt.input_id == "terminal-steer"
    finally:
        await service.close()


async def test_sdk_steering_replays_applied_input_after_terminal_transition(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    runtime_state = create_runtime_state()
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=runtime_state,
    )
    service = _service(session_factory, client, settings)
    try:
        handle = await service.spawn(
            "worker",
            "apply steering before completion",
            _parent_context(),
            idempotency_key="terminal-replay-child",
        )
        task = await client.mark_running(handle.execution_id)
        assert isinstance(task.task_run_id, str)

        async def ingress(
            input_id: str,
            input_parts: list[dict[str, Any]],
        ) -> EnqueueReceipt:
            _ = input_id, input_parts
            return EnqueueReceipt(
                logical_run_id="child-logical-run",
                input_id="durable-sdk-input",
                disposition=InputDisposition.applied,
                enqueue_id="durable-native-enqueue",
            )

        runtime_state.bind_input_ingress(task.task_run_id, ingress)
        first = await service.steer(
            handle.execution_id,
            "stable steering payload",
            caller_scope_id="parent-session",
            idempotency_key="terminal-replay-input",
        )
        await client.complete(handle.execution_id, "done")
        completed = await service.wait(
            handle.execution_id,
            caller_scope_id="parent-session",
            timeout=2,
        )
        assert completed.state is SubagentExecutionState.succeeded
        resumed = await service.resume(
            handle.execution_id,
            "continue on a new run",
            _parent_context(),
            idempotency_key="terminal-replay-resume",
        )
        resumed_task = await client.mark_running(resumed.execution_id)
        assert isinstance(resumed_task.task_run_id, str)
        assert resumed_task.task_run_id != task.task_run_id
        resumed_deliveries: list[str] = []

        async def resumed_ingress(
            input_id: str,
            input_parts: list[dict[str, Any]],
        ) -> EnqueueReceipt:
            _ = input_parts
            resumed_deliveries.append(input_id)
            return EnqueueReceipt(
                logical_run_id="resumed-logical-run",
                input_id="wrong-resumed-sdk-input",
                disposition=InputDisposition.applied,
                enqueue_id="wrong-resumed-native-enqueue",
            )

        runtime_state.bind_input_ingress(resumed_task.task_run_id, resumed_ingress)
        replay = await service.steer(
            handle.execution_id,
            "stable steering payload",
            caller_scope_id="parent-session",
            idempotency_key="terminal-replay-input",
        )

        assert replay == first
        assert replay.input_id == "durable-sdk-input"
        assert replay.enqueue_id == "durable-native-enqueue"
        assert replay.disposition is InputDisposition.applied
        async with session_factory() as db_session:
            rows = list(
                (
                    await db_session.execute(
                        select(RunInputInboxRecord).where(
                            RunInputInboxRecord.delivery_key == "terminal-replay-input",
                        )
                    )
                ).scalars()
            )
        assert resumed_deliveries == []
        assert len(rows) == 1
        assert rows[0].run_id == task.task_run_id
        assert rows[0].status == "applied"
    finally:
        await service.close()


async def test_sdk_steering_classifies_only_structured_self_api_closure(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)

    class StructuredErrorClient(_ControllerClient):
        def __init__(self, *, error_code: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.error_code = error_code

        async def steer_async_subagent(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            raise ClawSelfApiError(
                status_code=409,
                code=self.error_code,
                detail="message mentions run_input_closed",
            )

    closure_client = StructuredErrorClient(
        error_code="run_input_closed",
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    closure_service = _service(session_factory, closure_client, settings)
    handle = await closure_service.spawn(
        "worker",
        "closure classification",
        _parent_context(),
        idempotency_key="structured-closure-child",
    )
    receipt = await closure_service.steer(
        handle.execution_id,
        "late input",
        caller_scope_id="parent-session",
        idempotency_key="structured-closure-input",
    )
    assert receipt.disposition is InputDisposition.rejected
    await closure_service.close()

    unrelated_client = StructuredErrorClient(
        error_code="idempotency_conflict",
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    unrelated_service = _service(session_factory, unrelated_client, settings)
    unrelated = await unrelated_service.spawn(
        "worker",
        "unrelated classification",
        _parent_context(),
        idempotency_key="structured-unrelated-child",
    )
    with pytest.raises(ClawSelfApiError) as exc_info:
        await unrelated_service.steer(
            unrelated.execution_id,
            "input",
            caller_scope_id="parent-session",
            idempotency_key="structured-unrelated-input",
        )
    assert exc_info.value.code == "idempotency_conflict"
    await unrelated_service.close()


async def test_sdk_steering_does_not_hide_idempotency_conflicts(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    try:
        handle = await service.spawn(
            "worker",
            "keep running",
            _parent_context(),
            idempotency_key="active-child",
        )
        await client.mark_running(handle.execution_id)
        first = await service.steer(
            handle.execution_id,
            "first steering payload",
            caller_scope_id="parent-session",
            idempotency_key="stable-steer",
        )
        assert first.disposition is InputDisposition.accepted

        with pytest.raises(HTTPException) as exc_info:
            await service.steer(
                handle.execution_id,
                "different steering payload",
                caller_scope_id="parent-session",
                idempotency_key="stable-steer",
            )
        assert exc_info.value.status_code == 409
        assert "different content" in str(exc_info.value.detail)
    finally:
        await service.close()


async def test_sdk_service_foreground_is_idempotent_and_resumes_child_session(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    parent = _parent_context()

    first = await service.spawn(
        "worker",
        "first request",
        parent,
        mode=SubagentExecutionMode.foreground,
        idempotency_key="same-request",
    )
    queued = await service.get(
        first.execution_id,
        caller_scope_id="parent-session",
    )
    first_task = await client.task(first.execution_id)
    assert queued.input_state is SubagentInputState.applied
    assert first_task.sdk_input_state == SubagentInputState.applied.value
    assert first_task.sdk_owner_scope_id == "parent-session"
    assert first_task.sdk_idempotency_key == "same-request"

    duplicate = await service.spawn(
        "worker",
        "first request",
        parent,
        mode=SubagentExecutionMode.foreground,
        idempotency_key="same-request",
    )
    assert duplicate.execution_id == first.execution_id

    await client.mark_running(first.execution_id)
    running = await service.get(
        first.execution_id,
        caller_scope_id="parent-session",
    )
    assert running.state is SubagentExecutionState.running
    assert running.input_state is SubagentInputState.applied

    await client.complete(first.execution_id, "first result")
    completed = await service.wait(
        first.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )
    assert completed.state is SubagentExecutionState.succeeded
    assert completed.output == "first result"
    assert completed.delivery_state is SubagentDeliveryState.not_required
    assert await service.list(caller_scope_id="other-session") == ()
    with pytest.raises(KeyError, match="Unknown subagent execution"):
        await service.get(
            first.execution_id,
            caller_scope_id="other-session",
        )
    with pytest.raises(KeyError, match="Unknown subagent execution"):
        await service.resume(
            first.execution_id,
            "unauthorized continuation",
            AgentContext(delegation_scope_id="other-session"),
        )

    resumed = await service.resume(
        first.execution_id,
        "continue",
        parent,
        idempotency_key="resume-request",
    )
    first_task = await client.task(first.execution_id)
    resumed_task = await client.task(resumed.execution_id)
    assert resumed.execution_id != first.execution_id
    assert resumed_task.task_session_id == first_task.task_session_id

    await client.complete(resumed.execution_id, "resumed result")
    resumed_record = await service.wait(
        resumed.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )
    assert resumed_record.root_execution_id == first.execution_id
    assert resumed_record.resumed_from == first.execution_id
    assert resumed_record.output == "resumed result"

    async with session_factory() as db_session:
        task_count = (
            await db_session.execute(
                select(func.count(SessionAsyncTaskRecord.id)).where(
                    SessionAsyncTaskRecord.parent_session_id == "parent-session"
                )
            )
        ).scalar_one()
    assert task_count == 2
    await service.close()


async def test_sdk_spawn_is_sql_atomic_across_concurrent_sessions(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    parent = _parent_context()

    first, second = await asyncio.gather(
        service.spawn(
            "worker",
            "same concurrent request",
            parent,
            idempotency_key="concurrent-key",
        ),
        service.spawn(
            "worker",
            "same concurrent request",
            parent,
            idempotency_key="concurrent-key",
        ),
    )

    assert first.execution_id == second.execution_id
    async with session_factory() as db_session:
        task_count = (
            await db_session.execute(
                select(func.count(SessionAsyncTaskRecord.id)).where(
                    SessionAsyncTaskRecord.sdk_owner_scope_id == "parent-session",
                    SessionAsyncTaskRecord.sdk_idempotency_key == "concurrent-key",
                )
            )
        ).scalar_one()
        child_session_count = (
            await db_session.execute(
                select(func.count(SessionRecord.id)).where(
                    SessionRecord.parent_session_id == "parent-session",
                    SessionRecord.session_type == "async_task",
                )
            )
        ).scalar_one()
        child_run_count = (
            await db_session.execute(
                select(func.count(RunRecord.id))
                .join(
                    SessionRecord,
                    SessionRecord.id == RunRecord.session_id,
                )
                .where(
                    SessionRecord.parent_session_id == "parent-session",
                    SessionRecord.session_type == "async_task",
                )
            )
        ).scalar_one()
    assert (task_count, child_session_count, child_run_count) == (1, 1, 1)
    await service.close()


async def test_sdk_spawn_rejects_idempotency_intent_conflict(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    parent = _parent_context()

    await service.spawn(
        "worker",
        "original intent",
        parent,
        idempotency_key="conflicting-key",
    )
    with pytest.raises(ValueError, match="different intent"):
        await service.spawn(
            "worker",
            "changed intent",
            parent,
            idempotency_key="conflicting-key",
        )
    await service.close()


async def test_sdk_spawn_recovers_after_task_commit_before_client_response(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)

    class LostResponseClient(_ControllerClient):
        async def spawn_delegate(self, **kwargs: Any) -> dict[str, Any]:
            await super().spawn_delegate(**kwargs)
            raise RuntimeError("response lost after task commit")

    lost_client = LostResponseClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    lost_service = _service(session_factory, lost_client, settings)
    parent = _parent_context()
    with pytest.raises(RuntimeError, match="response lost after task commit"):
        await lost_service.spawn(
            "worker",
            "recover committed task",
            parent,
            idempotency_key="lost-response",
        )
    await lost_service.close()

    recovered_client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    recovered_service = _service(session_factory, recovered_client, settings)
    recovered = await recovered_service.spawn(
        "worker",
        "recover committed task",
        parent,
        idempotency_key="lost-response",
    )
    record = await recovered_service.get(
        recovered.execution_id,
        caller_scope_id="parent-session",
    )

    assert record.input_state is SubagentInputState.applied
    async with session_factory() as db_session:
        task_count = (
            await db_session.execute(
                select(func.count(SessionAsyncTaskRecord.id)).where(
                    SessionAsyncTaskRecord.sdk_idempotency_key == "lost-response"
                )
            )
        ).scalar_one()
    assert task_count == 1
    await recovered_service.close()


async def test_sdk_initial_input_is_rejected_before_create_and_stays_applied_after_failure(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)

    class PreCreateFailureClient(_ControllerClient):
        async def spawn_delegate(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("failed before host admission")

    failing_client = PreCreateFailureClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    failing_service = _service(session_factory, failing_client, settings)
    with pytest.raises(RuntimeError, match="failed before host admission"):
        await failing_service.spawn(
            "worker",
            "not admitted",
            _parent_context(),
            idempotency_key="pre-create-failure",
        )
    async with session_factory() as db_session:
        assert (
            await db_session.execute(
                select(func.count(SessionAsyncTaskRecord.id)).where(
                    SessionAsyncTaskRecord.sdk_idempotency_key == "pre-create-failure"
                )
            )
        ).scalar_one() == 0
    await failing_service.close()

    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    service = _service(session_factory, client, settings)
    handle = await service.spawn(
        "worker",
        "admitted then failed",
        _parent_context(),
        idempotency_key="post-admission-failure",
    )
    await client.fail(handle.execution_id, "model failed")
    failed = await service.wait(
        handle.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )
    assert failed.state is SubagentExecutionState.failed
    assert failed.input_state is SubagentInputState.applied
    assert failed.error == "model failed"
    await service.close()


async def test_sdk_resume_restores_prior_plan_after_profile_route_is_deleted(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    profile_resolver = _ProfileResolver()
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
        profile_resolver=profile_resolver,
    )
    first_service = _service(session_factory, client, settings)
    parent = _parent_context()
    first = await first_service.spawn(
        "worker",
        "first request",
        parent,
        idempotency_key="initial",
    )
    await client.complete(first.execution_id, "first result")
    completed = await first_service.wait(
        first.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )
    first_task = await client.task(first.execution_id)
    assert completed.state is SubagentExecutionState.succeeded
    assert isinstance(first_task.plan_descriptor_ref, str)
    await first_service.close()

    class DeletedRouteProfile:
        name = "general"
        subagent_specs: tuple[SubagentSpec, ...] = ()

    profile_resolver.profile = DeletedRouteProfile()
    restarted_service = _service(
        session_factory,
        client,
        settings,
        active_specs=(),
    )
    resumed = await restarted_service.resume(
        first.execution_id,
        "continue from the immutable plan",
        parent,
        idempotency_key="resume-after-delete",
    )
    resumed_task = await client.task(resumed.execution_id)

    assert resumed_task.plan_descriptor_ref == first_task.plan_descriptor_ref
    assert resumed_task.plan_fingerprint == first_task.plan_fingerprint
    await client.complete(resumed.execution_id, "resumed result")
    resumed_record = await restarted_service.wait(
        resumed.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )
    assert resumed_record.descriptor_id == completed.descriptor_id
    assert resumed_record.output == "resumed result"
    await restarted_service.close()


async def test_sdk_service_forwards_steering_and_cancellation(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    runtime_state = create_runtime_state()
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=runtime_state,
    )
    service = _service(session_factory, client, settings)
    handle = await service.spawn(
        "worker",
        "wait for direction",
        _parent_context(),
        mode=SubagentExecutionMode.foreground,
    )
    task = await client.mark_running(handle.execution_id)
    received: list[list[dict[str, Any]]] = []

    async def ingress(
        input_id: str,
        input_parts: list[dict[str, Any]],
    ) -> EnqueueReceipt:
        received.append(input_parts)
        return EnqueueReceipt(
            logical_run_id="child-logical-run",
            input_id=input_id,
            disposition=InputDisposition.enqueued,
            enqueue_id="enqueue-1",
        )

    assert isinstance(task.task_run_id, str)
    runtime_state.bind_input_ingress(task.task_run_id, ingress)
    receipt = await service.steer(
        handle.execution_id,
        "focus on tests",
        caller_scope_id="parent-session",
    )
    assert isinstance(receipt, EnqueueReceipt)
    assert receipt.disposition is InputDisposition.enqueued
    assert receipt.enqueue_id == "enqueue-1"
    async with session_factory() as db_session:
        inbox = (
            await db_session.execute(select(RunInputInboxRecord).where(RunInputInboxRecord.run_id == task.task_run_id))
        ).scalar_one()
    assert receipt.input_id == inbox.sdk_input_id == inbox.id
    assert received[-1][0]["text"] == "focus on tests"

    cancelled = await service.cancel(
        handle.execution_id,
        caller_scope_id="parent-session",
    )
    assert cancelled.state is SubagentExecutionState.cancelled
    persisted = await client.task(handle.execution_id)
    assert persisted.status == "cancelled"
    await service.close()


async def test_detached_background_completion_does_not_wake_interrupted_parent(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    runtime_state = create_runtime_state()
    detached_spec = SubagentSpec(
        route="worker",
        agent=AgentSpec(
            model="test",
            name="worker",
            description="Detached test worker",
            capabilities=[],
        ),
        execution_modes=(SubagentExecutionMode.background,),
        linkage=SubagentLinkagePolicy.detached,
        durability=SubagentDurability.restart,
    )
    profile = _Profile()
    profile.subagent_specs = (detached_spec,)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=runtime_state,
        profile_resolver=_ProfileResolver(profile),
    )
    service = _service(
        session_factory,
        client,
        settings,
        active_specs=(detached_spec,),
    )
    parent_run_id: str
    async with session_factory() as db_session:
        parent_run = await RunController().create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id="parent-session",
                profile_name="general",
                input_parts=[TextPart(type="text", text="parent work")],
            ),
        )
        parent_session = await db_session.get(SessionRecord, "parent-session")
        parent_record = await db_session.get(RunRecord, parent_run.id)
        assert isinstance(parent_session, SessionRecord)
        assert isinstance(parent_record, RunRecord)
        mark_run_running(parent_session, parent_record, claimed_by="adapter-test")
        await db_session.commit()
        parent_run_id = parent_record.id

    try:
        handle = await service.spawn(
            "worker",
            "finish independently",
            _parent_context(),
            mode=SubagentExecutionMode.background,
            idempotency_key="detached-child",
        )
        await client.mark_running(handle.execution_id)
        async with session_factory() as db_session:
            await RunController().interrupt(
                db_session,
                settings,
                runtime_state,
                parent_run_id,
            )

        completed_task = await client.complete(handle.execution_id, "detached result")
        completed = await service.wait(
            handle.execution_id,
            caller_scope_id="parent-session",
            timeout=2,
        )

        assert completed_task.wake_policy == "record_only"
        assert completed_task.delivery_id is None
        assert completed_task.delivery_run_id is None
        assert completed.delivery_state is SubagentDeliveryState.not_required
        async with session_factory() as db_session:
            parent_run_count = (
                await db_session.execute(
                    select(func.count(RunRecord.id)).where(
                        RunRecord.session_id == "parent-session",
                    )
                )
            ).scalar_one()
            parent_inbox_count = (
                await db_session.execute(
                    select(func.count(RunInputInboxRecord.id)).where(
                        RunInputInboxRecord.run_id == parent_run_id,
                    )
                )
            ).scalar_one()
        assert parent_run_count == 1
        assert parent_inbox_count == 0
    finally:
        await service.close()


async def test_sdk_service_recovers_background_work_after_restart(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    await _create_parent(session_factory)
    client = _ControllerClient(
        session_factory=session_factory,
        settings=settings,
        runtime_state=create_runtime_state(),
    )
    first_service = _service(session_factory, client, settings)
    parent = _parent_context()
    live_router = LogicalRunInputRouter(parent.run_input_ledger)
    registration = parent.active_run_registry.register(live_router)
    parent.input_router = live_router
    handle = await first_service.spawn(
        "worker",
        "durable request",
        parent,
        mode=SubagentExecutionMode.background,
    )
    await first_service.close()

    recovered_service = _service(session_factory, client, settings)
    assert await recovered_service.deliver_pending(parent) == 0
    task = await client.complete(handle.execution_id, "durable result")
    record = await recovered_service.wait(
        handle.execution_id,
        caller_scope_id="parent-session",
        timeout=2,
    )

    assert task.delivery_status in {"accepted", "enqueued", "applied"}
    assert record.state is SubagentExecutionState.succeeded
    assert record.output == "durable result"
    assert record.delivery_state is (
        SubagentDeliveryState.delivered if task.delivery_status == "applied" else SubagentDeliveryState.pending
    )
    assert await recovered_service.deliver_pending(parent) == 0
    assert parent.run_input_ledger.records == []
    await recovered_service.close()
    live_router.close()
    parent.active_run_registry.unregister(registration)
    parent.input_router = None
