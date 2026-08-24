from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent, AgentSpec, DeferredToolRequests, DeferredToolResults, Tool, ToolDenied
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo as FunctionAgentInfo
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.capabilities import SupportsDeferredOutput, build_default_capability_catalog
from ya_agent_sdk.inputs import InputDisposition, InputOrigin, LogicalRunInputRouter, RunInputLedger
from ya_agent_sdk.subagents import (
    ResolvedSubagentPlan,
    SubagentDeliveryState,
    SubagentDurability,
    SubagentExecutionIdConflict,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionService,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)
from yaacli.durable.models import InputState, SessionStatus
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.durable.store import TombstonedSessionError
from yaacli.durable.subagents import (
    DurableSubagentInboxCapability,
    FileSubagentExecutionStore,
    LocalProcessorSubagentExecutionHost,
    LocalSubagentDriver,
)
from yaacli.session import TUIContext


def _put_worker_descriptor(store: FileSubagentExecutionStore) -> ResolvedSubagentPlan:
    plan = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    ).resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(name="worker", model="test"),
            durability=SubagentDurability.process,
        )
    )
    store.put_descriptor(plan)
    return plan


@dataclass
class ApprovalCapability(SupportsDeferredOutput, AbstractCapability[TUIContext]):
    id: str | None = "approval"

    effects: ClassVar[list[str]] = []

    def get_toolset(self) -> FunctionToolset[TUIContext]:
        async def guarded_effect(value: str) -> str:
            self.effects.append(value)
            return value

        return FunctionToolset(
            [Tool(guarded_effect, requires_approval=True)],
            id="child-approval",
        )


def test_file_subagent_store_requires_initialized_product_store(tmp_path: Path) -> None:
    database_path = tmp_path / "missing.sqlite3"

    with pytest.raises(RuntimeError, match="Initialize SQLiteSessionStore"):
        FileSubagentExecutionStore(database_path)


def test_product_database_contains_no_subagent_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "product.sqlite3"
    with SQLiteSessionStore(database_path):
        pass

    with sqlite3.connect(database_path) as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert not any(name.startswith("subagent_") for name in names)


async def test_session_locks_do_not_block_subagents_in_other_sessions(tmp_path: Path) -> None:
    database_path = tmp_path / "independent-session-locks.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session-a")
    product_store.create_session(str(tmp_path), session_id="session-b")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        root_execution_id="child-b",
        execution_id="child-b",
        owner_scope_id="session-b",
        idempotency_key="child-b",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await store.create(record)
    completed = threading.Event()
    errors: list[BaseException] = []

    def accept_other_session_input() -> None:
        try:
            store.accept_input(
                record.execution_id,
                ["independent"],
                idempotency_key="independent",
                origin=InputOrigin.user,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            completed.set()

    with store.state_files.session_lock("session-a"):
        thread = threading.Thread(target=accept_other_session_input)
        thread.start()
        assert completed.wait(timeout=2)
    thread.join(timeout=2)

    assert errors == []
    assert store.list_inputs(record.execution_id)[0].content == ["independent"]
    await store.close()
    product_store.close()


async def test_same_session_children_keep_independent_state_under_concurrent_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "same-session-children.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    records = tuple(
        SubagentExecutionRecord(
            root_execution_id=f"child-{index}",
            execution_id=f"child-{index}",
            owner_scope_id="session",
            idempotency_key=f"child-{index}",
            descriptor_id=plan.descriptor_id,
            plan_fingerprint=plan.fingerprint,
            route="worker",
            mode=SubagentExecutionMode.background,
            state=SubagentExecutionState.running,
            parent_agent_id="main",
            parent_logical_run_id="parent-run",
            prompt="work",
        )
        for index in range(2)
    )
    for record in records:
        await store.create(record)
    errors: list[BaseException] = []

    def write_inputs(record: SubagentExecutionRecord) -> None:
        try:
            for index in range(20):
                store.accept_input(
                    record.execution_id,
                    [f"{record.execution_id}:{index}"],
                    idempotency_key=f"input-{index}",
                    origin=InputOrigin.user,
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=write_inputs, args=(record,)) for record in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert [len(store.list_inputs(record.execution_id)) for record in records] == [20, 20]
    assert all(store.state_files.subagent_path("session", record.execution_id).exists() for record in records)
    await store.close()
    product_store.close()


async def test_concurrent_cross_session_create_claims_execution_id_once(tmp_path: Path) -> None:
    database_path = tmp_path / "global-execution-id.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session-a")
    product_store.create_session(str(tmp_path), session_id="session-b")
    stores = (FileSubagentExecutionStore(database_path), FileSubagentExecutionStore(database_path))
    plans = tuple(_put_worker_descriptor(store) for store in stores)
    records = tuple(
        SubagentExecutionRecord(
            root_execution_id="shared-execution",
            execution_id="shared-execution",
            owner_scope_id=f"session-{'ab'[index]}",
            idempotency_key=f"child-{index}",
            descriptor_id=plans[index].descriptor_id,
            plan_fingerprint=plans[index].fingerprint,
            route="worker",
            mode=SubagentExecutionMode.background,
            state=SubagentExecutionState.running,
            parent_agent_id="main",
            parent_logical_run_id="parent-run",
            prompt="work",
        )
        for index in range(2)
    )
    successes: list[SubagentExecutionRecord] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def create(store: FileSubagentExecutionStore, record: SubagentExecutionRecord) -> None:
        barrier.wait()
        try:
            successes.append(asyncio.run(store.create(record)))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=create, args=(store, record)) for store, record in zip(stores, records, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], SubagentExecutionIdConflict)
    assert all(not thread.is_alive() for thread in threads)
    for store in stores:
        await store.close()
    product_store.close()


async def test_local_subagent_driver_persists_deferred_segments_and_cumulative_usage(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "subagents.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(
        catalog,
        default_model="test",
        host_capabilities=(ApprovalCapability(),),
        restart_durable=False,
    ).resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(
                name="worker",
                model="test",
                instructions="Call the available tool and then return the result.",
            ),
            execution_modes=(SubagentExecutionMode.foreground, SubagentExecutionMode.background),
            durability=SubagentDurability.process,
        )
    )
    store.put_descriptor(plan)
    registry = SubagentRegistry([plan])
    service = SubagentExecutionService(
        registry,
        store,
        LocalSubagentDriver(
            store=store,
            request_limit=2,
            default_model_cfg=TUIContext().model_cfg,
            custom_capability_types=catalog.custom_capability_types,
            runtime_capabilities=(DurableSubagentInboxCapability(store=store),),
        ),
        execution_host=LocalProcessorSubagentExecutionHost(),
    )
    model_call_count = 0

    async def child_model(
        _messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal model_call_count
        model_call_count += 1
        if model_call_count == 1:
            yield {
                0: DeltaToolCall(
                    name="guarded_effect",
                    json_args='{"value":"deferred"}',
                    tool_call_id="approval-call",
                )
            }
        else:
            yield "done"

    deterministic_model = FunctionModel(stream_function=child_model)

    def model_wrapper(_model: Model, _agent_name: str, _metadata: dict[str, object]) -> Model:
        return deterministic_model

    parent_ctx = TUIContext(
        run_input_ledger=RunInputLedger(logical_run_id="parent-run"),
        model_wrapper=model_wrapper,
    )
    parent_ctx.delegation_scope_id = "session"
    parent_ctx.note_manager.set("scope", "parent-a")
    ApprovalCapability.effects = []
    try:
        handle = await service.spawn(
            "worker",
            "do the bounded task",
            parent_ctx,
            mode=SubagentExecutionMode.background,
            idempotency_key="child-once",
        )
        suspended = await service.wait(handle.execution_id, caller_scope_id="session")
        assert suspended.state is SubagentExecutionState.suspended, suspended.error
        assert suspended.deferred is not None
        assert suspended.resumable_state["notes"] == {"scope": "parent-a"}

        steering = await service.steer(
            handle.execution_id,
            "apply after continuation",
            caller_scope_id="session",
            idempotency_key="suspended-steer",
        )
        assert steering.disposition is InputDisposition.accepted
        persisted_inputs = store.list_inputs(handle.execution_id)
        assert len(persisted_inputs) == 1
        assert persisted_inputs[0].content == ["apply after continuation"]

        parent_ctx.note_manager.set("scope", "parent-b")
        requests = TypeAdapter(DeferredToolRequests).validate_python(suspended.deferred)
        continued = await service.continue_deferred(
            handle.execution_id,
            DeferredToolResults(
                approvals={requests.approvals[0].tool_call_id: ToolDenied(message="Denied by local host test")}
            ),
            parent_ctx,
        )
        record = await service.wait(continued.execution_id, caller_scope_id="session")

        assert continued.execution_id == handle.execution_id
        assert record.state is SubagentExecutionState.succeeded, record.error
        assert record.segment_index == 1
        assert record.usage["requests"] == 2
        assert record.resumable_state["notes"] == {"scope": "parent-a"}
        assert parent_ctx.note_manager.get("scope") == "parent-b"
        assert store.list_inputs(handle.execution_id)[0].state is InputState.applied
        assert store.get_descriptor(record.descriptor_id) is not None
        assert ApprovalCapability.effects == []

        duplicate = await service.spawn(
            "worker",
            "do the bounded task",
            parent_ctx,
            mode=SubagentExecutionMode.background,
            idempotency_key="child-once",
        )
        assert duplicate.execution_id == handle.execution_id
        assert len(await store.list(owner_scope_id="session")) == 1
        page, total = await store.list_page(owner_scope_id="session", offset=0, limit=1)
        assert total == 1
        assert page[0].execution_id == handle.execution_id
    finally:
        await service.close()
        product_store.close()


async def test_local_subagent_steering_is_persisted_before_graph_application(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "steering.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.foreground,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="initial",
    )
    await store.create(record)
    driver = LocalSubagentDriver(
        store=store,
        request_limit=2,
        default_model_cfg=TUIContext().model_cfg,
    )

    first = await driver.steer(
        record,
        "steer once",
        origin=InputOrigin.user,
        idempotency_key="stable-steer-call",
    )
    duplicate = await driver.steer(
        record,
        "steer once",
        origin=InputOrigin.user,
        idempotency_key="stable-steer-call",
    )
    inputs = store.list_inputs(record.execution_id)

    assert first.input_id == duplicate.input_id
    assert first.disposition.value == "accepted"
    assert len(inputs) == 1
    assert inputs[0].content == ["steer once"]
    assert inputs[0].state is InputState.accepted

    await store.save(
        record.model_copy(
            update={
                "state": SubagentExecutionState.succeeded,
                "completed_at": datetime.now(UTC),
            }
        )
    )
    terminal_retry = await driver.steer(
        record,
        "steer once",
        origin=InputOrigin.user,
        idempotency_key="stable-steer-call",
    )
    assert terminal_retry.input_id == first.input_id
    assert terminal_retry.disposition is InputDisposition.rejected
    assert store.list_inputs(record.execution_id)[0].state is InputState.rejected
    await store.close()
    product_store.close()


async def test_local_subagent_steering_returns_rejected_when_input_closes_during_race(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "steering-race.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        child_logical_run_id="child-run",
        prompt="initial",
    )
    await store.create(record)
    assert store.close_and_list_inputs(record.execution_id) == ()
    driver = LocalSubagentDriver(
        store=store,
        request_limit=2,
        default_model_cfg=TUIContext().model_cfg,
    )

    receipt = await driver.steer(
        record,
        "too late",
        origin=InputOrigin.user,
        idempotency_key="stable-steer-call",
    )

    assert receipt.disposition is InputDisposition.rejected
    assert receipt.logical_run_id == "child-run"
    assert receipt.input_id == "stable-steer-call"
    assert store.list_inputs(record.execution_id) == ()
    await store.close()
    product_store.close()


async def test_session_tombstone_atomically_fences_and_requests_child_cancellation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tombstone-subagents.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    session = product_store.create_session(str(tmp_path), session_id="session")
    child_store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(child_store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id=session.session_id,
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await child_store.create(record)
    accepted = child_store.accept_input(
        record.execution_id,
        ["late steering"],
        idempotency_key="steer",
        origin=InputOrigin.user,
    )

    product_store.tombstone_session(session.session_id)

    child_state = child_store.state_files.read_subagent(
        session.session_id,
        record.execution_id,
    )
    assert child_state.input_open is False
    assert child_state.cancel_requested is True
    assert child_state.cancellation_reason == "owner session tombstoned"
    assert child_store.list_inputs(record.execution_id)[0].state is InputState.rejected
    assert child_store.list_inputs(record.execution_id)[0].input_id == accepted.input_id

    with pytest.raises(TombstonedSessionError):
        child_store.require_executable(record.execution_id)
    with pytest.raises(TombstonedSessionError):
        child_store.accept_input(
            record.execution_id,
            ["too late"],
            idempotency_key="too-late",
            origin=InputOrigin.user,
        )
    late_success = await child_store.save(
        record.model_copy(
            update={
                "state": SubagentExecutionState.succeeded,
                "input_state": SubagentInputState.applied,
                "output": "must be fenced",
            }
        )
    )

    persisted = await child_store.get(record.execution_id)
    assert persisted is not None
    assert persisted.state is SubagentExecutionState.cancelled
    assert persisted.input_state is SubagentInputState.rejected
    assert persisted.error == "owner session tombstoned"
    assert late_success == persisted

    maintenance = product_store.run_maintenance()
    assert maintenance.purged_sessions == 1
    assert await child_store.get(record.execution_id) is None

    await child_store.close()
    product_store.close()


async def test_age_retention_skips_session_with_nonterminal_child(tmp_path: Path) -> None:
    database_path = tmp_path / "child-retention.sqlite3"
    now = datetime(2026, 1, 31, tzinfo=UTC)
    product_store = SQLiteSessionStore(database_path, max_session_age_days=30)
    session = product_store.create_session(str(tmp_path), session_id="session")
    child_store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(child_store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id=session.session_id,
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await child_store.create(record)
    product_store._connection.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        ((now - timedelta(days=31)).isoformat(), session.session_id),
    )

    result = product_store.run_maintenance(now=now)

    assert result.tombstoned_sessions == 0
    assert product_store.get_session(session.session_id).status is SessionStatus.active  # type: ignore[union-attr]
    assert await child_store.get(record.execution_id) == record

    await child_store.close()
    product_store.close()


async def test_subagent_store_marks_process_orphans_lost_on_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "orphan.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        execution_id="child-orphan",
        root_execution_id="child-orphan",
        owner_scope_id="session",
        idempotency_key="child-orphan",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await store.create(record)
    accepted = store.accept_input(
        record.execution_id,
        ["pending input"],
        idempotency_key="pending-input",
        origin=InputOrigin.user,
    )
    state = store.state_files.read_subagent("session", record.execution_id)
    store.state_files.write_subagent(
        state.model_copy(
            update={
                "owner_pid": 99_999_999,
                "owner_token": "f" * 32,
            }
        )
    )
    await store.close()

    reopened = FileSubagentExecutionStore(database_path)
    assert reopened.recover_orphaned_executions() == (record.execution_id,)
    recovered = await reopened.get(record.execution_id)
    inputs = reopened.list_inputs(record.execution_id)
    assert recovered is not None
    assert recovered.state is SubagentExecutionState.lost
    assert recovered.input_state is SubagentInputState.rejected
    assert recovered.error == "Subagent execution was interrupted by process restart."
    assert inputs[0].input_id == accepted.input_id
    assert inputs[0].state is InputState.rejected
    assert inputs[0].rejection_reason == "Subagent execution was interrupted by process restart."
    await reopened.close()
    product_store.close()


async def test_opening_read_only_subagent_store_does_not_recover_live_execution(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "live-reader.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    owner = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(owner)
    record = SubagentExecutionRecord(
        execution_id="child-live",
        root_execution_id="child-live",
        owner_scope_id="session",
        idempotency_key="child-live",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await owner.create(record)

    reader = FileSubagentExecutionStore(database_path)
    try:
        assert reader.recover_orphaned_executions() == ()
        persisted = await reader.get(record.execution_id)
        assert persisted is not None
        assert persisted.state is SubagentExecutionState.running
    finally:
        await reader.close()
        await owner.close()
        product_store.close()


async def test_subagent_store_retains_exact_descriptors_for_terminal_records(
    tmp_path: Path,
) -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    )
    retained = resolver.resolve(
        SubagentSpec(
            route="worker",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="worker", instructions="retained"),
        )
    )
    terminal_plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="worker", instructions="terminal retained plan"),
        )
    )
    database_path = tmp_path / "descriptors.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    store.put_descriptor(retained)
    store.put_descriptor(terminal_plan)
    terminal = SubagentExecutionRecord(
        root_execution_id="execution-terminal",
        execution_id="execution-terminal",
        owner_scope_id="session",
        idempotency_key="terminal",
        descriptor_id=terminal_plan.descriptor_id,
        plan_fingerprint=terminal_plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.foreground,
        state=SubagentExecutionState.succeeded,
        input_state=SubagentInputState.applied,
        delivery_state=SubagentDeliveryState.pending,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="done",
        output="done",
    )
    await store.create(terminal)

    descriptors = store.list_referenced_descriptors()
    assert descriptors == (terminal_plan.to_descriptor(),)
    assert resolver.restore(descriptors[0]).fingerprint == terminal_plan.fingerprint
    assert store.get_descriptor(retained.descriptor_id) is not None
    await store.close()

    product_store.run_maintenance()
    reopened = FileSubagentExecutionStore(database_path)
    try:
        assert reopened.get_descriptor(retained.descriptor_id) is None
        assert reopened.get_descriptor(terminal_plan.descriptor_id) == terminal_plan.to_descriptor()
    finally:
        await reopened.close()
        product_store.close()


async def test_execution_create_keeps_active_descriptor_across_product_maintenance(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "descriptor-race.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)

    product_store.run_maintenance()
    assert store.get_descriptor(plan.descriptor_id) == plan.to_descriptor()

    record = SubagentExecutionRecord(
        root_execution_id="execution",
        execution_id="execution",
        owner_scope_id="session",
        idempotency_key="execution",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    assert await store.create(record) == record
    assert store.get_descriptor(plan.descriptor_id) == plan.to_descriptor()

    await store.close()
    product_store.close()


async def test_subagent_state_file_is_self_contained_with_exact_descriptor(tmp_path: Path) -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="worker"),
        )
    )
    database_path = tmp_path / "missing-descriptor.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    store.put_descriptor(plan)
    record = SubagentExecutionRecord(
        root_execution_id="execution",
        execution_id="execution",
        owner_scope_id="session",
        idempotency_key="execution",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await store.create(record)

    state = store.state_files.read_subagent("session", record.execution_id)
    assert state.descriptor == plan.to_descriptor()
    assert state.record == record
    assert store.list_referenced_descriptors() == (plan.to_descriptor(),)
    await store.close()
    product_store.close()


async def test_child_inbox_reuses_router_enqueue_on_new_native_attempt(tmp_path: Path) -> None:
    database_path = tmp_path / "child-inbox-attempt.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.foreground,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="initial",
    )
    await store.create(record)
    accepted = store.accept_input(
        record.execution_id,
        ["steer once"],
        idempotency_key="stable-steer-call",
        origin=InputOrigin.user,
    )
    context = TUIContext(run_input_ledger=RunInputLedger(logical_run_id=record.child_logical_run_id))
    context._agent_id = record.execution_id
    ctx = MagicMock()
    ctx.deps = context
    ctx.run_id = "native-run-1"
    ctx.enqueue.side_effect = ["enqueue-1", "unexpected-duplicate"]
    capability = DurableSubagentInboxCapability(store=store)

    capability._enqueue_pending(ctx, (accepted,))
    persisted = store.list_inputs(record.execution_id)[0]
    ledger_record = context.run_input_ledger.get(accepted.input_id)

    router = LogicalRunInputRouter(context.run_input_ledger)
    context.input_router = router
    native_run = MagicMock()
    native_run.enqueue.side_effect = ["enqueue-2", "enqueue-3"]
    await router.bind(native_run, native_attempt_id="native-attempt-2")

    ctx.run_id = "different-pydantic-run-id"
    capability._enqueue_pending(ctx, (persisted,))
    persisted = store.list_inputs(record.execution_id)[0]

    assert ctx.enqueue.call_count == 1
    native_run.enqueue.assert_called_once()
    assert persisted.native_enqueue_id == "enqueue-2"
    assert [attempt.native_attempt_id for attempt in ledger_record.enqueue_attempts] == [
        "native-run-1",
        "native-attempt-2",
    ]

    capability._mark_applied(context, "enqueue-1")
    assert store.list_inputs(record.execution_id)[0].state is InputState.applied

    accepted_before_hook = store.accept_input(
        record.execution_id,
        ["applied before capability hook"],
        idempotency_key="applied-before-hook",
        origin=InputOrigin.feature,
    )
    receipt = await router.enqueue(
        "applied before capability hook",
        input_id=accepted_before_hook.input_id,
        origin=InputOrigin.feature,
    )
    assert receipt.enqueue_id == "enqueue-3"
    context.run_input_ledger.mark_applied_by_enqueue_id("enqueue-3")
    capability._sync_applied_inputs(context)
    assert store.list_inputs(record.execution_id)[1].state is InputState.applied
    await store.close()
    product_store.close()


async def test_child_inbox_applies_idempotent_steering_at_graph_boundary(tmp_path: Path) -> None:
    database_path = tmp_path / "child-inbox.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.foreground,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="initial",
    )
    await store.create(record)
    accepted = store.accept_input(
        record.execution_id,
        ["steer once"],
        idempotency_key="stable-steer-call",
        origin=InputOrigin.user,
    )
    duplicate = store.accept_input(
        record.execution_id,
        ["steer once"],
        idempotency_key="stable-steer-call",
        origin=InputOrigin.user,
    )
    calls = 0
    seen_messages: list[list[ModelMessage]] = []

    async def stream_function(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        seen_messages.append(messages)
        yield f"pass-{calls}"

    context = TUIContext(run_input_ledger=RunInputLedger(logical_run_id=record.child_logical_run_id))
    context._agent_id = record.execution_id
    agent = Agent(
        FunctionModel(stream_function=stream_function),
        deps_type=TUIContext,
        capabilities=[DurableSubagentInboxCapability(store=store)],
    )
    result = await agent.run("initial", deps=context)
    inputs = store.list_inputs(record.execution_id)

    assert duplicate.input_id == accepted.input_id
    assert len(inputs) == 1
    assert inputs[0].state is InputState.applied
    assert result.output == "pass-1"
    assert calls == 1
    assert "steer once" in str(seen_messages[0])
    await store.close()
    product_store.close()


async def test_tombstone_retry_finishes_child_fence_after_filesystem_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "tombstone-retry.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    session = product_store.create_session(str(tmp_path), session_id="session")
    child_store = FileSubagentExecutionStore(database_path)
    plan = _put_worker_descriptor(child_store)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id=session.session_id,
        idempotency_key="child-running",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await child_store.create(record)
    original_fence = product_store.state_files.fence_subagents

    def fail_fence(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        raise OSError("injected child fence failure")

    monkeypatch.setattr(product_store.state_files, "fence_subagents", fail_fence)
    with pytest.raises(OSError, match="injected child fence failure"):
        product_store.tombstone_session(session.session_id)

    assert product_store.get_session(session.session_id).status is SessionStatus.tombstoned  # type: ignore[union-attr]
    assert (await child_store.get(record.execution_id)).state is SubagentExecutionState.running  # type: ignore[union-attr]

    monkeypatch.setattr(product_store.state_files, "fence_subagents", original_fence)
    product_store.tombstone_session(session.session_id)
    persisted = await child_store.get(record.execution_id)
    assert persisted is not None
    assert persisted.state is SubagentExecutionState.cancelled
    assert persisted.error == "owner session tombstoned"

    await child_store.close()
    product_store.close()
