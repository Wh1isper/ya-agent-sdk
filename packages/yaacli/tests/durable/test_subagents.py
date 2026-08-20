from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaacli.durable.sqlite as durable_sqlite
from pydantic import TypeAdapter
from pydantic_ai import Agent, AgentSpec, DeferredToolRequests, DeferredToolResults, Tool, ToolDenied
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.capabilities import SupportsDeferredOutput, build_default_capability_catalog
from ya_agent_sdk.inputs import InputOrigin, RunInputLedger
from ya_agent_sdk.subagents import (
    SubagentDeliveryState,
    SubagentDurability,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionService,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)
from yaacli.durable.models import InputState
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.durable.store import TombstonedSessionError
from yaacli.durable.subagents import (
    DurableSubagentInboxCapability,
    LocalProcessorSubagentExecutionHost,
    LocalSubagentDriver,
    SQLiteSubagentExecutionStore,
)
from yaacli.session import TUIContext


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


def test_subagent_store_rejects_partial_schema_before_running_ddl(tmp_path: Path) -> None:
    database_path = tmp_path / "partial-child.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.close()
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX subagent_executions_scope_idx")

    with pytest.raises(RuntimeError, match="missing index:subagent_executions_scope_idx"):
        SQLiteSubagentExecutionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        index = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'index' AND name = 'subagent_executions_scope_idx'"
        ).fetchone()
    assert index is None


def test_subagent_store_rejects_same_columns_without_constraints(tmp_path: Path) -> None:
    database_path = tmp_path / "malformed-child.sqlite3"
    malformed_schema = durable_sqlite._SCHEMA.replace(
        "input_open INTEGER NOT NULL CHECK(input_open IN (0, 1))",
        "input_open INTEGER NOT NULL",
        1,
    )
    assert malformed_schema != durable_sqlite._SCHEMA
    with sqlite3.connect(database_path) as connection:
        connection.executescript(malformed_schema)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    original_bytes = database_path.read_bytes()

    with pytest.raises(RuntimeError, match="definition mismatch for table:subagent_executions"):
        SQLiteSubagentExecutionStore(database_path)

    assert database_path.read_bytes() == original_bytes
    assert not database_path.with_name(f"{database_path.name}-wal").exists()
    assert not database_path.with_name(f"{database_path.name}-shm").exists()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_subagent_store_rejects_pre_v2_unscoped_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE subagent_executions (
                execution_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    with pytest.raises(
        RuntimeError,
        match="exact owner-scoped execution and durable child-inbox schema",
    ):
        SQLiteSubagentExecutionStore(database_path)


async def test_local_subagent_driver_persists_deferred_segments_and_cumulative_usage(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "subagents.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = SQLiteSubagentExecutionStore(database_path)
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
    parent_ctx = TUIContext(run_input_ledger=RunInputLedger(logical_run_id="parent-run"))
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
    finally:
        await service.close()
        product_store.close()


async def test_local_subagent_steering_is_persisted_before_graph_application(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "steering.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = SQLiteSubagentExecutionStore(database_path)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id="descriptor",
        plan_fingerprint="fingerprint",
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
    await store.close()
    product_store.close()


async def test_session_tombstone_atomically_fences_and_requests_child_cancellation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tombstone-subagents.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    session = product_store.create_session(str(tmp_path), session_id="session")
    child_store = SQLiteSubagentExecutionStore(database_path)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id=session.session_id,
        idempotency_key="child-running",
        descriptor_id="worker:descriptor",
        plan_fingerprint="descriptor",
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

    with sqlite3.connect(database_path) as connection:
        execution_row = connection.execute(
            "SELECT input_open, cancel_requested, cancellation_reason FROM subagent_executions WHERE execution_id = ?",
            (record.execution_id,),
        ).fetchone()
        command_row = connection.execute(
            "SELECT command_kind, aggregate_id, payload_json, state FROM outbox_commands WHERE command_id = ?",
            (f"cancel-subagent:{record.execution_id}:0",),
        ).fetchone()
    assert execution_row == (0, 1, "owner session tombstoned")
    assert command_row is not None
    assert command_row[0:2] == ("cancel_subagent_execution", record.execution_id)
    assert json.loads(command_row[2]) == {
        "execution_id": record.execution_id,
        "owner_scope_id": session.session_id,
        "reason": "owner session tombstoned",
    }
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
    with pytest.raises(TombstonedSessionError):
        await child_store.save(
            record.model_copy(
                update={
                    "state": SubagentExecutionState.succeeded,
                    "input_state": SubagentInputState.applied,
                    "output": "must be fenced",
                }
            )
        )

    cancelled = record.model_copy(
        update={
            "state": SubagentExecutionState.cancelled,
            "input_state": SubagentInputState.rejected,
            "error": "Subagent execution was cancelled",
            "completed_at": datetime.now(UTC),
        }
    )
    await child_store.save(cancelled)
    persisted = await child_store.get(record.execution_id)
    assert persisted is not None
    assert persisted.state is SubagentExecutionState.cancelled

    await child_store.close()
    product_store.close()


async def test_subagent_store_marks_process_orphans_lost_on_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "orphan.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = SQLiteSubagentExecutionStore(database_path)
    record = SubagentExecutionRecord(
        execution_id="child-orphan",
        root_execution_id="child-orphan",
        owner_scope_id="session",
        idempotency_key="child-orphan",
        descriptor_id="descriptor",
        plan_fingerprint="fingerprint",
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
    await store.close()

    reopened = SQLiteSubagentExecutionStore(database_path)
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
    owner = SQLiteSubagentExecutionStore(database_path)
    record = SubagentExecutionRecord(
        execution_id="child-live",
        root_execution_id="child-live",
        owner_scope_id="session",
        idempotency_key="child-live",
        descriptor_id="descriptor",
        plan_fingerprint="fingerprint",
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await owner.create(record)

    reader = SQLiteSubagentExecutionStore(database_path)
    try:
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
    store = SQLiteSubagentExecutionStore(database_path)
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
    await store.close()
    product_store.close()


async def test_subagent_store_fails_when_referenced_descriptor_is_missing(tmp_path: Path) -> None:
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
    store = SQLiteSubagentExecutionStore(database_path)
    store.put_descriptor(plan)
    await store.create(
        SubagentExecutionRecord(
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
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "DELETE FROM subagent_plan_descriptors WHERE descriptor_id = ?",
            (plan.descriptor_id,),
        )

    with pytest.raises(RuntimeError, match="missing descriptor"):
        store.list_referenced_descriptors()
    await store.close()
    product_store.close()


async def test_child_inbox_applies_idempotent_steering_at_graph_boundary(tmp_path: Path) -> None:
    database_path = tmp_path / "child-inbox.sqlite3"
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="session")
    store = SQLiteSubagentExecutionStore(database_path)
    record = SubagentExecutionRecord(
        execution_id="child-running",
        root_execution_id="child-running",
        owner_scope_id="session",
        idempotency_key="child-running",
        descriptor_id="descriptor",
        plan_fingerprint="fingerprint",
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
