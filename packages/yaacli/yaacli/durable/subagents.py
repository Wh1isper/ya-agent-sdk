"""YAACLI SQLite adapters for process-local portable SDK subagents."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    EnqueuedMessagesEvent,
    RunContext,
    UserContent,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_core import to_jsonable_python
from pydantic_graph import End
from ya_agent_sdk.context import (
    AgentContext,
    ModelConfig,
)
from ya_agent_sdk.inputs import (
    EnqueueReceipt,
    InputDisposition,
    InputOrigin,
    LogicalRunInputRouter,
)
from ya_agent_sdk.subagents import (
    AsyncioSubagentExecutionHost,
    InProcessSubagentDriver,
    SubagentExecutionIdConflict,
    SubagentExecutionStore,
    SubagentPlanResolver,
)
from ya_agent_sdk.subagents.spec import (
    ResolvedSubagentPlan,
    SubagentDriverOutcome,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanDescriptor,
)

from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.models import (
    InputPriority,
    InputRecord,
    InputState,
    SessionStatus,
)
from yaacli.durable.sqlite_schema import SUBAGENT_SCHEMA, validate_exact_schema_subset
from yaacli.durable.store import (
    InvalidTransitionError,
    TombstonedSessionError,
)
from yaacli.session import TUIContext
from yaacli.subagent_config import model_cfg_from_agent_spec

_DEFERRED_RESULTS = TypeAdapter(DeferredToolResults)
_USER_CONTENT = TypeAdapter(list[UserContent])


@dataclass(frozen=True, slots=True)
class SQLiteRetainedSubagentPlanProvider:
    """Lazily restore an exact historical child plan from product storage."""

    store: SQLiteSubagentExecutionStore
    resolver: SubagentPlanResolver

    async def load_retained_plan(
        self,
        record: SubagentExecutionRecord,
    ) -> ResolvedSubagentPlan | None:
        descriptor = self.store.get_descriptor(record.descriptor_id)
        if descriptor is None:
            return None
        return self.resolver.restore(descriptor)


class SQLiteSubagentExecutionStore(SubagentExecutionStore):
    """Portable child records stored in the YAACLI product database."""

    restart_durable = False

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize_or_validate_schema()
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            self._connection.close()
            raise

    def _initialize_or_validate_schema(self) -> None:
        error_prefix = (
            "Unsupported subagent execution schema; YAACLI 2.0 requires the exact "
            "owner-scoped execution and durable child-inbox schema"
        )
        session_table = self._connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        if session_table is None:
            raise RuntimeError(
                f"{error_prefix}; missing shared durable product schema. "
                "Initialize SQLiteSessionStore first; standalone child databases are not supported."
            )
        validate_exact_schema_subset(
            self._connection,
            SUBAGENT_SCHEMA,
            error_prefix=error_prefix,
        )

    def recover_orphaned_executions(self) -> tuple[str, ...]:
        """Mark process-owned child work lost at an explicit host startup boundary."""
        rows = self._connection.execute("SELECT execution_id, record_json FROM subagent_executions").fetchall()
        now = datetime.now(UTC)
        recovered_ids: list[str] = []
        for row in rows:
            record = SubagentExecutionRecord.model_validate_json(row["record_json"])
            if record.state not in {
                SubagentExecutionState.pending,
                SubagentExecutionState.running,
                SubagentExecutionState.suspended,
            }:
                continue
            recovered = record.model_copy(
                update={
                    "state": SubagentExecutionState.lost,
                    "input_state": (
                        SubagentInputState.applied
                        if record.input_state is SubagentInputState.applied
                        else SubagentInputState.rejected
                    ),
                    "error": "Subagent execution was interrupted by process restart.",
                    "completed_at": now,
                }
            )
            self._connection.execute(
                """
                UPDATE subagent_executions
                SET record_json = ?, input_open = 0, updated_at = ?
                WHERE execution_id = ?
                """,
                (recovered.model_dump_json(), now.isoformat(), recovered.execution_id),
            )
            recovered_ids.append(recovered.execution_id)
            self._connection.execute(
                """
                UPDATE subagent_inputs
                SET state = ?, rejection_reason = ?, updated_at = ?
                WHERE execution_id = ? AND state IN (?, ?)
                """,
                (
                    InputState.rejected.value,
                    "Subagent execution was interrupted by process restart.",
                    now.isoformat(),
                    recovered.execution_id,
                    InputState.accepted.value,
                    InputState.enqueued.value,
                ),
            )
        return tuple(recovered_ids)

    async def close(self) -> None:
        self.close_sync()

    def close_sync(self) -> None:
        """Close the shared-product synchronous SQLite adapter."""
        with self._lock:
            self._connection.close()

    @staticmethod
    def _ensure_owner_active(
        connection: sqlite3.Connection,
        owner_scope_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT status FROM sessions WHERE session_id = ?",
            (owner_scope_id,),
        ).fetchone()
        if row is None:
            raise KeyError(owner_scope_id)
        if SessionStatus(row["status"]) is not SessionStatus.active:
            raise TombstonedSessionError(f"Owner session {owner_scope_id!r} is tombstoned")

    @classmethod
    def _required_writable_execution(
        cls,
        connection: sqlite3.Connection,
        execution_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT e.*, s.status AS owner_status
            FROM subagent_executions AS e
            JOIN sessions AS s ON s.session_id = e.owner_scope_id
            WHERE e.execution_id = ?
            """,
            (execution_id,),
        ).fetchone()
        if row is None:
            raise KeyError(execution_id)
        if SessionStatus(row["owner_status"]) is not SessionStatus.active or bool(row["cancel_requested"]):
            raise TombstonedSessionError(
                f"Owner session {row['owner_scope_id']!r} no longer accepts child execution writes"
            )
        return row

    def require_executable(self, execution_id: str) -> SubagentExecutionRecord:
        """Fence model execution after owner tombstone or cancellation intent."""
        with self._lock:
            row = self._required_writable_execution(
                self._connection,
                execution_id,
            )
        return SubagentExecutionRecord.model_validate_json(row["record_json"])

    def put_descriptor(self, plan: ResolvedSubagentPlan) -> None:
        descriptor = plan.to_descriptor()
        payload = descriptor.model_dump_json()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT fingerprint, descriptor_json FROM subagent_plan_descriptors WHERE descriptor_id = ?",
                    (descriptor.descriptor_id,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO subagent_plan_descriptors "
                        "(descriptor_id, fingerprint, descriptor_json, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (descriptor.descriptor_id, descriptor.fingerprint, payload, now),
                    )
                elif row["fingerprint"] != descriptor.fingerprint or json.loads(row["descriptor_json"]) != json.loads(
                    payload
                ):
                    raise ValueError(f"Subagent descriptor collision for {descriptor.descriptor_id!r}")
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def get_descriptor(self, descriptor_id: str) -> SubagentPlanDescriptor | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT descriptor_json FROM subagent_plan_descriptors WHERE descriptor_id = ?",
                (descriptor_id,),
            ).fetchone()
        return SubagentPlanDescriptor.model_validate_json(row["descriptor_json"]) if row is not None else None

    def list_referenced_descriptors(self) -> tuple[SubagentPlanDescriptor, ...]:
        """Return every exact plan still referenced by an execution record."""
        with self._lock:
            record_rows = self._connection.execute(
                "SELECT record_json FROM subagent_executions ORDER BY created_at, execution_id"
            ).fetchall()
            descriptor_ids = sorted({
                SubagentExecutionRecord.model_validate_json(row["record_json"]).descriptor_id for row in record_rows
            })
            descriptors: list[SubagentPlanDescriptor] = []
            for descriptor_id in descriptor_ids:
                row = self._connection.execute(
                    "SELECT descriptor_json FROM subagent_plan_descriptors WHERE descriptor_id = ?",
                    (descriptor_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Subagent execution references missing descriptor {descriptor_id!r}")
                descriptors.append(SubagentPlanDescriptor.model_validate_json(row["descriptor_json"]))
        return tuple(descriptors)

    async def create(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
        now = datetime.now(UTC).isoformat()
        payload = record.model_dump_json()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_owner_active(
                    self._connection,
                    record.owner_scope_id,
                )
                existing = self._connection.execute(
                    "SELECT record_json FROM subagent_executions WHERE owner_scope_id = ? AND idempotency_key = ?",
                    (record.owner_scope_id, record.idempotency_key),
                ).fetchone()
                if existing is not None:
                    self._connection.execute("COMMIT")
                    return SubagentExecutionRecord.model_validate_json(existing["record_json"])
                self._connection.execute(
                    "INSERT INTO subagent_executions "
                    "(execution_id, owner_scope_id, idempotency_key, record_json, input_open, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.execution_id,
                        record.owner_scope_id,
                        record.idempotency_key,
                        payload,
                        int(
                            record.state
                            in {
                                SubagentExecutionState.pending,
                                SubagentExecutionState.running,
                                SubagentExecutionState.suspended,
                            }
                        ),
                        now,
                        now,
                    ),
                )
                self._connection.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._connection.execute("ROLLBACK")
                collision = self._connection.execute(
                    "SELECT 1 FROM subagent_executions WHERE execution_id = ?",
                    (record.execution_id,),
                ).fetchone()
                if collision is not None:
                    raise SubagentExecutionIdConflict(
                        f"Subagent execution {record.execution_id!r} already exists"
                    ) from exc
                raise
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return record.model_copy(deep=True)

    async def save(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT e.*, s.status AS owner_status
                    FROM subagent_executions AS e
                    JOIN sessions AS s ON s.session_id = e.owner_scope_id
                    WHERE e.execution_id = ? AND e.owner_scope_id = ?
                    """,
                    (record.execution_id, record.owner_scope_id),
                ).fetchone()
                if row is None:
                    raise KeyError(record.execution_id)
                current = SubagentExecutionRecord.model_validate_json(row["record_json"])
                if current.terminal and record.state is not current.state:
                    self._connection.execute("COMMIT")
                    return current.model_copy(deep=True)
                fenced = SessionStatus(row["owner_status"]) is not SessionStatus.active or bool(row["cancel_requested"])
                if fenced and record.state is not SubagentExecutionState.cancelled:
                    raise TombstonedSessionError(f"Owner session {record.owner_scope_id!r} fenced a late child commit")
                input_open = int(
                    not fenced
                    and record.state
                    in {
                        SubagentExecutionState.pending,
                        SubagentExecutionState.running,
                        SubagentExecutionState.suspended,
                    }
                )
                self._connection.execute(
                    "UPDATE subagent_executions "
                    "SET record_json = ?, input_open = ?, updated_at = ? "
                    "WHERE execution_id = ? AND owner_scope_id = ?",
                    (
                        record.model_dump_json(),
                        input_open,
                        now,
                        record.execution_id,
                        record.owner_scope_id,
                    ),
                )
                if record.terminal:
                    rejection_reason = (
                        row["cancellation_reason"]
                        if fenced and row["cancellation_reason"] is not None
                        else f"child terminated as {record.state.value} before input application"
                    )
                    self._connection.execute(
                        "UPDATE subagent_inputs SET state = ?, rejection_reason = ?, updated_at = ? "
                        "WHERE execution_id = ? AND state IN (?, ?)",
                        (
                            InputState.rejected.value,
                            rejection_reason,
                            now,
                            record.execution_id,
                            InputState.accepted.value,
                            InputState.enqueued.value,
                        ),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return record.model_copy(deep=True)

    async def get(
        self,
        execution_id: str,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord | None:
        query = "SELECT record_json FROM subagent_executions WHERE execution_id = ?"
        params: tuple[str, ...] = (execution_id,)
        if owner_scope_id is not None:
            query += " AND owner_scope_id = ?"
            params = (execution_id, owner_scope_id)
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return self._record(row)

    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        owner_scope_id: str,
    ) -> SubagentExecutionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM subagent_executions WHERE owner_scope_id = ? AND idempotency_key = ?",
                (owner_scope_id, idempotency_key),
            ).fetchone()
        return self._record(row)

    async def list(
        self,
        *,
        owner_scope_id: str | None = None,
    ) -> tuple[SubagentExecutionRecord, ...]:
        query = "SELECT record_json FROM subagent_executions"
        params: tuple[str, ...] = ()
        if owner_scope_id is not None:
            query += " WHERE owner_scope_id = ?"
            params = (owner_scope_id,)
        query += " ORDER BY created_at, execution_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(SubagentExecutionRecord.model_validate_json(row["record_json"]) for row in rows)

    def accept_input(
        self,
        execution_id: str,
        content: Sequence[Any],
        *,
        idempotency_key: str,
        origin: InputOrigin,
    ) -> InputRecord:
        normalized = cast(list[Any], to_jsonable_python(list(content)))
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                execution_row = self._required_writable_execution(
                    self._connection,
                    execution_id,
                )
                existing = self._connection.execute(
                    "SELECT * FROM subagent_inputs WHERE execution_id = ? AND idempotency_key = ?",
                    (execution_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    record = self._input(existing)
                    if record.content != normalized or record.origin != origin.value:
                        raise ValueError("Child input idempotency key was reused with different content")
                    self._connection.execute("COMMIT")
                    return record
                execution = SubagentExecutionRecord.model_validate_json(execution_row["record_json"])
                if not bool(execution_row["input_open"]) or execution.state not in {
                    SubagentExecutionState.pending,
                    SubagentExecutionState.running,
                    SubagentExecutionState.suspended,
                }:
                    raise InvalidTransitionError(f"Subagent execution {execution_id!r} is not accepting input")
                order_index = cast(
                    int,
                    self._connection.execute(
                        "SELECT COALESCE(MAX(order_index), -1) + 1 AS value "
                        "FROM subagent_inputs WHERE execution_id = ?",
                        (execution_id,),
                    ).fetchone()["value"],
                )
                input_id = str(uuid4())
                self._connection.execute(
                    "INSERT INTO subagent_inputs "
                    "(input_id, execution_id, order_index, idempotency_key, origin, priority, "
                    "content_json, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        input_id,
                        execution_id,
                        order_index,
                        idempotency_key,
                        origin.value,
                        InputPriority.asap.value,
                        json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                        InputState.accepted.value,
                        now,
                        now,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM subagent_inputs WHERE input_id = ?",
                    (input_id,),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        if row is None:  # pragma: no cover - SQLite insert invariant
            raise RuntimeError("Accepted child input disappeared")
        return self._input(row)

    def list_inputs(
        self,
        execution_id: str,
        *,
        states: Sequence[InputState] | None = None,
    ) -> tuple[InputRecord, ...]:
        query = "SELECT * FROM subagent_inputs WHERE execution_id = ?"
        params: list[Any] = [execution_id]
        if states is not None:
            if not states:
                return ()
            query += f" AND state IN ({','.join('?' for _ in states)})"
            params.extend(state.value for state in states)
        query += " ORDER BY CASE priority WHEN 'asap' THEN 0 ELSE 1 END, order_index"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(self._input(row) for row in rows)

    def transition_input(
        self,
        input_id: str,
        expected: InputState,
        target: InputState,
        *,
        native_enqueue_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> InputRecord:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM subagent_inputs WHERE input_id = ?",
                    (input_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(input_id)
                self._required_writable_execution(
                    self._connection,
                    row["execution_id"],
                )
                record = self._input(row)
                if record.state is target:
                    if (
                        target is InputState.enqueued
                        and native_enqueue_id is not None
                        and record.native_enqueue_id != native_enqueue_id
                    ):
                        self._connection.execute(
                            "UPDATE subagent_inputs SET native_enqueue_id = ?, updated_at = ? WHERE input_id = ?",
                            (native_enqueue_id, now, input_id),
                        )
                        refreshed = self._connection.execute(
                            "SELECT * FROM subagent_inputs WHERE input_id = ?",
                            (input_id,),
                        ).fetchone()
                        self._connection.execute("COMMIT")
                        if refreshed is None:  # pragma: no cover - SQLite update invariant
                            raise RuntimeError("Updated child input disappeared")
                        return self._input(refreshed)
                    self._connection.execute("COMMIT")
                    return record
                allowed = {
                    InputState.accepted: {InputState.enqueued, InputState.applied, InputState.rejected},
                    InputState.enqueued: {InputState.applied, InputState.rejected},
                    InputState.applied: set(),
                    InputState.rejected: set(),
                }
                if record.state is not expected or target not in allowed[record.state]:
                    raise InvalidTransitionError(
                        f"Cannot transition child input from {record.state.value} to {target.value}"
                    )
                if target is InputState.rejected and not rejection_reason:
                    raise InvalidTransitionError("Rejected child inputs require a reason")
                self._connection.execute(
                    "UPDATE subagent_inputs SET state = ?, native_enqueue_id = COALESCE(?, native_enqueue_id), "
                    "rejection_reason = COALESCE(?, rejection_reason), updated_at = ? WHERE input_id = ?",
                    (target.value, native_enqueue_id, rejection_reason, now, input_id),
                )
                updated = self._connection.execute(
                    "SELECT * FROM subagent_inputs WHERE input_id = ?",
                    (input_id,),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        if updated is None:  # pragma: no cover - SQLite update invariant
            raise RuntimeError("Transitioned child input disappeared")
        return self._input(updated)

    def close_and_list_inputs(self, execution_id: str) -> tuple[InputRecord, ...]:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._required_writable_execution(
                    self._connection,
                    execution_id,
                )
                execution = SubagentExecutionRecord.model_validate_json(row["record_json"])
                if execution.terminal or execution.state is SubagentExecutionState.suspended:
                    self._connection.execute("COMMIT")
                    return ()
                self._connection.execute(
                    "UPDATE subagent_executions SET input_open = 0, updated_at = ? WHERE execution_id = ?",
                    (now, execution_id),
                )
                rows = self._connection.execute(
                    "SELECT * FROM subagent_inputs WHERE execution_id = ? AND state IN (?, ?) "
                    "ORDER BY CASE priority WHEN 'asap' THEN 0 ELSE 1 END, order_index",
                    (execution_id, InputState.accepted.value, InputState.enqueued.value),
                ).fetchall()
                if rows:
                    self._connection.execute(
                        "UPDATE subagent_executions SET input_open = 1 WHERE execution_id = ?",
                        (execution_id,),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return tuple(self._input(item) for item in rows)

    @staticmethod
    def _record(row: sqlite3.Row | None) -> SubagentExecutionRecord | None:
        return SubagentExecutionRecord.model_validate_json(row["record_json"]) if row is not None else None

    @staticmethod
    def _input(row: sqlite3.Row) -> InputRecord:
        return InputRecord(
            input_id=row["input_id"],
            logical_run_id=row["execution_id"],
            order_index=row["order_index"],
            idempotency_key=row["idempotency_key"],
            origin=row["origin"],
            priority=InputPriority(row["priority"]),
            content=json.loads(row["content_json"]),
            state=InputState(row["state"]),
            native_enqueue_id=row["native_enqueue_id"],
            rejection_reason=row["rejection_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class DurableSubagentCompletionDelivery:
    """Persist child completion into the active compatible session run."""

    def __init__(self, binding_ref: str) -> None:
        self.binding_ref = binding_ref

    async def deliver(
        self,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        message: str,
    ) -> EnqueueReceipt | None:
        if not isinstance(parent_ctx, TUIContext):
            return None
        parent_logical_run_id = record.parent_logical_run_id
        if parent_logical_run_id is None:
            return None
        store = runtime_bindings.get(self.binding_ref).store
        source_run = store.get_run(parent_logical_run_id)
        if source_run is None:
            return None

        if record.delivery_logical_run_id is not None and record.delivery_input_id is not None:
            target_run = store.get_run(record.delivery_logical_run_id)
            if target_run is None or target_run.session_id != source_run.session_id:
                return EnqueueReceipt(
                    logical_run_id=record.delivery_logical_run_id,
                    input_id=record.delivery_input_id,
                    disposition=InputDisposition.rejected,
                )
            existing = next(
                (
                    item
                    for item in store.list_inputs(record.delivery_logical_run_id)
                    if item.input_id == record.delivery_input_id
                ),
                None,
            )
            if existing is None:
                return EnqueueReceipt(
                    logical_run_id=record.delivery_logical_run_id,
                    input_id=record.delivery_input_id,
                    disposition=InputDisposition.rejected,
                )
            return EnqueueReceipt(
                logical_run_id=existing.logical_run_id,
                input_id=existing.input_id,
                disposition=InputDisposition(existing.state.value),
                enqueue_id=existing.native_enqueue_id,
            )

        current_logical_run_id = parent_ctx.durable_logical_run_id
        if current_logical_run_id is None:
            return None
        current_run = store.get_run(current_logical_run_id)
        if current_run is None or current_run.session_id != source_run.session_id:
            return None
        try:
            accepted = store.accept_input(
                current_logical_run_id,
                [message],
                idempotency_key=f"subagent-completion:{record.execution_id}",
                priority=InputPriority.asap,
                origin="feature",
            )
        except (InvalidTransitionError, TombstonedSessionError):
            return None
        return EnqueueReceipt(
            logical_run_id=accepted.logical_run_id,
            input_id=accepted.input_id,
            disposition=InputDisposition(accepted.state.value),
            enqueue_id=accepted.native_enqueue_id,
        )


@dataclass(kw_only=True)
class DurableSubagentInboxCapability(AbstractCapability[TUIContext]):
    """Apply persisted process-local child steering at native graph boundaries."""

    store: SQLiteSubagentExecutionStore
    id: str | None = "yaacli_durable_subagent_inbox_v2"

    _safe_at_runtime: ClassVar[bool] = False

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    async def after_node_run(
        self,
        ctx: RunContext[TUIContext],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        del node
        execution_id = ctx.deps.agent_id
        self._sync_applied_inputs(ctx.deps)
        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            return result
        pending = (
            self.store.close_and_list_inputs(execution_id)
            if isinstance(result, End)
            else self.store.list_inputs(
                execution_id,
                states=(InputState.accepted, InputState.enqueued),
            )
        )
        self._enqueue_pending(ctx, pending)
        return result

    def _enqueue_pending(
        self,
        ctx: RunContext[TUIContext],
        pending: tuple[InputRecord, ...],
    ) -> None:
        for item in pending:
            content = _USER_CONTENT.validate_python(item.content)
            prompt_content: str | list[UserContent]
            prompt_content = content[0] if len(content) == 1 and isinstance(content[0], str) else content
            ledger_record = ctx.deps.run_input_ledger.accept(
                [ModelRequest(parts=[UserPromptPart(content=prompt_content)])],
                origin=(InputOrigin.user if item.origin == "user" else InputOrigin.feature),
                priority=item.priority.value,
                input_id=item.input_id,
            )
            if ledger_record.disposition is InputDisposition.applied:
                self.store.transition_input(item.input_id, item.state, InputState.applied)
                continue
            if ledger_record.disposition is InputDisposition.rejected:
                continue
            native_attempt_id = self._current_native_attempt_id(ctx)
            if native_attempt_id is None:
                continue
            current_attempt = next(
                (
                    attempt
                    for attempt in ledger_record.enqueue_attempts
                    if attempt.native_attempt_id == native_attempt_id
                ),
                None,
            )
            if current_attempt is not None:
                self.store.transition_input(
                    item.input_id,
                    item.state,
                    InputState.enqueued,
                    native_enqueue_id=current_attempt.enqueue_id,
                )
                continue
            enqueue_id = ctx.enqueue(*content, priority=item.priority.value)
            if enqueue_id is None:  # pragma: no cover - non-empty store invariant
                continue
            ctx.deps.run_input_ledger.mark_enqueued(
                ledger_record.input_id,
                native_attempt_id=native_attempt_id,
                enqueue_id=enqueue_id,
            )
            self.store.transition_input(
                item.input_id,
                item.state,
                InputState.enqueued,
                native_enqueue_id=enqueue_id,
            )

    @staticmethod
    def _current_native_attempt_id(ctx: RunContext[TUIContext]) -> str | None:
        router = ctx.deps.input_router
        if isinstance(router, LogicalRunInputRouter):
            return router.current_native_attempt_id
        return ctx.run_id or ctx.deps.run_id

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[TUIContext],
        *,
        stream: AsyncIterable[Any],
    ) -> AsyncIterator[Any]:
        async for event in stream:
            if isinstance(event, EnqueuedMessagesEvent):
                self._mark_applied(ctx.deps, event.enqueue_id)
            yield event

    def _mark_applied(self, deps: TUIContext, enqueue_id: str) -> None:
        deps.run_input_ledger.mark_applied_by_enqueue_id(enqueue_id)
        self._sync_applied_inputs(deps)

    def _sync_applied_inputs(self, deps: TUIContext) -> None:
        for item in self.store.list_inputs(
            deps.agent_id,
            states=(InputState.accepted, InputState.enqueued),
        ):
            ledger_record = deps.run_input_ledger.find(item.input_id)
            if ledger_record is not None and ledger_record.disposition is InputDisposition.applied:
                self.store.transition_input(
                    item.input_id,
                    item.state,
                    InputState.applied,
                )


class LocalProcessorSubagentExecutionHost(AsyncioSubagentExecutionHost):
    """Own fully asynchronous child tasks at the YAACLI processor boundary."""


class LocalSubagentDriver:
    """Persist host steering while composing SDK in-process child execution."""

    restart_durable = False

    def __init__(
        self,
        *,
        store: SQLiteSubagentExecutionStore,
        request_limit: int,
        default_model_cfg: ModelConfig,
        custom_capability_types: Sequence[type[Any]] = (),
        runtime_capabilities: Sequence[AbstractCapability[Any]] = (),
    ) -> None:
        self.store = store
        self.default_model_cfg = default_model_cfg.model_copy(deep=True)
        self.runtime_capabilities = tuple(runtime_capabilities)
        self._driver = InProcessSubagentDriver(
            custom_capability_types=custom_capability_types,
            request_limit=request_limit,
            child_context_configurer=self._configure_child_context,
        )

    def _configure_child_context(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        child_ctx: AgentContext,
    ) -> None:
        del parent_ctx
        if not isinstance(child_ctx, TUIContext):
            raise TypeError("YAACLI subagents require TUIContext")
        child_model_cfg = model_cfg_from_agent_spec(plan.normalized_agent_spec)
        child_ctx.model_cfg = (
            child_model_cfg if child_model_cfg is not None else self.default_model_cfg.model_copy(deep=True)
        )
        child_ctx.parent_run_id = record.parent_logical_run_id
        child_ctx.provider_session_id = record.execution_id
        child_ctx.provider_thread_id = record.root_execution_id
        child_ctx.runtime_descriptor_id = record.parent_runtime_descriptor_id
        child_ctx.durable_logical_run_id = record.child_logical_run_id
        child_ctx.model_profile_instructions = None
        child_ctx.shell_env = {}
        child_ctx.files_to_inspect = []
        child_ctx.goal_task = None
        child_ctx.goal_iteration = 0
        child_ctx.goal_needs_post_restore_audit = False
        child_ctx.goal_last_context_handoff_source = None

    async def run(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> SubagentDriverOutcome:
        self.store.put_descriptor(plan)
        self.store.require_executable(record.execution_id)
        execution_plan = replace(
            plan,
            host_capabilities=(*plan.host_capabilities, *self.runtime_capabilities),
        )
        return await self._driver.run(execution_plan, record, parent_ctx)

    async def steer(
        self,
        record: SubagentExecutionRecord,
        *content: Any,
        origin: InputOrigin,
        idempotency_key: str | None,
    ) -> EnqueueReceipt:
        input_key = idempotency_key or str(uuid4())
        try:
            accepted = self.store.accept_input(
                record.execution_id,
                content,
                idempotency_key=input_key,
                origin=origin,
            )
        except (InvalidTransitionError, TombstonedSessionError):
            return EnqueueReceipt(
                logical_run_id=record.child_logical_run_id,
                input_id=input_key,
                disposition=InputDisposition.rejected,
            )
        return EnqueueReceipt(
            logical_run_id=record.child_logical_run_id,
            input_id=accepted.input_id,
            disposition=InputDisposition(accepted.state.value),
            enqueue_id=accepted.native_enqueue_id,
        )

    async def cancel(self, record: SubagentExecutionRecord) -> None:
        await self._driver.cancel(record)
