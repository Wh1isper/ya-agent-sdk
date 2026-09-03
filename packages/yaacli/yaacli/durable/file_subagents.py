"""File-backed portable SDK subagent execution storage for YAACLI."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic_core import to_jsonable_python
from ya_agent_sdk.inputs import InputOrigin
from ya_agent_sdk.subagents import SubagentExecutionIdConflict, SubagentExecutionStore
from ya_agent_sdk.subagents.spec import (
    ResolvedSubagentPlan,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanDescriptor,
)

from yaacli.durable.models import InputPriority, InputRecord, InputState, SessionStatus
from yaacli.durable.sqlite_schema import SESSION_SCHEMA_VERSION
from yaacli.durable.state_files import SessionStateFiles, SubagentStateFile
from yaacli.durable.store import InvalidTransitionError, TombstonedSessionError

_NONTERMINAL_STATES = {
    SubagentExecutionState.pending,
    SubagentExecutionState.running,
    SubagentExecutionState.suspended,
}
_INPUT_TRANSITIONS: dict[InputState, frozenset[InputState]] = {
    InputState.accepted: frozenset({InputState.enqueued, InputState.applied, InputState.rejected}),
    InputState.enqueued: frozenset({InputState.applied, InputState.rejected}),
    InputState.applied: frozenset(),
    InputState.rejected: frozenset(),
}


class FileSubagentExecutionStore(SubagentExecutionStore):
    """One self-contained, grep-friendly state file per child execution."""

    restart_durable = False

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.state_files = SessionStateFiles(self.database_path)
        self.process_owner_token = self.state_files.current_process_owner_token()
        self._active_descriptors: dict[str, SubagentPlanDescriptor] = {}
        self._retained_descriptors: dict[str, SubagentPlanDescriptor] | None = None
        self._validate_product_store()

    def _validate_product_store(self) -> None:
        if not self.database_path.exists():
            raise RuntimeError("Initialize SQLiteSessionStore before the file subagent execution store")
        with sqlite3.connect(self.database_path) as connection:
            session_table = connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'sessions'"
            ).fetchone()
            metadata_table = connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_metadata'"
            ).fetchone()
            marker = (
                connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
                if metadata_table is not None
                else None
            )
        if session_table is None or marker is None or marker[0] != str(SESSION_SCHEMA_VERSION):
            raise RuntimeError("File subagent execution storage requires the current durable product schema")

    def _owner_status(self, owner_scope_id: str) -> SessionStatus:
        with sqlite3.connect(self.database_path, timeout=30) as connection:
            row = connection.execute(
                "SELECT status FROM sessions WHERE session_id = ?",
                (owner_scope_id,),
            ).fetchone()
        if row is None:
            raise KeyError(owner_scope_id)
        return SessionStatus(str(row[0]))

    def _ensure_owner_active(self, owner_scope_id: str) -> None:
        if self._owner_status(owner_scope_id) is not SessionStatus.active:
            raise TombstonedSessionError(f"Owner session {owner_scope_id!r} is tombstoned")

    def _state_for_execution(self, execution_id: str) -> SubagentStateFile:
        state = self.state_files.find_subagent(execution_id)
        if state is None:
            raise KeyError(execution_id)
        return state

    def _writable_state(self, execution_id: str) -> SubagentStateFile:
        state = self._state_for_execution(execution_id)
        if self._owner_status(state.record.owner_scope_id) is not SessionStatus.active or state.cancel_requested:
            raise TombstonedSessionError(
                f"Owner session {state.record.owner_scope_id!r} no longer accepts child execution writes"
            )
        return state

    def recover_orphaned_executions(self) -> tuple[str, ...]:
        """Index retained plans and mark orphaned child work lost in one startup scan."""
        snapshots = self.state_files.list_subagents()
        self._cache_retained_descriptors(snapshots)
        recovered_ids: list[str] = []
        for snapshot in snapshots:
            if snapshot.record.state not in _NONTERMINAL_STATES or self.state_files.process_owner_is_alive(
                snapshot.owner_token
            ):
                continue
            owner_scope_id = snapshot.record.owner_scope_id
            state = self.state_files.read_subagent(
                owner_scope_id,
                snapshot.record.execution_id,
            )
            if state.record.state not in _NONTERMINAL_STATES or self.state_files.process_owner_is_alive(
                state.owner_token
            ):
                continue
            now = datetime.now(UTC)
            reason = "Subagent execution was interrupted by process restart."
            record = state.record.model_copy(
                update={
                    "state": SubagentExecutionState.lost,
                    "input_state": (
                        SubagentInputState.applied
                        if state.record.input_state is SubagentInputState.applied
                        else SubagentInputState.rejected
                    ),
                    "error": reason,
                    "completed_at": now,
                }
            )
            inputs = _reject_pending_inputs(state.inputs, reason=reason, now=now)
            self.state_files.write_subagent(
                state.model_copy(
                    update={
                        "record": record,
                        "inputs": inputs,
                        "input_open": False,
                        "updated_at": now,
                    }
                )
            )
            recovered_ids.append(record.execution_id)
        return tuple(recovered_ids)

    async def close(self) -> None:
        return None

    def close_sync(self) -> None:
        """Close the no-op synchronous adapter boundary."""

    def require_executable(self, execution_id: str) -> SubagentExecutionRecord:
        """Fence model execution after owner tombstone or cancellation intent."""
        self._state_for_execution(execution_id)
        return self._writable_state(execution_id).record.model_copy(deep=True)

    def put_descriptor(self, plan: ResolvedSubagentPlan) -> None:
        descriptor = plan.to_descriptor()
        existing = self._active_descriptors.get(descriptor.descriptor_id)
        if existing is not None and existing != descriptor:
            raise ValueError(f"Subagent descriptor collision for {descriptor.descriptor_id!r}")
        retained = self.get_descriptor(descriptor.descriptor_id)
        if retained is not None and retained != descriptor:
            raise ValueError(f"Subagent descriptor collision for {descriptor.descriptor_id!r}")
        self._active_descriptors[descriptor.descriptor_id] = descriptor

    def _cache_retained_descriptors(
        self,
        states: Sequence[SubagentStateFile],
    ) -> dict[str, SubagentPlanDescriptor]:
        descriptors = dict(self._retained_descriptors or {})
        for state in states:
            descriptor = state.descriptor
            existing = descriptors.get(descriptor.descriptor_id)
            if existing is not None and existing != descriptor:
                raise RuntimeError(f"Conflicting retained subagent descriptor {descriptor.descriptor_id!r}")
            descriptors[descriptor.descriptor_id] = descriptor
        self._retained_descriptors = descriptors
        return descriptors

    def _retained_descriptor_index(self) -> dict[str, SubagentPlanDescriptor]:
        """Index historical descriptors once instead of rescanning large child payloads per plan."""
        if self._retained_descriptors is None:
            self._cache_retained_descriptors(self.state_files.list_subagents())
        return self._retained_descriptors or {}

    def get_descriptor(self, descriptor_id: str) -> SubagentPlanDescriptor | None:
        descriptor = self._active_descriptors.get(descriptor_id)
        if descriptor is None:
            descriptor = self._retained_descriptor_index().get(descriptor_id)
        return descriptor.model_copy(deep=True) if descriptor is not None else None

    def list_referenced_descriptors(self) -> tuple[SubagentPlanDescriptor, ...]:
        descriptors = self._retained_descriptor_index()
        return tuple(descriptors[key].model_copy(deep=True) for key in sorted(descriptors))

    async def create(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
        owner_scope_id = record.owner_scope_id
        self._ensure_owner_active(owner_scope_id)
        for state in self.state_files.list_subagents(owner_scope_id):
            if state.record.idempotency_key == record.idempotency_key:
                return state.record.model_copy(deep=True)
        existing_id = self.state_files.find_subagent(record.execution_id)
        if existing_id is not None:
            raise SubagentExecutionIdConflict(f"Subagent execution {record.execution_id!r} already exists")
        descriptor = self.get_descriptor(record.descriptor_id)
        if descriptor is None:
            raise RuntimeError(f"Subagent execution references missing descriptor {record.descriptor_id!r}")
        if descriptor.fingerprint != record.plan_fingerprint:
            raise ValueError("Subagent execution plan fingerprint does not match its descriptor")
        now = datetime.now(UTC)
        self.state_files.write_subagent(
            SubagentStateFile(
                descriptor=descriptor,
                record=record,
                owner_pid=os.getpid(),
                owner_token=self.process_owner_token,
                input_open=record.state in _NONTERMINAL_STATES,
                created_at=record.created_at,
                updated_at=now,
            )
        )
        if self._retained_descriptors is not None:
            self._retained_descriptors[descriptor.descriptor_id] = descriptor
        return record.model_copy(deep=True)

    async def save(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
        owner_scope_id = record.owner_scope_id
        state = self.state_files.read_subagent(owner_scope_id, record.execution_id)
        current = state.record
        fenced = self._owner_status(owner_scope_id) is not SessionStatus.active or state.cancel_requested
        if current.terminal and (record.state is not current.state or fenced):
            return current.model_copy(deep=True)
        if fenced and record.state is not SubagentExecutionState.cancelled:
            raise TombstonedSessionError(f"Owner session {owner_scope_id!r} fenced a late child commit")
        now = datetime.now(UTC)
        input_open = not fenced and record.state in _NONTERMINAL_STATES
        inputs = state.inputs
        if record.terminal:
            reason = (
                state.cancellation_reason
                if fenced and state.cancellation_reason is not None
                else f"child terminated as {record.state.value} before input application"
            )
            inputs = _reject_pending_inputs(inputs, reason=reason, now=now)
        self.state_files.write_subagent(
            state.model_copy(
                update={
                    "record": record,
                    "inputs": inputs,
                    "input_open": input_open,
                    "updated_at": now,
                }
            )
        )
        return record.model_copy(deep=True)

    async def get(
        self,
        execution_id: str,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord | None:
        state = self.state_files.find_subagent(execution_id)
        if state is None or (owner_scope_id is not None and state.record.owner_scope_id != owner_scope_id):
            return None
        return state.record.model_copy(deep=True)

    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        owner_scope_id: str,
    ) -> SubagentExecutionRecord | None:
        for state in self.state_files.list_subagents(owner_scope_id):
            if state.record.idempotency_key == idempotency_key:
                return state.record.model_copy(deep=True)
        return None

    async def list(
        self,
        *,
        owner_scope_id: str | None = None,
    ) -> tuple[SubagentExecutionRecord, ...]:
        return tuple(state.record.model_copy(deep=True) for state in self.state_files.list_subagents(owner_scope_id))

    async def list_page(
        self,
        *,
        owner_scope_id: str,
        offset: int,
        limit: int,
    ) -> tuple[tuple[SubagentExecutionRecord, ...], int]:
        records = await self.list(owner_scope_id=owner_scope_id)
        return records[offset : offset + limit], len(records)

    async def list_nonterminal_ids(
        self,
        *,
        owner_scope_id: str,
    ) -> tuple[str, ...]:
        return tuple(
            state.record.execution_id for state in self.state_files.list_subagents(owner_scope_id) if state.input_open
        )

    def accept_input(
        self,
        execution_id: str,
        content: Sequence[Any],
        *,
        idempotency_key: str,
        origin: InputOrigin,
    ) -> InputRecord:
        normalized = cast(list[Any], to_jsonable_python(list(content)))
        self._state_for_execution(execution_id)
        state = self._writable_state(execution_id)
        for item in state.inputs:
            if item.idempotency_key != idempotency_key:
                continue
            if item.content != normalized or item.origin != origin.value:
                raise ValueError("Child input idempotency key was reused with different content")
            return item.model_copy(deep=True)
        if not state.input_open or state.record.state not in _NONTERMINAL_STATES:
            raise InvalidTransitionError(f"Subagent execution {execution_id!r} is not accepting input")
        now = datetime.now(UTC)
        record = InputRecord(
            input_id=str(uuid4()),
            logical_run_id=execution_id,
            order_index=max((item.order_index for item in state.inputs), default=-1) + 1,
            idempotency_key=idempotency_key,
            origin=origin.value,
            priority=InputPriority.asap,
            content=normalized,
            state=InputState.accepted,
            created_at=now,
            updated_at=now,
        )
        self.state_files.write_subagent(
            state.model_copy(
                update={
                    "inputs": (*state.inputs, record),
                    "updated_at": now,
                }
            )
        )
        return record.model_copy(deep=True)

    def list_inputs(
        self,
        execution_id: str,
        *,
        states: Sequence[InputState] | None = None,
    ) -> tuple[InputRecord, ...]:
        state = self._state_for_execution(execution_id)
        allowed = None if states is None else set(states)
        records = (item for item in state.inputs if allowed is None or item.state in allowed)
        return tuple(
            item.model_copy(deep=True)
            for item in sorted(
                records,
                key=lambda item: (
                    0 if item.priority is InputPriority.asap else 1,
                    item.order_index,
                ),
            )
        )

    def transition_input(
        self,
        input_id: str,
        expected: InputState,
        target: InputState,
        *,
        native_enqueue_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> InputRecord:
        snapshot, index = self._find_input(input_id)
        state = self._writable_state(snapshot.record.execution_id)
        try:
            index = next(position for position, item in enumerate(state.inputs) if item.input_id == input_id)
        except StopIteration as exc:
            raise KeyError(input_id) from exc
        record = state.inputs[index]
        now = datetime.now(UTC)
        if record.state is target:
            if (
                target is InputState.enqueued
                and native_enqueue_id is not None
                and record.native_enqueue_id != native_enqueue_id
            ):
                record = record.model_copy(
                    update={
                        "native_enqueue_id": native_enqueue_id,
                        "updated_at": now,
                    }
                )
            else:
                return record.model_copy(deep=True)
        else:
            if record.state is not expected or target not in _INPUT_TRANSITIONS[record.state]:
                raise InvalidTransitionError(
                    f"Cannot transition child input from {record.state.value} to {target.value}"
                )
            if target is InputState.rejected and not rejection_reason:
                raise InvalidTransitionError("Rejected child inputs require a reason")
            record = record.model_copy(
                update={
                    "state": target,
                    "native_enqueue_id": native_enqueue_id or record.native_enqueue_id,
                    "rejection_reason": rejection_reason or record.rejection_reason,
                    "updated_at": now,
                }
            )
        inputs = list(state.inputs)
        inputs[index] = record
        self.state_files.write_subagent(state.model_copy(update={"inputs": tuple(inputs), "updated_at": now}))
        return record.model_copy(deep=True)

    def close_and_list_inputs(self, execution_id: str) -> tuple[InputRecord, ...]:
        self._state_for_execution(execution_id)
        state = self._writable_state(execution_id)
        if state.record.terminal or state.record.state is SubagentExecutionState.suspended:
            return ()
        pending = tuple(item for item in state.inputs if item.state in {InputState.accepted, InputState.enqueued})
        now = datetime.now(UTC)
        self.state_files.write_subagent(
            state.model_copy(
                update={
                    "input_open": bool(pending),
                    "updated_at": now,
                }
            )
        )
        return tuple(item.model_copy(deep=True) for item in pending)

    def _find_input(self, input_id: str) -> tuple[SubagentStateFile, int]:
        for state in self.state_files.list_subagents():
            for index, item in enumerate(state.inputs):
                if item.input_id == input_id:
                    return state, index
        raise KeyError(input_id)


def _reject_pending_inputs(
    inputs: tuple[InputRecord, ...],
    *,
    reason: str,
    now: datetime,
) -> tuple[InputRecord, ...]:
    return tuple(
        item.model_copy(
            update={
                "state": InputState.rejected,
                "rejection_reason": reason,
                "updated_at": now,
            }
        )
        if item.state in {InputState.accepted, InputState.enqueued}
        else item
        for item in inputs
    )
