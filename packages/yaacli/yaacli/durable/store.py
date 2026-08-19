"""Product persistence contract for durable YAACLI sessions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from yaacli.durable.models import (
    ActionBatch,
    EventRecord,
    ExecutionCheckpointRecord,
    ExecutionRecord,
    InputPriority,
    InputRecord,
    InputState,
    LogicalRunRecord,
    LogicalRunStatus,
    OutboxCommand,
    RevisionPayload,
    RevisionRecord,
    RuntimeDescriptor,
    SessionRecord,
    StartRunRequest,
)


class HeadConflictError(RuntimeError):
    """The session head changed after the caller read it."""


class TombstonedSessionError(RuntimeError):
    """A late operation targeted a tombstoned session."""


class InvalidTransitionError(RuntimeError):
    """A persisted state transition violated the durable state machine."""


@runtime_checkable
class SessionStore(Protocol):
    """Typed product truth independent from a workflow engine."""

    def close(self) -> None: ...

    def create_session(self, workspace_ref: str, *, session_id: str | None = None) -> SessionRecord: ...

    def get_session(self, session_id: str) -> SessionRecord | None: ...

    def list_sessions(self, *, limit: int = 100) -> tuple[SessionRecord, ...]: ...

    def tombstone_session(self, session_id: str) -> SessionRecord: ...

    def put_descriptor(self, descriptor: RuntimeDescriptor) -> RuntimeDescriptor: ...

    def get_descriptor(self, descriptor_id: str) -> RuntimeDescriptor | None: ...

    def list_nonterminal_descriptors(self) -> tuple[RuntimeDescriptor, ...]: ...

    def start_run(self, request: StartRunRequest) -> LogicalRunRecord: ...

    def get_run(self, logical_run_id: str) -> LogicalRunRecord | None: ...

    def get_execution(self, execution_id: str) -> ExecutionRecord | None: ...

    def put_execution_checkpoint(
        self,
        checkpoint: ExecutionCheckpointRecord,
    ) -> ExecutionCheckpointRecord: ...

    def get_execution_checkpoint(self, execution_id: str) -> ExecutionCheckpointRecord | None: ...

    def set_run_status(
        self,
        logical_run_id: str,
        status: LogicalRunStatus,
        *,
        pending_action_batch_id: str | None = None,
        cancellation_reason: str | None = None,
    ) -> LogicalRunRecord: ...

    def accept_input(
        self,
        logical_run_id: str,
        content: Sequence[JsonValue],
        *,
        idempotency_key: str,
        priority: InputPriority,
        origin: str = "user",
        wake_execution: bool = True,
    ) -> InputRecord: ...

    def list_inputs(
        self,
        logical_run_id: str,
        *,
        states: Sequence[InputState] | None = None,
    ) -> tuple[InputRecord, ...]: ...

    def transition_input(
        self,
        input_id: str,
        expected: InputState,
        target: InputState,
        *,
        native_enqueue_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> InputRecord: ...

    def close_and_list_inputs(self, logical_run_id: str) -> tuple[InputRecord, ...]: ...

    def enqueue_command(
        self,
        command_kind: str,
        aggregate_id: str,
        payload: dict[str, JsonValue],
        *,
        command_id: str | None = None,
    ) -> OutboxCommand: ...

    def recover_outbox(self) -> int:
        """Make unfinished delivery claims immediately available after worker restart."""
        ...

    def claim_outbox(self, *, limit: int = 50) -> tuple[OutboxCommand, ...]: ...

    def complete_outbox(self, command_id: str) -> OutboxCommand: ...

    def retry_outbox(self, command_id: str, error: str) -> OutboxCommand: ...

    def import_revision(
        self,
        session_id: str,
        *,
        descriptor: RuntimeDescriptor,
        payload: RevisionPayload,
        source: str,
    ) -> RevisionRecord: ...

    def commit_revision(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
    ) -> RevisionRecord: ...

    def commit_terminal(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
        event_type: str,
    ) -> tuple[RevisionRecord, EventRecord]:
        """Atomically publish a terminal revision and its canonical session event."""
        ...

    def get_revision(self, revision_id: str) -> RevisionRecord | None: ...

    def get_revision_for_run(self, logical_run_id: str) -> RevisionRecord | None: ...

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        *,
        event_id: str,
        logical_run_id: str | None = None,
    ) -> EventRecord: ...

    def read_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[EventRecord, ...]: ...

    def create_action_batch(
        self,
        logical_run_id: str,
        items: Sequence[dict[str, JsonValue]],
        *,
        batch_id: str | None = None,
    ) -> ActionBatch: ...

    def get_action_batch(self, batch_id: str) -> ActionBatch | None: ...

    def decide_action(
        self,
        action_item_id: str,
        *,
        decision_id: str,
        decision: dict[str, JsonValue],
        actor: str | None = None,
    ) -> ActionBatch: ...
