"""SQLite implementation of the YAACLI durable product store."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue
from ya_agent_sdk.subagents.spec import SubagentExecutionRecord

from yaacli.durable.models import (
    ActionBatch,
    ActionItem,
    ActionState,
    EventRecord,
    ExecutionCheckpointRecord,
    ExecutionRecord,
    InputPriority,
    InputRecord,
    InputState,
    LogicalRunRecord,
    LogicalRunStatus,
    OutboxCommand,
    OutboxState,
    RevisionPayload,
    RevisionRecord,
    RuntimeDescriptor,
    SessionRecord,
    SessionStatus,
    StartRunRequest,
    utc_now,
)
from yaacli.durable.sqlite_schema import (
    SUBAGENT_SCHEMA,
    user_schema_object_names,
    validate_exact_schema_subset,
)
from yaacli.durable.store import (
    HeadConflictError,
    InvalidTransitionError,
    TombstonedSessionError,
)

_SCHEMA_VERSION = 4
_TERMINAL_RUN_STATES = {
    LogicalRunStatus.completed,
    LogicalRunStatus.failed,
    LogicalRunStatus.cancelled,
    LogicalRunStatus.interrupted,
}
_RUN_TRANSITIONS: dict[LogicalRunStatus, frozenset[LogicalRunStatus]] = {
    LogicalRunStatus.pending: frozenset({
        LogicalRunStatus.running,
        LogicalRunStatus.cancelling,
        LogicalRunStatus.failed,
        LogicalRunStatus.cancelled,
        LogicalRunStatus.interrupted,
    }),
    LogicalRunStatus.running: frozenset({
        LogicalRunStatus.suspended,
        LogicalRunStatus.cancelling,
        LogicalRunStatus.completed,
        LogicalRunStatus.failed,
        LogicalRunStatus.cancelled,
        LogicalRunStatus.interrupted,
    }),
    LogicalRunStatus.suspended: frozenset({
        LogicalRunStatus.running,
        LogicalRunStatus.cancelling,
        LogicalRunStatus.failed,
        LogicalRunStatus.cancelled,
        LogicalRunStatus.interrupted,
    }),
    LogicalRunStatus.cancelling: frozenset({
        LogicalRunStatus.cancelled,
        LogicalRunStatus.failed,
        LogicalRunStatus.interrupted,
    }),
    LogicalRunStatus.completed: frozenset(),
    LogicalRunStatus.failed: frozenset(),
    LogicalRunStatus.cancelled: frozenset(),
    LogicalRunStatus.interrupted: frozenset(),
}
_INPUT_TRANSITIONS: dict[InputState, frozenset[InputState]] = {
    InputState.accepted: frozenset({InputState.enqueued, InputState.applied, InputState.rejected}),
    InputState.enqueued: frozenset({InputState.applied, InputState.rejected}),
    InputState.applied: frozenset(),
    InputState.rejected: frozenset(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    workspace_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    head_revision_id TEXT,
    active_execution_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tombstoned_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS runtime_descriptors (
    descriptor_id TEXT PRIMARY KEY,
    plan_fingerprint TEXT NOT NULL,
    descriptor_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS logical_runs (
    logical_run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    execution_id TEXT NOT NULL UNIQUE,
    expected_head_revision_id TEXT,
    descriptor_id TEXT NOT NULL REFERENCES runtime_descriptors(descriptor_id),
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL,
    input_open INTEGER NOT NULL DEFAULT 1,
    cancellation_reason TEXT,
    pending_action_batch_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS logical_runs_session_idx
    ON logical_runs(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL UNIQUE REFERENCES logical_runs(logical_run_id),
    executable_version TEXT NOT NULL,
    plan_fingerprint TEXT NOT NULL,
    descriptor_id TEXT NOT NULL REFERENCES runtime_descriptors(descriptor_id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_checkpoints (
    execution_id TEXT PRIMARY KEY REFERENCES executions(execution_id),
    logical_run_id TEXT NOT NULL UNIQUE REFERENCES logical_runs(logical_run_id),
    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
    segment_status TEXT NOT NULL CHECK(segment_status IN ('completed', 'suspended')),
    payload_json TEXT NOT NULL,
    deferred_requests_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_inputs (
    input_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    order_index INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    origin TEXT NOT NULL,
    priority TEXT NOT NULL,
    content_json TEXT NOT NULL,
    state TEXT NOT NULL,
    native_enqueue_id TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(logical_run_id, order_index),
    UNIQUE(logical_run_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS run_inputs_drain_idx
    ON run_inputs(logical_run_id, state, priority, order_index);

CREATE TABLE IF NOT EXISTS outbox_commands (
    command_id TEXT PRIMARY KEY,
    command_kind TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    delivered_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outbox_delivery_idx
    ON outbox_commands(state, available_at, created_at);

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    commit_kind TEXT NOT NULL,
    parent_revision_id TEXT,
    message_history_json TEXT NOT NULL,
    resumable_state_json TEXT NOT NULL,
    input_ledger_json TEXT NOT NULL,
    display_projection_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    terminal_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(logical_run_id, commit_kind)
);
CREATE INDEX IF NOT EXISTS revisions_session_idx
    ON revisions(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS session_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    logical_run_id TEXT,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, sequence)
);
CREATE INDEX IF NOT EXISTS session_events_read_idx
    ON session_events(session_id, sequence);

CREATE TABLE IF NOT EXISTS action_batches (
    batch_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    state TEXT NOT NULL,
    deadline_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_items (
    action_item_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES action_batches(batch_id),
    tool_call_id TEXT NOT NULL,
    decision_kind TEXT NOT NULL,
    request_json TEXT NOT NULL,
    state TEXT NOT NULL,
    decision_id TEXT UNIQUE,
    decision_json TEXT,
    actor TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT,
    UNIQUE(batch_id, tool_call_id)
);
CREATE INDEX IF NOT EXISTS action_items_batch_idx
    ON action_items(batch_id, created_at, action_item_id);
"""
_SCHEMA += SUBAGENT_SCHEMA


class SQLiteSessionStore:
    """Transactional SQLite product truth for sessions and logical runs."""

    def __init__(self, database_path: Path | str) -> None:
        path = Path(database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            path,
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            with self._lock:
                self._initialize_or_validate_schema()
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
                self._connection.execute("PRAGMA busy_timeout = 30000")
        except BaseException:
            self._connection.close()
            raise

    def _initialize_or_validate_schema(self) -> None:
        error_prefix = "Unsupported YAACLI store schema; YAACLI 2.0 requires the exact durable product schema"
        if not user_schema_object_names(self._connection):
            self._connection.executescript(
                "BEGIN IMMEDIATE;\n"
                f"{_SCHEMA}\n"
                "INSERT INTO schema_metadata(key, value) "
                f"VALUES('schema_version', '{_SCHEMA_VERSION}');\n"
                "COMMIT;"
            )
            validate_exact_schema_subset(
                self._connection,
                _SCHEMA,
                error_prefix=error_prefix,
            )
            return

        validate_exact_schema_subset(
            self._connection,
            _SCHEMA,
            error_prefix=error_prefix,
        )
        current = self._connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
        if current is None or current["value"] != str(_SCHEMA_VERSION):
            value = None if current is None else current["value"]
            raise RuntimeError(
                f"Unsupported YAACLI store schema marker {value!r}; expected {_SCHEMA_VERSION}. "
                "Migrate the database offline or recreate it; runtime schema compatibility is not supported."
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteSessionStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def create_session(
        self,
        workspace_ref: str,
        *,
        session_id: str | None = None,
    ) -> SessionRecord:
        resolved_id = session_id or uuid.uuid4().hex[:12]
        now = utc_now()
        with self._write() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, workspace_ref, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_id,
                        workspace_ref,
                        SessionStatus.active.value,
                        _dt(now),
                        _dt(now),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Session {resolved_id!r} already exists") from exc
            return self._session_from_row(self._required_row(connection, "sessions", "session_id", resolved_id))

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return self._session_from_row(row) if row is not None else None

    def list_sessions(self, *, limit: int = 100) -> tuple[SessionRecord, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM sessions
                WHERE status = ?
                ORDER BY updated_at DESC, session_id
                LIMIT ?
                """,
                (SessionStatus.active.value, limit),
            ).fetchall()
        return tuple(self._session_from_row(row) for row in rows)

    def tombstone_session(self, session_id: str) -> SessionRecord:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "sessions", "session_id", session_id)
            session = self._session_from_row(row)
            if session.status is SessionStatus.tombstoned:
                return session
            if session.active_execution_id is not None:
                run_row = connection.execute(
                    "SELECT * FROM logical_runs WHERE execution_id = ?",
                    (session.active_execution_id,),
                ).fetchone()
                if run_row is not None and LogicalRunStatus(run_row["status"]) not in _TERMINAL_RUN_STATES:
                    connection.execute(
                        """
                        UPDATE logical_runs
                        SET status = ?, input_open = 0,
                            cancellation_reason = ?, updated_at = ?
                        WHERE logical_run_id = ?
                        """,
                        (
                            LogicalRunStatus.cancelling.value,
                            "session tombstoned",
                            _dt(now),
                            run_row["logical_run_id"],
                        ),
                    )
                    connection.execute(
                        "UPDATE executions SET status = ?, updated_at = ? WHERE execution_id = ?",
                        (
                            LogicalRunStatus.cancelling.value,
                            _dt(now),
                            session.active_execution_id,
                        ),
                    )
                    self._insert_outbox(
                        connection,
                        "cancel_execution",
                        session.active_execution_id,
                        {
                            "execution_id": session.active_execution_id,
                            "logical_run_id": run_row["logical_run_id"],
                            "reason": "session tombstoned",
                        },
                    )
            self._fence_subagents_for_tombstone(
                connection,
                session_id=session_id,
                now=now,
            )
            connection.execute(
                """
                UPDATE sessions
                SET status = ?, active_execution_id = NULL,
                    tombstoned_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    SessionStatus.tombstoned.value,
                    _dt(now),
                    _dt(now),
                    session_id,
                ),
            )
            return self._session_from_row(self._required_row(connection, "sessions", "session_id", session_id))

    def _fence_subagents_for_tombstone(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        now: datetime,
    ) -> None:
        reason = "owner session tombstoned"
        rows = connection.execute(
            "SELECT execution_id, record_json FROM subagent_executions WHERE owner_scope_id = ?",
            (session_id,),
        ).fetchall()
        for row in rows:
            record = SubagentExecutionRecord.model_validate_json(row["record_json"])
            if record.terminal:
                continue
            connection.execute(
                """
                UPDATE subagent_executions
                SET input_open = 0, cancel_requested = 1,
                    cancellation_reason = ?, updated_at = ?
                WHERE execution_id = ?
                """,
                (reason, _dt(now), record.execution_id),
            )
            connection.execute(
                """
                UPDATE subagent_inputs
                SET state = ?, rejection_reason = ?, updated_at = ?
                WHERE execution_id = ? AND state IN (?, ?)
                """,
                (
                    InputState.rejected.value,
                    reason,
                    _dt(now),
                    record.execution_id,
                    InputState.accepted.value,
                    InputState.enqueued.value,
                ),
            )
            self._insert_outbox(
                connection,
                "cancel_subagent_execution",
                record.execution_id,
                {
                    "execution_id": record.execution_id,
                    "owner_scope_id": record.owner_scope_id,
                    "reason": reason,
                },
                command_id=f"cancel-subagent:{record.execution_id}:{record.segment_index}",
            )

    def put_descriptor(self, descriptor: RuntimeDescriptor) -> RuntimeDescriptor:
        with self._write() as connection:
            return self._put_descriptor(connection, descriptor)

    def _put_descriptor(
        self,
        connection: sqlite3.Connection,
        descriptor: RuntimeDescriptor,
    ) -> RuntimeDescriptor:
        descriptor.assert_integrity()
        payload = descriptor.model_dump_json()
        existing = connection.execute(
            "SELECT descriptor_json FROM runtime_descriptors WHERE descriptor_id = ?",
            (descriptor.descriptor_id,),
        ).fetchone()
        if existing is not None:
            stored = RuntimeDescriptor.model_validate_json(existing["descriptor_json"])
            if (
                stored.plan_fingerprint != descriptor.plan_fingerprint
                or stored.behavior_payload() != descriptor.behavior_payload()
            ):
                raise ValueError(f"Descriptor ID {descriptor.descriptor_id!r} has different content")
            return stored
        connection.execute(
            """
            INSERT INTO runtime_descriptors(
                descriptor_id, plan_fingerprint, descriptor_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                descriptor.descriptor_id,
                descriptor.plan_fingerprint,
                payload,
                _dt(descriptor.created_at),
            ),
        )
        return descriptor

    def get_descriptor(self, descriptor_id: str) -> RuntimeDescriptor | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT descriptor_json FROM runtime_descriptors WHERE descriptor_id = ?",
                (descriptor_id,),
            ).fetchone()
        return RuntimeDescriptor.model_validate_json(row["descriptor_json"]) if row is not None else None

    def list_nonterminal_descriptors(self) -> tuple[RuntimeDescriptor, ...]:
        """Return every exact plan still referenced by unfinished main work."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT descriptor.descriptor_json
                FROM runtime_descriptors AS descriptor
                JOIN logical_runs AS run ON run.descriptor_id = descriptor.descriptor_id
                WHERE run.status NOT IN ('completed', 'failed', 'cancelled', 'interrupted')
                ORDER BY descriptor.descriptor_id
                """
            ).fetchall()
        return tuple(RuntimeDescriptor.model_validate_json(row["descriptor_json"]) for row in rows)

    def start_run(self, request: StartRunRequest) -> LogicalRunRecord:
        if request.plan_fingerprint != request.descriptor.plan_fingerprint:
            raise ValueError("Start-run fingerprint does not match its descriptor")
        if request.executable_version != request.descriptor.executable_version:
            raise ValueError("Start-run executable version does not match its descriptor")
        now = utc_now()
        with self._write() as connection:
            session_row = self._required_row(connection, "sessions", "session_id", request.session_id)
            session = self._session_from_row(session_row)
            self._ensure_active_session(session)
            if session.head_revision_id != request.expected_head_revision_id:
                raise HeadConflictError(
                    f"Session {request.session_id!r} head changed from "
                    f"{request.expected_head_revision_id!r} to {session.head_revision_id!r}"
                )
            existing = connection.execute(
                """
                SELECT * FROM logical_runs
                WHERE session_id = ? AND idempotency_key = ?
                """,
                (request.session_id, request.idempotency_key),
            ).fetchone()
            if existing is not None:
                record = self._run_from_row(existing)
                initial_row = connection.execute(
                    """
                    SELECT content_json FROM run_inputs
                    WHERE logical_run_id = ? AND order_index = 0
                    """,
                    (record.logical_run_id,),
                ).fetchone()
                initial_content = _array(initial_row["content_json"]) if initial_row is not None else None
                if (
                    record.expected_head_revision_id != request.expected_head_revision_id
                    or record.descriptor_id != request.descriptor.descriptor_id
                    or initial_content != list(request.initial_content)
                ):
                    raise ValueError("Start-run idempotency key was reused with different intent")
                return record
            if session.active_execution_id is not None:
                raise InvalidTransitionError(
                    f"Session {request.session_id!r} already has active execution {session.active_execution_id!r}"
                )

            self._put_descriptor(connection, request.descriptor)
            logical_run_id = str(uuid.uuid4())
            execution_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO logical_runs(
                    logical_run_id, session_id, execution_id,
                    expected_head_revision_id, descriptor_id, idempotency_key,
                    status, input_open, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    logical_run_id,
                    request.session_id,
                    execution_id,
                    request.expected_head_revision_id,
                    request.descriptor.descriptor_id,
                    request.idempotency_key,
                    LogicalRunStatus.pending.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            connection.execute(
                """
                INSERT INTO executions(
                    execution_id, logical_run_id, executable_version,
                    plan_fingerprint, descriptor_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    logical_run_id,
                    request.executable_version,
                    request.plan_fingerprint,
                    request.descriptor.descriptor_id,
                    LogicalRunStatus.pending.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            input_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO run_inputs(
                    input_id, logical_run_id, order_index, idempotency_key,
                    origin, priority, content_json, state, created_at, updated_at
                ) VALUES (?, ?, 0, ?, 'user', ?, ?, ?, ?, ?)
                """,
                (
                    input_id,
                    logical_run_id,
                    f"{request.idempotency_key}:initial",
                    InputPriority.asap.value,
                    _json(list(request.initial_content)),
                    InputState.accepted.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            self._insert_outbox(
                connection,
                "start_execution",
                execution_id,
                {
                    "command_id": request.idempotency_key,
                    "execution_id": execution_id,
                    "logical_run_id": logical_run_id,
                    "session_id": request.session_id,
                    "initial_input_id": input_id,
                    "descriptor_id": request.descriptor.descriptor_id,
                },
            )
            connection.execute(
                """
                UPDATE sessions
                SET active_execution_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (execution_id, _dt(now), request.session_id),
            )
            return self._run_from_row(self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id))

    def get_run(self, logical_run_id: str) -> LogicalRunRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM logical_runs WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
        return self._execution_from_row(row) if row is not None else None

    def put_execution_checkpoint(
        self,
        checkpoint: ExecutionCheckpointRecord,
    ) -> ExecutionCheckpointRecord:
        with self._write() as connection:
            execution = self._execution_from_row(
                self._required_row(connection, "executions", "execution_id", checkpoint.execution_id)
            )
            if execution.logical_run_id != checkpoint.logical_run_id:
                raise ValueError("Execution checkpoint does not match its logical run")
            existing_row = connection.execute(
                "SELECT * FROM execution_checkpoints WHERE execution_id = ?",
                (checkpoint.execution_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._execution_checkpoint_from_row(existing_row)
                if existing.segment_index > checkpoint.segment_index:
                    raise InvalidTransitionError("Execution checkpoint segment index cannot move backwards")
                if existing.segment_index == checkpoint.segment_index:
                    comparable_existing = existing.model_copy(
                        update={"created_at": checkpoint.created_at, "updated_at": checkpoint.updated_at}
                    )
                    if comparable_existing != checkpoint:
                        raise ValueError("Execution checkpoint segment was reused with different content")
                    return existing
            created_at = existing_row["created_at"] if existing_row is not None else _dt(checkpoint.created_at)
            connection.execute(
                """
                INSERT INTO execution_checkpoints(
                    execution_id, logical_run_id, segment_index, segment_status,
                    payload_json, deferred_requests_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    segment_index = excluded.segment_index,
                    segment_status = excluded.segment_status,
                    payload_json = excluded.payload_json,
                    deferred_requests_json = excluded.deferred_requests_json,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.execution_id,
                    checkpoint.logical_run_id,
                    checkpoint.segment_index,
                    checkpoint.segment_status,
                    checkpoint.payload.model_dump_json(),
                    (_json(checkpoint.deferred_requests) if checkpoint.deferred_requests is not None else None),
                    created_at,
                    _dt(checkpoint.updated_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM execution_checkpoints WHERE execution_id = ?",
                (checkpoint.execution_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - insert invariant
                raise RuntimeError("Execution checkpoint insert did not persist")
            return self._execution_checkpoint_from_row(row)

    def get_execution_checkpoint(self, execution_id: str) -> ExecutionCheckpointRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM execution_checkpoints WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        return self._execution_checkpoint_from_row(row) if row is not None else None

    def set_run_status(
        self,
        logical_run_id: str,
        status: LogicalRunStatus,
        *,
        pending_action_batch_id: str | None = None,
        cancellation_reason: str | None = None,
    ) -> LogicalRunRecord:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id)
            run = self._run_from_row(row)
            session = self._required_session(connection, run.session_id)
            self._ensure_active_session(session)
            if status is run.status:
                return run
            if status not in _RUN_TRANSITIONS[run.status]:
                raise InvalidTransitionError(f"Cannot transition logical run from {run.status.value} to {status.value}")
            if status is LogicalRunStatus.suspended and pending_action_batch_id is None:
                raise InvalidTransitionError("Suspended runs require a pending action batch")
            if status is not LogicalRunStatus.suspended and pending_action_batch_id is not None:
                raise InvalidTransitionError("Only suspended runs may reference a pending action batch")
            connection.execute(
                """
                UPDATE logical_runs
                SET status = ?, pending_action_batch_id = ?,
                    cancellation_reason = COALESCE(?, cancellation_reason),
                    input_open = CASE WHEN ? IN ('cancelling', 'completed', 'failed', 'cancelled', 'interrupted')
                                      THEN 0 ELSE input_open END,
                    updated_at = ?
                WHERE logical_run_id = ?
                """,
                (
                    status.value,
                    pending_action_batch_id,
                    cancellation_reason,
                    status.value,
                    _dt(now),
                    logical_run_id,
                ),
            )
            connection.execute(
                "UPDATE executions SET status = ?, updated_at = ? WHERE execution_id = ?",
                (status.value, _dt(now), run.execution_id),
            )
            return self._run_from_row(self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id))

    def accept_input(
        self,
        logical_run_id: str,
        content: Sequence[JsonValue],
        *,
        idempotency_key: str,
        priority: InputPriority,
        origin: str = "user",
        wake_execution: bool = True,
    ) -> InputRecord:
        if origin not in {"user", "feature"}:
            raise ValueError("origin must be 'user' or 'feature'")
        now = utc_now()
        normalized_content = list(content)
        with self._write() as connection:
            run_row = self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id)
            run = self._run_from_row(run_row)
            session = self._required_session(connection, run.session_id)
            self._ensure_active_session(session)
            existing = connection.execute(
                """
                SELECT * FROM run_inputs
                WHERE logical_run_id = ? AND idempotency_key = ?
                """,
                (logical_run_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                record = self._input_from_row(existing)
                if record.content != normalized_content or record.priority is not priority or record.origin != origin:
                    raise ValueError("Input idempotency key was reused with different content")
                return record
            if not bool(run_row["input_open"]) or run.terminal or run.status is LogicalRunStatus.cancelling:
                raise InvalidTransitionError(f"Logical run {logical_run_id!r} is not accepting input")
            next_order = cast(
                int,
                connection.execute(
                    """
                    SELECT COALESCE(MAX(order_index), -1) + 1 AS next_order
                    FROM run_inputs WHERE logical_run_id = ?
                    """,
                    (logical_run_id,),
                ).fetchone()["next_order"],
            )
            input_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO run_inputs(
                    input_id, logical_run_id, order_index, idempotency_key,
                    origin, priority, content_json, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_id,
                    logical_run_id,
                    next_order,
                    idempotency_key,
                    origin,
                    priority.value,
                    _json(normalized_content),
                    InputState.accepted.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            if wake_execution:
                self._insert_outbox(
                    connection,
                    "notify_input",
                    run.execution_id,
                    {
                        "execution_id": run.execution_id,
                        "logical_run_id": logical_run_id,
                        "input_id": input_id,
                    },
                )
            return self._input_from_row(self._required_row(connection, "run_inputs", "input_id", input_id))

    def list_inputs(
        self,
        logical_run_id: str,
        *,
        states: Sequence[InputState] | None = None,
    ) -> tuple[InputRecord, ...]:
        query = "SELECT * FROM run_inputs WHERE logical_run_id = ?"
        params: list[object] = [logical_run_id]
        if states is not None:
            if not states:
                return ()
            query += f" AND state IN ({','.join('?' for _ in states)})"
            params.extend(state.value for state in states)
        query += " ORDER BY CASE priority WHEN 'asap' THEN 0 ELSE 1 END, order_index"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(self._input_from_row(row) for row in rows)

    def transition_input(
        self,
        input_id: str,
        expected: InputState,
        target: InputState,
        *,
        native_enqueue_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> InputRecord:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "run_inputs", "input_id", input_id)
            record = self._input_from_row(row)
            run = self._run_from_row(
                self._required_row(connection, "logical_runs", "logical_run_id", record.logical_run_id)
            )
            self._ensure_active_session(self._required_session(connection, run.session_id))
            if record.state is target:
                if (
                    target is InputState.enqueued
                    and native_enqueue_id is not None
                    and record.native_enqueue_id != native_enqueue_id
                ):
                    connection.execute(
                        "UPDATE run_inputs SET native_enqueue_id = ?, updated_at = ? WHERE input_id = ?",
                        (native_enqueue_id, _dt(now), input_id),
                    )
                    return self._input_from_row(self._required_row(connection, "run_inputs", "input_id", input_id))
                return record
            if record.state is not expected:
                raise InvalidTransitionError(f"Input {input_id!r} is {record.state.value}, expected {expected.value}")
            if target not in _INPUT_TRANSITIONS[record.state]:
                raise InvalidTransitionError(f"Cannot transition input from {record.state.value} to {target.value}")
            if target is InputState.rejected and not rejection_reason:
                raise InvalidTransitionError("Rejected inputs require a reason")
            connection.execute(
                """
                UPDATE run_inputs
                SET state = ?, native_enqueue_id = COALESCE(?, native_enqueue_id),
                    rejection_reason = COALESCE(?, rejection_reason), updated_at = ?
                WHERE input_id = ?
                """,
                (
                    target.value,
                    native_enqueue_id,
                    rejection_reason,
                    _dt(now),
                    input_id,
                ),
            )
            return self._input_from_row(self._required_row(connection, "run_inputs", "input_id", input_id))

    def close_and_list_inputs(self, logical_run_id: str) -> tuple[InputRecord, ...]:
        """Fence ingress and return unresolved input accepted before the fence."""
        now = utc_now()
        with self._write() as connection:
            run_row = self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id)
            run = self._run_from_row(run_row)
            self._ensure_active_session(self._required_session(connection, run.session_id))
            if run.terminal or run.status is LogicalRunStatus.cancelling:
                return ()
            connection.execute(
                "UPDATE logical_runs SET input_open = 0, updated_at = ? WHERE logical_run_id = ?",
                (_dt(now), logical_run_id),
            )
            rows = connection.execute(
                """
                SELECT * FROM run_inputs
                WHERE logical_run_id = ? AND state IN (?, ?)
                ORDER BY CASE priority WHEN 'asap' THEN 0 ELSE 1 END, order_index
                """,
                (
                    logical_run_id,
                    InputState.accepted.value,
                    InputState.enqueued.value,
                ),
            ).fetchall()
            if rows:
                connection.execute(
                    "UPDATE logical_runs SET input_open = 1 WHERE logical_run_id = ?",
                    (logical_run_id,),
                )
            return tuple(self._input_from_row(row) for row in rows)

    def enqueue_command(
        self,
        command_kind: str,
        aggregate_id: str,
        payload: dict[str, JsonValue],
        *,
        command_id: str | None = None,
    ) -> OutboxCommand:
        with self._write() as connection:
            return self._insert_outbox(
                connection,
                command_kind,
                aggregate_id,
                payload,
                command_id=command_id,
            )

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        command_kind: str,
        aggregate_id: str,
        payload: dict[str, JsonValue],
        *,
        command_id: str | None = None,
    ) -> OutboxCommand:
        resolved_id = command_id or str(uuid.uuid4())
        now = utc_now()
        existing = connection.execute("SELECT * FROM outbox_commands WHERE command_id = ?", (resolved_id,)).fetchone()
        if existing is not None:
            record = self._outbox_from_row(existing)
            if record.command_kind != command_kind or record.aggregate_id != aggregate_id or record.payload != payload:
                raise ValueError("Outbox command ID was reused with different intent")
            return record
        connection.execute(
            """
            INSERT INTO outbox_commands(
                command_id, command_kind, aggregate_id, payload_json,
                state, attempt_count, available_at, created_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                resolved_id,
                command_kind,
                aggregate_id,
                _json(payload),
                OutboxState.pending.value,
                _dt(now),
                _dt(now),
            ),
        )
        return self._outbox_from_row(self._required_row(connection, "outbox_commands", "command_id", resolved_id))

    def recover_outbox(self) -> int:
        """Release delivery claims owned by a previous worker process."""
        now = utc_now()
        with self._write() as connection:
            cursor = connection.execute(
                """
                UPDATE outbox_commands
                SET state = ?, available_at = ?, claimed_at = NULL
                WHERE state = ?
                """,
                (
                    OutboxState.pending.value,
                    _dt(now),
                    OutboxState.delivering.value,
                ),
            )
            return cursor.rowcount

    def claim_outbox(self, *, limit: int = 50) -> tuple[OutboxCommand, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()
        now = utc_now()
        stale_before = now - timedelta(seconds=30)
        with self._write() as connection:
            rows = connection.execute(
                """
                SELECT * FROM outbox_commands
                WHERE available_at <= ?
                  AND (
                    state = ? OR (state = ? AND claimed_at <= ?)
                  )
                ORDER BY created_at, command_id
                LIMIT ?
                """,
                (
                    _dt(now),
                    OutboxState.pending.value,
                    OutboxState.delivering.value,
                    _dt(stale_before),
                    limit,
                ),
            ).fetchall()
            if not rows:
                return ()
            command_ids = [row["command_id"] for row in rows]
            connection.executemany(
                """
                UPDATE outbox_commands
                SET state = ?, attempt_count = attempt_count + 1, claimed_at = ?
                WHERE command_id = ?
                """,
                [(OutboxState.delivering.value, _dt(now), command_id) for command_id in command_ids],
            )
            return tuple(
                self._outbox_from_row(self._required_row(connection, "outbox_commands", "command_id", command_id))
                for command_id in command_ids
            )

    def complete_outbox(self, command_id: str) -> OutboxCommand:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "outbox_commands", "command_id", command_id)
            record = self._outbox_from_row(row)
            if record.state is OutboxState.delivered:
                return record
            if record.state is not OutboxState.delivering:
                raise InvalidTransitionError("Only claimed outbox commands can complete")
            connection.execute(
                """
                UPDATE outbox_commands
                SET state = ?, delivered_at = ?, last_error = NULL
                WHERE command_id = ?
                """,
                (OutboxState.delivered.value, _dt(now), command_id),
            )
            return self._outbox_from_row(self._required_row(connection, "outbox_commands", "command_id", command_id))

    def retry_outbox(self, command_id: str, error: str) -> OutboxCommand:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "outbox_commands", "command_id", command_id)
            record = self._outbox_from_row(row)
            if record.state is OutboxState.delivered:
                return record
            if record.state is not OutboxState.delivering:
                raise InvalidTransitionError("Only claimed outbox commands can be retried")
            delay = min(30, 2 ** min(record.attempt_count, 5))
            connection.execute(
                """
                UPDATE outbox_commands
                SET state = ?, available_at = ?, claimed_at = NULL, last_error = ?
                WHERE command_id = ?
                """,
                (
                    OutboxState.pending.value,
                    _dt(now + timedelta(seconds=delay)),
                    error,
                    command_id,
                ),
            )
            return self._outbox_from_row(self._required_row(connection, "outbox_commands", "command_id", command_id))

    def import_revision(
        self,
        session_id: str,
        *,
        descriptor: RuntimeDescriptor,
        payload: RevisionPayload,
        source: str,
    ) -> RevisionRecord:
        """Import an offline snapshot without creating executable outbox work."""
        now = utc_now()
        logical_run_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        with self._write() as connection:
            session = self._required_session(connection, session_id)
            self._ensure_active_session(session)
            if session.active_execution_id is not None:
                raise InvalidTransitionError(f"Session {session_id!r} already has an active execution")
            self._put_descriptor(connection, descriptor)
            connection.execute(
                """
                INSERT INTO logical_runs(
                    logical_run_id, session_id, execution_id,
                    expected_head_revision_id, descriptor_id, idempotency_key,
                    status, input_open, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    logical_run_id,
                    session_id,
                    execution_id,
                    session.head_revision_id,
                    descriptor.descriptor_id,
                    f"offline-import:{revision_id}",
                    LogicalRunStatus.completed.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            connection.execute(
                """
                INSERT INTO executions(
                    execution_id, logical_run_id, executable_version,
                    plan_fingerprint, descriptor_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    logical_run_id,
                    descriptor.executable_version,
                    descriptor.plan_fingerprint,
                    descriptor.descriptor_id,
                    LogicalRunStatus.completed.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            terminal = dict(payload.terminal)
            terminal.setdefault("status", LogicalRunStatus.completed.value)
            terminal["import_source"] = source
            connection.execute(
                """
                INSERT INTO revisions(
                    revision_id, session_id, logical_run_id, commit_kind,
                    parent_revision_id, message_history_json,
                    resumable_state_json, input_ledger_json,
                    display_projection_json, usage_json, terminal_json, created_at
                ) VALUES (?, ?, ?, 'offline_import', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    session_id,
                    logical_run_id,
                    session.head_revision_id,
                    _json(payload.message_history),
                    _json(payload.resumable_state),
                    _json(payload.input_ledger),
                    _json(payload.display_projection),
                    _json(payload.usage),
                    _json(terminal),
                    _dt(now),
                ),
            )
            connection.execute(
                "UPDATE sessions SET head_revision_id = ?, updated_at = ? WHERE session_id = ?",
                (revision_id, _dt(now), session_id),
            )
            return self._revision_from_row(self._required_row(connection, "revisions", "revision_id", revision_id))

    def commit_revision(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
    ) -> RevisionRecord:
        now = utc_now()
        with self._write() as connection:
            return self._commit_revision(
                connection,
                logical_run_id,
                commit_kind=commit_kind,
                payload=payload,
                terminal_status=terminal_status,
                now=now,
            )

    def commit_terminal(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
        event_type: str,
    ) -> tuple[RevisionRecord, EventRecord]:
        now = utc_now()
        with self._write() as connection:
            revision = self._commit_revision(
                connection,
                logical_run_id,
                commit_kind=commit_kind,
                payload=payload,
                terminal_status=terminal_status,
                now=now,
            )
            event = self._append_event(
                connection,
                revision.session_id,
                event_type,
                {"revision_id": revision.revision_id, **payload.terminal},
                event_id=f"terminal:{logical_run_id}",
                logical_run_id=logical_run_id,
                now=now,
            )
            return revision, event

    def _commit_revision(
        self,
        connection: sqlite3.Connection,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
        now: datetime,
    ) -> RevisionRecord:
        if terminal_status not in _TERMINAL_RUN_STATES:
            raise ValueError("A committed revision requires a terminal run status")
        run = self._run_from_row(self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id))
        session = self._required_session(connection, run.session_id)
        self._ensure_active_session(session)
        existing = connection.execute(
            """
            SELECT * FROM revisions
            WHERE logical_run_id = ? AND commit_kind = ?
            """,
            (logical_run_id, commit_kind),
        ).fetchone()
        if existing is not None:
            revision = self._revision_from_row(existing)
            if self._revision_payload(revision) != payload:
                raise ValueError("Terminal revision idempotency key has different payload")
            return revision
        if terminal_status not in _RUN_TRANSITIONS[run.status]:
            raise InvalidTransitionError(f"Cannot transition run from {run.status.value} to {terminal_status.value}")
        if session.head_revision_id != run.expected_head_revision_id:
            raise HeadConflictError(
                f"Session {run.session_id!r} head changed from "
                f"{run.expected_head_revision_id!r} to {session.head_revision_id!r}"
            )
        revision_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO revisions(
                revision_id, session_id, logical_run_id, commit_kind,
                parent_revision_id, message_history_json, resumable_state_json,
                input_ledger_json, display_projection_json, usage_json,
                terminal_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                run.session_id,
                logical_run_id,
                commit_kind,
                run.expected_head_revision_id,
                _json(payload.message_history),
                _json(payload.resumable_state),
                _json(payload.input_ledger),
                _json(payload.display_projection),
                _json(payload.usage),
                _json(payload.terminal),
                _dt(now),
            ),
        )
        cursor = connection.execute(
            """
            UPDATE sessions
            SET head_revision_id = ?, active_execution_id = NULL, updated_at = ?
            WHERE session_id = ? AND status = ?
              AND head_revision_id IS ? AND active_execution_id = ?
            """,
            (
                revision_id,
                _dt(now),
                run.session_id,
                SessionStatus.active.value,
                run.expected_head_revision_id,
                run.execution_id,
            ),
        )
        if cursor.rowcount != 1:
            raise HeadConflictError(f"Session {run.session_id!r} head or active execution changed")
        connection.execute(
            """
            UPDATE run_inputs
            SET state = ?, rejection_reason = ?, updated_at = ?
            WHERE logical_run_id = ? AND state IN (?, ?)
            """,
            (
                InputState.rejected.value,
                f"run terminated as {terminal_status.value} before input application",
                _dt(now),
                logical_run_id,
                InputState.accepted.value,
                InputState.enqueued.value,
            ),
        )
        connection.execute(
            """
            UPDATE logical_runs
            SET status = ?, input_open = 0, pending_action_batch_id = NULL,
                updated_at = ? WHERE logical_run_id = ?
            """,
            (terminal_status.value, _dt(now), logical_run_id),
        )
        connection.execute(
            """
            UPDATE executions SET status = ?, updated_at = ?
            WHERE execution_id = ?
            """,
            (terminal_status.value, _dt(now), run.execution_id),
        )
        return self._revision_from_row(self._required_row(connection, "revisions", "revision_id", revision_id))

    def get_revision(self, revision_id: str) -> RevisionRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM revisions WHERE revision_id = ?", (revision_id,)).fetchone()
        return self._revision_from_row(row) if row is not None else None

    def get_revision_for_run(self, logical_run_id: str) -> RevisionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM revisions WHERE logical_run_id = ? ORDER BY created_at DESC LIMIT 1",
                (logical_run_id,),
            ).fetchone()
        return self._revision_from_row(row) if row is not None else None

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        *,
        event_id: str,
        logical_run_id: str | None = None,
    ) -> EventRecord:
        now = utc_now()
        with self._write() as connection:
            return self._append_event(
                connection,
                session_id,
                event_type,
                payload,
                event_id=event_id,
                logical_run_id=logical_run_id,
                now=now,
            )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        *,
        event_id: str,
        logical_run_id: str | None,
        now: datetime,
    ) -> EventRecord:
        session = self._required_session(connection, session_id)
        self._ensure_active_session(session)
        existing = connection.execute("SELECT * FROM session_events WHERE event_id = ?", (event_id,)).fetchone()
        if existing is not None:
            event = self._event_from_row(existing)
            if (
                event.session_id != session_id
                or event.logical_run_id != logical_run_id
                or event.event_type != event_type
                or event.payload != payload
            ):
                raise ValueError("Event ID was reused with different content")
            return event
        sequence = cast(
            int,
            connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM session_events WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()["next_sequence"],
        )
        connection.execute(
            """
            INSERT INTO session_events(
                event_id, session_id, logical_run_id, sequence,
                event_type, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                logical_run_id,
                sequence,
                event_type,
                _json(payload),
                _dt(now),
            ),
        )
        return self._event_from_row(self._required_row(connection, "session_events", "event_id", event_id))

    def read_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[EventRecord, ...]:
        if after_sequence < 0 or limit < 0:
            raise ValueError("after_sequence and limit must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM session_events
                WHERE session_id = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (session_id, after_sequence, limit),
            ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def create_action_batch(
        self,
        logical_run_id: str,
        items: Sequence[dict[str, JsonValue]],
        *,
        batch_id: str | None = None,
    ) -> ActionBatch:
        if not items:
            raise ValueError("Action batches cannot be empty")
        resolved_id = batch_id or str(uuid.uuid4())
        now = utc_now()
        with self._write() as connection:
            run = self._run_from_row(self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id))
            self._ensure_active_session(self._required_session(connection, run.session_id))
            existing = connection.execute(
                "SELECT batch_id FROM action_batches WHERE batch_id = ?", (resolved_id,)
            ).fetchone()
            if existing is not None:
                batch = self._action_batch(connection, resolved_id)
                comparable = [
                    {
                        "tool_call_id": item.tool_call_id,
                        "decision_kind": item.decision_kind,
                        "request": item.request,
                    }
                    for item in batch.items
                ]
                normalized = [self._normalize_action_input(item) for item in items]
                if comparable != normalized or batch.logical_run_id != logical_run_id:
                    raise ValueError("Action batch ID was reused with different content")
                return batch
            connection.execute(
                """
                INSERT INTO action_batches(
                    batch_id, logical_run_id, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    logical_run_id,
                    ActionState.pending.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            for raw_item in items:
                item = self._normalize_action_input(raw_item)
                connection.execute(
                    """
                    INSERT INTO action_items(
                        action_item_id, batch_id, tool_call_id, decision_kind,
                        request_json, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        resolved_id,
                        item["tool_call_id"],
                        item["decision_kind"],
                        _json(item["request"]),
                        ActionState.pending.value,
                        _dt(now),
                    ),
                )
            connection.execute(
                """
                UPDATE logical_runs
                SET status = ?, pending_action_batch_id = ?, updated_at = ?
                WHERE logical_run_id = ?
                """,
                (
                    LogicalRunStatus.suspended.value,
                    resolved_id,
                    _dt(now),
                    logical_run_id,
                ),
            )
            connection.execute(
                "UPDATE executions SET status = ?, updated_at = ? WHERE execution_id = ?",
                (LogicalRunStatus.suspended.value, _dt(now), run.execution_id),
            )
            return self._action_batch(connection, resolved_id)

    def get_action_batch(self, batch_id: str) -> ActionBatch | None:
        with self._lock:
            exists = self._connection.execute("SELECT 1 FROM action_batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if exists is None:
                return None
            return self._action_batch(self._connection, batch_id)

    def decide_action(
        self,
        action_item_id: str,
        *,
        decision_id: str,
        decision: dict[str, JsonValue],
        actor: str | None = None,
    ) -> ActionBatch:
        now = utc_now()
        with self._write() as connection:
            row = self._required_row(connection, "action_items", "action_item_id", action_item_id)
            batch_row = self._required_row(connection, "action_batches", "batch_id", row["batch_id"])
            run = self._run_from_row(
                self._required_row(
                    connection,
                    "logical_runs",
                    "logical_run_id",
                    batch_row["logical_run_id"],
                )
            )
            self._ensure_active_session(self._required_session(connection, run.session_id))
            item = self._action_item_from_row(row)
            if item.state is not ActionState.pending:
                if item.decision_id == decision_id and item.decision == decision:
                    return self._action_batch(connection, item.batch_id)
                raise InvalidTransitionError(f"Action item {action_item_id!r} is already decided")
            reused = connection.execute(
                "SELECT action_item_id FROM action_items WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if reused is not None:
                raise ValueError("Decision ID is already used by another action item")
            connection.execute(
                """
                UPDATE action_items
                SET state = ?, decision_id = ?, decision_json = ?, actor = ?, decided_at = ?
                WHERE action_item_id = ?
                """,
                (
                    ActionState.resolved.value,
                    decision_id,
                    _json(decision),
                    actor,
                    _dt(now),
                    action_item_id,
                ),
            )
            pending = connection.execute(
                """
                SELECT COUNT(*) AS count FROM action_items
                WHERE batch_id = ? AND state = ?
                """,
                (item.batch_id, ActionState.pending.value),
            ).fetchone()["count"]
            if pending == 0:
                connection.execute(
                    """
                    UPDATE action_batches SET state = ?, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (ActionState.resolved.value, _dt(now), item.batch_id),
                )
                self._insert_outbox(
                    connection,
                    "notify_action",
                    run.execution_id,
                    {
                        "execution_id": run.execution_id,
                        "logical_run_id": run.logical_run_id,
                        "batch_id": item.batch_id,
                    },
                )
            else:
                connection.execute(
                    "UPDATE action_batches SET updated_at = ? WHERE batch_id = ?",
                    (_dt(now), item.batch_id),
                )
            return self._action_batch(connection, item.batch_id)

    def _required_session(self, connection: sqlite3.Connection, session_id: str) -> SessionRecord:
        return self._session_from_row(self._required_row(connection, "sessions", "session_id", session_id))

    @staticmethod
    def _ensure_active_session(session: SessionRecord) -> None:
        if session.status is SessionStatus.tombstoned:
            raise TombstonedSessionError(f"Session {session.session_id!r} is tombstoned")

    @staticmethod
    def _required_row(
        connection: sqlite3.Connection,
        table: str,
        key: str,
        value: str,
    ) -> sqlite3.Row:
        allowed = {
            ("sessions", "session_id"),
            ("logical_runs", "logical_run_id"),
            ("executions", "execution_id"),
            ("run_inputs", "input_id"),
            ("outbox_commands", "command_id"),
            ("revisions", "revision_id"),
            ("session_events", "event_id"),
            ("action_batches", "batch_id"),
            ("action_items", "action_item_id"),
        }
        if (table, key) not in allowed:  # pragma: no cover - internal invariant
            raise AssertionError(f"Unsafe internal row lookup: {table}.{key}")
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {key} = ?",  # noqa: S608 - allowlisted above
            (value,),
        ).fetchone()
        if row is None:
            raise KeyError(value)
        return row

    @staticmethod
    def _normalize_action_input(
        value: dict[str, JsonValue],
    ) -> dict[str, Any]:
        tool_call_id = value.get("tool_call_id")
        decision_kind = value.get("decision_kind")
        request = value.get("request")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Action item tool_call_id must be a non-empty string")
        if decision_kind not in {"approval", "external_result"}:
            raise ValueError("Action item decision_kind is invalid")
        if not isinstance(request, dict):
            raise TypeError("Action item request must be an object")
        return {
            "tool_call_id": tool_call_id,
            "decision_kind": decision_kind,
            "request": request,
        }

    def _action_batch(self, connection: sqlite3.Connection, batch_id: str) -> ActionBatch:
        row = self._required_row(connection, "action_batches", "batch_id", batch_id)
        item_rows = connection.execute(
            "SELECT * FROM action_items WHERE batch_id = ? ORDER BY created_at, action_item_id",
            (batch_id,),
        ).fetchall()
        return ActionBatch(
            batch_id=row["batch_id"],
            logical_run_id=row["logical_run_id"],
            state=ActionState(row["state"]),
            deadline_at=_parse_dt(row["deadline_at"]),
            items=tuple(self._action_item_from_row(item) for item in item_rows),
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
        )

    @staticmethod
    def _action_item_from_row(row: sqlite3.Row) -> ActionItem:
        return ActionItem(
            action_item_id=row["action_item_id"],
            batch_id=row["batch_id"],
            tool_call_id=row["tool_call_id"],
            decision_kind=row["decision_kind"],
            request=_object(row["request_json"]),
            state=ActionState(row["state"]),
            decision_id=row["decision_id"],
            decision=(_object(row["decision_json"]) if row["decision_json"] is not None else None),
            actor=row["actor"],
            created_at=_parse_required_dt(row["created_at"]),
            decided_at=_parse_dt(row["decided_at"]),
            consumed_at=_parse_dt(row["consumed_at"]),
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            workspace_ref=row["workspace_ref"],
            status=SessionStatus(row["status"]),
            head_revision_id=row["head_revision_id"],
            active_execution_id=row["active_execution_id"],
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
            tombstoned_at=_parse_dt(row["tombstoned_at"]),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> LogicalRunRecord:
        return LogicalRunRecord(
            logical_run_id=row["logical_run_id"],
            session_id=row["session_id"],
            execution_id=row["execution_id"],
            expected_head_revision_id=row["expected_head_revision_id"],
            descriptor_id=row["descriptor_id"],
            status=LogicalRunStatus(row["status"]),
            cancellation_reason=row["cancellation_reason"],
            pending_action_batch_id=row["pending_action_batch_id"],
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
        )

    @staticmethod
    def _execution_from_row(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=row["execution_id"],
            logical_run_id=row["logical_run_id"],
            executable_version=row["executable_version"],
            plan_fingerprint=row["plan_fingerprint"],
            descriptor_id=row["descriptor_id"],
            status=LogicalRunStatus(row["status"]),
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
        )

    @staticmethod
    def _execution_checkpoint_from_row(row: sqlite3.Row) -> ExecutionCheckpointRecord:
        return ExecutionCheckpointRecord(
            execution_id=row["execution_id"],
            logical_run_id=row["logical_run_id"],
            segment_index=row["segment_index"],
            segment_status=row["segment_status"],
            payload=RevisionPayload.model_validate_json(row["payload_json"]),
            deferred_requests=(
                _object(row["deferred_requests_json"]) if row["deferred_requests_json"] is not None else None
            ),
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
        )

    @staticmethod
    def _input_from_row(row: sqlite3.Row) -> InputRecord:
        return InputRecord(
            input_id=row["input_id"],
            logical_run_id=row["logical_run_id"],
            order_index=row["order_index"],
            idempotency_key=row["idempotency_key"],
            origin=row["origin"],
            priority=InputPriority(row["priority"]),
            content=_array(row["content_json"]),
            state=InputState(row["state"]),
            native_enqueue_id=row["native_enqueue_id"],
            rejection_reason=row["rejection_reason"],
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
        )

    @staticmethod
    def _outbox_from_row(row: sqlite3.Row) -> OutboxCommand:
        return OutboxCommand(
            command_id=row["command_id"],
            command_kind=row["command_kind"],
            aggregate_id=row["aggregate_id"],
            payload=_object(row["payload_json"]),
            state=OutboxState(row["state"]),
            attempt_count=row["attempt_count"],
            available_at=_parse_required_dt(row["available_at"]),
            claimed_at=_parse_dt(row["claimed_at"]),
            delivered_at=_parse_dt(row["delivered_at"]),
            last_error=row["last_error"],
            created_at=_parse_required_dt(row["created_at"]),
        )

    @staticmethod
    def _revision_from_row(row: sqlite3.Row) -> RevisionRecord:
        return RevisionRecord(
            revision_id=row["revision_id"],
            session_id=row["session_id"],
            logical_run_id=row["logical_run_id"],
            commit_kind=row["commit_kind"],
            parent_revision_id=row["parent_revision_id"],
            message_history=_array(row["message_history_json"]),
            resumable_state=_object(row["resumable_state_json"]),
            input_ledger=_object(row["input_ledger_json"]),
            display_projection=_array(row["display_projection_json"]),
            usage=_object(row["usage_json"]),
            terminal=_object(row["terminal_json"]),
            created_at=_parse_required_dt(row["created_at"]),
        )

    @staticmethod
    def _revision_payload(revision: RevisionRecord) -> RevisionPayload:
        return RevisionPayload(
            message_history=revision.message_history,
            resumable_state=revision.resumable_state,
            input_ledger=revision.input_ledger,
            display_projection=revision.display_projection,
            usage=revision.usage,
            terminal=revision.terminal,
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            session_id=row["session_id"],
            logical_run_id=row["logical_run_id"],
            sequence=row["sequence"],
            event_type=row["event_type"],
            payload=_object(row["payload_json"]),
            created_at=_parse_required_dt(row["created_at"]),
        )


def _dt(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_required_dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _parse_dt(value: str | None) -> datetime | None:
    return _parse_required_dt(value) if value is not None else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _array(value: str) -> list[Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):  # pragma: no cover - database invariant
        raise TypeError("Expected a JSON array in durable store")
    return decoded


def _object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):  # pragma: no cover - database invariant
        raise TypeError("Expected a JSON object in durable store")
    return decoded
