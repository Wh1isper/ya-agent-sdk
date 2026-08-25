"""SQLite implementation of the YAACLI durable product store."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

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
    RevisionPayload,
    RevisionRecord,
    SessionRecord,
    SessionStatus,
    SessionSummary,
    StartRunRequest,
    utc_now,
)
from yaacli.durable.sqlite_schema import (
    SESSION_SCHEMA_VERSION,
    user_schema_object_names,
    validate_exact_schema_subset,
)
from yaacli.durable.state_files import (
    CheckpointStateFile,
    RevisionStateFile,
    SessionStateFiles,
    exclusive_file_lock,
)
from yaacli.durable.store import (
    InvalidTransitionError,
    TombstonedSessionError,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = SESSION_SCHEMA_VERSION
_LEGACY_RESET_SCHEMA_VERSIONS = frozenset({"5"})
_LEGACY_SCHEMA_V5_FINGERPRINT = "cdc841c9a9b51a4ddf01ab91f66e0f5df01562c53b30a87252edb67afb4ad951"
_DEFAULT_MAX_TURNS_PER_SESSION = 20
_DEFAULT_MAX_SESSIONS = 100
_VACUUM_MIN_INTERVAL = timedelta(days=7)
_VACUUM_MIN_RECLAIM_BYTES = 8 * 1024 * 1024
_VACUUM_BUSY_TIMEOUT_MS = 100
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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tombstoned_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS logical_runs (
    logical_run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    execution_id TEXT NOT NULL UNIQUE,
    expected_head_revision_id TEXT,
    model TEXT,
    model_profile_id TEXT,
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
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_checkpoints (
    execution_id TEXT PRIMARY KEY REFERENCES executions(execution_id),
    logical_run_id TEXT NOT NULL UNIQUE REFERENCES logical_runs(logical_run_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
    segment_status TEXT NOT NULL CHECK(segment_status IN ('completed', 'suspended')),
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

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    commit_kind TEXT NOT NULL,
    parent_revision_id TEXT,
    message_count INTEGER NOT NULL CHECK(message_count >= 0),
    display_event_count INTEGER NOT NULL CHECK(display_event_count >= 0),
    input_preview TEXT,
    output_preview TEXT,
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


@dataclass(frozen=True, slots=True)
class StoreMaintenanceResult:
    """Observable result of one bounded retention and SQLite maintenance pass."""

    pruned_turns: int = 0
    purged_sessions: int = 0
    tombstoned_sessions: int = 0
    removed_orphan_files: int = 0
    vacuumed: bool = False


class SQLiteSessionStore:
    """Transactional SQLite product truth for sessions and logical runs."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        max_turns_per_session: int = _DEFAULT_MAX_TURNS_PER_SESSION,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        max_session_age_days: int | None = None,
    ) -> None:
        if max_turns_per_session <= 0:
            raise ValueError("max_turns_per_session must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_session_age_days is not None and max_session_age_days <= 0:
            raise ValueError("max_session_age_days must be positive when configured")
        path = Path(database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.state_files = SessionStateFiles(path)
        self.max_turns_per_session = max_turns_per_session
        self.max_sessions = max_sessions
        self.max_session_age_days = max_session_age_days
        self._lock = threading.RLock()
        cutover_lock_path = path.with_name(f".{path.name}.cutover.lock")
        with exclusive_file_lock(cutover_lock_path):
            _reset_disposable_legacy_database(path)
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
        try:
            maintenance = self.run_maintenance()
            if maintenance != StoreMaintenanceResult():
                logger.info("SQLite maintenance completed: %s", maintenance)
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

    def run_maintenance(
        self,
        *,
        now: datetime | None = None,
        force_vacuum: bool = False,
    ) -> StoreMaintenanceResult:
        """Apply bounded retention, orphan cleanup, WAL checkpointing, and optional vacuum."""
        maintenance_time = now or utc_now()
        purged_sessions = self._purge_tombstoned_sessions()
        with self._write() as connection:
            pruned_turns = self._prune_all_session_turns(connection)
        tombstoned_sessions = self._tombstone_expired_sessions(maintenance_time)
        removed_orphan_files = self._cleanup_orphan_state_files()
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            vacuumed = self._maybe_vacuum(maintenance_time, force=force_vacuum)
        return StoreMaintenanceResult(
            pruned_turns=pruned_turns,
            purged_sessions=purged_sessions,
            tombstoned_sessions=tombstoned_sessions,
            removed_orphan_files=removed_orphan_files,
            vacuumed=vacuumed,
        )

    def _cleanup_orphan_state_files(self) -> int:
        removed = 0
        for session_id in self.state_files.list_managed_session_ids():
            if not self.state_files.session_dir(session_id).exists():
                continue
            try:
                with self.state_files.session_lock(session_id):
                    with self._lock:
                        session_exists = (
                            self._connection.execute(
                                "SELECT 1 FROM sessions WHERE session_id = ?",
                                (session_id,),
                            ).fetchone()
                            is not None
                        )
                        revision_ids = {
                            str(row["revision_id"])
                            for row in self._connection.execute(
                                "SELECT revision_id FROM revisions WHERE session_id = ?",
                                (session_id,),
                            )
                        }
                        checkpoint_ids = {
                            str(row["execution_id"])
                            for row in self._connection.execute(
                                "SELECT execution_id FROM execution_checkpoints WHERE session_id = ?",
                                (session_id,),
                            )
                        }
                    removed += self.state_files.remove_session_orphans(
                        session_id,
                        revision_ids=revision_ids,
                        checkpoint_ids=checkpoint_ids,
                        session_exists=session_exists,
                    )
            except OSError as exc:
                logger.warning(
                    "Deferred durable state-file cleanup for session %s: %s",
                    session_id,
                    exc,
                )
        return removed

    def _prune_all_session_turns(self, connection: sqlite3.Connection) -> int:
        session_rows = connection.execute(
            "SELECT session_id FROM sessions WHERE status = ?",
            (SessionStatus.active.value,),
        ).fetchall()
        return sum(self._prune_session_turns(connection, row["session_id"]) for row in session_rows)

    def _prune_session_turns(self, connection: sqlite3.Connection, session_id: str) -> int:
        session = self._required_session(connection, session_id)
        rows = connection.execute(
            """
            SELECT lr.logical_run_id, r.revision_id
            FROM logical_runs AS lr
            JOIN revisions AS r ON r.logical_run_id = lr.logical_run_id
            WHERE lr.session_id = ?
              AND lr.status IN ('completed', 'failed', 'cancelled', 'interrupted')
            ORDER BY r.created_at DESC, r.revision_id DESC
            """,
            (session_id,),
        ).fetchall()
        retained_run_ids = {row["logical_run_id"] for row in rows[: self.max_turns_per_session]}
        if session.head_revision_id is not None:
            retained_run_ids.update(
                row["logical_run_id"] for row in rows if row["revision_id"] == session.head_revision_id
            )
        candidates = [row["logical_run_id"] for row in rows if row["logical_run_id"] not in retained_run_ids]
        self._delete_run_bundles(connection, candidates)
        return len(candidates)

    @staticmethod
    def _delete_run_bundles(connection: sqlite3.Connection, logical_run_ids: Sequence[str]) -> None:
        for logical_run_id in logical_run_ids:
            connection.execute(
                "DELETE FROM action_items WHERE batch_id IN "
                "(SELECT batch_id FROM action_batches WHERE logical_run_id = ?)",
                (logical_run_id,),
            )
            connection.execute("DELETE FROM action_batches WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM session_events WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM execution_checkpoints WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM run_inputs WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM revisions WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM executions WHERE logical_run_id = ?", (logical_run_id,))
            connection.execute("DELETE FROM logical_runs WHERE logical_run_id = ?", (logical_run_id,))

    def _purge_tombstoned_sessions(self) -> int:
        with self._lock:
            rows = self._connection.execute(
                "SELECT session_id FROM sessions WHERE status = ? ORDER BY tombstoned_at, session_id",
                (SessionStatus.tombstoned.value,),
            ).fetchall()
        purged = 0
        now = utc_now()
        for row in rows:
            session_id = str(row["session_id"])
            with self.state_files.session_lock(session_id), self._write() as connection:
                current = connection.execute(
                    "SELECT status FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if current is None or current["status"] != SessionStatus.tombstoned.value:
                    continue
                self.state_files.fence_subagents(
                    session_id,
                    reason="owner session tombstoned",
                    now=now,
                )
                if not self._session_is_quiescent(connection, session_id):
                    continue
                run_rows = connection.execute(
                    "SELECT logical_run_id FROM logical_runs WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                self._delete_run_bundles(
                    connection,
                    [run["logical_run_id"] for run in run_rows],
                )
                connection.execute(
                    "DELETE FROM session_events WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                purged += 1
        return purged

    def _tombstone_expired_sessions(self, now: datetime) -> int:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM sessions
                WHERE status = ?
                ORDER BY updated_at DESC, session_id
                """,
                (SessionStatus.active.value,),
            ).fetchall()
        excess_ids = {row["session_id"] for row in rows[self.max_sessions :]}
        cutoff = now - timedelta(days=self.max_session_age_days) if self.max_session_age_days is not None else None
        if cutoff is not None:
            excess_ids.update(row["session_id"] for row in rows if _parse_required_dt(row["updated_at"]) < cutoff)

        tombstoned = 0
        for row in reversed(rows):
            session_id = str(row["session_id"])
            if session_id not in excess_ids:
                continue
            with self.state_files.session_lock(session_id), self._write() as connection:
                current = self._required_session(connection, session_id)
                if current.status is not SessionStatus.active or not self._session_is_quiescent(connection, session_id):
                    continue
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, tombstoned_at = ?, updated_at = ?
                    WHERE session_id = ? AND status = ?
                    """,
                    (
                        SessionStatus.tombstoned.value,
                        _dt(now),
                        _dt(now),
                        session_id,
                        SessionStatus.active.value,
                    ),
                )
                tombstoned += 1
        return tombstoned

    @staticmethod
    def _session_has_nonterminal_main_run(connection: sqlite3.Connection, session_id: str) -> bool:
        row = connection.execute(
            """
            SELECT 1 FROM logical_runs
            WHERE session_id = ?
              AND status NOT IN ('completed', 'failed', 'cancelled', 'interrupted')
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return row is not None

    def _session_is_quiescent(self, connection: sqlite3.Connection, session_id: str) -> bool:
        return not self._session_has_nonterminal_main_run(
            connection, session_id
        ) and not self.state_files.session_has_nonterminal_subagents(session_id)

    def _maybe_vacuum(self, now: datetime, *, force: bool) -> bool:
        page_size = int(self._connection.execute("PRAGMA page_size").fetchone()[0])
        free_pages = int(self._connection.execute("PRAGMA freelist_count").fetchone()[0])
        if not force and free_pages * page_size < _VACUUM_MIN_RECLAIM_BYTES:
            return False
        row = self._connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'retention_last_vacuum_attempt_at'"
        ).fetchone()
        if not force and row is not None:
            last_attempt = datetime.fromisoformat(row["value"])
            if now - last_attempt < _VACUUM_MIN_INTERVAL:
                return False

        previous_timeout_ms = int(self._connection.execute("PRAGMA busy_timeout").fetchone()[0])
        try:
            self._connection.execute(f"PRAGMA busy_timeout = {_VACUUM_BUSY_TIMEOUT_MS}")
            self._connection.execute(
                """
                INSERT INTO schema_metadata(key, value) VALUES('retention_last_vacuum_attempt_at', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (_dt(now),),
            )
            self._connection.execute("VACUUM")
            self._connection.execute(
                """
                INSERT INTO schema_metadata(key, value) VALUES('retention_last_vacuum_at', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (_dt(now),),
            )
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode & 0xFF not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            logger.info("SQLite VACUUM deferred because the database is busy")
            return False
        finally:
            self._connection.execute(f"PRAGMA busy_timeout = {previous_timeout_ms}")
        return True

    def create_session(
        self,
        workspace_ref: str,
        *,
        session_id: str | None = None,
    ) -> SessionRecord:
        resolved_id = session_id or uuid.uuid4().hex[:12]
        self.state_files.session_dir(resolved_id)
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

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT s.*, r.input_preview, r.output_preview,
                       COALESCE(r.message_count, 0) AS message_count,
                       COALESCE(r.display_event_count, 0) AS display_event_count,
                       lr.model, lr.model_profile_id
                FROM sessions AS s
                LEFT JOIN revisions AS r ON r.revision_id = s.head_revision_id
                LEFT JOIN logical_runs AS lr ON lr.logical_run_id = r.logical_run_id
                WHERE s.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionSummary(
            session_id=row["session_id"],
            workspace_ref=row["workspace_ref"],
            status=SessionStatus(row["status"]),
            head_revision_id=row["head_revision_id"],
            created_at=_parse_required_dt(row["created_at"]),
            updated_at=_parse_required_dt(row["updated_at"]),
            input_preview=row["input_preview"],
            output_preview=row["output_preview"],
            message_count=row["message_count"],
            display_event_count=row["display_event_count"],
            model=row["model"],
            model_profile_id=row["model_profile_id"],
        )

    def tombstone_session(self, session_id: str) -> SessionRecord:
        now = utc_now()
        with self.state_files.session_lock(session_id):
            with self._write() as connection:
                row = self._required_row(connection, "sessions", "session_id", session_id)
                session = self._session_from_row(row)
                if session.status is SessionStatus.active:
                    if self._session_has_nonterminal_main_run(connection, session_id):
                        raise InvalidTransitionError(
                            f"Cannot tombstone session {session_id!r} while a main run is nonterminal"
                        )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = ?, tombstoned_at = ?, updated_at = ?
                        WHERE session_id = ?
                        """,
                        (
                            SessionStatus.tombstoned.value,
                            _dt(now),
                            _dt(now),
                            session_id,
                        ),
                    )
                    tombstoned = self._session_from_row(
                        self._required_row(connection, "sessions", "session_id", session_id)
                    )
                else:
                    tombstoned = session
            self.state_files.fence_subagents(
                session_id,
                reason="owner session tombstoned",
                now=now,
            )
            return tombstoned

    def start_run(self, request: StartRunRequest) -> LogicalRunRecord:
        now = utc_now()
        with self._write() as connection:
            session_row = self._required_row(connection, "sessions", "session_id", request.session_id)
            session = self._session_from_row(session_row)
            self._ensure_active_session(session)
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
                    or record.model != request.model
                    or record.model_profile_id != request.model_profile_id
                    or initial_content != list(request.initial_content)
                ):
                    raise ValueError("Start-run idempotency key was reused with different intent")
                return record
            logical_run_id = str(uuid.uuid4())
            execution_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO logical_runs(
                    logical_run_id, session_id, execution_id,
                    expected_head_revision_id, model, model_profile_id, idempotency_key,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    logical_run_id,
                    request.session_id,
                    execution_id,
                    request.expected_head_revision_id,
                    request.model,
                    request.model_profile_id,
                    request.idempotency_key,
                    LogicalRunStatus.pending.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            connection.execute(
                """
                INSERT INTO executions(
                    execution_id, logical_run_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    logical_run_id,
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
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_dt(now), request.session_id),
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
        with self._lock:
            row = self._connection.execute(
                """
                SELECT e.logical_run_id, lr.session_id
                FROM executions AS e
                JOIN logical_runs AS lr ON lr.logical_run_id = e.logical_run_id
                WHERE e.execution_id = ?
                """,
                (checkpoint.execution_id,),
            ).fetchone()
        if row is None:
            raise KeyError(checkpoint.execution_id)
        if row["logical_run_id"] != checkpoint.logical_run_id:
            raise ValueError("Execution checkpoint does not match its logical run")
        session_id = str(row["session_id"])
        with self.state_files.session_lock(session_id):
            session = self.get_session(session_id)
            if session is None:
                raise KeyError(session_id)
            self._ensure_active_session(session)
            existing = self.get_execution_checkpoint(checkpoint.execution_id)
            if existing is not None:
                if existing.segment_index > checkpoint.segment_index:
                    raise InvalidTransitionError("Execution checkpoint segment index cannot move backwards")
                if existing.segment_index == checkpoint.segment_index:
                    comparable_existing = existing.model_copy(
                        update={"created_at": checkpoint.created_at, "updated_at": checkpoint.updated_at}
                    )
                    if comparable_existing != checkpoint:
                        raise ValueError("Execution checkpoint segment was reused with different content")
                    return existing
            stored_checkpoint = (
                checkpoint if existing is None else checkpoint.model_copy(update={"created_at": existing.created_at})
            )
            self.state_files.write_checkpoint(
                session_id,
                CheckpointStateFile(
                    checkpoint=stored_checkpoint,
                    previous_checkpoint=existing,
                ),
            )
            with self._write() as connection:
                self._ensure_active_session(self._required_session(connection, session_id))
                created_at = _dt(existing.created_at) if existing is not None else _dt(checkpoint.created_at)
                connection.execute(
                    """
                    INSERT INTO execution_checkpoints(
                        execution_id, logical_run_id, session_id, segment_index,
                        segment_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(execution_id) DO UPDATE SET
                        segment_index = excluded.segment_index,
                        segment_status = excluded.segment_status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        checkpoint.execution_id,
                        checkpoint.logical_run_id,
                        session_id,
                        stored_checkpoint.segment_index,
                        stored_checkpoint.segment_status,
                        created_at,
                        _dt(stored_checkpoint.updated_at),
                    ),
                )
            try:
                self.state_files.write_checkpoint(
                    session_id,
                    CheckpointStateFile(checkpoint=stored_checkpoint),
                )
            except OSError as exc:
                logger.warning(
                    "Deferred committed checkpoint compaction for execution %s: %s",
                    stored_checkpoint.execution_id,
                    exc,
                )
            return stored_checkpoint.model_copy(deep=True)

    def get_execution_checkpoint(self, execution_id: str) -> ExecutionCheckpointRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM execution_checkpoints WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            return None
        state = self.state_files.read_checkpoint(str(row["session_id"]), execution_id)
        for checkpoint in (state.checkpoint, state.previous_checkpoint):
            if checkpoint is not None and self._checkpoint_matches_row(checkpoint, row):
                return checkpoint
        raise RuntimeError(f"Checkpoint file metadata mismatch for {execution_id!r}")

    @staticmethod
    def _checkpoint_matches_row(
        checkpoint: ExecutionCheckpointRecord,
        row: sqlite3.Row,
    ) -> bool:
        return (
            checkpoint.execution_id == row["execution_id"]
            and checkpoint.logical_run_id == row["logical_run_id"]
            and checkpoint.segment_index == row["segment_index"]
            and checkpoint.segment_status == row["segment_status"]
            and _dt(checkpoint.created_at) == row["created_at"]
            and _dt(checkpoint.updated_at) == row["updated_at"]
        )

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
                    updated_at = ?
                WHERE logical_run_id = ?
                """,
                (
                    status.value,
                    pending_action_batch_id,
                    cancellation_reason,
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
            terminal_won = run.terminal or run.status is LogicalRunStatus.cancelling
            input_state = InputState.rejected if terminal_won else InputState.accepted
            rejection_reason = f"logical run is already {run.status.value}" if terminal_won else None
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
                    origin, priority, content_json, state, rejection_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_id,
                    logical_run_id,
                    next_order,
                    idempotency_key,
                    origin,
                    priority.value,
                    _json(normalized_content),
                    input_state.value,
                    rejection_reason,
                    _dt(now),
                    _dt(now),
                ),
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

    def list_pending_inputs(self, logical_run_id: str) -> tuple[InputRecord, ...]:
        """Return the unresolved input snapshot at a serialized graph boundary."""
        with self._write() as connection:
            run_row = self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id)
            run = self._run_from_row(run_row)
            self._ensure_active_session(self._required_session(connection, run.session_id))
            if run.terminal or run.status is LogicalRunStatus.cancelling:
                return ()
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
            return tuple(self._input_from_row(row) for row in rows)

    def import_revision(
        self,
        session_id: str,
        *,
        payload: RevisionPayload,
        source: str,
        model: str | None = None,
        model_profile_id: str | None = None,
    ) -> RevisionRecord:
        """Import an offline snapshot without creating executable outbox work."""
        now = utc_now()
        logical_run_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        with self.state_files.session_lock(session_id):
            with self._lock:
                row = self._connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if row is None:
                raise KeyError(session_id)
            session = self._session_from_row(row)
            self._ensure_active_session(session)
            terminal = dict(payload.terminal)
            terminal.setdefault("status", LogicalRunStatus.completed.value)
            terminal["import_source"] = source
            revision = RevisionRecord(
                revision_id=revision_id,
                session_id=session_id,
                logical_run_id=logical_run_id,
                commit_kind="offline_import",
                parent_revision_id=session.head_revision_id,
                message_history=payload.message_history,
                resumable_state=payload.resumable_state,
                input_ledger=payload.input_ledger,
                display_projection=payload.display_projection,
                usage=payload.usage,
                terminal=terminal,
                created_at=now,
            )
            self.state_files.write_revision(RevisionStateFile(revision=revision))
            with self._write() as connection:
                current = self._required_session(connection, session_id)
                self._ensure_active_session(current)
                if current.head_revision_id != session.head_revision_id:
                    raise InvalidTransitionError("Session head changed during offline revision import")
                connection.execute(
                    """
                    INSERT INTO logical_runs(
                        logical_run_id, session_id, execution_id,
                        expected_head_revision_id, model, model_profile_id, idempotency_key,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        logical_run_id,
                        session_id,
                        execution_id,
                        session.head_revision_id,
                        model,
                        model_profile_id,
                        f"offline-import:{revision_id}",
                        LogicalRunStatus.completed.value,
                        _dt(now),
                        _dt(now),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO executions(
                        execution_id, logical_run_id, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        execution_id,
                        logical_run_id,
                        LogicalRunStatus.completed.value,
                        _dt(now),
                        _dt(now),
                    ),
                )
                self._insert_revision_metadata(connection, revision, input_preview=None)
                connection.execute(
                    "UPDATE sessions SET head_revision_id = ?, updated_at = ? WHERE session_id = ?",
                    (revision_id, _dt(now), session_id),
                )
                self._prune_session_turns(connection, session_id)
            self._cleanup_orphan_state_files()
            return revision

    def commit_revision(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
    ) -> RevisionRecord:
        revision, _event = self._commit_revision_document(
            logical_run_id,
            commit_kind=commit_kind,
            payload=payload,
            terminal_status=terminal_status,
            event_type=None,
        )
        return revision

    def commit_terminal(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
        event_type: str,
    ) -> tuple[RevisionRecord, EventRecord]:
        revision, event = self._commit_revision_document(
            logical_run_id,
            commit_kind=commit_kind,
            payload=payload,
            terminal_status=terminal_status,
            event_type=event_type,
        )
        if event is None:  # pragma: no cover - caller invariant
            raise RuntimeError("Terminal commit did not publish its canonical event")
        return revision, event

    def _commit_revision_document(
        self,
        logical_run_id: str,
        *,
        commit_kind: str,
        payload: RevisionPayload,
        terminal_status: LogicalRunStatus,
        event_type: str | None,
    ) -> tuple[RevisionRecord, EventRecord | None]:
        if terminal_status not in _TERMINAL_RUN_STATES:
            raise ValueError("A committed revision requires a terminal run status")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM logical_runs WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(logical_run_id)
        initial_run = self._run_from_row(row)
        now = utc_now()
        with self.state_files.session_lock(initial_run.session_id):
            session = self.get_session(initial_run.session_id)
            if session is None:
                raise KeyError(initial_run.session_id)
            self._ensure_active_session(session)
            with self._lock:
                existing_row = self._connection.execute(
                    """
                    SELECT * FROM revisions
                    WHERE logical_run_id = ? AND commit_kind = ?
                    """,
                    (logical_run_id, commit_kind),
                ).fetchone()
            if existing_row is not None:
                revision = self._revision_from_row(existing_row)
                if self._revision_payload(revision) != payload:
                    raise ValueError("Terminal revision idempotency key has different payload")
            else:
                revision = RevisionRecord(
                    revision_id=str(uuid.uuid4()),
                    session_id=initial_run.session_id,
                    logical_run_id=logical_run_id,
                    commit_kind=commit_kind,
                    parent_revision_id=initial_run.expected_head_revision_id,
                    message_history=payload.message_history,
                    resumable_state=payload.resumable_state,
                    input_ledger=payload.input_ledger,
                    display_projection=payload.display_projection,
                    usage=payload.usage,
                    terminal=payload.terminal,
                    created_at=now,
                )
                self.state_files.write_revision(RevisionStateFile(revision=revision))

            with self._write() as connection:
                run = self._run_from_row(
                    self._required_row(connection, "logical_runs", "logical_run_id", logical_run_id)
                )
                session = self._required_session(connection, run.session_id)
                self._ensure_active_session(session)
                published = connection.execute(
                    """
                    SELECT revision_id FROM revisions
                    WHERE logical_run_id = ? AND commit_kind = ?
                    """,
                    (logical_run_id, commit_kind),
                ).fetchone()
                if published is None:
                    if terminal_status not in _RUN_TRANSITIONS[run.status]:
                        raise InvalidTransitionError(
                            f"Cannot transition run from {run.status.value} to {terminal_status.value}"
                        )
                    input_row = connection.execute(
                        "SELECT content_json FROM run_inputs WHERE logical_run_id = ? AND order_index = 0",
                        (logical_run_id,),
                    ).fetchone()
                    input_preview = (
                        _preview_json_values(_array(input_row["content_json"])) if input_row is not None else None
                    )
                    self._insert_revision_metadata(
                        connection,
                        revision,
                        input_preview=input_preview,
                    )
                    cursor = connection.execute(
                        """
                        UPDATE sessions
                        SET head_revision_id = ?, updated_at = ?
                        WHERE session_id = ? AND status = ?
                        """,
                        (
                            revision.revision_id,
                            _dt(now),
                            run.session_id,
                            SessionStatus.active.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise TombstonedSessionError(f"Session {run.session_id!r} is tombstoned")
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
                        SET status = ?, pending_action_batch_id = NULL,
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
                elif published["revision_id"] != revision.revision_id:
                    raise RuntimeError("Revision publication disagrees with its state file")
                connection.execute(
                    "DELETE FROM execution_checkpoints WHERE logical_run_id = ?",
                    (logical_run_id,),
                )
                event = (
                    self._append_event(
                        connection,
                        revision.session_id,
                        event_type,
                        {"revision_id": revision.revision_id, **payload.terminal},
                        event_id=f"terminal:{logical_run_id}",
                        logical_run_id=logical_run_id,
                        now=now,
                    )
                    if event_type is not None
                    else None
                )
                self._prune_session_turns(connection, run.session_id)
            try:
                self.state_files.remove_checkpoint(run.session_id, run.execution_id)
            except OSError as exc:
                logger.warning("Deferred checkpoint-file cleanup after filesystem error: %s", exc)
            self._cleanup_orphan_state_files()
            return revision, event

    def _insert_revision_metadata(
        self,
        connection: sqlite3.Connection,
        revision: RevisionRecord,
        *,
        input_preview: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO revisions(
                revision_id, session_id, logical_run_id, commit_kind,
                parent_revision_id, message_count, display_event_count,
                input_preview, output_preview, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                revision.session_id,
                revision.logical_run_id,
                revision.commit_kind,
                revision.parent_revision_id,
                len(revision.message_history),
                len(revision.display_projection),
                input_preview,
                _preview_json_value(revision.terminal.get("output")),
                _dt(revision.created_at),
            ),
        )

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
            model=row["model"],
            model_profile_id=row["model_profile_id"],
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
            status=LogicalRunStatus(row["status"]),
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

    def _revision_from_row(self, row: sqlite3.Row) -> RevisionRecord:
        state = self.state_files.read_revision(str(row["session_id"]), str(row["revision_id"]))
        revision = state.revision
        if (
            revision.logical_run_id != row["logical_run_id"]
            or revision.commit_kind != row["commit_kind"]
            or revision.parent_revision_id != row["parent_revision_id"]
            or len(revision.message_history) != row["message_count"]
            or len(revision.display_projection) != row["display_event_count"]
        ):
            raise RuntimeError(f"Revision file metadata mismatch for {revision.revision_id!r}")
        return revision

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


def _reset_disposable_legacy_database(path: Path) -> None:
    """Replace the disposable schema-v5 store at the explicit v6 cutover boundary."""
    if not path.exists():
        return
    try:
        with closing(sqlite3.connect(path)) as connection:
            marker_table = connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_metadata'"
            ).fetchone()
            if marker_table is None:
                return
            marker = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
            schema_fingerprint = _schema_fingerprint(connection)
    except sqlite3.DatabaseError:
        return
    if (
        marker is None
        or str(marker[0]) not in _LEGACY_RESET_SCHEMA_VERSIONS
        or schema_fingerprint != _LEGACY_SCHEMA_V5_FINGERPRINT
    ):
        return
    logger.warning(
        "Resetting disposable YAACLI schema-v%s data at %s for the file-state cutover",
        marker[0],
        path,
    )
    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        candidate.unlink(missing_ok=True)


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_schema
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY type, name
        """
    ).fetchall()
    payload = "\n".join(f"{row[0]}:{row[1]}:{' '.join(str(row[2]).strip().removesuffix(';').split())}" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _preview_json_values(values: Sequence[JsonValue]) -> str | None:
    if len(values) == 1:
        return _preview_json_value(values[0])
    return _bounded_preview(json.dumps(list(values), ensure_ascii=False))


def _preview_json_value(value: JsonValue | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _bounded_preview(value)
    return _bounded_preview(json.dumps(value, ensure_ascii=False))


def _bounded_preview(value: str, *, limit: int = 2000) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


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
