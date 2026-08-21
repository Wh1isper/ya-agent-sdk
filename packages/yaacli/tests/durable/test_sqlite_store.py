from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from yaacli.durable import (
    ActionState,
    HeadConflictError,
    InputPriority,
    InputState,
    InvalidTransitionError,
    LogicalRunStatus,
    OutboxState,
    RevisionPayload,
    RuntimeDescriptor,
    SQLiteSessionStore,
    StartRunRequest,
    TombstonedSessionError,
)
from yaacli.durable import sqlite as sqlite_store_module
from yaacli.durable.application import build_runtime_descriptor


def test_store_bootstraps_only_a_truly_empty_database(tmp_path: Path) -> None:
    database_path = tmp_path / "nonempty.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(RuntimeError, match="exact durable product schema"):
        SQLiteSessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert names == {"unrelated"}


def test_store_rejects_legacy_schema_without_modifying_database(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        connection.executescript("""
            CREATE TABLE schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO schema_metadata(key, value) VALUES('schema_version', '1');
            CREATE TABLE executions (
                execution_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                application_version TEXT NOT NULL,
                workflow_version TEXT NOT NULL
            );
            INSERT INTO executions(
                execution_id, workflow_id, application_version, workflow_version
            ) VALUES('legacy-execution', 'legacy-workflow', '1', '1');
        """)
    assert journal_mode is not None and journal_mode[0] == "delete"
    original_bytes = database_path.read_bytes()

    with pytest.raises(RuntimeError, match="exact durable product schema"):
        SQLiteSessionStore(database_path)

    assert database_path.read_bytes() == original_bytes
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 1


def test_store_rejects_exact_objects_without_schema_marker(tmp_path: Path) -> None:
    database_path = tmp_path / "missing-marker.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(sqlite_store_module._SCHEMA)

    with pytest.raises(RuntimeError, match="schema marker None"):
        SQLiteSessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        marker = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
    assert marker is None


def test_store_rejects_same_columns_when_constraints_are_missing(tmp_path: Path) -> None:
    database_path = tmp_path / "missing-constraint.sqlite3"
    malformed_schema = sqlite_store_module._SCHEMA.replace(
        "    updated_at TEXT NOT NULL,\n    UNIQUE(session_id, idempotency_key)\n);",
        "    updated_at TEXT NOT NULL\n);",
        1,
    )
    assert malformed_schema != sqlite_store_module._SCHEMA
    with sqlite3.connect(database_path) as connection:
        connection.executescript(malformed_schema)
        connection.execute(
            "INSERT INTO schema_metadata(key, value) VALUES('schema_version', ?)",
            (str(sqlite_store_module._SCHEMA_VERSION),),
        )

    with pytest.raises(RuntimeError, match="definition mismatch for table:logical_runs"):
        SQLiteSessionStore(database_path)


def _descriptor(suffix: str = "one") -> RuntimeDescriptor:
    return build_runtime_descriptor(
        executable_version=f"executable-{suffix}",
        agent_spec={"name": "main", "model": "test"},
        host_envelope={"workspace": suffix},
    ).model_copy(update={"created_at": datetime(2026, 8, 18, tzinfo=UTC)})


def _request(
    session_id: str,
    *,
    descriptor: RuntimeDescriptor | None = None,
    idempotency_key: str = "turn-1",
    expected_head: str | None = None,
) -> StartRunRequest:
    resolved = descriptor or _descriptor()
    return StartRunRequest(
        session_id=session_id,
        expected_head_revision_id=expected_head,
        idempotency_key=idempotency_key,
        descriptor=resolved,
        initial_content=[{"kind": "text", "text": "hello"}],
        plan_fingerprint=resolved.plan_fingerprint,
        executable_version=resolved.executable_version,
    )


def test_session_catalog_and_descriptor_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    with SQLiteSessionStore(path) as store:
        created = store.create_session("/workspace", session_id="session-a")
        stored_descriptor = store.put_descriptor(_descriptor())
        repeated_descriptor = store.put_descriptor(
            _descriptor().model_copy(update={"created_at": datetime(2026, 8, 19, tzinfo=UTC)})
        )
        assert created.workspace_ref == "/workspace"
        assert stored_descriptor == _descriptor()
        assert repeated_descriptor == stored_descriptor

    with SQLiteSessionStore(path) as reopened:
        assert reopened.get_session("session-a") == created
        assert reopened.list_sessions() == (created,)
        assert reopened.get_descriptor(_descriptor().descriptor_id) == _descriptor()


def test_start_run_atomically_persists_execution_input_and_outbox(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        request = _request(session.session_id)

        run = store.start_run(request)
        repeated = store.start_run(request)
        with pytest.raises(ValueError, match="different intent"):
            store.start_run(request.model_copy(update={"initial_content": ["different prompt"]}))

        assert repeated == run
        execution = store.get_execution(run.execution_id)
        assert execution is not None
        assert execution.logical_run_id == run.logical_run_id
        assert execution.executable_version == request.executable_version
        assert store.get_session(session.session_id).active_execution_id == run.execution_id  # type: ignore[union-attr]

        inputs = store.list_inputs(run.logical_run_id)
        assert len(inputs) == 1
        assert inputs[0].order_index == 0
        assert inputs[0].state is InputState.accepted
        commands = store.claim_outbox()
        assert len(commands) == 1
        assert commands[0].command_kind == "start_execution"
        assert commands[0].state is OutboxState.delivering
        assert commands[0].payload["logical_run_id"] == run.logical_run_id


def test_start_run_rejects_stale_head_and_competing_execution(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        with pytest.raises(HeadConflictError):
            store.start_run(_request(session.session_id, expected_head="stale"))

        store.start_run(_request(session.session_id))
        with pytest.raises(InvalidTransitionError, match="active execution"):
            store.start_run(
                _request(
                    session.session_id,
                    descriptor=_descriptor("two"),
                    idempotency_key="turn-2",
                )
            )


def test_input_inbox_is_idempotent_ordered_and_notified(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        run = store.start_run(_request(session.session_id))
        store.claim_outbox()

        queued = store.accept_input(
            run.logical_run_id,
            [{"kind": "text", "text": "later"}],
            idempotency_key="input-later",
            priority=InputPriority.when_idle,
        )
        urgent = store.accept_input(
            run.logical_run_id,
            [{"kind": "text", "text": "now"}],
            idempotency_key="input-now",
            priority=InputPriority.asap,
        )
        repeated = store.accept_input(
            run.logical_run_id,
            [{"kind": "text", "text": "now"}],
            idempotency_key="input-now",
            priority=InputPriority.asap,
        )

        assert repeated == urgent
        assert [item.input_id for item in store.list_inputs(run.logical_run_id)] == [
            store.list_inputs(run.logical_run_id)[0].input_id,
            urgent.input_id,
            queued.input_id,
        ]
        enqueued = store.transition_input(
            urgent.input_id,
            InputState.accepted,
            InputState.enqueued,
            native_enqueue_id="native-1",
        )
        applied = store.transition_input(
            urgent.input_id,
            InputState.enqueued,
            InputState.applied,
        )
        assert enqueued.native_enqueue_id == "native-1"
        assert applied.state is InputState.applied
        assert len(store.claim_outbox()) == 2


def test_outbox_claim_complete_retry_and_idempotency(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        command = store.enqueue_command(
            "wake",
            "run-1",
            {"run_id": "run-1"},
            command_id="command-1",
        )
        assert store.enqueue_command("wake", "run-1", {"run_id": "run-1"}, command_id="command-1") == command

        claimed = store.claim_outbox()[0]
        assert claimed.attempt_count == 1
        retried = store.retry_outbox(claimed.command_id, "temporary")
        assert retried.state is OutboxState.pending
        assert retried.last_error == "temporary"

        second = store.enqueue_command("wake", "run-2", {"run_id": "run-2"})
        delivered = store.complete_outbox(store.claim_outbox()[0].command_id)
        assert delivered.command_id == second.command_id
        assert delivered.state is OutboxState.delivered
        assert store.complete_outbox(delivered.command_id) == delivered


def test_outbox_restart_releases_unfinished_delivery_claims(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        command = store.enqueue_command(
            "wake",
            "run-1",
            {"run_id": "run-1"},
        )
        claimed = store.claim_outbox()[0]
        assert claimed.command_id == command.command_id
        assert claimed.state is OutboxState.delivering

        assert store.recover_outbox() == 1
        reclaimed = store.claim_outbox()[0]
        assert reclaimed.command_id == command.command_id
        assert reclaimed.attempt_count == 2


def test_action_batch_survives_partial_idempotent_decisions(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        batch = store.create_action_batch(
            run.logical_run_id,
            [
                {
                    "tool_call_id": "call-1",
                    "decision_kind": "approval",
                    "request": {"tool": "shell"},
                },
                {
                    "tool_call_id": "call-2",
                    "decision_kind": "external_result",
                    "request": {"tool": "ask"},
                },
            ],
            batch_id="batch-1",
        )
        assert batch.state is ActionState.pending
        assert store.get_run(run.logical_run_id).status is LogicalRunStatus.suspended  # type: ignore[union-attr]

        partial = store.decide_action(
            batch.items[0].action_item_id,
            decision_id="decision-1",
            decision={"approved": True},
            actor="user",
        )
        repeated = store.decide_action(
            batch.items[0].action_item_id,
            decision_id="decision-1",
            decision={"approved": True},
            actor="user",
        )
        assert partial == repeated
        assert partial.state is ActionState.pending

        resolved = store.decide_action(
            batch.items[1].action_item_id,
            decision_id="decision-2",
            decision={"result": "answer"},
        )
        assert resolved.state is ActionState.resolved
        assert all(item.state is ActionState.resolved for item in resolved.items)
        assert any(command.command_kind == "notify_action" for command in store.claim_outbox())


def test_terminal_revision_and_event_are_atomic_and_idempotent(tmp_path: Path) -> None:
    payload = RevisionPayload(
        message_history=[{"kind": "request"}, {"kind": "response"}],
        resumable_state={"schema": 2},
        input_ledger={"applied": ["input-1"]},
        display_projection=[{"type": "RUN_FINISHED"}],
        usage={"requests": 1},
        terminal={"status": "completed", "output_text": "done"},
    )
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        initial = store.list_inputs(run.logical_run_id)[0]
        store.transition_input(
            initial.input_id,
            InputState.accepted,
            InputState.applied,
        )
        late = store.accept_input(
            run.logical_run_id,
            ["too late"],
            idempotency_key="late-input",
            priority=InputPriority.asap,
        )
        revision, event = store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=payload,
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )
        repeated_revision, repeated_event = store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=payload,
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )

        assert repeated_revision == revision
        assert repeated_event == event
        assert event.payload == {
            "revision_id": revision.revision_id,
            **payload.terminal,
        }
        assert store.read_events(session.session_id) == (event,)
        current = store.get_session(session.session_id)
        assert current is not None
        assert current.head_revision_id == revision.revision_id
        assert current.active_execution_id is None
        assert store.get_run(run.logical_run_id).status is LogicalRunStatus.completed  # type: ignore[union-attr]
        assert store.get_revision(revision.revision_id) == revision
        assert store.get_revision_for_run(run.logical_run_id) == revision
        rejected = next(item for item in store.list_inputs(run.logical_run_id) if item.input_id == late.input_id)
        assert rejected.state is InputState.rejected
        assert rejected.rejection_reason == ("run terminated as completed before input application")

        cancelled_race = store.start_run(
            _request(
                session.session_id,
                idempotency_key="cancel-race",
                expected_head=revision.revision_id,
            )
        )
        store.set_run_status(
            cancelled_race.logical_run_id,
            LogicalRunStatus.running,
        )
        store.set_run_status(
            cancelled_race.logical_run_id,
            LogicalRunStatus.cancelling,
        )
        with pytest.raises(
            InvalidTransitionError,
            match="cancelling to completed",
        ):
            store.commit_terminal(
                cancelled_race.logical_run_id,
                commit_kind="success",
                payload=payload,
                terminal_status=LogicalRunStatus.completed,
                event_type="RUN_FINISHED",
            )
        assert store.get_run(cancelled_race.logical_run_id).status is LogicalRunStatus.cancelling  # type: ignore[union-attr]


def test_events_are_ordered_idempotent_and_tombstone_fences_late_writes(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        first = store.append_event(session.session_id, "started", {"value": 1}, event_id="event-1")
        assert store.append_event(session.session_id, "started", {"value": 1}, event_id="event-1") == first
        second = store.append_event(session.session_id, "progress", {"value": 2}, event_id="event-2")
        assert [event.sequence for event in store.read_events(session.session_id)] == [1, 2]
        assert store.read_events(session.session_id, after_sequence=1) == (second,)

        tombstoned = store.tombstone_session(session.session_id)
        assert tombstoned.tombstoned_at is not None
        assert store.list_sessions() == ()
        with pytest.raises(TombstonedSessionError):
            store.append_event(session.session_id, "late", {}, event_id="event-3")
        with pytest.raises(TombstonedSessionError):
            store.start_run(_request(session.session_id))


def test_offline_import_publishes_revision_without_outbox_work(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    try:
        session = store.create_session("/workspace", session_id="import-session")
        descriptor = _descriptor("import-plan")
        revision = store.import_revision(
            session.session_id,
            descriptor=descriptor,
            payload=RevisionPayload(
                message_history=[{"kind": "request", "parts": []}],
                display_projection=[{"type": "RUN_STARTED"}],
                terminal={"status": "completed", "output": None},
            ),
            source="/offline/bundle",
        )

        assert store.get_session(session.session_id).head_revision_id == revision.revision_id  # type: ignore[union-attr]
        assert revision.commit_kind == "offline_import"
        assert revision.terminal["import_source"] == "/offline/bundle"
        assert store.claim_outbox() == ()
        run = store.get_run(revision.logical_run_id)
        assert run is not None
        assert run.status is LogicalRunStatus.completed
    finally:
        store.close()
