from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from yaacli.durable import (
    ActionState,
    InputPriority,
    InputState,
    InvalidTransitionError,
    LogicalRunStatus,
    RevisionPayload,
    SessionStatus,
    SQLiteSessionStore,
    StartRunRequest,
    TombstonedSessionError,
)
from yaacli.durable import sqlite as sqlite_store_module


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


def _request(
    session_id: str,
    *,
    idempotency_key: str = "turn-1",
    expected_head: str | None = None,
    model: str | None = "test:model",
    model_profile_id: str | None = "test-profile",
) -> StartRunRequest:
    return StartRunRequest(
        session_id=session_id,
        expected_head_revision_id=expected_head,
        idempotency_key=idempotency_key,
        initial_content=["hello"],
        model=model,
        model_profile_id=model_profile_id,
    )


def test_session_catalog_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    with SQLiteSessionStore(path) as store:
        created = store.create_session("/workspace", session_id="session-a")
        assert created.workspace_ref == "/workspace"

    with SQLiteSessionStore(path) as reopened:
        assert reopened.get_session("session-a") == created
        assert reopened.list_sessions() == (created,)


def test_start_run_atomically_persists_execution_and_input(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        request = _request(session.session_id)

        run = store.start_run(request)
        repeated = store.start_run(request)
        with pytest.raises(ValueError, match="different intent"):
            store.start_run(request.model_copy(update={"initial_content": ["different prompt"]}))

        assert repeated == run
        assert run.model == "test:model"
        assert run.model_profile_id == "test-profile"
        execution = store.get_execution(run.execution_id)
        assert execution is not None
        assert execution.logical_run_id == run.logical_run_id
        assert execution.status is LogicalRunStatus.pending
        inputs = store.list_inputs(run.logical_run_id)
        assert len(inputs) == 1
        assert inputs[0].order_index == 0
        assert inputs[0].state is InputState.accepted


def test_start_run_allows_independent_process_owned_runs(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        first = store.start_run(_request(session.session_id))
        second = store.start_run(_request(session.session_id, idempotency_key="turn-2"))

        assert first.logical_run_id != second.logical_run_id
        assert first.execution_id != second.execution_id


def test_input_inbox_is_idempotent_and_priority_ordered(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        run = store.start_run(_request(session.session_id))

        queued = store.accept_input(
            run.logical_run_id,
            ["later"],
            idempotency_key="input-later",
            priority=InputPriority.when_idle,
        )
        urgent = store.accept_input(
            run.logical_run_id,
            ["now"],
            idempotency_key="input-now",
            priority=InputPriority.asap,
        )
        repeated = store.accept_input(
            run.logical_run_id,
            ["now"],
            idempotency_key="input-now",
            priority=InputPriority.asap,
        )

        assert repeated == urgent
        inputs = store.list_inputs(run.logical_run_id)
        assert [item.order_index for item in inputs] == [0, urgent.order_index, queued.order_index]
        enqueued = store.transition_input(
            urgent.input_id,
            InputState.accepted,
            InputState.enqueued,
            native_enqueue_id="native-1",
        )
        applied = store.transition_input(urgent.input_id, InputState.enqueued, InputState.applied)
        assert enqueued.native_enqueue_id == "native-1"
        assert applied.state is InputState.applied


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
        store.transition_input(initial.input_id, InputState.accepted, InputState.applied)
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
        assert event.payload == {"revision_id": revision.revision_id, **payload.terminal}
        assert store.read_events(session.session_id) == (event,)
        current = store.get_session(session.session_id)
        assert current is not None
        assert current.head_revision_id == revision.revision_id
        assert store.get_run(run.logical_run_id).status is LogicalRunStatus.completed  # type: ignore[union-attr]
        rejected = next(item for item in store.list_inputs(run.logical_run_id) if item.input_id == late.input_id)
        assert rejected.state is InputState.rejected
        assert rejected.rejection_reason == "run terminated as completed before input application"

        cancelling = store.start_run(
            _request(
                session.session_id,
                idempotency_key="cancel-race",
                expected_head=revision.revision_id,
            )
        )
        store.set_run_status(cancelling.logical_run_id, LogicalRunStatus.running)
        store.set_run_status(cancelling.logical_run_id, LogicalRunStatus.cancelling)
        with pytest.raises(InvalidTransitionError, match="cancelling to completed"):
            store.commit_terminal(
                cancelling.logical_run_id,
                commit_kind="success",
                payload=payload,
                terminal_status=LogicalRunStatus.completed,
                event_type="RUN_FINISHED",
            )


def test_terminal_retention_prunes_complete_run_bundles(tmp_path: Path) -> None:
    path = tmp_path / "retention.sqlite3"
    with SQLiteSessionStore(path, max_turns_per_session=2) as store:
        session = store.create_session("/workspace", session_id="retained-session")
        run_ids: list[str] = []
        revision_ids: list[str] = []
        expected_head: str | None = None
        for index in range(3):
            run = store.start_run(
                _request(
                    session.session_id,
                    idempotency_key=f"turn-{index}",
                    expected_head=expected_head,
                )
            )
            store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
            revision, _event = store.commit_terminal(
                run.logical_run_id,
                commit_kind="success",
                payload=RevisionPayload(
                    message_history=[{"turn": index}],
                    display_projection=[{"type": "RUN_FINISHED", "turn": index}],
                    terminal={"status": "completed", "output": f"answer-{index}"},
                ),
                terminal_status=LogicalRunStatus.completed,
                event_type="RUN_FINISHED",
            )
            run_ids.append(run.logical_run_id)
            revision_ids.append(revision.revision_id)
            expected_head = revision.revision_id

        assert store.get_run(run_ids[0]) is None
        assert store.get_revision(revision_ids[0]) is None
        assert store.get_run(run_ids[1]) is not None
        assert store.get_run(run_ids[2]) is not None
        assert store.get_session(session.session_id).head_revision_id == revision_ids[2]  # type: ignore[union-attr]
        assert [event.payload["output"] for event in store.read_events(session.session_id)] == [
            "answer-1",
            "answer-2",
        ]
        for table in (
            "logical_runs",
            "executions",
            "run_inputs",
            "revisions",
            "session_events",
            "execution_checkpoints",
            "action_batches",
        ):
            column = "logical_run_id"
            count = store._connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",  # noqa: S608
                (run_ids[0],),
            ).fetchone()[0]
            assert count == 0, table


def test_session_retention_tombstones_then_purges_on_next_open(tmp_path: Path) -> None:
    path = tmp_path / "session-retention.sqlite3"
    session_ids = ["session-a", "session-b", "session-c"]
    with SQLiteSessionStore(path, max_sessions=10) as store:
        for session_id in session_ids:
            store.create_session("/workspace", session_id=session_id)

    with SQLiteSessionStore(path, max_sessions=2) as store:
        active_ids = {session.session_id for session in store.list_sessions(limit=10)}
        tombstoned_ids = {
            session_id
            for session_id in session_ids
            if store.get_session(session_id) is not None and store.get_session(session_id).status.value == "tombstoned"  # type: ignore[union-attr]
        }
        assert len(active_ids) == 2
        assert len(tombstoned_ids) == 1
        purged_id = tombstoned_ids.pop()

    with SQLiteSessionStore(path, max_sessions=2) as store:
        assert store.get_session(purged_id) is None
        assert len(store.list_sessions(limit=10)) == 2


def test_age_retention_skips_session_with_nonterminal_main_run(tmp_path: Path) -> None:
    now = datetime(2026, 1, 31, tzinfo=UTC)
    stale_at = now - timedelta(days=31)
    with SQLiteSessionStore(
        tmp_path / "age-retention.sqlite3",
        max_session_age_days=30,
    ) as store:
        quiescent = store.create_session("/workspace", session_id="quiescent")
        running = store.create_session("/workspace", session_id="running")
        active_run = store.start_run(_request(running.session_id))
        store._connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id IN (?, ?)",
            (stale_at.isoformat(), quiescent.session_id, running.session_id),
        )

        result = store.run_maintenance(now=now)

        assert result.tombstoned_sessions == 1
        assert store.get_session(quiescent.session_id).status is SessionStatus.tombstoned  # type: ignore[union-attr]
        assert store.get_session(running.session_id).status is SessionStatus.active  # type: ignore[union-attr]
        assert store.get_run(active_run.logical_run_id) is not None


def test_tombstone_and_purge_refuse_nonterminal_main_run(tmp_path: Path) -> None:
    now = datetime(2026, 1, 31, tzinfo=UTC)
    with SQLiteSessionStore(tmp_path / "main-run-safety.sqlite3") as store:
        session = store.create_session("/workspace", session_id="running")
        run = store.start_run(_request(session.session_id))

        with pytest.raises(InvalidTransitionError, match="main run is nonterminal"):
            store.tombstone_session(session.session_id)

        store._connection.execute(
            "UPDATE sessions SET status = ?, tombstoned_at = ?, updated_at = ? WHERE session_id = ?",
            (SessionStatus.tombstoned.value, now.isoformat(), now.isoformat(), session.session_id),
        )
        result = store.run_maintenance(now=now)

        assert result.purged_sessions == 0
        assert store.get_session(session.session_id) is not None
        assert store.get_run(run.logical_run_id) is not None
        assert store.list_inputs(run.logical_run_id)


def test_vacuum_is_explicitly_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite_store_module, "_VACUUM_MIN_RECLAIM_BYTES", 0)
    with SQLiteSessionStore(tmp_path / "vacuum.sqlite3") as store:
        first = store.run_maintenance(force_vacuum=True)
        second = store.run_maintenance()

        assert first.vacuumed is True
        assert second.vacuumed is False
        marker = store._connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'retention_last_vacuum_at'"
        ).fetchone()
        assert marker is not None


def test_automatic_vacuum_defers_when_another_writer_holds_the_database(tmp_path: Path) -> None:
    path = tmp_path / "busy-vacuum.sqlite3"
    now = datetime(2026, 1, 31, tzinfo=UTC)
    with SQLiteSessionStore(path) as store, sqlite3.connect(path, isolation_level=None) as blocker:
        session = store.create_session("/workspace")
        blocker.execute("BEGIN IMMEDIATE")
        try:
            assert store._maybe_vacuum(now, force=True) is False
        finally:
            blocker.rollback()

        assert store.get_session(session.session_id) == session
        assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000


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


def test_offline_import_publishes_revision_with_model_metadata(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "sessions.sqlite3") as store:
        session = store.create_session("/workspace", session_id="import-session")
        revision = store.import_revision(
            session.session_id,
            payload=RevisionPayload(
                message_history=[{"kind": "request", "parts": []}],
                display_projection=[{"type": "RUN_STARTED"}],
                terminal={"status": "completed", "output": None},
            ),
            source="/offline/bundle",
            model="test:model",
            model_profile_id="test-profile",
        )

        assert store.get_session(session.session_id).head_revision_id == revision.revision_id  # type: ignore[union-attr]
        assert revision.commit_kind == "offline_import"
        assert revision.terminal["import_source"] == "/offline/bundle"
        run = store.get_run(revision.logical_run_id)
        assert run is not None
        assert run.status is LogicalRunStatus.completed
        assert run.model == "test:model"
        assert run.model_profile_id == "test-profile"
