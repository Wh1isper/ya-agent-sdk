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
from yaacli.durable.application import SessionApplicationService
from yaacli.durable.models import ExecutionCheckpointRecord


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


def test_terminal_and_input_writes_use_transaction_order_without_ingress_fence(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "store.sqlite3") as store:
        session = store.create_session("/workspace")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        initial = store.list_inputs(run.logical_run_id)[0]
        store.transition_input(initial.input_id, InputState.accepted, InputState.applied)

        assert store.list_pending_inputs(run.logical_run_id) == ()
        before_terminal = store.accept_input(
            run.logical_run_id,
            ["late guidance"],
            idempotency_key="before-terminal",
            priority=InputPriority.asap,
        )
        assert before_terminal.state is InputState.accepted

        store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=RevisionPayload(terminal={"status": "completed", "output": "done"}),
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )
        terminal_winner = next(
            item for item in store.list_inputs(run.logical_run_id) if item.input_id == before_terminal.input_id
        )
        assert terminal_winner.state is InputState.rejected

        after_terminal = store.accept_input(
            run.logical_run_id,
            ["too late"],
            idempotency_key="after-terminal",
            priority=InputPriority.asap,
        )
        assert after_terminal.state is InputState.rejected
        assert after_terminal.rejection_reason == "logical run is already completed"


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


def test_session_retention_tombstones_then_purges_on_explicit_maintenance(tmp_path: Path) -> None:
    path = tmp_path / "session-retention.sqlite3"
    session_ids = ["session-a", "session-b", "session-c"]
    with SQLiteSessionStore(path, max_sessions=10) as store:
        for session_id in session_ids:
            store.create_session("/workspace", session_id=session_id)

    with SQLiteSessionStore(path, max_sessions=2) as store:
        assert store.run_maintenance().tombstoned_sessions == 1
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
        assert store.run_maintenance().purged_sessions == 1
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


def test_schema_v5_store_is_rejected_without_runtime_reset(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions-v2.sqlite3"
    legacy_schema = (Path(__file__).parents[1] / "fixtures" / "session_schema_v5.sql").read_text(encoding="utf-8")
    with sqlite3.connect(database_path) as connection:
        connection.executescript(legacy_schema)
        connection.execute(
            """
            INSERT INTO sessions(
                session_id, workspace_ref, status, created_at, updated_at
            ) VALUES('preserve-me', '/workspace', 'active', ?, ?)
            """,
            (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat()),
        )

    with pytest.raises(RuntimeError, match="exact durable product schema"):
        SQLiteSessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        marker = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
        assert marker == ("5",)
        assert connection.execute("SELECT session_id FROM sessions").fetchall() == [("preserve-me",)]


def test_sqlite_schema_contains_only_revision_and_checkpoint_metadata(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata-only.sqlite3"
    with SQLiteSessionStore(database_path):
        pass

    with sqlite3.connect(database_path) as connection:
        revision_columns = {row[1] for row in connection.execute("PRAGMA table_info(revisions)")}
        checkpoint_columns = {row[1] for row in connection.execute("PRAGMA table_info(execution_checkpoints)")}
        table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}

    assert {
        "message_history_json",
        "resumable_state_json",
        "input_ledger_json",
        "display_projection_json",
        "usage_json",
        "terminal_json",
    }.isdisjoint(revision_columns)
    assert {"payload_json", "deferred_requests_json"}.isdisjoint(checkpoint_columns)
    assert not any(name.startswith("subagent_") for name in table_names)


def test_revision_and_checkpoint_payloads_are_pretty_grepable_files(tmp_path: Path) -> None:
    database_path = tmp_path / "grepable.sqlite3"
    payload = RevisionPayload(
        message_history=[{"kind": "request", "content": "你好 durable history"}],
        resumable_state={"notes": {"scope": "grep-me"}},
        display_projection=[{"type": "RUN_STARTED"}],
        terminal={"status": "completed", "output": "完成"},
    )
    now = datetime.now(UTC)
    with SQLiteSessionStore(database_path) as store:
        session = store.create_session("/workspace", session_id="grep-session")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        checkpoint = ExecutionCheckpointRecord(
            execution_id=run.execution_id,
            logical_run_id=run.logical_run_id,
            segment_index=0,
            segment_status="completed",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        store.put_execution_checkpoint(checkpoint)
        checkpoint_path = store.state_files.checkpoint_path(
            session.session_id,
            run.execution_id,
        )
        checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
        assert "你好 durable history" in checkpoint_text
        assert '\n  "checkpoint"' in checkpoint_text
        assert store.get_execution_checkpoint(run.execution_id) == checkpoint

        revision, _event = store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=payload,
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )
        revision_path = store.state_files.revision_path(
            session.session_id,
            revision.revision_id,
        )
        revision_text = revision_path.read_text(encoding="utf-8")
        assert "你好 durable history" in revision_text
        assert "grep-me" in revision_text
        assert "完成" in revision_text
        assert checkpoint_path.exists() is False
        assert store.get_revision(revision.revision_id) == revision


def test_failed_revision_metadata_publish_leaves_only_cleanup_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "file-first.sqlite3"
    with SQLiteSessionStore(database_path) as store:
        session = store.create_session("/workspace", session_id="file-first")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)

        def fail_metadata_publish(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected metadata failure")

        monkeypatch.setattr(store, "_insert_revision_metadata", fail_metadata_publish)
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            store.commit_terminal(
                run.logical_run_id,
                commit_kind="success",
                payload=RevisionPayload(message_history=[{"orphan": "searchable"}]),
                terminal_status=LogicalRunStatus.completed,
                event_type="RUN_FINISHED",
            )

        orphan_paths = tuple((tmp_path / session.session_id / "revisions").glob("*/state.json"))
        assert len(orphan_paths) == 1
        assert store.get_revision_for_run(run.logical_run_id) is None
        result = store.run_maintenance()
        assert result.removed_orphan_files == 1
        assert orphan_paths[0].exists() is False


def test_malformed_schema_v5_store_is_rejected_without_reset(tmp_path: Path) -> None:
    database_path = tmp_path / "malformed-v5.sqlite3"
    legacy_schema = (Path(__file__).parents[1] / "fixtures" / "session_schema_v5.sql").read_text(encoding="utf-8")
    with sqlite3.connect(database_path) as connection:
        connection.executescript(legacy_schema)
        connection.execute("DROP INDEX sessions_updated_idx")

    with pytest.raises(RuntimeError, match="exact durable product schema"):
        SQLiteSessionStore(database_path)

    with sqlite3.connect(database_path) as connection:
        marker = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
        assert marker == ("5",)
        assert connection.execute("SELECT 1 FROM sqlite_schema WHERE name = 'sessions_updated_idx'").fetchone() is None


def test_orphan_cleanup_never_claims_unmarked_sibling_directories(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated" / "checkpoints"
    unrelated.mkdir(parents=True)
    important = unrelated / "important.txt"
    important.write_text("must survive", encoding="utf-8")
    empty_unrelated = tmp_path / "empty-unrelated"
    empty_unrelated.mkdir()

    with SQLiteSessionStore(tmp_path / "sessions.sqlite3") as store:
        with pytest.raises(KeyError, match="empty-unrelated"):
            store.tombstone_session("empty-unrelated")
        store.run_maintenance()

    assert important.read_text(encoding="utf-8") == "must survive"
    assert empty_unrelated.is_dir()
    assert tuple(empty_unrelated.iterdir()) == ()


def test_failed_checkpoint_metadata_update_reads_previous_committed_segment(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "checkpoint-failure.sqlite3"
    now = datetime.now(UTC)
    with SQLiteSessionStore(database_path) as store:
        session = store.create_session("/workspace", session_id="checkpoint-session")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        first = ExecutionCheckpointRecord(
            execution_id=run.execution_id,
            logical_run_id=run.logical_run_id,
            segment_index=0,
            segment_status="completed",
            payload=RevisionPayload(message_history=[{"segment": 0}]),
            created_at=now,
            updated_at=now,
        )
        assert store.put_execution_checkpoint(first) == first
        second = ExecutionCheckpointRecord(
            execution_id=run.execution_id,
            logical_run_id=run.logical_run_id,
            segment_index=1,
            segment_status="suspended",
            payload=RevisionPayload(message_history=[{"segment": 1}]),
            deferred_requests={"approvals": []},
            created_at=now + timedelta(seconds=1),
            updated_at=now + timedelta(seconds=1),
        )
        store._connection.execute(
            """
            CREATE TRIGGER fail_checkpoint_update
            BEFORE UPDATE ON execution_checkpoints
            BEGIN
                SELECT RAISE(ABORT, 'injected checkpoint metadata failure');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="injected checkpoint metadata failure"):
            store.put_execution_checkpoint(second)

        assert store.get_execution_checkpoint(run.execution_id) == first
        staged = store.state_files.read_checkpoint(session.session_id, run.execution_id)
        assert staged.checkpoint.segment_index == 1
        assert staged.previous_checkpoint == first

        store._connection.execute("DROP TRIGGER fail_checkpoint_update")
        committed = store.put_execution_checkpoint(second)
        assert committed.segment_index == 1
        assert committed.created_at == first.created_at
        compacted = store.state_files.read_checkpoint(session.session_id, run.execution_id)
        assert compacted.checkpoint == committed
        assert compacted.previous_checkpoint is None


def test_session_summary_reads_sqlite_metadata_without_loading_revision_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "metadata-summary.sqlite3"
    with SQLiteSessionStore(database_path) as store:
        session = store.create_session("/workspace", session_id="summary-session")
        run = store.start_run(_request(session.session_id))
        store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        revision, _event = store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=RevisionPayload(
                message_history=[{"kind": "request"}, {"kind": "response"}],
                display_projection=[{"type": "RUN_FINISHED"}],
                terminal={"status": "completed", "output": "metadata answer"},
            ),
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )

        def reject_revision_read(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("session summary loaded the revision state file")

        monkeypatch.setattr(store.state_files, "read_revision", reject_revision_read)
        summary = SessionApplicationService(store).get_session_summary(session.session_id)

        assert summary.head_revision_id == revision.revision_id
        assert summary.input_preview == "hello"
        assert summary.output_preview == "metadata answer"
        assert summary.message_count == 2
        assert summary.display_event_count == 1
