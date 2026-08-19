"""CLI coverage for the single durable YAACLI session store."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from yaacli.cli import cli
from yaacli.durable.application import SessionApplicationService, build_runtime_descriptor
from yaacli.durable.models import RevisionPayload
from yaacli.durable.sqlite import SQLiteSessionStore


def _seed_session(database_path: Path, *, session_id: str = "abc123def456") -> None:
    with SQLiteSessionStore(database_path) as store:
        service = SessionApplicationService(store)
        service.create_session("/workspace", session_id=session_id)
        service.import_snapshot(
            session_id,
            descriptor=build_runtime_descriptor(
                agent_spec={"name": "yaacli", "model": "test:model"},
                host_envelope={"model_profile_id": "test-profile"},
            ),
            payload=RevisionPayload(
                message_history=[{"kind": "request", "parts": []}],
                display_projection=[{"type": "RUN_FINISHED"}],
                terminal={"status": "completed", "output": "done"},
            ),
            source="test",
        )


def _manager(database_path: Path) -> MagicMock:
    manager = MagicMock()
    manager.get_session_database_path.return_value = database_path
    return manager


def test_cli_sessions_list_show_and_delete_use_sqlite_store(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "sessions.sqlite3"
    _seed_session(database_path)
    monkeypatch.setattr(
        "yaacli.cli._prepare_session_cli_runtime",
        MagicMock(return_value=_manager(database_path)),
    )
    runner = CliRunner()

    listed_result = runner.invoke(cli, ["sessions", "list", "--json"])
    assert listed_result.exit_code == 0
    listed = json.loads(listed_result.output)
    assert listed[0]["session_id"] == "abc123def456"
    assert listed[0]["workspace_ref"] == "/workspace"
    assert listed[0]["message_count"] == 1
    assert listed[0]["display_event_count"] == 1
    assert listed[0]["output_preview"] == "done"
    assert listed[0]["model"] == "test:model"
    assert "path" not in listed[0]

    shown_result = runner.invoke(cli, ["sessions", "show", "abc123", "--json"])
    assert shown_result.exit_code == 0
    shown = json.loads(shown_result.output)
    assert shown["session_id"] == "abc123def456"
    assert shown["model_profile_id"] == "test-profile"

    deleted_result = runner.invoke(cli, ["sessions", "delete", "abc123", "--yes"])
    assert deleted_result.exit_code == 0
    assert "Deleted session: abc123def456" in deleted_result.output

    empty_result = runner.invoke(cli, ["sessions", "list", "--json"])
    assert empty_result.exit_code == 0
    assert json.loads(empty_result.output) == []


def test_cli_sessions_rejects_ambiguous_prefix(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "sessions.sqlite3"
    with SQLiteSessionStore(database_path) as store:
        service = SessionApplicationService(store)
        service.create_session("/workspace", session_id="abc123aaaaaa")
        service.create_session("/workspace", session_id="abc123bbbbbb")
    monkeypatch.setattr(
        "yaacli.cli._prepare_session_cli_runtime",
        MagicMock(return_value=_manager(database_path)),
    )

    result = CliRunner().invoke(cli, ["sessions", "show", "abc"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Ambiguous session prefix" in str(result.exception)
