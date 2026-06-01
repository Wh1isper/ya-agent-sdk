from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from yaacli.cli import cli
from yaacli.sessions import delete_session, get_session_info, list_sessions


class DummyConfigManager:
    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = sessions_dir

    def get_sessions_dir(self) -> Path:
        return self._sessions_dir


def _write_session(root: Path, session_id: str, *, updated_at: str = "2026-01-01T00:00:00+00:00") -> Path:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "session_id": session_id,
            "working_dir": "/workspace",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": updated_at,
            "output_text": "done",
        })
    )
    turn_dir = session_dir / "turns" / "turn-1"
    turn_dir.mkdir(parents=True)
    (turn_dir / "message_history.json").write_text("[]")
    (turn_dir / "context_state.json").write_text("{}")
    (turn_dir / "display_messages.json").write_text(json.dumps([{"type": "RUN_FINISHED"}]))
    (turn_dir / "metadata.json").write_text(json.dumps({"turn_id": "turn-1", "updated_at": updated_at}))
    metadata = json.loads((session_dir / "metadata.json").read_text())
    metadata["schema_version"] = 2
    metadata["head_turn_id"] = "turn-1"
    (session_dir / "metadata.json").write_text(json.dumps(metadata))
    return session_dir


def test_session_helpers_list_show_delete(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "abc123")
    manager = DummyConfigManager(sessions_dir)

    entries = list_sessions(manager)  # type: ignore[arg-type]
    assert [entry.id for entry in entries] == ["abc123"]
    assert entries[0].message_count == 0
    assert entries[0].display_event_count == 1

    entry = get_session_info(manager, "abc")  # type: ignore[arg-type]
    assert entry.id == "abc123"

    deleted = delete_session(manager, "abc")  # type: ignore[arg-type]
    assert deleted.id == "abc123"
    assert not (sessions_dir / "abc123").exists()


def test_cli_sessions_list_show_delete(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "abc123")
    manager = DummyConfigManager(sessions_dir)

    monkeypatch.setattr("yaacli.cli._prepare_session_cli_runtime", MagicMock(return_value=manager))
    runner = CliRunner()

    list_result = runner.invoke(cli, ["sessions", "list", "--json"])
    assert list_result.exit_code == 0
    listed = json.loads(list_result.output)
    assert listed[0]["id"] == "abc123"
    assert listed[0]["display_event_count"] == 1

    show_result = runner.invoke(cli, ["sessions", "show", "abc", "--json"])
    assert show_result.exit_code == 0
    shown = json.loads(show_result.output)
    assert shown["id"] == "abc123"

    delete_result = runner.invoke(cli, ["sessions", "delete", "abc", "--yes"])
    assert delete_result.exit_code == 0
    assert "Deleted session: abc123" in delete_result.output
