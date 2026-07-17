from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from yaacli.cli import _prepare_session_cli_runtime, cli
from yaacli.config import ConfigManager
from yaacli.sessions import delete_session, get_session_info, list_sessions, resolve_session_dir


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


def test_session_helpers_reject_symlinked_session_without_touching_target(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    history_file = outside / "message_history.json"
    history_file.write_text("[]")
    linked = sessions_dir / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    manager = DummyConfigManager(sessions_dir)

    assert list_sessions(manager) == []  # type: ignore[arg-type]
    assert history_file.read_text() == "[]"
    assert not (outside / "turns").exists()
    assert not (outside / "metadata.json").exists()
    with pytest.raises(FileNotFoundError):
        resolve_session_dir(manager, "linked")  # type: ignore[arg-type]


def test_session_helpers_reject_escaping_head_turn_id(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "escape123"
    turns_dir = session_dir / "turns"
    escaped_turn = session_dir / "outside-turn"
    turns_dir.mkdir(parents=True)
    escaped_turn.mkdir()
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": "escape123",
            "head_turn_id": "../outside-turn",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
    )
    manager = DummyConfigManager(sessions_dir)

    assert list_sessions(manager) == []  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        resolve_session_dir(manager, "escape123")  # type: ignore[arg-type]


def test_session_helpers_ignore_unrelated_directories(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "abc123")
    for name in ("skills", "subagents", "config-cache"):
        unknown = sessions_dir / name
        unknown.mkdir()
        (unknown / "keep.txt").write_text(name)
    manager = DummyConfigManager(sessions_dir)

    assert [entry.id for entry in list_sessions(manager)] == ["abc123"]  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        resolve_session_dir(manager, "skills")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        delete_session(manager, "subagents")  # type: ignore[arg-type]
    assert (sessions_dir / "skills" / "keep.txt").read_text() == "skills"
    assert (sessions_dir / "subagents" / "keep.txt").read_text() == "subagents"


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


def test_prepare_session_cli_runtime_loads_env_session_dir(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    env_sessions = tmp_path / "env-sessions"
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    monkeypatch.setenv("YAACLI_SESSION_DIR", str(env_sessions))
    monkeypatch.setattr("yaacli.cli.ConfigManager", MagicMock(return_value=manager))
    monkeypatch.setattr("yaacli.cli.load_package_env_files", MagicMock())

    prepared = _prepare_session_cli_runtime(verbose=False)

    assert prepared is manager
    assert prepared.get_sessions_dir() == env_sessions.resolve()


def test_cli_sessions_commands_use_configured_session_dir(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    configured_sessions = tmp_path / "custom-sessions"
    config_dir.mkdir()
    project_dir.mkdir()
    (config_dir / "config.toml").write_text(f"[session]\nsession_dir = {json.dumps(str(configured_sessions))}\n")
    _write_session(configured_sessions, "configured123")
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    monkeypatch.setattr("yaacli.cli.ConfigManager", MagicMock(return_value=manager))
    monkeypatch.setattr("yaacli.cli.load_package_env_files", MagicMock())
    runner = CliRunner()

    list_result = runner.invoke(cli, ["sessions", "list", "--json"])
    show_result = runner.invoke(cli, ["sessions", "show", "configured", "--json"])
    delete_result = runner.invoke(cli, ["sessions", "delete", "configured", "--yes"])

    assert list_result.exit_code == 0
    assert json.loads(list_result.output)[0]["id"] == "configured123"
    assert show_result.exit_code == 0
    assert json.loads(show_result.output)["id"] == "configured123"
    assert delete_result.exit_code == 0
    assert not (configured_sessions / "configured123").exists()
