from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaacli.sessions as sessions_module
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
            "input_text": "build the feature",
            "output_text": "done",
        })
    )
    turn_dir = session_dir / "turns" / "turn-1"
    turn_dir.mkdir(parents=True)
    (turn_dir / "message_history.json").write_text("[]")
    (turn_dir / "context_state.json").write_text("{}")
    (turn_dir / "display_messages.json").write_text(json.dumps([{"type": "RUN_FINISHED"}]))
    (turn_dir / "metadata.json").write_text(
        json.dumps({
            "turn_id": "turn-1",
            "updated_at": updated_at,
            "input_text": "build the feature",
            "output_text": "done",
            "message_count": 0,
            "display_event_count": 1,
        })
    )
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
    assert entries[0].input_text == "build the feature"
    assert entries[0].output_text == "done"
    assert entries[0].message_count == 0
    assert entries[0].display_event_count == 1

    entry = get_session_info(manager, "abc")  # type: ignore[arg-type]
    assert entry.id == "abc123"

    deleted = delete_session(manager, "abc")  # type: ignore[arg-type]
    assert deleted.id == "abc123"
    assert not (sessions_dir / "abc123").exists()


def test_list_sessions_reads_counts_from_metadata_without_opening_large_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = _write_session(sessions_dir, "abc123")
    manager = DummyConfigManager(sessions_dir)
    artifact_names = {"message_history.json", "context_state.json", "display_messages.json"}
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name in artifact_names:
            raise AssertionError(f"session listing read artifact: {path.name}")
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name in artifact_names:
            raise AssertionError(f"session listing read artifact: {path.name}")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert session_dir.is_dir()
    assert entries[0].input_text == "build the feature"
    assert entries[0].output_text == "done"
    assert entries[0].message_count == 0
    assert entries[0].display_event_count == 1


def test_list_sessions_defers_legacy_artifact_reads_and_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "legacy123"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "session_id": session_dir.name,
            "working_dir": "/legacy-workspace",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
    )
    (session_dir / "message_history.json").write_text("[]")
    (session_dir / "context_state.json").write_text("{}")
    (session_dir / "display_messages.json").write_text("[]")
    manager = DummyConfigManager(sessions_dir)
    artifact_names = {"message_history.json", "context_state.json", "display_messages.json"}
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name in artifact_names:
            raise AssertionError(f"session listing read artifact: {path.name}")
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name in artifact_names:
            raise AssertionError(f"session listing read artifact: {path.name}")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert [entry.id for entry in entries] == [session_dir.name]
    assert entries[0].working_dir == "/legacy-workspace"
    assert entries[0].head_turn_id is None
    assert entries[0].turn_count == 0
    assert not (session_dir / "turns").exists()


def test_list_sessions_uses_one_root_metadata_snapshot_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = _write_session(sessions_dir, "abc123")
    manager = DummyConfigManager(sessions_dir)
    root_metadata_file = session_dir / "metadata.json"
    root_reads = 0
    original_read = sessions_module._read_json_object_bounded

    def count_root_reads(path: Path, *, max_bytes: int) -> dict[str, Any] | None:
        nonlocal root_reads
        if path == root_metadata_file:
            root_reads += 1
        return original_read(path, max_bytes=max_bytes)

    monkeypatch.setattr(sessions_module, "_read_json_object_bounded", count_root_reads)

    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert [entry.id for entry in entries] == ["abc123"]
    assert root_reads == 1


def test_session_index_does_not_fallback_to_a_different_head_than_root_snapshot(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = _write_session(sessions_dir, "abc123")
    root_metadata = json.loads((session_dir / "metadata.json").read_text())
    old_turn = session_dir / "turns" / "turn-1"
    for artifact in old_turn.iterdir():
        artifact.unlink()
    old_turn.rmdir()
    new_turn = session_dir / "turns" / "turn-2"
    new_turn.mkdir()
    (new_turn / "message_history.json").write_text("[]")
    (new_turn / "metadata.json").write_text(
        json.dumps({
            "turn_id": "turn-2",
            "message_count": 99,
            "display_event_count": 101,
        })
    )

    info = sessions_module._read_session_info(session_dir, metadata=root_metadata)

    assert info.head_turn_id == "turn-1"
    assert info.message_count is None
    assert info.display_event_count is None


def test_list_sessions_does_not_wait_for_artifact_writer_global_lock(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "abc123")
    manager = DummyConfigManager(sessions_dir)
    writer_holds_lock = threading.Event()
    release_writer = threading.Event()
    reader_finished = threading.Event()
    result: list[str] = []

    def hold_writer_lock() -> None:
        with sessions_module.local_file_lock(sessions_dir / ".sessions.lock"):
            writer_holds_lock.set()
            release_writer.wait(timeout=5)

    def read_index() -> None:
        result.extend(entry.id for entry in list_sessions(manager))  # type: ignore[arg-type]
        reader_finished.set()

    writer = threading.Thread(target=hold_writer_lock)
    reader = threading.Thread(target=read_index)
    writer.start()
    assert writer_holds_lock.wait(timeout=2)
    reader.start()
    try:
        assert reader_finished.wait(timeout=1), "session index waited for the artifact writer lock"
    finally:
        release_writer.set()
        writer.join(timeout=2)
        reader.join(timeout=2)

    assert result == ["abc123"]


def test_list_sessions_bounds_previews_from_older_unbounded_metadata(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = _write_session(sessions_dir, "abc123")
    metadata_file = session_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["input_text"] = "i" * 5000
    metadata["output_text"] = "o" * 5000
    metadata_file.write_text(json.dumps(metadata))
    manager = DummyConfigManager(sessions_dir)

    entry = list_sessions(manager)[0]  # type: ignore[arg-type]

    assert entry.input_text is not None and len(entry.input_text) == 2000
    assert entry.output_text is not None and len(entry.output_text) == 2000
    assert entry.input_text.endswith("...")
    assert entry.output_text.endswith("...")
    assert len(entry.metadata["input_text"]) == 2000
    assert len(entry.metadata["output_text"]) == 2000


def test_list_sessions_skips_older_oversized_metadata_index_payload(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = _write_session(sessions_dir, "abc123")
    metadata_file = session_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["output_text"] = "o" * (300 * 1024)
    metadata_file.write_text(json.dumps(metadata))
    manager = DummyConfigManager(sessions_dir)

    assert list_sessions(manager) == []  # type: ignore[arg-type]

    explicit = get_session_info(manager, "abc123")  # type: ignore[arg-type]
    assert explicit.output_text is not None
    assert len(explicit.output_text) == 2000


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
    assert listed[0]["input_text"] == "build the feature"
    assert listed[0]["output_text"] == "done"
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
