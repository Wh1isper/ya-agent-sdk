from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaacli.sessions as sessions_module
from yaacli.sessions import local_file_lock, read_head_artifacts, save_session_turn


class DummyConfigManager:
    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = sessions_dir

    def get_sessions_dir(self) -> Path:
        return self._sessions_dir


def test_local_file_lock_creates_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "nested" / ".session.lock"

    with local_file_lock(lock_path):
        assert lock_path.exists()

    assert lock_path.exists()


def test_local_file_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "outside.lock"
    target.write_bytes(b"sentinel")
    lock_path = tmp_path / "nested" / ".session.lock"
    lock_path.parent.mkdir()
    try:
        lock_path.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    with pytest.raises(OSError, match="must not be a symbolic link"):
        with local_file_lock(lock_path):
            pytest.fail("symlink lock unexpectedly acquired")

    assert target.read_bytes() == b"sentinel"


def test_read_head_artifacts_returns_stable_bytes_during_concurrent_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committing writer cannot retain-delete the selected turn mid-read."""
    manager = DummyConfigManager(tmp_path / "sessions")
    old_history = b"[]"
    save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=old_history,
        context_state_json='{"turn": 0}',
        display_messages=[{"type": "RUN_FINISHED", "turn": 0}],
        output_text="old",
        save_reason="test",
        turn_id="turn-0",
        max_turns=1,
        max_sessions=10,
    )

    original_read = sessions_module._read_head_artifacts_unlocked
    snapshot_ready = threading.Event()
    release_reader = threading.Event()

    def paused_read(
        session_dir: Path,
        *,
        max_display_messages_bytes: int | None,
    ) -> sessions_module.SessionHeadArtifacts:
        snapshot = original_read(
            session_dir,
            max_display_messages_bytes=max_display_messages_bytes,
        )
        snapshot_ready.set()
        if not release_reader.wait(timeout=5):
            raise TimeoutError("test did not release snapshot reader")
        return snapshot

    monkeypatch.setattr(sessions_module, "_read_head_artifacts_unlocked", paused_read)
    reader_results: list[sessions_module.SessionHeadArtifacts] = []
    reader_errors: list[BaseException] = []

    def read_snapshot() -> None:
        try:
            reader_results.append(
                read_head_artifacts(
                    manager,  # type: ignore[arg-type]
                    "session-1",
                    max_display_messages_bytes=1024,
                )
            )
        except BaseException as exc:
            reader_errors.append(exc)

    reader = threading.Thread(target=read_snapshot, name="session-reader")
    reader.start()
    assert snapshot_ready.wait(timeout=5)

    original_create_turn = sessions_module._create_turn_directory
    writer_entered_session = threading.Event()

    def observed_create_turn(session_dir: Path, turn_id: str) -> Path:
        writer_entered_session.set()
        return original_create_turn(session_dir, turn_id)

    monkeypatch.setattr(sessions_module, "_create_turn_directory", observed_create_turn)
    writer_started = threading.Event()
    writer_errors: list[BaseException] = []

    def commit_new_head() -> None:
        writer_started.set()
        try:
            save_session_turn(
                config_manager=manager,  # type: ignore[arg-type]
                session_id="session-1",
                working_dir=tmp_path,
                message_history_json=b"[]",
                context_state_json='{"turn": 1}',
                display_messages=[{"type": "RUN_FINISHED", "turn": 1}],
                output_text="new",
                save_reason="test",
                turn_id="turn-1",
                max_turns=1,
                max_sessions=10,
            )
        except BaseException as exc:
            writer_errors.append(exc)

    writer = threading.Thread(target=commit_new_head, name="session-writer")
    writer.start()
    try:
        assert writer_started.wait(timeout=5)
        assert not writer_entered_session.wait(timeout=0.5), "writer bypassed the reader's global/session lock"
    finally:
        release_reader.set()

    reader.join(timeout=5)
    writer.join(timeout=5)
    assert not reader.is_alive()
    assert not writer.is_alive()
    assert reader_errors == []
    assert writer_errors == []

    snapshot = reader_results[0]
    assert snapshot.session_id == "session-1"
    assert snapshot.turn_id == "turn-0"
    assert snapshot.message_history_json == old_history
    assert json.loads(snapshot.context_state_json or b"null") == {"turn": 0}
    assert json.loads(snapshot.display_messages_json or b"null") == [{"type": "RUN_FINISHED", "turn": 0}]
    assert not (manager.get_sessions_dir() / "session-1" / "turns" / "turn-0").exists()
    assert read_head_artifacts(manager, "session-1").turn_id == "turn-1"  # type: ignore[arg-type]


def test_read_head_artifacts_bounds_display_without_affecting_required_state(tmp_path: Path) -> None:
    manager = DummyConfigManager(tmp_path / "sessions")
    save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json='{"restorable": true}',
        display_messages=[{"type": "TEXT_MESSAGE_CHUNK", "delta": "oversized"}],
        output_text=None,
        save_reason="test",
        turn_id="turn-0",
        max_turns=1,
        max_sessions=10,
    )

    snapshot = read_head_artifacts(
        manager,  # type: ignore[arg-type]
        "session-1",
        max_display_messages_bytes=1,
    )

    assert snapshot.message_history_json == b"[]"
    assert snapshot.context_state_json == b'{"restorable": true}'
    assert snapshot.display_messages_json is None


def test_read_head_artifacts_returns_none_for_missing_optional_artifacts(tmp_path: Path) -> None:
    manager = DummyConfigManager(tmp_path / "sessions")
    turn_dir = save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[],
        output_text=None,
        save_reason="test",
        turn_id="turn-0",
        max_turns=1,
        max_sessions=10,
    )
    (turn_dir / "context_state.json").unlink()
    (turn_dir / "display_messages.json").unlink()

    snapshot = read_head_artifacts(manager, "session-1")  # type: ignore[arg-type]

    assert snapshot.message_history_json == b"[]"
    assert snapshot.context_state_json is None
    assert snapshot.display_messages_json is None


def test_read_head_artifacts_upgrades_legacy_snapshot_under_lock(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "legacy-session"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": session_dir.name}))
    (session_dir / "message_history.json").write_bytes(b"[]")
    manager = DummyConfigManager(sessions_dir)

    snapshot = read_head_artifacts(manager, "legacy")  # type: ignore[arg-type]

    assert snapshot.session_id == session_dir.name
    assert snapshot.message_history_json == b"[]"
    assert snapshot.context_state_json == b"{}"
    assert snapshot.display_messages_json == b"[]"
    assert not (session_dir / "message_history.json").exists()
    assert (session_dir / "turns" / snapshot.turn_id / "message_history.json").is_file()


def test_read_head_artifacts_rejects_symlinked_head_artifact(tmp_path: Path) -> None:
    manager = DummyConfigManager(tmp_path / "sessions")
    turn_dir = save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[],
        output_text=None,
        save_reason="test",
        turn_id="turn-0",
        max_turns=1,
        max_sessions=10,
    )
    outside = tmp_path / "outside.json"
    outside.write_text('[{"type": "RUN_FINISHED"}]')
    display_file = turn_dir / "display_messages.json"
    display_file.unlink()
    try:
        display_file.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    with pytest.raises(FileNotFoundError, match="Session not found"):
        read_head_artifacts(manager, "session-1")  # type: ignore[arg-type]

    assert outside.read_text() == '[{"type": "RUN_FINISHED"}]'
