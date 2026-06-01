from __future__ import annotations

import json
from pathlib import Path

from yaacli.sessions import get_head_artifact_paths, list_sessions, save_session_turn


class DummyConfigManager:
    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = sessions_dir

    def get_sessions_dir(self) -> Path:
        return self._sessions_dir


def test_save_session_turn_retains_latest_turns_and_head_artifacts(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)

    for index in range(3):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "TEXT_MESSAGE_CHUNK", "messageId": f"m{index}", "delta": f"turn {index}"}],
            output_text=f"done {index}",
            save_reason="test",
            turn_id=f"turn-{index}",
            max_turns=2,
            max_sessions=10,
        )

    session_dir = sessions_dir / "session-1"
    turn_names = sorted(path.name for path in (session_dir / "turns").iterdir())
    assert turn_names == ["turn-1", "turn-2"]
    metadata = json.loads((session_dir / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["head_turn_id"] == "turn-2"

    paths = get_head_artifact_paths(manager, "session")  # type: ignore[arg-type]
    assert paths.turn_id == "turn-2"
    assert paths.display_messages_file is not None
    assert json.loads(paths.display_messages_file.read_text())[0]["delta"] == "turn 2"


def test_legacy_session_upgrades_and_removes_root_artifacts(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "legacy123456"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({"session_id": "legacy123456", "updated_at": "2026-01-01T00:00:00+00:00"})
    )
    (session_dir / "message_history.json").write_bytes(b"[]")
    (session_dir / "context_state.json").write_text("{}")
    (session_dir / "display_messages.json").write_text(json.dumps([{"type": "RUN_FINISHED"}]))

    manager = DummyConfigManager(sessions_dir)
    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert entries[0].id == "legacy123456"
    assert entries[0].turn_count == 1
    assert entries[0].head_turn_id is not None
    assert not (session_dir / "message_history.json").exists()
    assert not (session_dir / "context_state.json").exists()
    assert not (session_dir / "display_messages.json").exists()
    assert (session_dir / "turns" / entries[0].head_turn_id / "display_messages.json").exists()


def test_global_session_trim_uses_updated_at(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)

    for index in range(3):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id=f"session-{index}",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text=None,
            save_reason="test",
            turn_id="turn-1",
            max_turns=2,
            max_sessions=2,
        )

    remaining = sorted(path.name for path in sessions_dir.iterdir() if path.is_dir())
    assert remaining == ["session-1", "session-2"]
