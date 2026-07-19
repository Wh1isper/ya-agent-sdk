from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaacli.sessions as sessions_module
from yaacli.sessions import get_head_artifact_paths, list_sessions, save_session_turn, trim_session_turns


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


def test_save_session_turn_persists_bounded_input_and_output_previews(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    turn_dir = save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-preview",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[],
        input_text=f"  input line\n{'i' * 2500}  ",
        output_text=f"  output line\n{'o' * 2500}  ",
        save_reason="test",
        max_sessions=10,
    )

    root_metadata = json.loads((turn_dir.parents[1] / "metadata.json").read_text())
    turn_metadata = json.loads((turn_dir / "metadata.json").read_text())
    for metadata in (root_metadata, turn_metadata):
        assert metadata["input_text"].startswith("input line\n")
        assert metadata["output_text"].startswith("output line\n")
        assert metadata["input_text"].endswith("...")
        assert metadata["output_text"].endswith("...")
        assert len(metadata["input_text"]) <= 2000
        assert len(metadata["output_text"]) <= 2000

    info = list_sessions(manager)[0]  # type: ignore[arg-type]
    assert info.input_text == root_metadata["input_text"]
    assert info.output_text == root_metadata["output_text"]


def test_failed_initial_save_does_not_poison_session_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed first write must leave no unrecognized artifacts behind."""
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    original_write_text_atomic = sessions_module._write_text_atomic
    failed = False

    def fail_first_context_write(path: Path, content: str) -> None:
        nonlocal failed
        if path.name == "context_state.json" and not failed:
            failed = True
            raise OSError("injected write failure")
        original_write_text_atomic(path, content)

    monkeypatch.setattr(sessions_module, "_write_text_atomic", fail_first_context_write)
    save_kwargs: dict[str, Any] = {
        "config_manager": manager,
        "session_id": "retryable-session",
        "working_dir": tmp_path,
        "message_history_json": b"[]",
        "context_state_json": "{}",
        "display_messages": [{"type": "RUN_FINISHED"}],
        "output_text": "done",
        "save_reason": "test",
        "turn_id": "turn-1",
        "max_turns": 2,
        "max_sessions": 10,
    }

    with pytest.raises(OSError, match="injected write failure"):
        save_session_turn(**save_kwargs)

    assert not (sessions_dir / "retryable-session").exists()

    turn_dir = save_session_turn(**save_kwargs)

    assert turn_dir.is_dir()
    assert [entry.id for entry in list_sessions(manager)] == ["retryable-session"]  # type: ignore[arg-type]


def test_failed_artifact_write_removes_only_new_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial turn must be removed without disturbing the committed head."""
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[{"type": "RUN_FINISHED"}],
        output_text="committed",
        save_reason="test",
        turn_id="turn-0",
        max_turns=2,
        max_sessions=10,
    )
    session_dir = sessions_dir / "session-1"
    root_metadata = (session_dir / "metadata.json").read_bytes()
    original_write_text_atomic = sessions_module._write_text_atomic

    def fail_display_write(path: Path, content: str) -> None:
        if path.name == "display_messages.json":
            raise OSError("injected artifact failure")
        original_write_text_atomic(path, content)

    monkeypatch.setattr(sessions_module, "_write_text_atomic", fail_display_write)

    with pytest.raises(OSError, match="injected artifact failure"):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text="not committed",
            save_reason="test",
            turn_id="turn-failed",
            max_turns=2,
            max_sessions=10,
        )

    assert (session_dir / "metadata.json").read_bytes() == root_metadata
    assert (session_dir / "turns" / "turn-0").is_dir()
    assert not (session_dir / "turns" / "turn-failed").exists()
    assert get_head_artifact_paths(manager, "session-1").turn_id == "turn-0"  # type: ignore[arg-type]


def test_failed_root_commit_does_not_consume_turn_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete but unpublished turn must be deleted before later retention."""
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    for turn_id in ("turn-0", "turn-1"):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text=turn_id,
            save_reason="test",
            turn_id=turn_id,
            max_turns=10,
            max_sessions=10,
        )

    session_dir = sessions_dir / "session-1"
    root_metadata_file = session_dir / "metadata.json"
    committed_metadata = root_metadata_file.read_bytes()
    original_write_text_atomic = sessions_module._write_text_atomic
    failed = False

    def fail_root_metadata_write(path: Path, content: str) -> None:
        nonlocal failed
        if path == root_metadata_file and not failed:
            failed = True
            raise OSError("injected root metadata failure")
        original_write_text_atomic(path, content)

    monkeypatch.setattr(sessions_module, "_write_text_atomic", fail_root_metadata_write)

    with pytest.raises(OSError, match="injected root metadata failure"):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text="not committed",
            save_reason="test",
            turn_id="turn-failed",
            max_turns=2,
            max_sessions=10,
        )

    assert root_metadata_file.read_bytes() == committed_metadata
    assert not (session_dir / "turns" / "turn-failed").exists()
    assert get_head_artifact_paths(manager, "session-1").turn_id == "turn-1"  # type: ignore[arg-type]

    save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[{"type": "RUN_FINISHED"}],
        output_text="turn-2",
        save_reason="test",
        turn_id="turn-2",
        max_turns=2,
        max_sessions=10,
    )

    assert sorted(path.name for path in (session_dir / "turns").iterdir()) == ["turn-1", "turn-2"]
    assert get_head_artifact_paths(manager, "session-1").turn_id == "turn-2"  # type: ignore[arg-type]


def test_existing_explicit_turn_id_is_never_partially_overwritten(tmp_path: Path) -> None:
    """Reusing an explicit ID must preserve all artifacts and the committed head."""
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    turn_dir = save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id="session-1",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json='{"original": true}',
        display_messages=[{"type": "RUN_FINISHED"}],
        output_text="original",
        save_reason="test",
        turn_id="turn-0",
        max_turns=2,
        max_sessions=10,
    )
    session_dir = sessions_dir / "session-1"
    original_artifacts = {name: (turn_dir / name).read_bytes() for name in sessions_module.TURN_ARTIFACT_NAMES}
    original_root_metadata = (session_dir / "metadata.json").read_bytes()

    with pytest.raises(ValueError, match="Turn ID already exists"):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json='{"replacement": true}',
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text="replacement",
            save_reason="test",
            turn_id="turn-0",
            max_turns=2,
            max_sessions=10,
        )

    assert {name: (turn_dir / name).read_bytes() for name in sessions_module.TURN_ARTIFACT_NAMES} == original_artifacts
    assert (session_dir / "metadata.json").read_bytes() == original_root_metadata
    assert get_head_artifact_paths(manager, "session-1").turn_id == "turn-0"  # type: ignore[arg-type]


def test_turn_retention_ignores_historical_turn_with_symlinked_artifact(tmp_path: Path) -> None:
    """Unsafe historical artifacts must not be read or removed by retention."""
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    for index in range(3):
        save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[{"type": "RUN_FINISHED"}],
            output_text=None,
            save_reason="test",
            turn_id=f"turn-{index}",
            max_turns=10,
            max_sessions=10,
        )

    session_dir = sessions_dir / "session-1"
    unsafe_turn = session_dir / "turns" / "turn-0"
    outside_metadata = tmp_path / "outside-metadata.json"
    outside_metadata.write_text(json.dumps({"updated_at": "1900-01-01T00:00:00+00:00"}))
    (unsafe_turn / "metadata.json").unlink()
    try:
        (unsafe_turn / "metadata.json").symlink_to(outside_metadata)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    trim_session_turns(session_dir, max_turns=1)

    assert unsafe_turn.is_dir()
    assert outside_metadata.read_text() == json.dumps({"updated_at": "1900-01-01T00:00:00+00:00"})
    assert not (session_dir / "turns" / "turn-1").exists()
    assert (session_dir / "turns" / "turn-2").is_dir()


def test_turn_retention_accepts_turns_missing_optional_artifacts(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    for index in range(3):
        turn_dir = save_session_turn(
            config_manager=manager,  # type: ignore[arg-type]
            session_id="session-1",
            working_dir=tmp_path,
            message_history_json=b"[]",
            context_state_json="{}",
            display_messages=[],
            output_text=None,
            save_reason="test",
            turn_id=f"turn-{index}",
            max_turns=10,
            max_sessions=10,
        )
        (turn_dir / "context_state.json").unlink()
        (turn_dir / "display_messages.json").unlink()

    session_dir = sessions_dir / "session-1"
    trim_session_turns(session_dir, max_turns=1)

    assert sorted(path.name for path in (session_dir / "turns").iterdir()) == ["turn-2"]
    assert get_head_artifact_paths(manager, "session-1").turn_id == "turn-2"  # type: ignore[arg-type]


def test_legacy_session_listing_defers_upgrade_until_artifacts_are_loaded(tmp_path: Path) -> None:
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
    assert entries[0].turn_count == 0
    assert entries[0].head_turn_id is None
    assert (session_dir / "message_history.json").exists()
    assert (session_dir / "context_state.json").exists()
    assert (session_dir / "display_messages.json").exists()

    paths = get_head_artifact_paths(manager, "legacy123456")  # type: ignore[arg-type]

    assert paths.turn_id is not None
    assert not (session_dir / "message_history.json").exists()
    assert not (session_dir / "context_state.json").exists()
    assert not (session_dir / "display_messages.json").exists()
    assert paths.display_messages_file is not None
    assert paths.display_messages_file.exists()


def test_legacy_upgrade_replaces_destination_symlink_without_touching_target(tmp_path: Path) -> None:
    """Legacy migration must never write through a pre-existing turn artifact symlink."""
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "legacy-safe"
    session_dir.mkdir(parents=True)
    updated_at = "2026-01-01T00:00:00+00:00"
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": session_dir.name, "updated_at": updated_at}))
    (session_dir / "message_history.json").write_text("[]")

    safe_timestamp = "".join(character for character in updated_at if character.isalnum())[:20]
    turn_dir = session_dir / "turns" / f"legacy-{safe_timestamp}"
    turn_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('"do not overwrite"')
    try:
        (turn_dir / "message_history.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    manager = DummyConfigManager(sessions_dir)
    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert [entry.id for entry in entries] == [session_dir.name]
    assert outside.read_text() == '"do not overwrite"'
    get_head_artifact_paths(manager, session_dir.name)  # type: ignore[arg-type]
    migrated_history = turn_dir / "message_history.json"
    assert migrated_history.is_file()
    assert not migrated_history.is_symlink()
    assert migrated_history.read_text() == "[]"


def test_legacy_listing_uses_safe_signature_and_defers_malformed_artifact_validation(tmp_path: Path) -> None:
    """Listing is metadata-only; explicit load still rejects malformed legacy artifacts."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    context_only = sessions_dir / "context-only"
    context_only.mkdir()
    (context_only / "metadata.json").write_text(json.dumps({"session_id": context_only.name}))
    (context_only / "context_state.json").write_text("{}")

    message_only = sessions_dir / "message-only"
    message_only.mkdir()
    (message_only / "message_history.json").write_text("[]")

    missing_identity = sessions_dir / "missing-identity"
    missing_identity.mkdir()
    (missing_identity / "metadata.json").write_text("{}")
    (missing_identity / "message_history.json").write_text("[]")

    invalid_message = sessions_dir / "invalid-message"
    invalid_message.mkdir()
    (invalid_message / "metadata.json").write_text(json.dumps({"session_id": invalid_message.name}))
    (invalid_message / "message_history.json").write_text("not-json")

    wrong_id = sessions_dir / "wrong-id"
    wrong_id.mkdir()
    (wrong_id / "metadata.json").write_text(json.dumps({"session_id": "another-directory"}))
    (wrong_id / "message_history.json").write_text("[]")

    invalid_optional = sessions_dir / "invalid-optional"
    invalid_optional.mkdir()
    (invalid_optional / "metadata.json").write_text(json.dumps({"session_id": invalid_optional.name}))
    (invalid_optional / "message_history.json").write_text("[]")
    (invalid_optional / "context_state.json").write_text("not-json")

    invalid_display = sessions_dir / "invalid-display"
    invalid_display.mkdir()
    (invalid_display / "metadata.json").write_text(json.dumps({"session_id": invalid_display.name}))
    (invalid_display / "message_history.json").write_text("[]")
    (invalid_display / "display_messages.json").write_text("not-json")

    malformed_v2 = sessions_dir / "malformed-v2"
    malformed_v2.mkdir()
    (malformed_v2 / "metadata.json").write_text(json.dumps({"schema_version": 2}))
    (malformed_v2 / "message_history.json").write_text("[]")

    manager = DummyConfigManager(sessions_dir)
    listed_ids = sorted(entry.id for entry in list_sessions(manager))  # type: ignore[arg-type]
    assert listed_ids == ["invalid-display", "invalid-message", "invalid-optional"]
    for session_id in listed_ids:
        with pytest.raises(FileNotFoundError):
            get_head_artifact_paths(manager, session_id)  # type: ignore[arg-type]
    for directory in (
        context_only,
        message_only,
        missing_identity,
        invalid_message,
        wrong_id,
        invalid_optional,
        invalid_display,
        malformed_v2,
    ):
        assert not (directory / "turns").exists()
    assert (missing_identity / "message_history.json").read_text() == "[]"
    assert (invalid_message / "message_history.json").read_text() == "not-json"
    assert (invalid_optional / "context_state.json").read_text() == "not-json"
    assert (invalid_display / "display_messages.json").read_text() == "not-json"


def test_schema_v2_session_requires_regular_identity_but_allows_missing_optional_artifacts(tmp_path: Path) -> None:
    """Turn metadata and history identify a v2 turn; context and display are optional."""
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "partial-v2"
    turn_dir = session_dir / "turns" / "turn-1"
    turn_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": session_dir.name,
            "head_turn_id": turn_dir.name,
        })
    )

    manager = DummyConfigManager(sessions_dir)
    assert list_sessions(manager) == []  # type: ignore[arg-type]

    (turn_dir / "message_history.json").write_text("[]")
    assert list_sessions(manager) == []  # type: ignore[arg-type]

    (turn_dir / "metadata.json").write_text("{}")
    assert [entry.id for entry in list_sessions(manager)] == [session_dir.name]  # type: ignore[arg-type]
    paths = get_head_artifact_paths(manager, session_dir.name)  # type: ignore[arg-type]
    assert paths.turn_id == turn_dir.name
    assert paths.context_state_file is None
    assert paths.display_messages_file is None


def test_minimal_parseable_legacy_session_upgrades(tmp_path: Path) -> None:
    """Metadata plus valid history is the minimum accepted legacy signature."""
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "legacy-minimal"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": session_dir.name}))
    (session_dir / "message_history.json").write_text("[]")

    manager = DummyConfigManager(sessions_dir)
    entries = list_sessions(manager)  # type: ignore[arg-type]

    assert [entry.id for entry in entries] == [session_dir.name]
    assert entries[0].head_turn_id is None
    paths = get_head_artifact_paths(manager, session_dir.name)  # type: ignore[arg-type]
    assert paths.turn_id is not None
    turn_dir = session_dir / "turns" / paths.turn_id
    assert (turn_dir / "message_history.json").read_text() == "[]"
    assert (turn_dir / "context_state.json").read_text() == "{}"
    assert (turn_dir / "display_messages.json").read_text() == "[]"


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
