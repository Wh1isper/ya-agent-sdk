from __future__ import annotations

import contextlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from pydantic_ai.messages import ModelMessagesTypeAdapter

from yaacli.agui import validate_display_events
from yaacli.config import ConfigManager

SESSION_SCHEMA_VERSION = 2
TURN_STORE_DIRNAME = "turns"
LEGACY_ARTIFACT_NAMES = ("message_history.json", "context_state.json", "display_messages.json")


@dataclass(slots=True)
class SessionInfo:
    id: str
    path: Path
    updated_at: str
    created_at: str | None
    working_dir: str | None
    output_text: str | None
    message_count: int | None
    display_event_count: int | None
    metadata: dict[str, Any]
    head_turn_id: str | None = None
    turn_count: int = 0


@dataclass(slots=True)
class SessionArtifactPaths:
    session_id: str
    session_dir: Path
    turn_id: str | None
    turn_dir: Path | None
    message_history_file: Path | None
    context_state_file: Path | None
    display_messages_file: Path | None


def list_sessions(config_manager: ConfigManager) -> list[SessionInfo]:
    sessions_dir = config_manager.get_sessions_dir()
    if not sessions_dir.exists():
        return []
    sessions = []
    with local_file_lock(_global_lock_path(sessions_dir)):
        for path in sessions_dir.iterdir():
            if path.is_dir():
                upgrade_legacy_session(path)
                sessions.append(_read_session_info(path))
    sessions.sort(key=lambda item: item.updated_at, reverse=True)
    return sessions


def resolve_session_dir(config_manager: ConfigManager, session_id: str) -> Path:
    sessions_dir = config_manager.get_sessions_dir()
    exact = sessions_dir / session_id
    if exact.is_dir():
        upgrade_legacy_session(exact)
        return exact
    matches = (
        [path for path in sessions_dir.iterdir() if path.is_dir() and path.name.startswith(session_id)]
        if sessions_dir.exists()
        else []
    )
    if len(matches) == 1:
        upgrade_legacy_session(matches[0])
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous session ID {session_id!r}: {', '.join(sorted(path.name for path in matches))}")
    raise FileNotFoundError(f"Session not found: {session_id}")


def get_session_info(config_manager: ConfigManager, session_id: str) -> SessionInfo:
    return _read_session_info(resolve_session_dir(config_manager, session_id))


def delete_session(config_manager: ConfigManager, session_id: str) -> SessionInfo:
    sessions_dir = config_manager.get_sessions_dir()
    with local_file_lock(_global_lock_path(sessions_dir)):
        session_dir = resolve_session_dir(config_manager, session_id)
        info = _read_session_info(session_dir)
        shutil.rmtree(session_dir)
        return info


def get_head_artifact_paths(config_manager: ConfigManager, session_id: str) -> SessionArtifactPaths:
    session_dir = resolve_session_dir(config_manager, session_id)
    turn_dir = _head_turn_dir(session_dir)
    turn_id = turn_dir.name if turn_dir is not None else None
    return SessionArtifactPaths(
        session_id=session_dir.name,
        session_dir=session_dir,
        turn_id=turn_id,
        turn_dir=turn_dir,
        message_history_file=(turn_dir / "message_history.json") if turn_dir is not None else None,
        context_state_file=(turn_dir / "context_state.json") if turn_dir is not None else None,
        display_messages_file=(turn_dir / "display_messages.json") if turn_dir is not None else None,
    )


def save_session_turn(
    *,
    config_manager: ConfigManager,
    session_id: str,
    working_dir: Path,
    message_history_json: bytes,
    context_state_json: str,
    display_messages: list[dict[str, Any]],
    output_text: str | None,
    save_reason: str,
    turn_id: str | None = None,
    max_turns: int = 20,
    max_sessions: int = 100,
    max_session_age_days: int | None = None,
) -> Path:
    sessions_dir = config_manager.get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    with local_file_lock(_global_lock_path(sessions_dir)):
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        with local_file_lock(_session_lock_path(session_dir)):
            _upgrade_legacy_session_unlocked(session_dir)
            resolved_turn_id = turn_id or uuid.uuid4().hex[:12]
            turn_dir = session_dir / TURN_STORE_DIRNAME / resolved_turn_id
            turn_dir.mkdir(parents=True, exist_ok=True)
            _write_bytes_atomic(turn_dir / "message_history.json", message_history_json)
            _write_text_atomic(turn_dir / "context_state.json", context_state_json)
            _write_text_atomic(
                turn_dir / "display_messages.json",
                json.dumps(display_messages, ensure_ascii=False, indent=2),
            )

            now = datetime.now(UTC).isoformat()
            turn_metadata = {
                "turn_id": resolved_turn_id,
                "session_id": session_id,
                "working_dir": str(working_dir),
                "created_at": now,
                "updated_at": now,
                "save_reason": save_reason,
                "output_text": output_text,
                "message_count": _read_message_count(turn_dir / "message_history.json"),
                "display_event_count": len(display_messages),
            }
            _write_text_atomic(turn_dir / "metadata.json", json.dumps(turn_metadata, ensure_ascii=False, indent=2))

            metadata_file = session_dir / "metadata.json"
            metadata = _read_json_object(metadata_file)
            created_at = metadata.get("created_at") if isinstance(metadata.get("created_at"), str) else now
            metadata.update({
                "schema_version": SESSION_SCHEMA_VERSION,
                "session_id": session_id,
                "working_dir": str(working_dir),
                "created_at": created_at,
                "updated_at": now,
                "head_turn_id": resolved_turn_id,
                "last_save_reason": save_reason,
                "output_text": output_text,
            })
            _write_text_atomic(metadata_file, json.dumps(metadata, ensure_ascii=False, indent=2))
            trim_session_turns(session_dir, max_turns=max_turns)

        trim_sessions(
            sessions_dir,
            max_sessions=max_sessions,
            max_session_age_days=max_session_age_days,
            protected_session_id=session_id,
        )
        return turn_dir


def upgrade_legacy_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    with local_file_lock(_session_lock_path(session_dir)):
        _upgrade_legacy_session_unlocked(session_dir)


def trim_session_turns(session_dir: Path, *, max_turns: int) -> None:
    if max_turns <= 0:
        max_turns = 1
    turns_dir = session_dir / TURN_STORE_DIRNAME
    if not turns_dir.exists():
        return
    turn_dirs = [path for path in turns_dir.iterdir() if path.is_dir()]
    if len(turn_dirs) <= max_turns:
        return
    metadata = _read_json_object(session_dir / "metadata.json")
    head_turn_id = metadata.get("head_turn_id") if isinstance(metadata.get("head_turn_id"), str) else None
    protected = {head_turn_id} if head_turn_id else set()
    removable = [path for path in turn_dirs if path.name not in protected]
    removable.sort(key=_path_updated_at)
    remove_count = max(0, len(turn_dirs) - max_turns)
    for path in removable[:remove_count]:
        shutil.rmtree(path, ignore_errors=True)


def trim_sessions(
    sessions_dir: Path,
    *,
    max_sessions: int,
    max_session_age_days: int | None = None,
    protected_session_id: str | None = None,
) -> None:
    if not sessions_dir.exists():
        return
    if max_sessions <= 0:
        max_sessions = 1
    now = datetime.now(UTC)
    cutoff = (
        now - timedelta(days=max_session_age_days)
        if max_session_age_days is not None and max_session_age_days > 0
        else None
    )
    session_dirs = [path for path in sessions_dir.iterdir() if path.is_dir()]
    protected = {protected_session_id} if protected_session_id else set()

    for path in session_dirs:
        if path.name in protected:
            continue
        if cutoff is not None and _path_updated_datetime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)

    session_dirs = [path for path in sessions_dir.iterdir() if path.is_dir()]
    if len(session_dirs) <= max_sessions:
        return
    removable = [path for path in session_dirs if path.name not in protected]
    removable.sort(key=_path_updated_at)
    remove_count = max(0, len(session_dirs) - max_sessions)
    for path in removable[:remove_count]:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def local_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            _lock_file(lock_file)
            yield
        finally:
            with contextlib.suppress(OSError):
                _unlock_file(lock_file)


def _lock_file(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _upgrade_legacy_session_unlocked(session_dir: Path) -> None:
    legacy_files = [session_dir / name for name in LEGACY_ARTIFACT_NAMES]
    if not any(path.exists() for path in legacy_files):
        return

    metadata_file = session_dir / "metadata.json"
    metadata = _read_json_object(metadata_file)
    updated_at = str(metadata.get("updated_at") or _mtime_iso(session_dir))
    created_at = metadata.get("created_at") if isinstance(metadata.get("created_at"), str) else updated_at
    turn_id = _legacy_turn_id(updated_at)
    turn_dir = session_dir / TURN_STORE_DIRNAME / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)

    for name in LEGACY_ARTIFACT_NAMES:
        source = session_dir / name
        target = turn_dir / name
        if source.exists():
            shutil.copy2(source, target)
        elif name == "message_history.json" or name == "display_messages.json":
            _write_text_atomic(target, "[]")
        else:
            _write_text_atomic(target, "{}")

    turn_metadata = {
        "turn_id": turn_id,
        "session_id": session_dir.name,
        "working_dir": metadata.get("working_dir"),
        "created_at": created_at,
        "updated_at": updated_at,
        "save_reason": metadata.get("last_save_reason") or "legacy_upgrade",
        "output_text": metadata.get("output_text"),
        "message_count": _read_message_count(turn_dir / "message_history.json"),
        "display_event_count": _read_display_event_count(turn_dir / "display_messages.json"),
    }
    _write_text_atomic(turn_dir / "metadata.json", json.dumps(turn_metadata, ensure_ascii=False, indent=2))

    metadata.update({
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": metadata.get("session_id") or session_dir.name,
        "created_at": created_at,
        "updated_at": updated_at,
        "head_turn_id": turn_id,
        "last_save_reason": "legacy_upgrade",
    })
    _write_text_atomic(metadata_file, json.dumps(metadata, ensure_ascii=False, indent=2))

    for source in legacy_files:
        if source.exists():
            source.unlink()


def _read_session_info(path: Path) -> SessionInfo:
    metadata = _read_json_object(path / "metadata.json")
    head_turn_id = metadata.get("head_turn_id") if isinstance(metadata.get("head_turn_id"), str) else None
    head_turn_dir = _head_turn_dir(path)
    updated_at = str(metadata.get("updated_at") or _mtime_iso(path))
    created_at = metadata.get("created_at") if isinstance(metadata.get("created_at"), str) else None
    working_dir = metadata.get("working_dir") if isinstance(metadata.get("working_dir"), str) else None
    output_text = metadata.get("output_text") if isinstance(metadata.get("output_text"), str) else None
    return SessionInfo(
        id=path.name,
        path=path,
        updated_at=updated_at,
        created_at=created_at,
        working_dir=working_dir,
        output_text=output_text,
        message_count=_read_message_count(head_turn_dir / "message_history.json")
        if head_turn_dir is not None
        else None,
        display_event_count=_read_display_event_count(head_turn_dir / "display_messages.json")
        if head_turn_dir is not None
        else None,
        metadata=metadata,
        head_turn_id=head_turn_id,
        turn_count=len(_turn_dirs(path)),
    )


def _head_turn_dir(session_dir: Path) -> Path | None:
    metadata = _read_json_object(session_dir / "metadata.json")
    head_turn_id = metadata.get("head_turn_id") if isinstance(metadata.get("head_turn_id"), str) else None
    if head_turn_id:
        head_turn_dir = session_dir / TURN_STORE_DIRNAME / head_turn_id
        if head_turn_dir.is_dir():
            return head_turn_dir
    turn_dirs = _turn_dirs(session_dir)
    if not turn_dirs:
        return None
    return max(turn_dirs, key=_path_updated_at)


def _turn_dirs(session_dir: Path) -> list[Path]:
    turns_dir = session_dir / TURN_STORE_DIRNAME
    if not turns_dir.exists():
        return []
    return [path for path in turns_dir.iterdir() if path.is_dir()]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_message_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(ModelMessagesTypeAdapter.validate_json(path.read_bytes()))
    except Exception:
        return None


def _read_display_event_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(validate_display_events(json.loads(path.read_text(encoding="utf-8"))))
    except Exception:
        return None


def _path_updated_at(path: Path) -> str:
    metadata = _read_json_object(path / "metadata.json")
    updated_at = metadata.get("updated_at")
    return updated_at if isinstance(updated_at, str) else _mtime_iso(path)


def _path_updated_datetime(path: Path) -> datetime:
    value = _path_updated_at(path)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _legacy_turn_id(updated_at: str) -> str:
    safe = "".join(ch for ch in updated_at if ch.isalnum())[:20]
    return f"legacy-{safe or uuid.uuid4().hex[:12]}"


def _global_lock_path(sessions_dir: Path) -> Path:
    return sessions_dir / ".sessions.lock"


def _session_lock_path(session_dir: Path) -> Path:
    return session_dir / ".session.lock"


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(path)
