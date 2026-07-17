from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from pydantic_ai.messages import ModelMessagesTypeAdapter
from ya_agent_sdk.context import AgentContext, ResumableState
from ya_agent_stream_protocol.agui import validate_display_events

from yaacli.config import ConfigManager

SESSION_SCHEMA_VERSION = 2
TURN_STORE_DIRNAME = "turns"
LEGACY_ARTIFACT_NAMES = ("message_history.json", "context_state.json", "display_messages.json")
TURN_ARTIFACT_NAMES = (*LEGACY_ARTIFACT_NAMES, "metadata.json")
_REQUIRED_TURN_ARTIFACT_NAMES = ("metadata.json", "message_history.json")
_OPTIONAL_TURN_ARTIFACT_NAMES = ("context_state.json", "display_messages.json")
_RESUMABLE_CONTEXT_FIELDS = (
    "subagent_history",
    "usage_snapshot_entries",
    "user_prompts",
    "previous_assistant_response_reference",
    "steering_messages",
    "handoff_message",
    "shell_env",
    "deferred_tool_metadata",
    "agent_registry",
    "need_user_approve_tools",
    "need_user_approve_mcps",
    "auto_load_files",
    "task_manager",
    "note_manager",
    "tool_search_loaded_tools",
    "tool_search_loaded_namespaces",
)
_RUNTIME_APPROVAL_POLICY_FIELDS = ("need_user_approve_tools", "need_user_approve_mcps")


@dataclass(slots=True)
class SessionInfo:
    id: str
    path: Path
    updated_at: str
    created_at: str | None
    working_dir: str | None
    model_profile_id: str | None
    model_label: str | None
    model: str | None
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


@dataclass(frozen=True, slots=True)
class SessionHeadArtifacts:
    """Immutable bytes from one committed session head."""

    session_id: str
    turn_id: str
    message_history_json: bytes
    context_state_json: bytes | None
    display_messages_json: bytes | None


def restore_resumable_state_safely(state: ResumableState, ctx: AgentContext) -> None:
    """Restore conversation state transactionally without weakening runtime approval policy."""
    previous_values = {field_name: getattr(ctx, field_name) for field_name in _RESUMABLE_CONTEXT_FIELDS}
    runtime_policy = {field_name: list(getattr(ctx, field_name)) for field_name in _RUNTIME_APPROVAL_POLICY_FIELDS}
    try:
        state.restore(ctx)
    except Exception:
        for field_name, value in previous_values.items():
            setattr(ctx, field_name, value)
        raise
    for field_name, value in runtime_policy.items():
        setattr(ctx, field_name, value)


def list_sessions(config_manager: ConfigManager) -> list[SessionInfo]:
    sessions_dir = config_manager.get_sessions_dir()
    if not sessions_dir.exists():
        return []
    sessions = []
    with local_file_lock(_global_lock_path(sessions_dir)):
        for path in sessions_dir.iterdir():
            if _is_session_dir(path):
                upgrade_legacy_session(path)
                sessions.append(_read_session_info(path))
    sessions.sort(key=lambda item: item.updated_at, reverse=True)
    return sessions


def resolve_session_dir(config_manager: ConfigManager, session_id: str) -> Path:
    _validate_session_id(session_id)
    sessions_dir = config_manager.get_sessions_dir()
    exact = sessions_dir / session_id
    if _is_session_dir(exact):
        upgrade_legacy_session(exact)
        return exact
    matches = (
        [path for path in sessions_dir.iterdir() if _is_session_dir(path) and path.name.startswith(session_id)]
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
    """Resolve head paths coherently; callers needing stable content must use ``read_head_artifacts``."""
    with _locked_session_dir(config_manager, session_id) as session_dir:
        turn_dir = _head_turn_dir(session_dir)
        turn_id = turn_dir.name if turn_dir is not None else None
        return SessionArtifactPaths(
            session_id=session_dir.name,
            session_dir=session_dir,
            turn_id=turn_id,
            turn_dir=turn_dir,
            message_history_file=(turn_dir / "message_history.json") if turn_dir is not None else None,
            context_state_file=_optional_artifact_path(turn_dir, "context_state.json"),
            display_messages_file=_optional_artifact_path(turn_dir, "display_messages.json"),
        )


def read_head_artifacts(
    config_manager: ConfigManager,
    session_id: str,
    *,
    max_display_messages_bytes: int | None = None,
) -> SessionHeadArtifacts:
    """Read one committed head while commit, retention, and deletion are excluded.

    Lock order is always the global sessions lock followed by the resolved
    session lock, matching ``save_session_turn``. The result owns immutable
    bytes, so it remains valid after the locks are released and an old turn is
    removed. A missing or oversized display replay is represented by ``None``;
    oversized payloads are not read.
    """
    if max_display_messages_bytes is not None and max_display_messages_bytes < 0:
        raise ValueError("max_display_messages_bytes must be non-negative")

    with _locked_session_dir(config_manager, session_id) as session_dir:
        return _read_head_artifacts_unlocked(
            session_dir,
            max_display_messages_bytes=max_display_messages_bytes,
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
    model_profile_id: str | None = None,
    model_label: str | None = None,
    model: str | None = None,
    turn_id: str | None = None,
    max_turns: int = 20,
    max_sessions: int = 100,
    max_session_age_days: int | None = None,
) -> Path:
    _validate_session_id(session_id)
    sessions_dir = config_manager.get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    with local_file_lock(_global_lock_path(sessions_dir)):
        session_dir = sessions_dir / session_id
        if session_dir.is_symlink():
            raise ValueError(f"Session directory must not be a symbolic link: {session_dir}")
        session_dir_existed = session_dir.exists()
        if session_dir.is_dir() and any(session_dir.iterdir()) and not _is_session_dir(session_dir):
            raise ValueError(f"Refusing to overwrite an unrecognized session directory: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)
        created_turn_dir: Path | None = None
        root_metadata_committed = False
        try:
            with local_file_lock(_session_lock_path(session_dir)):
                _upgrade_legacy_session_unlocked(session_dir)
                resolved_turn_id = turn_id or uuid.uuid4().hex[:12]
                turn_dir = _create_turn_directory(session_dir, resolved_turn_id)
                created_turn_dir = turn_dir
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
                    "model_profile_id": model_profile_id,
                    "model_label": model_label,
                    "model": model,
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
                    "model_profile_id": model_profile_id,
                    "model_label": model_label,
                    "model": model,
                    "output_text": output_text,
                })
                _write_text_atomic(metadata_file, json.dumps(metadata, ensure_ascii=False, indent=2))
                root_metadata_committed = True
                trim_session_turns(session_dir, max_turns=max_turns)
        except BaseException:
            # Until the root metadata commit publishes this turn as the new head,
            # it is private to this save attempt and must not affect a later
            # retention pass. Only remove a directory that this attempt created;
            # an explicit ID that already exists is rejected before any writes.
            if created_turn_dir is not None and not root_metadata_committed:
                shutil.rmtree(created_turn_dir, ignore_errors=True)

            # A first save has no committed root metadata to recover from. Remove
            # every artifact created by this attempt so a retry is not rejected
            # as an unrelated, non-empty directory. A failure after the metadata
            # commit retains the valid session and its new head.
            if not _is_session_dir(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
                if session_dir_existed:
                    session_dir.mkdir(parents=True, exist_ok=True)
            raise

        trim_sessions(
            sessions_dir,
            max_sessions=max_sessions,
            max_session_age_days=max_session_age_days,
            protected_session_id=session_id,
        )
        return turn_dir


def upgrade_legacy_session(session_dir: Path) -> None:
    if not session_dir.is_dir() or not _has_legacy_artifacts(session_dir):
        return
    with local_file_lock(_session_lock_path(session_dir)):
        _upgrade_legacy_session_unlocked(session_dir)


def trim_session_turns(session_dir: Path, *, max_turns: int) -> None:
    if max_turns <= 0:
        max_turns = 1
    turn_dirs = _turn_dirs(session_dir)
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
    session_dirs = [path for path in sessions_dir.iterdir() if _is_session_dir(path)]
    protected = {protected_session_id} if protected_session_id else set()

    for path in session_dirs:
        if path.name in protected:
            continue
        if cutoff is not None and _path_updated_datetime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)

    session_dirs = [path for path in sessions_dir.iterdir() if _is_session_dir(path)]
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
    if lock_path.is_symlink():
        raise OSError(f"Lock file must not be a symbolic link: {lock_path}")
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


def _is_safe_path_segment(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and Path(value).name == value
    )


def _validate_session_id(session_id: str) -> None:
    if not _is_safe_path_segment(session_id):
        raise ValueError(f"Invalid session ID: {session_id!r}")


def _is_real_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _has_legacy_artifacts(session_dir: Path) -> bool:
    """Recognize only a complete, parseable legacy-session signature."""
    if not _is_real_directory(session_dir):
        return False

    metadata_file = session_dir / "metadata.json"
    history_file = session_dir / "message_history.json"
    context_file = session_dir / "context_state.json"
    display_file = session_dir / "display_messages.json"
    turns_dir = session_dir / TURN_STORE_DIRNAME
    lock_file = _session_lock_path(session_dir)
    if not _is_regular_file(metadata_file) or not _is_regular_file(history_file):
        return False
    if (
        turns_dir.is_symlink()
        or (turns_dir.exists() and not turns_dir.is_dir())
        or lock_file.is_symlink()
        or (lock_file.exists() and not lock_file.is_file())
    ):
        return False
    for optional_file in (context_file, display_file):
        if optional_file.is_symlink() or (optional_file.exists() and not optional_file.is_file()):
            return False

    try:
        metadata_payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata_payload, dict) or "schema_version" in metadata_payload:
            return False
        metadata_session_id = metadata_payload.get("session_id")
        if not isinstance(metadata_session_id, str) or metadata_session_id != session_dir.name:
            return False
        ModelMessagesTypeAdapter.validate_json(history_file.read_bytes())
        if context_file.exists():
            ResumableState.model_validate_json(context_file.read_text(encoding="utf-8"))
        if display_file.exists():
            validate_display_events(json.loads(display_file.read_text(encoding="utf-8")))
    except Exception:
        return False
    return True


def _is_session_dir(path: Path) -> bool:
    """Return whether a directory is a confirmed schema-v2 or legacy session."""
    if not _is_real_directory(path) or not _is_safe_path_segment(path.name):
        return False
    if _has_legacy_artifacts(path):
        return True

    metadata_file = path / "metadata.json"
    if not _is_regular_file(metadata_file):
        return False
    metadata = _read_json_object(metadata_file)
    schema_version = metadata.get("schema_version")
    session_id = metadata.get("session_id")
    head_turn_id = metadata.get("head_turn_id")
    if isinstance(schema_version, bool) or schema_version != SESSION_SCHEMA_VERSION:
        return False
    if not isinstance(session_id, str) or session_id != path.name:
        return False
    if not isinstance(head_turn_id, str) or not _is_safe_path_segment(head_turn_id):
        return False
    turns_dir = path / TURN_STORE_DIRNAME
    head_turn_dir = turns_dir / head_turn_id
    return _is_real_directory(turns_dir) and _is_turn_dir(path, head_turn_dir)


def _prepare_turn_directory(session_dir: Path, turn_id: str) -> Path:
    if not _is_real_directory(session_dir):
        raise ValueError(f"Session directory is not a real directory: {session_dir}")
    if not _is_safe_path_segment(turn_id):
        raise ValueError(f"Invalid turn ID: {turn_id!r}")
    turns_dir = session_dir / TURN_STORE_DIRNAME
    if turns_dir.is_symlink() or (turns_dir.exists() and not turns_dir.is_dir()):
        raise ValueError(f"Turn store must be a real directory: {turns_dir}")
    turns_dir.mkdir(exist_ok=True)
    turn_dir = turns_dir / turn_id
    if turn_dir.is_symlink() or (turn_dir.exists() and not turn_dir.is_dir()):
        raise ValueError(f"Turn directory must be a real directory: {turn_dir}")
    turn_dir.mkdir(exist_ok=True)
    try:
        turn_dir.resolve(strict=True).relative_to(session_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Turn directory escapes its session: {turn_dir}") from exc
    return turn_dir


def _create_turn_directory(session_dir: Path, turn_id: str) -> Path:
    """Create a fresh turn without ever reusing or partially overwriting one."""
    if not _is_real_directory(session_dir):
        raise ValueError(f"Session directory is not a real directory: {session_dir}")
    if not _is_safe_path_segment(turn_id):
        raise ValueError(f"Invalid turn ID: {turn_id!r}")
    turns_dir = session_dir / TURN_STORE_DIRNAME
    if turns_dir.is_symlink() or (turns_dir.exists() and not turns_dir.is_dir()):
        raise ValueError(f"Turn store must be a real directory: {turns_dir}")
    turns_dir.mkdir(exist_ok=True)
    turn_dir = turns_dir / turn_id
    try:
        turn_dir.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"Turn ID already exists: {turn_id!r}") from exc
    try:
        turn_dir.resolve(strict=True).relative_to(session_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        shutil.rmtree(turn_dir, ignore_errors=True)
        raise ValueError(f"Turn directory escapes its session: {turn_dir}") from exc
    return turn_dir


def _upgrade_legacy_session_unlocked(session_dir: Path) -> None:
    if not _has_legacy_artifacts(session_dir):
        return
    legacy_files = [session_dir / name for name in LEGACY_ARTIFACT_NAMES]

    metadata_file = session_dir / "metadata.json"
    metadata = _read_json_object(metadata_file)
    updated_at = str(metadata.get("updated_at") or _mtime_iso(session_dir))
    created_at = metadata.get("created_at") if isinstance(metadata.get("created_at"), str) else updated_at
    turn_id = _legacy_turn_id(updated_at)
    turn_dir = _prepare_turn_directory(session_dir, turn_id)

    for name in LEGACY_ARTIFACT_NAMES:
        source = session_dir / name
        target = turn_dir / name
        if _is_regular_file(source):
            # Atomic replacement never follows a pre-existing destination
            # symlink or hard link outside the session boundary.
            _write_bytes_atomic(target, source.read_bytes())
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
        "model_profile_id": metadata.get("model_profile_id"),
        "model_label": metadata.get("model_label"),
        "model": metadata.get("model"),
        "output_text": metadata.get("output_text"),
        "message_count": _read_message_count(turn_dir / "message_history.json"),
        "display_event_count": _read_display_event_count(turn_dir / "display_messages.json"),
    }
    _write_text_atomic(turn_dir / "metadata.json", json.dumps(turn_metadata, ensure_ascii=False, indent=2))

    metadata.update({
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": session_dir.name,
        "created_at": created_at,
        "updated_at": updated_at,
        "head_turn_id": turn_id,
        "last_save_reason": "legacy_upgrade",
    })
    _write_text_atomic(metadata_file, json.dumps(metadata, ensure_ascii=False, indent=2))

    for source in legacy_files:
        if _is_regular_file(source):
            source.unlink()


def _read_session_info(path: Path) -> SessionInfo:
    metadata = _read_json_object(path / "metadata.json")
    head_turn_id = metadata.get("head_turn_id") if isinstance(metadata.get("head_turn_id"), str) else None
    head_turn_dir = _head_turn_dir(path)
    updated_at = str(metadata.get("updated_at") or _mtime_iso(path))
    created_at = metadata.get("created_at") if isinstance(metadata.get("created_at"), str) else None
    working_dir = metadata.get("working_dir") if isinstance(metadata.get("working_dir"), str) else None
    model_profile_id = metadata.get("model_profile_id") if isinstance(metadata.get("model_profile_id"), str) else None
    model_label = metadata.get("model_label") if isinstance(metadata.get("model_label"), str) else None
    model = metadata.get("model") if isinstance(metadata.get("model"), str) else None
    output_text = metadata.get("output_text") if isinstance(metadata.get("output_text"), str) else None
    return SessionInfo(
        id=path.name,
        path=path,
        updated_at=updated_at,
        created_at=created_at,
        working_dir=working_dir,
        model_profile_id=model_profile_id,
        model_label=model_label,
        model=model,
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


@contextmanager
def _locked_session_dir(config_manager: ConfigManager, session_id: str) -> Iterator[Path]:
    """Yield a resolved session under the canonical global-then-session lock order."""
    _validate_session_id(session_id)
    sessions_dir = config_manager.get_sessions_dir()
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with local_file_lock(_global_lock_path(sessions_dir)):
        session_dir = resolve_session_dir(config_manager, session_id)
        with local_file_lock(_session_lock_path(session_dir)):
            if not _is_session_dir(session_dir):
                raise FileNotFoundError(f"Session not found: {session_id}")
            yield session_dir


def _read_head_artifacts_unlocked(
    session_dir: Path,
    *,
    max_display_messages_bytes: int | None,
) -> SessionHeadArtifacts:
    turn_dir = _head_turn_dir(session_dir)
    if turn_dir is None:
        raise ValueError(f"Session {session_dir.name} has no committed head turn")

    message_history_json = _read_regular_file_bytes(turn_dir / "message_history.json")
    context_state_json = _read_optional_regular_file_bytes(turn_dir / "context_state.json")
    display_messages_json = _read_optional_regular_file_bytes(
        turn_dir / "display_messages.json",
        max_bytes=max_display_messages_bytes,
    )
    if message_history_json is None:
        # No bound is supplied for required history, so this is defensive only.
        raise ValueError(f"Session {session_dir.name} is missing required message_history.json")
    return SessionHeadArtifacts(
        session_id=session_dir.name,
        turn_id=turn_dir.name,
        message_history_json=message_history_json,
        context_state_json=context_state_json,
        display_messages_json=display_messages_json,
    )


def _head_turn_dir(session_dir: Path) -> Path | None:
    metadata = _read_json_object(session_dir / "metadata.json")
    head_turn_id = metadata.get("head_turn_id") if isinstance(metadata.get("head_turn_id"), str) else None
    if head_turn_id and _is_safe_path_segment(head_turn_id):
        head_turn_dir = session_dir / TURN_STORE_DIRNAME / head_turn_id
        if _is_turn_dir(session_dir, head_turn_dir):
            return head_turn_dir
    turn_dirs = _turn_dirs(session_dir)
    if not turn_dirs:
        return None
    return max(turn_dirs, key=_path_updated_at)


def _is_turn_dir(session_dir: Path, path: Path) -> bool:
    """Return whether a turn stays inside its session and has only safe artifacts."""
    if not _is_safe_path_segment(path.name) or not _is_real_directory(path):
        return False
    try:
        path.resolve(strict=True).relative_to(session_dir.resolve(strict=True))
    except (OSError, ValueError):
        return False
    if not all(_is_regular_file(path / name) for name in _REQUIRED_TURN_ARTIFACT_NAMES):
        return False
    return all(_is_missing_or_regular_file(path / name) for name in _OPTIONAL_TURN_ARTIFACT_NAMES)


def _turn_dirs(session_dir: Path) -> list[Path]:
    turns_dir = session_dir / TURN_STORE_DIRNAME
    if not _is_real_directory(turns_dir):
        return []
    return [path for path in turns_dir.iterdir() if _is_turn_dir(session_dir, path)]


def _optional_artifact_path(turn_dir: Path | None, name: str) -> Path | None:
    if turn_dir is None:
        return None
    path = turn_dir / name
    return path if _is_regular_file(path) else None


def _is_missing_or_regular_file(path: Path) -> bool:
    return not path.is_symlink() and (not path.exists() or path.is_file())


def _read_optional_regular_file_bytes(path: Path, *, max_bytes: int | None = None) -> bytes | None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Session artifact must be a regular file: {path}")
    if not path.exists():
        return None
    return _read_regular_file_bytes(path, max_bytes=max_bytes)


def _read_regular_file_bytes(path: Path, *, max_bytes: int | None = None) -> bytes | None:
    """Read a regular file through one descriptor without following symlinks."""
    if not _is_regular_file(path):
        raise ValueError(f"Session artifact must be a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Unable to open session artifact safely: {path}") from exc
    with os.fdopen(descriptor, "rb") as artifact_file:
        file_stat = os.fstat(artifact_file.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Session artifact must be a regular file: {path}")
        if max_bytes is not None and file_stat.st_size > max_bytes:
            return None
        payload = artifact_file.read() if max_bytes is None else artifact_file.read(max_bytes + 1)
    if max_bytes is not None and len(payload) > max_bytes:
        return None
    return payload


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
