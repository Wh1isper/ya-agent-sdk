"""Atomic, grep-friendly state files for durable YAACLI sessions."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from ya_agent_sdk.subagents.spec import (
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanDescriptor,
)

from yaacli.durable.models import ExecutionCheckpointRecord, InputRecord, InputState, RevisionRecord

_FILE_SCHEMA_VERSION = 1
_SESSION_MARKER_NAME = ".yaacli-session-state"
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[Path, threading.RLock] = {}
_LOCK_DEPTHS = threading.local()
_PROCESS_OWNER_TOKEN = uuid.uuid4().hex
_PROCESS_OWNER_FDS: dict[Path, int] = {}


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold a process- and filesystem-coordinated reentrant lock for one path."""
    with _LOCKS_GUARD:
        process_lock = _SESSION_LOCKS.setdefault(path, threading.RLock())
    with process_lock:
        depths = getattr(_LOCK_DEPTHS, "paths", None)
        if depths is None:
            depths = {}
            _LOCK_DEPTHS.paths = depths
        depth = depths.get(path, 0)
        if depth:
            depths[path] = depth + 1
            try:
                yield
            finally:
                depths[path] -= 1
            return

        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        depths[path] = 1
        locked = False
        try:
            if not _lock_descriptor(descriptor, blocking=True):  # pragma: no cover - blocking invariant
                raise RuntimeError(f"Failed to acquire durable state lock: {path}")
            locked = True
            yield
        finally:
            depths.pop(path, None)
            if locked:
                _unlock_descriptor(descriptor)
            os.close(descriptor)


class RevisionStateFile(BaseModel):
    """Self-contained immutable revision state document."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = _FILE_SCHEMA_VERSION
    revision: RevisionRecord


class CheckpointStateFile(BaseModel):
    """Self-contained mutable execution checkpoint document."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = _FILE_SCHEMA_VERSION
    checkpoint: ExecutionCheckpointRecord
    previous_checkpoint: ExecutionCheckpointRecord | None = None


class SubagentStateFile(BaseModel):
    """Self-contained child execution, descriptor, and persisted inbox."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = _FILE_SCHEMA_VERSION
    descriptor: SubagentPlanDescriptor
    record: SubagentExecutionRecord
    owner_pid: int = Field(gt=0)
    owner_token: str = Field(pattern=r"^[0-9a-f]{32}$")
    inputs: tuple[InputRecord, ...] = ()
    input_open: bool = True
    cancel_requested: bool = False
    cancellation_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class SessionStateFiles:
    """Deterministic state paths and atomic JSON publication beside the product DB."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.root = self.database_path.parent
        self.marker_content = (
            json.dumps(
                {"database": self.database_path.name, "schema_version": 1},
                sort_keys=True,
            )
            + "\n"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def process_owner_is_alive(self, owner_token: str) -> bool:
        if len(owner_token) != 32 or any(character not in "0123456789abcdef" for character in owner_token):
            raise ValueError(f"Invalid process owner token: {owner_token!r}")
        lock_path = self.root / f".subagent-owner-{owner_token}.lock"
        with _LOCKS_GUARD:
            if owner_token == _PROCESS_OWNER_TOKEN and lock_path in _PROCESS_OWNER_FDS:
                return True
        if not lock_path.exists():
            return False
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if not _lock_descriptor(descriptor, blocking=False):
                return True
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        return False

    def hold_process_owner_lock(self) -> str:
        lock_path = self.root / f".subagent-owner-{_PROCESS_OWNER_TOKEN}.lock"
        with _LOCKS_GUARD:
            if lock_path not in _PROCESS_OWNER_FDS:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    if not _lock_descriptor(descriptor, blocking=False):
                        raise RuntimeError(f"Failed to acquire process owner lock: {lock_path}")
                except BaseException:
                    os.close(descriptor)
                    raise
                _PROCESS_OWNER_FDS[lock_path] = descriptor
        return _PROCESS_OWNER_TOKEN

    @contextmanager
    def subagent_creation_lock(self) -> Iterator[None]:
        """Serialize only the short global child-execution ID claim boundary."""
        lock_path = self.root / ".subagent-create.lock"
        with exclusive_file_lock(lock_path):
            yield

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        """Serialize one session's SQLite lifecycle and file-state mutations."""
        component = _safe_component(session_id, label="session ID")
        self.session_dir(component, create=True)
        lock_digest = hashlib.sha256(component.encode()).hexdigest()[:32]
        lock_path = self.root / f".session-lock-{lock_digest}.lock"
        if lock_path.is_symlink():
            raise RuntimeError(f"Session lock path must not be a symlink: {lock_path}")
        with exclusive_file_lock(lock_path):
            yield

    def session_dir(self, session_id: str, *, create: bool = False) -> Path:
        component = _safe_component(session_id, label="session ID")
        path = self.root / component
        if create:
            marker = path / _SESSION_MARKER_NAME
            if marker.exists():
                self._validate_session_marker(path)
                return path
            claim_digest = hashlib.sha256(component.encode()).hexdigest()[:32]
            claim_lock = self.root / f".session-claim-{claim_digest}.lock"
            with exclusive_file_lock(claim_lock):
                if marker.exists():
                    self._validate_session_marker(path)
                    return path
                if path.exists():
                    raise RuntimeError(f"Refusing to claim unmarked durable session directory: {path}")
                temporary_prefix = f".session-{claim_digest}.tmp."
                for stale in self.root.glob(f"{temporary_prefix}*"):
                    if stale.is_dir() and not stale.is_symlink():
                        shutil.rmtree(stale)
                temporary_dir = self.root / f"{temporary_prefix}{uuid.uuid4().hex}"
                temporary_dir.mkdir()
                try:
                    temporary_marker = temporary_dir / _SESSION_MARKER_NAME
                    with temporary_marker.open("x", encoding="utf-8") as stream:
                        stream.write(self.marker_content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    _fsync_directory(temporary_dir)
                    os.rename(temporary_dir, path)
                    _fsync_directory(self.root)
                finally:
                    if temporary_dir.exists():
                        shutil.rmtree(temporary_dir)
                self._validate_session_marker(path)
        elif path.exists():
            self._validate_session_marker(path)
        return path

    def revision_path(self, session_id: str, revision_id: str, *, create: bool = False) -> Path:
        directory = (
            self.session_dir(session_id, create=create)
            / "revisions"
            / _safe_component(revision_id, label="revision ID")
        )
        if create:
            self._ensure_directory(directory)
        return directory / "state.json"

    def checkpoint_path(self, session_id: str, execution_id: str, *, create: bool = False) -> Path:
        directory = (
            self.session_dir(session_id, create=create)
            / "checkpoints"
            / _safe_component(execution_id, label="execution ID")
        )
        if create:
            self._ensure_directory(directory)
        return directory / "state.json"

    def subagent_path(self, session_id: str, execution_id: str, *, create: bool = False) -> Path:
        directory = (
            self.session_dir(session_id, create=create)
            / "subagents"
            / _safe_component(execution_id, label="execution ID")
        )
        if create:
            self._ensure_directory(directory)
        return directory / "state.json"

    def write_revision(self, state: RevisionStateFile) -> Path:
        path = self.revision_path(state.revision.session_id, state.revision.revision_id, create=True)
        self._write_model(path, state)
        return path

    def read_revision(self, session_id: str, revision_id: str) -> RevisionStateFile:
        return self._read_model(self.revision_path(session_id, revision_id), RevisionStateFile)

    def write_checkpoint(self, session_id: str, state: CheckpointStateFile) -> Path:
        path = self.checkpoint_path(session_id, state.checkpoint.execution_id, create=True)
        self._write_model(path, state)
        return path

    def read_checkpoint(self, session_id: str, execution_id: str) -> CheckpointStateFile:
        return self._read_model(self.checkpoint_path(session_id, execution_id), CheckpointStateFile)

    def write_subagent(self, state: SubagentStateFile) -> Path:
        path = self.subagent_path(state.record.owner_scope_id, state.record.execution_id, create=True)
        self._write_model(path, state)
        return path

    def read_subagent(self, session_id: str, execution_id: str) -> SubagentStateFile:
        return self._read_model(self.subagent_path(session_id, execution_id), SubagentStateFile)

    def find_subagent(self, execution_id: str) -> SubagentStateFile | None:
        component = _safe_component(execution_id, label="execution ID")
        matches = tuple(
            path
            for session_id in self.list_managed_session_ids()
            if (path := self.root / session_id / "subagents" / component / "state.json").exists()
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(f"Subagent execution ID is not globally unique: {execution_id!r}")
        return self._read_model(matches[0], SubagentStateFile)

    def list_subagents(self, session_id: str | None = None) -> tuple[SubagentStateFile, ...]:
        managed_ids = self.list_managed_session_ids()
        if session_id is not None:
            component = _safe_component(session_id, label="session ID")
            managed_ids = (component,) if component in managed_ids else ()
        paths = (
            path for managed_id in managed_ids for path in (self.root / managed_id / "subagents").glob("*/state.json")
        )
        states = [self._read_model(path, SubagentStateFile) for path in paths]
        return tuple(sorted(states, key=lambda state: (state.created_at, state.record.execution_id)))

    def session_has_nonterminal_subagents(self, session_id: str) -> bool:
        return any(not state.record.terminal for state in self.list_subagents(session_id))

    def fence_subagents(self, session_id: str, *, reason: str, now: datetime) -> tuple[str, ...]:
        """Cancel nonterminal children and reject unresolved child input under the session lock."""
        changed: list[str] = []
        for state in self.list_subagents(session_id):
            record = state.record
            if not record.terminal:
                record = record.model_copy(
                    update={
                        "state": SubagentExecutionState.cancelled,
                        "input_state": (
                            SubagentInputState.applied
                            if record.input_state is SubagentInputState.applied
                            else SubagentInputState.rejected
                        ),
                        "error": reason,
                        "completed_at": now,
                    }
                )
            inputs = tuple(
                item.model_copy(
                    update={
                        "state": InputState.rejected,
                        "rejection_reason": reason,
                        "updated_at": now,
                    }
                )
                if item.state in {InputState.accepted, InputState.enqueued}
                else item
                for item in state.inputs
            )
            updated = state.model_copy(
                update={
                    "record": record,
                    "inputs": inputs,
                    "input_open": False,
                    "cancel_requested": True,
                    "cancellation_reason": reason,
                    "updated_at": now,
                }
            )
            if updated != state:
                self.write_subagent(updated)
                changed.append(record.execution_id)
        return tuple(changed)

    def remove_revision(self, session_id: str, revision_id: str) -> None:
        self._remove_state_parent(self.revision_path(session_id, revision_id))

    def remove_checkpoint(self, session_id: str, execution_id: str) -> None:
        self._remove_state_parent(self.checkpoint_path(session_id, execution_id))

    def remove_session(self, session_id: str) -> None:
        path = self.session_dir(session_id)
        if path.exists():
            if path.is_symlink():
                raise RuntimeError(f"Session state directory must not be a symlink: {path}")
            shutil.rmtree(path)
            _fsync_directory(self.root)

    def list_managed_session_ids(self) -> tuple[str, ...]:
        """Return only directories carrying this store's explicit ownership marker."""
        session_ids: list[str] = []
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_dir():
                continue
            marker = path / _SESSION_MARKER_NAME
            if (
                marker.is_file()
                and not marker.is_symlink()
                and marker.read_text(encoding="utf-8") == self.marker_content
            ):
                session_ids.append(path.name)
        return tuple(sorted(session_ids))

    def remove_session_orphans(
        self,
        session_id: str,
        *,
        revision_ids: set[str],
        checkpoint_ids: set[str],
        session_exists: bool,
    ) -> int:
        """Clean one marked session directory while its lifecycle lock is held."""
        if session_id not in self.list_managed_session_ids():
            return 0
        if not session_exists:
            self.remove_session(session_id)
            return 1

        removed = 0
        session_dir = self.session_dir(session_id)
        removed += self._clean_state_category(
            session_dir / "revisions",
            retained_ids=revision_ids,
        )
        removed += self._clean_state_category(
            session_dir / "checkpoints",
            retained_ids=checkpoint_ids,
        )
        removed += self._clean_state_category(
            session_dir / "subagents",
            retained_ids=None,
        )
        return removed

    def _clean_state_category(
        self,
        category_dir: Path,
        *,
        retained_ids: set[str] | None,
    ) -> int:
        if not category_dir.exists():
            return 0
        self._assert_safe_path(category_dir)
        if category_dir.is_symlink() or not category_dir.is_dir():
            raise RuntimeError(f"Durable state category must be a real directory: {category_dir}")
        removed = 0
        for state_dir in tuple(category_dir.iterdir()):
            self._assert_safe_path(state_dir)
            if state_dir.is_symlink() or not state_dir.is_dir():
                raise RuntimeError(f"Durable state entry must be a real directory: {state_dir}")
            state_path = state_dir / "state.json"
            if retained_ids is not None and state_dir.name not in retained_ids:
                shutil.rmtree(state_dir)
                removed += 1
                continue
            if retained_ids is None and not state_path.exists():
                shutil.rmtree(state_dir)
                removed += 1
                continue
            for temporary in state_dir.glob(".state.json.*.tmp"):
                self._assert_safe_path(temporary)
                temporary.unlink(missing_ok=True)
                removed += 1
        if removed:
            _fsync_directory(category_dir)
        return removed

    def _validate_session_marker(self, session_dir: Path) -> None:
        marker = session_dir / _SESSION_MARKER_NAME
        if not marker.is_file() or marker.is_symlink() or marker.read_text(encoding="utf-8") != self.marker_content:
            raise RuntimeError(f"Invalid durable session directory marker: {marker}")

    def _ensure_directory(self, path: Path) -> None:
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            if current.exists():
                if current.is_symlink() or not current.is_dir():
                    raise RuntimeError(f"Durable state path must be a real directory: {current}")
            else:
                current.mkdir()
                _fsync_directory(current.parent)

    def _write_model(self, path: Path, model: BaseModel) -> None:
        self._assert_safe_path(path)
        if path.exists() and path.is_symlink():
            raise RuntimeError(f"Durable state file must not be a symlink: {path}")
        payload = (
            json.dumps(
                model.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_model(self, path: Path, model_type: type[_ModelT]) -> _ModelT:
        self._assert_safe_path(path)
        if path.is_symlink():
            raise RuntimeError(f"Durable state file must not be a symlink: {path}")
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def _remove_state_parent(self, path: Path) -> None:
        self._assert_safe_path(path)
        directory = path.parent
        if directory.exists():
            if directory.is_symlink():
                raise RuntimeError(f"Durable state directory must not be a symlink: {directory}")
            shutil.rmtree(directory)
            _fsync_directory(directory.parent)

    def _assert_safe_path(self, path: Path) -> None:
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise RuntimeError(f"Durable state path must not contain symlinks: {current}")


def _lock_descriptor(descriptor: int, *, blocking: bool) -> bool:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(descriptor, mode, 1)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    import fcntl

    mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, mode)
    except BlockingIOError:
        return False
    return True


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _safe_component(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
