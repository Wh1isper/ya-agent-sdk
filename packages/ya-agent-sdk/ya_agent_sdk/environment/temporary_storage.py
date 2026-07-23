"""Shared conventions for workspace-backed temporary storage."""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_TMP_DIR_NAME = ".tmp"
TMP_DIR_PREFIX = "ya-agent-"
TMP_GITIGNORE_CONTENT = b"*\n"
_POSIX_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable filesystem identity captured for one owned directory."""

    device: int
    inode: int


def _identity_from_stat(path_stat: os.stat_result) -> DirectoryIdentity:
    return DirectoryIdentity(device=path_stat.st_dev, inode=path_stat.st_ino)


def capture_directory_identity(path: Path) -> DirectoryIdentity:
    """Capture a directory identity without following a final symlink."""
    path_stat = path.lstat()
    if not stat.S_ISDIR(path_stat.st_mode):
        raise RuntimeError(f"Temporary storage path is not a directory: {path}")
    return _identity_from_stat(path_stat)


def verify_owned_tmp_directory(
    path: Path,
    *,
    parent_identity: DirectoryIdentity,
    directory_identity: DirectoryIdentity,
) -> bool:
    """Verify that a path still names the directory owned by this environment.

    Returns ``False`` when the owned instance was already removed. Parent replacement
    is always an error because the original instance may have been moved elsewhere.
    """
    current_parent_identity = capture_directory_identity(path.parent)
    if current_parent_identity != parent_identity:
        raise RuntimeError("Temporary storage parent ownership changed")
    try:
        current_directory_identity = capture_directory_identity(path)
    except FileNotFoundError:
        return False
    if current_directory_identity != directory_identity:
        raise RuntimeError("Temporary storage instance ownership changed")
    return True


def prepare_workspace_tmp_parent(workspace_root: Path) -> tuple[Path, DirectoryIdentity]:
    """Create the shared temporary parent and return its stable identity."""
    resolved_root = workspace_root.resolve()
    tmp_parent = resolved_root / WORKSPACE_TMP_DIR_NAME
    if os.name == "posix":
        directory_flags = _POSIX_DIRECTORY_FLAGS
        root_fd = os.open(resolved_root, directory_flags)
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(WORKSPACE_TMP_DIR_NAME, dir_fd=root_fd)
            try:
                parent_fd = os.open(WORKSPACE_TMP_DIR_NAME, directory_flags, dir_fd=root_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise RuntimeError("Temporary storage escapes the workspace") from exc
                raise
            try:
                parent_identity = _identity_from_stat(os.fstat(parent_fd))
            finally:
                os.close(parent_fd)
        finally:
            os.close(root_fd)
        return tmp_parent, parent_identity

    if tmp_parent.exists() or tmp_parent.is_symlink():
        lexical_parent = Path(os.path.abspath(tmp_parent))
        if tmp_parent.resolve() != lexical_parent:
            raise RuntimeError("Temporary storage escapes the workspace")
    tmp_parent.mkdir(exist_ok=True)
    try:
        tmp_parent.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("Temporary storage escapes the workspace") from exc
    return tmp_parent, capture_directory_identity(tmp_parent)


def create_owned_tmp_directory(
    parent: Path,
    *,
    parent_identity: DirectoryIdentity,
    instance_name: str,
    mode: int = 0o700,
) -> tuple[Path, DirectoryIdentity]:
    """Create one owned instance without following a replaced POSIX parent."""
    path = parent / instance_name
    if os.name == "posix":
        directory_flags = _POSIX_DIRECTORY_FLAGS
        parent_fd = os.open(parent, directory_flags)
        try:
            if _identity_from_stat(os.fstat(parent_fd)) != parent_identity:
                raise RuntimeError("Temporary storage parent ownership changed")
            os.mkdir(instance_name, mode=mode, dir_fd=parent_fd)
            try:
                directory_fd = os.open(instance_name, directory_flags, dir_fd=parent_fd)
                try:
                    directory_identity = _identity_from_stat(os.fstat(directory_fd))
                finally:
                    os.close(directory_fd)
            except Exception as create_error:
                try:
                    os.rmdir(instance_name, dir_fd=parent_fd)
                except OSError as cleanup_error:
                    raise ExceptionGroup(
                        f"Temporary storage creation failed and left an unclaimed instance: {path}",
                        [create_error, cleanup_error],
                    ) from create_error
                raise
        finally:
            os.close(parent_fd)
        return path, directory_identity

    if capture_directory_identity(parent) != parent_identity:
        raise RuntimeError("Temporary storage parent ownership changed")
    path.mkdir(mode=mode)
    directory_identity = capture_directory_identity(path)
    if capture_directory_identity(parent) != parent_identity:
        raise RuntimeError("Temporary storage parent ownership changed")
    return path, directory_identity


def configure_owned_tmp_directory(
    path: Path,
    *,
    parent_identity: DirectoryIdentity,
    directory_identity: DirectoryIdentity,
    mode: int,
    owner: tuple[int, int] | None = None,
) -> None:
    """Configure an already-registered instance through stable POSIX handles."""
    if os.name == "posix":
        directory_flags = _POSIX_DIRECTORY_FLAGS
        parent_fd = os.open(path.parent, directory_flags)
        try:
            if _identity_from_stat(os.fstat(parent_fd)) != parent_identity:
                raise RuntimeError("Temporary storage parent ownership changed")
            directory_fd = os.open(path.name, directory_flags, dir_fd=parent_fd)
            try:
                if _identity_from_stat(os.fstat(directory_fd)) != directory_identity:
                    raise RuntimeError("Temporary storage instance ownership changed")
                os.fchmod(directory_fd, mode)
                if owner is not None:
                    os.fchown(directory_fd, owner[0], owner[1])
            finally:
                os.close(directory_fd)
        finally:
            os.close(parent_fd)
        return

    if not verify_owned_tmp_directory(
        path,
        parent_identity=parent_identity,
        directory_identity=directory_identity,
    ):
        raise RuntimeError("Temporary storage instance disappeared during setup")
    path.chmod(mode)


def _write_all(descriptor: int, content: bytes) -> None:
    """Write every byte or propagate the underlying write failure."""
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Temporary storage marker write made no progress")
        remaining = remaining[written:]


def write_tmp_gitignore(
    tmp_dir: Path,
    *,
    parent_identity: DirectoryIdentity,
    directory_identity: DirectoryIdentity,
    owner: tuple[int, int] | None = None,
) -> Path:
    """Exclusively create the Git ignore marker without following a final symlink."""
    if not verify_owned_tmp_directory(
        tmp_dir,
        parent_identity=parent_identity,
        directory_identity=directory_identity,
    ):
        raise RuntimeError("Temporary storage instance disappeared during setup")
    gitignore_path = tmp_dir / ".gitignore"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    if os.name == "posix":
        directory_flags = _POSIX_DIRECTORY_FLAGS
        parent_fd = os.open(tmp_dir.parent, directory_flags)
        try:
            if _identity_from_stat(os.fstat(parent_fd)) != parent_identity:
                raise RuntimeError("Temporary storage parent ownership changed")
            directory_fd = os.open(tmp_dir.name, directory_flags, dir_fd=parent_fd)
            try:
                if _identity_from_stat(os.fstat(directory_fd)) != directory_identity:
                    raise RuntimeError("Temporary storage instance ownership changed")
                descriptor = os.open(".gitignore", flags, 0o600, dir_fd=directory_fd)
                try:
                    _write_all(descriptor, TMP_GITIGNORE_CONTENT)
                    if owner is not None:
                        os.fchown(descriptor, owner[0], owner[1])
                finally:
                    os.close(descriptor)
            finally:
                os.close(directory_fd)
        finally:
            os.close(parent_fd)
        return gitignore_path

    descriptor = os.open(gitignore_path, flags, 0o600)
    try:
        _write_all(descriptor, TMP_GITIGNORE_CONTENT)
    finally:
        os.close(descriptor)
    return gitignore_path


def _remove_directory_contents_fd(directory_fd: int) -> None:
    """Remove one owned directory tree without following symlink entries."""
    directory_flags = _POSIX_DIRECTORY_FLAGS
    for name in os.listdir(directory_fd):
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                _remove_directory_contents_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _remove_owned_tmp_directory_posix(
    path: Path,
    *,
    parent_identity: DirectoryIdentity,
    directory_identity: DirectoryIdentity,
) -> None:
    """Remove an instance through stable parent and instance directory handles."""
    directory_flags = _POSIX_DIRECTORY_FLAGS
    parent_fd = os.open(path.parent, directory_flags)
    try:
        if _identity_from_stat(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError("Temporary storage parent ownership changed")
        try:
            directory_fd = os.open(path.name, directory_flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return
        try:
            if _identity_from_stat(os.fstat(directory_fd)) != directory_identity:
                raise RuntimeError("Temporary storage instance ownership changed")
            _remove_directory_contents_fd(directory_fd)
            try:
                current_identity = _identity_from_stat(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False))
            except FileNotFoundError as exc:
                raise RuntimeError("Temporary storage instance ownership changed") from exc
            if current_identity != directory_identity:
                raise RuntimeError("Temporary storage instance ownership changed")
            os.rmdir(path.name, dir_fd=parent_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def remove_owned_tmp_directory(
    path: Path,
    *,
    parent_identity: DirectoryIdentity,
    directory_identity: DirectoryIdentity,
) -> None:
    """Remove one owned instance without following a replaced POSIX parent path."""
    if os.name == "posix":
        _remove_owned_tmp_directory_posix(
            path,
            parent_identity=parent_identity,
            directory_identity=directory_identity,
        )
        return
    if not verify_owned_tmp_directory(
        path,
        parent_identity=parent_identity,
        directory_identity=directory_identity,
    ):
        return
    shutil.rmtree(path)
