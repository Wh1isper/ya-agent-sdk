"""Local environment implementations.

This module provides local file system and shell implementations
using standard library functions.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import anyio
from ya_agent_environment import (
    DEFAULT_CHUNK_SIZE,
    Environment,
    ExecutionHandle,
    FileEntry,
    FileOperationError,
    FileOperator,
    FileStat,
    PathNotAllowedError,
    ResourceFactory,
    ResourceRegistryState,
    Shell,
    ShellExecutionError,
    StdinAdapter,
)

from ya_agent_sdk.environment.process import (
    kill_process_tree,
    process_group_kwargs,
    send_process_tree_signal,
)
from ya_agent_sdk.environment.virtual_path import (
    VirtualPath,
    as_virtual_path,
    normalize_virtual_path,
)

if TYPE_CHECKING:
    from ya_agent_sdk.environment.shell_sandbox.policy import ShellSandboxRuntimePolicy


def _default_shell_executable() -> str | None:
    """Return the default local shell executable for create_subprocess_shell."""
    if os.name != "posix":
        return None

    bash_path = Path("/bin/bash")
    if bash_path.exists():
        return str(bash_path)

    return shutil.which("bash")


def _resolve_shell_executable(shell_executable: str | None) -> str | None:
    """Resolve the configured shell executable.

    None selects the YA Agent SDK platform default. An empty string delegates
    shell selection to Python's platform default.
    """
    if shell_executable == "":
        return None
    if shell_executable is None:
        return _default_shell_executable()
    return shell_executable


_LOCAL_GUARDIAN_CATCHABLE_SIGNAL_NAMES = (
    "SIGHUP",
    "SIGINT",
    "SIGQUIT",
    "SIGTERM",
    "SIGUSR1",
    "SIGUSR2",
    "SIGALRM",
    "SIGPIPE",
    "SIGPOLL",
    "SIGIO",
    "SIGPROF",
    "SIGVTALRM",
    "SIGXCPU",
    "SIGXFSZ",
    "SIGWINCH",
    "SIGURG",
    "SIGTSTP",
    "SIGTTIN",
    "SIGTTOU",
    "SIGCONT",
)
_LOCAL_GUARDIAN_CATCHABLE_SIGNALS = frozenset(
    int(member)
    for name in _LOCAL_GUARDIAN_CATCHABLE_SIGNAL_NAMES
    if isinstance((member := getattr(signal, name, None)), signal.Signals)
)
_LOCAL_GUARDIAN_SUPPORTED_SIGNALS = _LOCAL_GUARDIAN_CATCHABLE_SIGNALS | frozenset(
    int(member)
    for name in ("SIGKILL", "SIGSTOP")
    if isinstance((member := getattr(signal, name, None)), signal.Signals)
)


_LOCAL_PROCESS_GUARDIAN = """
import os
import signal
import subprocess
import sys
import traceback

ready_fd = int(sys.argv[1])
status_fd = int(sys.argv[2])
catchable_signals = tuple(int(value) for value in sys.argv[3].split(",") if value)
try:
    try:
        child = subprocess.Popen(
            sys.argv[4:],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            close_fds=True,
        )
    except BaseException:
        traceback.print_exc()
        command_status = 125
    else:
        def _keep_guardian_alive(_signum, _frame):
            return None

        for signum in catchable_signals:
            signal.signal(signum, _keep_guardian_alive)
        os.write(ready_fd, b"ready\\n")
        os.close(ready_fd)
        ready_fd = -1
        command_status = child.wait()
    os.write(status_fd, f"{command_status}\\n".encode("ascii"))
finally:
    if ready_fd >= 0:
        os.close(ready_fd)
    os.close(status_fd)
while True:
    signal.pause()
""".strip()


def _read_local_guardian_ready(read_fd: int) -> None:
    """Require readiness after the guardian has installed every accepted handler."""
    with os.fdopen(read_fd, "rb", closefd=True) as stream:
        readiness = stream.readline(32)
    if readiness != b"ready\n":
        raise RuntimeError("Local process guardian exited before publishing signal readiness")


def _read_local_guardian_status(read_fd: int) -> int:
    """Read one command status while the stable process-group guardian remains alive."""
    with os.fdopen(read_fd, "rb", closefd=True) as stream:
        raw_status = stream.readline(32)
    if not raw_status:
        raise RuntimeError("Local process guardian exited before reporting command completion")
    try:
        status = int(raw_status)
    except ValueError as exc:
        raise RuntimeError("Local process guardian returned an invalid command status") from exc
    if not -(2**31) <= status < 2**31:
        raise RuntimeError("Local process guardian returned an invalid command status")
    return status


def _read_path_bytes(path: Path, *, offset: int = 0, length: int | None = None) -> bytes:
    """Read bytes from a local path using seek for bounded reads."""
    with path.open("rb") as stream:
        if offset > 0:
            stream.seek(offset)
        if length is not None:
            return stream.read(length)
        return stream.read()


def _shell_type_from_executable(shell_executable: str | None) -> str:
    """Return a shell type label for context instructions."""
    if shell_executable is None:
        return "platform-default"

    shell_name = Path(shell_executable).name
    if shell_name.endswith(".exe"):
        shell_name = shell_name[:-4]
    return shell_name or "custom"


class LocalFileOperator(FileOperator):
    """Local file system operator with path validation.

    Implements the FileOperator ABC for local file system access.
    Validates all paths against a list of allowed directories.

    Temporary directories are ordinary allowed paths supplied by an Environment.
    """

    def __init__(
        self,
        default_path: Path | None = None,
        allowed_paths: Sequence[Path | PurePath] | None = None,
        instructions_paths: Sequence[Path | PurePath] | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = 3,
    ):
        """Initialize LocalFileOperator.

        Args:
            default_path: Default working directory for operations.
                If None, no real filesystem access is available (only tmp operations).
            allowed_paths: Directories accessible for file operations.
                If None, defaults to [default_path] when default_path is set.
            instructions_paths: Directories included in generated file-tree context.
                If None, all allowed_paths are included.
            instructions_skip_dirs: Directories to skip in file tree generation.
            instructions_max_depth: Maximum depth for file tree generation.
        """
        # Fallback: use first allowed_path as default when only allowed_paths is provided
        if default_path is None and allowed_paths:
            first_allowed_path = allowed_paths[0]
            default_path = first_allowed_path if isinstance(first_allowed_path, Path) else None

        super().__init__(
            default_path=default_path,
            allowed_paths=allowed_paths,
            instructions_paths=instructions_paths,
            instructions_skip_dirs=instructions_skip_dirs,
            instructions_max_depth=instructions_max_depth,
        )

    def _local_default_path(self) -> Path | None:
        default_path = self._default_path
        return default_path if isinstance(default_path, Path) else None

    def _local_allowed_paths(self) -> list[Path]:
        return [path for path in self._allowed_paths if isinstance(path, Path)]

    def _resolve_path(self, path: str) -> Path:
        """Normalize a lexical operand and validate its resolved target."""
        default_path = self._local_default_path()
        if default_path is None:
            raise PathNotAllowedError(path, [])
        target = Path(path)
        if not target.is_absolute():
            target = default_path / target
        lexical = Path(os.path.abspath(target))
        if not self._is_path_allowed(lexical.resolve()):
            raise PathNotAllowedError(
                path,
                [str(p) for p in self._local_allowed_paths()],
            )
        return lexical

    def _is_path_allowed(self, resolved: Path) -> bool:
        """Check if resolved path is within allowed directories."""
        for allowed in self._local_allowed_paths():
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Read file content as string.

        Args:
            path: File path.
            encoding: Text encoding (default: utf-8).
            offset: Character offset to start reading from (default: 0).
            length: Maximum number of characters to read (default: None = read all).

        Returns:
            File content as string (or substring if offset/length specified).
        """
        resolved = self._resolve_path(path)
        try:
            content = await anyio.Path(resolved).read_text(encoding=encoding)
            if offset > 0 or length is not None:
                end = None if length is None else offset + length
                content = content[offset:end]
            return content
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read file content as bytes.

        Args:
            path: File path.
            offset: Byte offset to start reading from (default: 0).
            length: Maximum number of bytes to read (default: None = read all).

        Returns:
            File content as bytes (or slice if offset/length specified).
        """
        resolved = self._resolve_path(path)
        try:
            return await anyio.to_thread.run_sync(  # type: ignore[reportAttributeAccessIssue]
                lambda: _read_path_bytes(resolved, offset=offset, length=length)
            )
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read a local file as a bounded-memory byte stream."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        resolved = self._resolve_path(path)
        try:
            file = await anyio.open_file(resolved, "rb")
            try:
                while chunk := await file.read(chunk_size):
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await file.aclose()
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write content to file."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            if isinstance(content, bytes):
                await apath.write_bytes(content)
            else:
                await apath.write_text(content, encoding=encoding)
        except PermissionError as e:
            raise FileOperationError("write", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("write", path, str(e)) from e

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Append content to file."""
        resolved = self._resolve_path(path)
        try:
            # anyio.Path doesn't support append mode, use sync in thread
            def _append():
                mode = "ab" if isinstance(content, bytes) else "a"
                with open(resolved, mode, encoding=None if isinstance(content, bytes) else encoding) as f:
                    f.write(content)

            await anyio.to_thread.run_sync(_append)  # type: ignore[reportAttributeAccessIssue]
        except PermissionError as e:
            raise FileOperationError("append", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("append", path, str(e)) from e

    async def delete(self, path: str) -> None:
        """Delete file or empty directory."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            if await apath.is_symlink() or not await apath.is_dir():
                await apath.unlink()
            else:
                await apath.rmdir()
        except FileNotFoundError as e:
            raise FileOperationError("delete", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("delete", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("delete", path, str(e)) from e

    async def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            entries = []
            async for entry in apath.iterdir():
                entries.append(entry.name)
            return sorted(entries)
        except FileNotFoundError as e:
            raise FileOperationError("list", path, "directory not found") from e
        except NotADirectoryError as e:
            raise FileOperationError("list", path, "not a directory") from e
        except PermissionError as e:
            raise FileOperationError("list", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("list", path, str(e)) from e

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).exists()

    async def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).is_file()

    async def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).is_dir()

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create directory."""
        resolved = self._resolve_path(path)
        try:
            await anyio.Path(resolved).mkdir(parents=parents, exist_ok=True)
        except PermissionError as e:
            raise FileOperationError("mkdir", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("mkdir", path, str(e)) from e

    async def move(self, src: str, dst: str) -> None:
        """Move file or directory."""
        src_resolved = self._resolve_path(src)
        dst_resolved = self._resolve_path(dst)
        try:
            await anyio.to_thread.run_sync(lambda: shutil.move(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("move", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("move", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("move", src, str(e)) from e

    async def copy(self, src: str, dst: str) -> None:
        """Copy file or directory."""
        src_resolved = self._resolve_path(src)
        dst_resolved = self._resolve_path(dst)
        try:
            if src_resolved.is_dir():
                await anyio.to_thread.run_sync(lambda: shutil.copytree(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
            else:
                await anyio.to_thread.run_sync(lambda: shutil.copy2(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("copy", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("copy", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("copy", src, str(e)) from e

    async def stat(self, path: str) -> FileStat:
        """Get file/directory status information."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            st = await apath.stat()
            return FileStat(
                size=st.st_size,
                mtime=st.st_mtime,
                is_file=await apath.is_file(),
                is_dir=await apath.is_dir(),
            )
        except FileNotFoundError as e:
            raise FileOperationError("stat", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("stat", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("stat", path, str(e)) from e

    async def walk_files(  # noqa: C901
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[FileEntry]:
        """Walk files and directories under the local default path."""
        default_path = self._local_default_path()
        if default_path is None:
            return
        unresolved_root = Path(root)
        if not unresolved_root.is_absolute():
            unresolved_root = default_path / unresolved_root
        if not follow_symlinks and unresolved_root.is_symlink():
            return
        resolved_root = self._resolve_path(root)
        if not await anyio.Path(resolved_root).exists():
            return

        def _walk() -> list[FileEntry]:  # noqa: C901
            entries: list[FileEntry] = []
            if resolved_root.is_file():
                stat = resolved_root.stat()
                path = resolved_root.relative_to(default_path).as_posix()
                entries.append(FileEntry(path=path, is_file=True, is_dir=False, size=stat.st_size, mtime=stat.st_mtime))
                return entries

            root_depth = len(resolved_root.parts)
            for current, dirnames, filenames in os.walk(resolved_root, followlinks=follow_symlinks):
                current_path = Path(current)
                depth = len(current_path.parts) - root_depth
                if max_depth is not None and depth >= max_depth:
                    dirnames[:] = []
                if not include_hidden:
                    dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                    filenames = [name for name in filenames if not name.startswith(".")]
                for name in sorted(dirnames):
                    path = current_path / name
                    try:
                        if follow_symlinks:
                            resolved = path.resolve()
                            if not self._is_path_allowed(resolved):
                                continue
                            rel = resolved.relative_to(default_path).as_posix()
                            stat = path.stat()
                        else:
                            rel = path.relative_to(default_path).as_posix()
                            stat = path.lstat()
                    except (OSError, ValueError):
                        continue
                    entries.append(
                        FileEntry(path=rel, is_file=False, is_dir=True, size=stat.st_size, mtime=stat.st_mtime)
                    )
                for name in sorted(filenames):
                    path = current_path / name
                    try:
                        if follow_symlinks:
                            resolved = path.resolve()
                            if not self._is_path_allowed(resolved):
                                continue
                            rel = resolved.relative_to(default_path).as_posix()
                            stat = path.stat()
                        else:
                            rel = path.relative_to(default_path).as_posix()
                            stat = path.lstat()
                    except (OSError, ValueError):
                        continue
                    entries.append(
                        FileEntry(path=rel, is_file=True, is_dir=False, size=stat.st_size, mtime=stat.st_mtime)
                    )
            return entries

        for entry in await anyio.to_thread.run_sync(_walk):  # type: ignore[reportAttributeAccessIssue]
            yield entry


@dataclass(frozen=True)
class VirtualMount:
    """Maps a host directory to a virtual path.

    Used by VirtualLocalFileOperator and SandboxEnvironment to define
    path mappings between host filesystem and virtual path space.

    Attributes:
        host_path: Actual directory on the host filesystem.
        virtual_path: Virtual path presented to the agent. Must be absolute.
    """

    host_path: Path
    virtual_path: Path | VirtualPath

    def __post_init__(self) -> None:
        virtual_path = normalize_virtual_path(self.virtual_path)
        if not virtual_path.is_absolute():
            raise ValueError(f"virtual_path must be absolute, got: {self.virtual_path}")
        object.__setattr__(self, "virtual_path", virtual_path)


class VirtualLocalFileOperator(FileOperator):
    """File operator that presents a virtual path space while performing I/O on the host filesystem.

    Supports multiple mount mappings between virtual paths (what the agent sees)
    and host paths (where actual I/O happens). This enables symmetric path spaces
    between file operations and shell execution in sandboxed environments.

    Path resolution uses longest-prefix matching when multiple mounts are configured.

    Example:
        Single mount:

        ```python
        op = VirtualLocalFileOperator(
            mounts=[VirtualMount(Path("/home/user/project"), Path("/workspace"))],
        )
        # Agent reads "/workspace/test.py" -> reads /home/user/project/test.py
        content = await op.read_file("test.py")
        ```

        Multiple mounts:

        ```python
        op = VirtualLocalFileOperator(
            mounts=[
                VirtualMount(Path("/home/user/project"), Path("/workspace/project")),
                VirtualMount(Path("/home/user/.config"), Path("/workspace/.config")),
            ],
        )
        await op.read_file("/workspace/project/main.py")   # -> /home/user/project/main.py
        await op.read_file("/workspace/.config/settings")   # -> /home/user/.config/settings
        ```
    """

    def __init__(
        self,
        mounts: list[VirtualMount],
        default_virtual_path: Path | VirtualPath | None = None,
        instructions_paths: Sequence[Path | PurePath] | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = 3,
    ):
        """Initialize VirtualLocalFileOperator.

        Args:
            mounts: List of mount mappings from host paths to virtual paths.
                At least one mount is required. All virtual_paths must be absolute.
            default_virtual_path: Default virtual path for relative path resolution.
                If None, uses the first mount's virtual_path.
            instructions_paths: Virtual paths included in generated file-tree context.
            instructions_skip_dirs: Directories to skip in file tree generation.
            instructions_max_depth: Maximum depth for file tree generation.
        """
        self._mounts = mounts
        default_vp = (
            normalize_virtual_path(default_virtual_path)
            if default_virtual_path is not None
            else (mounts[0].virtual_path if mounts else None)
        )

        super().__init__(
            default_path=default_vp,
            allowed_paths=[m.virtual_path for m in mounts],
            instructions_paths=instructions_paths,
            instructions_skip_dirs=instructions_skip_dirs,
            instructions_max_depth=instructions_max_depth,
        )

    def _find_mount(self, normalized_virtual: VirtualPath) -> VirtualMount:
        """Find the mount whose virtual_path is the longest prefix of the given path.

        Args:
            normalized_virtual: Normalized absolute virtual path.

        Returns:
            The best-matching VirtualMount.

        Raises:
            PathNotAllowedError: If no mount matches the path.
        """
        best: VirtualMount | None = None
        best_depth = -1
        for mount in self._mounts:
            try:
                normalized_virtual.relative_to(mount.virtual_path)
                depth = len(mount.virtual_path.parts)
                if depth > best_depth:
                    best = mount
                    best_depth = depth
            except ValueError:
                continue
        if best is None:
            raise PathNotAllowedError(
                str(normalized_virtual),
                [str(m.virtual_path) for m in self._mounts],
            )
        return best

    def _resolve_virtual(self, path: str) -> VirtualPath:
        """Resolve a virtual path to a normalized absolute virtual path.

        Args:
            path: Virtual path (relative or absolute).

        Returns:
            Normalized absolute virtual Path.

        Raises:
            PathNotAllowedError: If the path is outside all mount virtual paths,
                or if no default path is configured for relative path resolution.
        """
        if self._default_path is None:
            raise PathNotAllowedError(path, [])
        target = as_virtual_path(path)
        if not target.is_absolute():
            target = as_virtual_path(self._default_path) / target
        normalized = normalize_virtual_path(target)

        # Validate: must be under at least one mount
        self._find_mount(normalized)
        return normalized

    def _to_host(self, path: str) -> Path:
        """Translate a virtual path to a validated lexical host operand.

        Uses longest-prefix matching to find the appropriate mount. The resolved
        target is checked for symlink escapes, while the returned path preserves
        the final symlink so delete and move operate on the link itself.

        Args:
            path: Virtual path (relative or absolute).

        Returns:
            Lexically normalized host Path for actual I/O.
        """
        virtual = self._resolve_virtual(path)
        mount = self._find_mount(virtual)
        rel = virtual.relative_to(mount.virtual_path)
        mount_root = mount.host_path.resolve()
        lexical = Path(os.path.abspath(mount_root / Path(rel.as_posix())))

        # Security: verify the resolved target has not escaped via symlinks.
        try:
            lexical.resolve().relative_to(mount_root)
        except ValueError as exc:
            raise PathNotAllowedError(f"Path escapes mount boundary via symlink: {path}") from exc

        return lexical

    def _find_mount_for_host(self, host_path: Path) -> VirtualMount | None:
        """Find the mount that contains a host path.

        Args:
            host_path: Absolute host path.

        Returns:
            The matching VirtualMount, or None if no mount matches.
        """
        best: VirtualMount | None = None
        best_depth = -1
        for mount in self._mounts:
            resolved_host = mount.host_path.resolve()
            try:
                host_path.relative_to(resolved_host)
                depth = len(resolved_host.parts)
                if depth > best_depth:
                    best = mount
                    best_depth = depth
            except ValueError:
                continue
        return best

    def _to_virtual_rel(self, host_path: Path) -> str | None:
        """Translate a host path back to a virtual-relative path string.

        Uses longest-prefix matching to find the appropriate mount.

        Args:
            host_path: Absolute or relative host path.

        Returns:
            Path string relative to the default virtual path, or None if
            the host path is outside all mounts.
        """
        if not host_path.is_absolute():
            # For relative paths, try default mount first
            mount = self._mounts[0]
            host_path = mount.host_path.resolve() / host_path

        found = self._find_mount_for_host(host_path)
        if found is not None:
            rel = host_path.relative_to(found.host_path.resolve())
            virtual_abs = found.virtual_path / rel.as_posix()
            # Return relative to default_path if possible
            if self._default_path is not None:
                try:
                    return str(virtual_abs.relative_to(self._default_path))
                except ValueError:
                    pass
            return str(virtual_abs)
        # Path is outside all mounts - return None to avoid leaking host paths
        return None

    # --- FileOperator _impl methods: translate then perform local I/O ---

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        host = self._to_host(path)
        try:
            content = await anyio.Path(host).read_text(encoding=encoding)
            if offset > 0 or length is not None:
                end = None if length is None else offset + length
                content = content[offset:end]
            return content
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        host = self._to_host(path)
        try:
            return await anyio.to_thread.run_sync(  # type: ignore[reportAttributeAccessIssue]
                lambda: _read_path_bytes(host, offset=offset, length=length)
            )
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read a virtual-path file as a bounded-memory byte stream."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        host = self._to_host(path)
        try:
            file = await anyio.open_file(host, "rb")
            try:
                while chunk := await file.read(chunk_size):
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await file.aclose()
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        host = self._to_host(path)
        try:
            apath = anyio.Path(host)
            if isinstance(content, bytes):
                await apath.write_bytes(content)
            else:
                await apath.write_text(content, encoding=encoding)
        except PermissionError as e:
            raise FileOperationError("write", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("write", path, str(e)) from e

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        host = self._to_host(path)
        try:

            def _append() -> None:
                mode = "ab" if isinstance(content, bytes) else "a"
                with open(host, mode, encoding=None if isinstance(content, bytes) else encoding) as f:
                    f.write(content)

            await anyio.to_thread.run_sync(_append)  # type: ignore[reportAttributeAccessIssue]
        except PermissionError as e:
            raise FileOperationError("append", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("append", path, str(e)) from e

    async def delete(self, path: str) -> None:
        host = self._to_host(path)
        try:
            apath = anyio.Path(host)
            if await apath.is_symlink() or not await apath.is_dir():
                await apath.unlink()
            else:
                await apath.rmdir()
        except FileNotFoundError as e:
            raise FileOperationError("delete", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("delete", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("delete", path, str(e)) from e

    async def list_dir(self, path: str) -> list[str]:
        host = self._to_host(path)
        try:
            entries: list[str] = []
            async for entry in anyio.Path(host).iterdir():
                entries.append(entry.name)
            return sorted(entries)
        except FileNotFoundError as e:
            raise FileOperationError("list", path, "directory not found") from e
        except NotADirectoryError as e:
            raise FileOperationError("list", path, "not a directory") from e
        except PermissionError as e:
            raise FileOperationError("list", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("list", path, str(e)) from e

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        host = self._to_host(path)
        try:
            result: list[tuple[str, bool]] = []
            async for entry in anyio.Path(host).iterdir():
                is_dir = await entry.is_dir()
                result.append((entry.name, is_dir))
            return sorted(result, key=lambda x: x[0])
        except FileNotFoundError as e:
            raise FileOperationError("list", path, "directory not found") from e
        except NotADirectoryError as e:
            raise FileOperationError("list", path, "not a directory") from e
        except PermissionError as e:
            raise FileOperationError("list", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("list", path, str(e)) from e

    async def exists(self, path: str) -> bool:
        host = self._to_host(path)
        return await anyio.Path(host).exists()

    async def is_file(self, path: str) -> bool:
        host = self._to_host(path)
        return await anyio.Path(host).is_file()

    async def is_dir(self, path: str) -> bool:
        host = self._to_host(path)
        return await anyio.Path(host).is_dir()

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        host = self._to_host(path)
        try:
            await anyio.Path(host).mkdir(parents=parents, exist_ok=True)
        except PermissionError as e:
            raise FileOperationError("mkdir", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("mkdir", path, str(e)) from e

    async def move(self, src: str, dst: str) -> None:
        src_host = self._to_host(src)
        dst_host = self._to_host(dst)
        try:
            await anyio.to_thread.run_sync(lambda: shutil.move(src_host, dst_host))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("move", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("move", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("move", src, str(e)) from e

    async def copy(self, src: str, dst: str) -> None:
        src_host = self._to_host(src)
        dst_host = self._to_host(dst)
        try:
            if src_host.is_dir():
                await anyio.to_thread.run_sync(lambda: shutil.copytree(src_host, dst_host, symlinks=True))  # type: ignore[reportAttributeAccessIssue]
            else:
                await anyio.to_thread.run_sync(lambda: shutil.copy2(src_host, dst_host))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("copy", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("copy", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("copy", src, str(e)) from e

    async def stat(self, path: str) -> FileStat:
        host = self._to_host(path)
        try:
            apath = anyio.Path(host)
            st = await apath.stat()
            return FileStat(
                size=st.st_size,
                mtime=st.st_mtime,
                is_file=await apath.is_file(),
                is_dir=await apath.is_dir(),
            )
        except FileNotFoundError as e:
            raise FileOperationError("stat", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("stat", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("stat", path, str(e)) from e

    async def walk_files(  # noqa: C901
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[FileEntry]:
        """Walk files and directories under the virtual default path."""
        if self._default_path is None:
            return
        root_virtual = as_virtual_path(root)
        if not root_virtual.is_absolute():
            root_virtual = normalize_virtual_path(as_virtual_path(self._default_path) / root_virtual)
        mount = self._find_mount(root_virtual)
        mount_relative_root = root_virtual.relative_to(mount.virtual_path)
        unresolved_host_root = mount.host_path.resolve() / Path(mount_relative_root.as_posix())
        if not follow_symlinks and unresolved_host_root.is_symlink():
            return
        host_root = self._to_host(str(root_virtual))
        default_virtual = as_virtual_path(self._default_path)
        if not await anyio.Path(host_root).exists():
            return

        def _logical_path(host_path: Path) -> str | None:
            virtual = self._to_virtual_rel(host_path.resolve())
            if virtual is None:
                return None
            virtual_path = as_virtual_path(virtual)
            if virtual_path.is_absolute():
                try:
                    return virtual_path.relative_to(default_virtual).as_posix()
                except ValueError:
                    return virtual_path.as_posix()
            return virtual

        def _walk() -> list[FileEntry]:  # noqa: C901
            entries: list[FileEntry] = []
            if host_root.is_file():
                logical = _logical_path(host_root)
                if logical is None:
                    return entries
                stat = host_root.stat()
                entries.append(
                    FileEntry(path=logical, is_file=True, is_dir=False, size=stat.st_size, mtime=stat.st_mtime)
                )
                return entries

            root_depth = len(host_root.parts)
            for current, dirnames, filenames in os.walk(host_root, followlinks=follow_symlinks):
                current_path = Path(current)
                depth = len(current_path.parts) - root_depth
                if max_depth is not None and depth >= max_depth:
                    dirnames[:] = []
                if not include_hidden:
                    dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                    filenames = [name for name in filenames if not name.startswith(".")]
                for name in sorted(dirnames):
                    path = current_path / name
                    logical = _logical_path(path)
                    if logical is None:
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    entries.append(
                        FileEntry(path=logical, is_file=False, is_dir=True, size=stat.st_size, mtime=stat.st_mtime)
                    )
                for name in sorted(filenames):
                    path = current_path / name
                    logical = _logical_path(path)
                    if logical is None:
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    entries.append(
                        FileEntry(path=logical, is_file=True, is_dir=False, size=stat.st_size, mtime=stat.st_mtime)
                    )
            return entries

        for entry in await anyio.to_thread.run_sync(_walk):  # type: ignore[reportAttributeAccessIssue]
            yield entry


class LocalShell(Shell):
    """Local shell command executor with optional sandbox policy.

    LocalShell is the single SDK implementation for host shell execution. With
    no sandbox policy it preserves raw subprocess behavior. When a
    ShellSandboxRuntimePolicy is provided and enabled, LocalShell routes command
    creation through the selected sandbox backend.
    """

    def __init__(
        self,
        default_cwd: Path | None = None,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        include_os_env: bool = True,
        shell_executable: str | None = None,
        environment_overrides: dict[str, str] | None = None,
        sandbox_policy: ShellSandboxRuntimePolicy | None = None,
    ):
        """Initialize LocalShell.

        Args:
            default_cwd: Default working directory for command execution.
                If None, commands cannot be executed (shell is non-functional).
            allowed_paths: Directories allowed as working directories.
                If None, defaults to [default_cwd] when default_cwd is set.
            default_timeout: Default timeout in seconds.
            include_os_env: Whether to include the parent process environment
                variables when an explicit env dict is provided to execute().
                When True (default), os.environ is merged as the base layer.
                When False, only the explicitly provided env dict is used.
                Note: raw mode with env=None naturally inherits os.environ;
                sandbox mode always builds an explicit policy-filtered env.
            shell_executable: Shell executable used by local shell execution.
                Defaults to /bin/bash on POSIX systems when available and
                Python's platform default shell otherwise.
            environment_overrides: Environment values injected before per-call
                env values. Used by workspace providers to pass runtime secrets
                and tool configuration into shell commands.
            sandbox_policy: Optional resolved shell sandbox policy. When
                provided and enabled, backend, network, mounts, environment
                allowlist, and raw host allowance affect process creation.
        """
        # Fallback: use first allowed_path as default when only allowed_paths is provided
        if default_cwd is None and allowed_paths:
            default_cwd = allowed_paths[0]

        super().__init__(
            default_cwd=default_cwd,
            allowed_paths=allowed_paths,
            default_timeout=default_timeout,
        )
        self._include_os_env = include_os_env
        self._shell_executable = _resolve_shell_executable(shell_executable)
        self._platform_name = sys.platform
        self._environment_overrides = dict(environment_overrides or {})
        self._sandbox_policy = sandbox_policy

    def _resolve_cwd(self, cwd: str | None) -> Path:
        """Resolve and validate working directory."""
        if cwd is None:
            if self._default_cwd is None:
                raise ShellExecutionError("", stderr="No working directory configured")
            return self._default_cwd

        target = Path(cwd)
        if not target.is_absolute():
            if self._default_cwd is None:
                raise PathNotAllowedError(cwd, [])
            target = self._default_cwd / target
        resolved = target.resolve()

        if not self._is_path_allowed(resolved):
            raise PathNotAllowedError(
                cwd,
                [str(p) for p in self._allowed_paths],
            )
        return resolved

    def _is_path_allowed(self, resolved: Path) -> bool:
        """Check if resolved path is within allowed directories."""
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def _build_effective_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Build effective environment for subprocess."""
        requested = {**self._environment_overrides, **dict(env or {})}
        policy = self._sandbox_policy
        if policy is not None and policy.enabled:
            return self._build_sandbox_env(requested, policy.env_allowlist)
        return self._build_raw_env(env, requested)

    def _build_raw_env(self, env: dict[str, str] | None, requested: dict[str, str]) -> dict[str, str] | None:
        if requested:
            return {**os.environ, **requested} if self._include_os_env else requested
        if env is not None and self._include_os_env:
            return {**os.environ, **env}
        if env is None and not self._include_os_env:
            return {}
        return env

    def _build_sandbox_env(self, requested: dict[str, str], env_allowlist: tuple[str, ...]) -> dict[str, str]:
        allowlist = set(env_allowlist)
        if "*" in allowlist:
            return {**os.environ, **requested} if self._include_os_env else requested
        filtered = {key: value for key, value in requested.items() if key in allowlist}
        if self._include_os_env:
            for key in allowlist:
                if key in os.environ and key not in filtered:
                    filtered[key] = os.environ[key]
        if "HOME" not in filtered and "HOME" in os.environ:
            filtered["HOME"] = os.environ["HOME"]
        if "PATH" not in filtered and "PATH" in os.environ:
            filtered["PATH"] = os.environ["PATH"]
        return filtered

    def _shell_environment_instruction(self) -> str:
        shell_type = _shell_type_from_executable(self._shell_executable)
        parts = [
            "  <shell-environment>",
            f"    <platform>{self._platform_name}</platform>",
            f"    <shell-type>{shell_type}</shell-type>",
        ]
        if self._shell_executable is not None:
            parts.append(f"    <shell-executable>{self._shell_executable}</shell-executable>")
        parts.append("  </shell-environment>")
        return "\n".join(parts)

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about local shell capabilities."""
        instructions = await super().get_context_instructions()
        if instructions is None:
            return None
        instructions = instructions.replace(
            "\n  <default-timeout>", f"\n{self._shell_environment_instruction()}\n  <default-timeout>"
        )
        if self._sandbox_policy is None:
            return instructions
        metadata = self._sandbox_policy.to_metadata()
        sandbox_lines = [
            "  <shell-sandbox>",
            f"    <enabled>{str(self._sandbox_policy.enabled).lower()}</enabled>",
            f"    <profile>{self._sandbox_policy.profile}</profile>",
            f"    <backend>{self._sandbox_policy.backend}</backend>",
            f"    <network>{self._sandbox_policy.network}</network>",
            f"    <raw-host-allowed>{str(self._sandbox_policy.raw_shell_allowed).lower()}</raw-host-allowed>",
            "  </shell-sandbox>",
        ]
        insertion = "\n".join(sandbox_lines)
        note = (
            f"Commands run through YA shell sandbox policy {metadata['profile']} on backend {metadata['backend']}."
            if self._sandbox_policy.enabled
            else "Commands will be executed with the working directory validated."
        )
        return instructions.replace("\n  <note>", f"\n{insertion}\n  <note>").replace(
            "Commands will be executed with the working directory validated.",
            note,
        )

    def _resolve_execute_timeout(self, timeout: float | None) -> float | None:
        """Apply the sandbox default while preserving raw-shell no-timeout semantics."""
        sandbox_enabled = self._sandbox_policy is not None and self._sandbox_policy.enabled
        if timeout is None and sandbox_enabled:
            return self._default_timeout
        return timeout

    async def _create_process(  # noqa: C901
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a local subprocess and return an ExecutionHandle.

        Validates the command and working directory, creates an async
        subprocess with piped stdout/stderr, and returns stream handles
        and lifecycle callbacks.
        """
        if not command:
            raise ShellExecutionError("", stderr="Empty command")

        resolved_cwd = self._resolve_cwd(cwd)
        effective_env = self._build_effective_env(env)
        cleanup = lambda: None
        ready_read_fd: int | None = None
        ready_write_fd: int | None = None
        status_read_fd: int | None = None
        status_write_fd: int | None = None
        ready_task: asyncio.Task[None] | None = None
        status_task: asyncio.Task[int] | None = None

        try:
            sandbox_enabled = self._sandbox_policy is not None and self._sandbox_policy.enabled
            policy = self._sandbox_policy if sandbox_enabled else None
            process_args: list[str] | None = None
            if policy is not None:
                if policy.backend == "raw_host" and not policy.raw_shell_allowed:
                    raise ShellExecutionError(
                        command, stderr="Raw host shell backend is disabled by shell sandbox policy"
                    )
                if policy.backend != "raw_host":
                    from ya_agent_sdk.environment.shell_sandbox.backend import build_sandbox_command

                    process_args, cleanup = build_sandbox_command(
                        command=command,
                        cwd=resolved_cwd,
                        policy=policy,
                        shell_executable=self._shell_executable,
                    )

            if os.name == "posix":
                if process_args is None:
                    process_args = [self._shell_executable or "/bin/sh", "-c", command]
                ready_read_fd, ready_write_fd = os.pipe()
                status_read_fd, status_write_fd = os.pipe()
                catchable_signals = ",".join(str(signum) for signum in _LOCAL_GUARDIAN_CATCHABLE_SIGNALS)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    _LOCAL_PROCESS_GUARDIAN,
                    str(ready_write_fd),
                    str(status_write_fd),
                    catchable_signals,
                    *process_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resolved_cwd,
                    env=effective_env,
                    pass_fds=(ready_write_fd, status_write_fd),
                    **process_group_kwargs(),
                )
                os.close(ready_write_fd)
                ready_write_fd = None
                os.close(status_write_fd)
                status_write_fd = None
                ready_task = asyncio.create_task(
                    asyncio.to_thread(_read_local_guardian_ready, ready_read_fd),
                    name=f"local-shell-guardian-ready-{process.pid or id(process)}",
                )
                ready_read_fd = None
                status_task = asyncio.create_task(
                    asyncio.to_thread(_read_local_guardian_status, status_read_fd),
                    name=f"local-shell-guardian-status-{process.pid or id(process)}",
                )
                status_read_fd = None
            elif process_args is not None:
                process = await asyncio.create_subprocess_exec(
                    *process_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resolved_cwd,
                    env=effective_env,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resolved_cwd,
                    env=effective_env,
                    executable=self._shell_executable,
                )
        except Exception as e:
            for pipe_fd in (ready_read_fd, ready_write_fd, status_read_fd, status_write_fd):
                if pipe_fd is not None:
                    os.close(pipe_fd)
            cleanup()
            raise ShellExecutionError(command, stderr=str(e)) from e

        stdout = process.stdout
        if stdout is None:
            stdout = asyncio.StreamReader()
            stdout.feed_eof()
        stderr = process.stderr
        if stderr is None:
            stderr = asyncio.StreamReader()
            stderr.feed_eof()

        # On POSIX the direct child is a stable group/session guardian. It
        # reports the real command status over a private pipe, then remains
        # alive until this handle terminates the group. The PGID therefore
        # cannot be released and reused before cleanup.
        process_group_id = process.pid if status_task is not None else None
        group_termination_confirmed = False
        cleanup_completed = False
        process_signal_lock = asyncio.Lock()

        async def _terminate_group() -> None:
            nonlocal group_termination_confirmed
            async with process_signal_lock:
                if group_termination_confirmed:
                    return
                if os.name == "posix" and process_group_id is not None and process.returncode is not None:
                    try:
                        os.killpg(process_group_id, 0)
                    except ProcessLookupError:
                        await process.wait()
                        group_termination_confirmed = True
                        return
                    raise RuntimeError(
                        "Local process guardian exited before its process group could be safely terminated"
                    )
                await kill_process_tree(process, process_group_id=process_group_id)
                group_termination_confirmed = True

        def _cleanup() -> None:
            nonlocal cleanup_completed
            if cleanup_completed:
                return
            cleanup()
            cleanup_completed = True

        async def _wait() -> int:
            if status_task is None or ready_task is None:
                await process.wait()
                exit_code = process.returncode or 0
            else:
                try:
                    await asyncio.shield(ready_task)
                    exit_code = await asyncio.shield(status_task)
                except BaseException:
                    await _terminate_group()
                    _cleanup()
                    raise
            # The guardian still owns the numeric PGID here. Terminate residual
            # members before reaping it and releasing ownership.
            await _terminate_group()
            _cleanup()
            return exit_code

        async def _kill() -> None:
            await _terminate_group()
            if ready_task is not None:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(ready_task)
            if status_task is not None:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(status_task)
            _cleanup()

        async def _send_signal(sig: int) -> None:
            if ready_task is not None:
                if sig not in _LOCAL_GUARDIAN_SUPPORTED_SIGNALS:
                    raise ValueError(f"Unsupported LocalShell guardian signal number: {sig}")
                await asyncio.shield(ready_task)
                if sig == signal.SIGKILL:
                    await _kill()
                    return
            async with process_signal_lock:
                if group_termination_confirmed:
                    return
                if process_group_id is not None and process.returncode is not None:
                    return
                send_process_tree_signal(process, sig, process_group_id=process_group_id)

        stdin = StdinAdapter(process.stdin) if process.stdin is not None else None

        return ExecutionHandle(
            stdout=stdout,
            stderr=stderr,
            wait=_wait,
            kill=_kill,
            stdin=stdin,
            pid=process.pid,
            send_signal=_send_signal,
            communicate=process.communicate if status_task is None else None,
        )


class LocalEnvironment(Environment):
    """Local environment with filesystem and shell access.

    Creates LocalFileOperator and LocalShell with shared configuration,
    and manages temporary directory lifecycle.

    Example:
        Using AsyncExitStack (recommended for dependent contexts):

        ```python
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            env = await stack.enter_async_context(
                LocalEnvironment(allowed_paths=[Path("/workspace")])
            )
            ctx = await stack.enter_async_context(
                AgentContext(env=env)
            )
            await ctx.file_operator.read_file("test.txt")
        # Resources cleaned up when stack exits
        ```
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        instructions_paths: list[Path] | None = None,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        enable_tmp_dir: bool = True,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
        include_os_env: bool = True,
        shell_executable: str | None = None,
        environment_overrides: dict[str, str] | None = None,
        shell_sandbox_policy: ShellSandboxRuntimePolicy | None = None,
    ):
        """Initialize LocalEnvironment.

        Args:
            allowed_paths: Directories accessible by both file and shell operations.
            default_path: Default working directory for operations.
            instructions_paths: Directories included in generated file-tree context.
                If None, all allowed_paths are included.
            shell_timeout: Default shell command timeout.
            tmp_base_dir: Base directory for creating session temporary directory.
                If None, uses system default temp directory.
            enable_tmp_dir: Whether to create a session temporary directory.
                Defaults to True.
            resource_state: Optional state to restore resources from.
                Resources will be restored when entering the context.
            resource_factories: Optional dictionary of resource factories.
                Required for any resources in resource_state.
            include_os_env: Whether shell subprocesses include parent process
                environment variables when explicit env is provided.
                Passed through to LocalShell. See LocalShell for details.
            shell_executable: Shell executable used by LocalShell.
                Defaults to /bin/bash on POSIX systems when available and
                Python's platform default shell otherwise.
            environment_overrides: Environment values injected into shell commands.
            shell_sandbox_policy: Optional LocalShell sandbox policy. The default
                is raw local subprocess behavior for SDK and YAACLI compatibility.
        """
        super().__init__(
            resource_state=resource_state,
            resource_factories=resource_factories,
        )
        self._allowed_paths = allowed_paths
        self._default_path = default_path
        self._instructions_paths = instructions_paths
        self._shell_timeout = shell_timeout
        self._tmp_base_dir = tmp_base_dir
        self._enable_tmp_dir = enable_tmp_dir
        self._include_os_env = include_os_env
        self._shell_executable = shell_executable
        self._environment_overrides = dict(environment_overrides or {})
        self._shell_sandbox_policy = shell_sandbox_policy
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    async def _setup(self) -> None:
        """Initialize file operator, shell, and tmp directory."""
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="ya_agent_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name).resolve()
            self._tmp_dir = tmp_dir_path

        # Determine default_path: use provided value, or infer from allowed_paths.
        # Never fall back to Path.cwd() to avoid exposing the process working directory.
        default_path = self._default_path
        if default_path is None and self._allowed_paths:
            default_path = self._allowed_paths[0]

        # Build allowed_paths list
        allowed = list(self._allowed_paths) if self._allowed_paths else []
        if tmp_dir_path:
            allowed.append(tmp_dir_path)
        if default_path is not None and default_path.resolve() not in [p.resolve() for p in allowed]:
            allowed.append(default_path)

        # Always create file_operator when tmp_dir is available, so the agent
        # can still access temporary files (e.g., large output storage).
        # When default_path is None, the operator runs in "empty folder" mode:
        # only tmp operations are accessible, all other paths are rejected.
        if default_path is not None or tmp_dir_path is not None:
            instruction_paths = (
                self._instructions_paths
                if self._instructions_paths is not None
                else [path for path in allowed if path != tmp_dir_path]
            )
            self._file_operator = LocalFileOperator(
                default_path=default_path,
                allowed_paths=allowed or None,
                instructions_paths=instruction_paths,
            )

        # Shell requires a real working directory - not created with only tmp_dir.
        if default_path is not None:
            self._shell = LocalShell(
                default_cwd=default_path,
                allowed_paths=allowed or None,
                default_timeout=self._shell_timeout,
                include_os_env=self._include_os_env,
                shell_executable=self._shell_executable,
                environment_overrides=self._environment_overrides,
                sandbox_policy=self._shell_sandbox_policy,
            )

    async def _teardown(self) -> None:
        """Clean up tmp directory.

        Note: Do NOT null _file_operator or _shell here.
        The base Environment.__aexit__ calls close() on them after
        _teardown returns.  Nulling here would skip close() and
        leak background processes.
        """
        try:
            if self._tmp_dir_obj is not None:
                self._tmp_dir_obj.cleanup()
        finally:
            self._tmp_dir_obj = None
            self._tmp_dir = None
            self._file_operator = None
            self._shell = None
