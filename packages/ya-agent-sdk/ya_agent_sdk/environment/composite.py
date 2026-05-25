"""Composite file operator for multi-backend routing.

Routes file operations to different FileOperator backends based on
virtual mount path prefixes. Similar to a Unix VFS.

Each mount maps a virtual path prefix to a backend FileOperator.
The CompositeFileOperator resolves paths to the appropriate mount
using longest-prefix matching, then delegates operations to that
mount's backend.

Example:
    Agent has a local workspace and a remote PC mount:

    ```python
    local_op = LocalFileOperator(default_path=Path("/data/workspace"))
    remote_op = RemoteFileOperator(client=remote_client)

    composite = CompositeFileOperator(
        mounts=[
            Mount(Path("/workspace"), local_op, label="Agent workspace"),
            Mount(Path("/mnt/pc"), remote_op, label="User PC"),
        ],
        default_mount=Path("/workspace"),
    )

    # Routes to local_op
    await composite.read_file("main.py")  # relative -> default mount

    # Routes to remote_op
    await composite.read_file("/mnt/pc/project/src/app.py")

    # Cross-mount copy (streaming)
    await composite.copy("/mnt/pc/project/src/app.py", "/workspace/app.py")
    ```
"""

import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from ya_agent_environment import FileOperationError, FileOperator, FileStat, PathNotAllowedError, TmpFileOperator
from ya_agent_environment.file_operator import DEFAULT_CHUNK_SIZE

from ya_agent_sdk.environment.virtual_path import VirtualPath, as_virtual_path, normalize_virtual_path


class MountBackend(ABC):
    """Protocol for mount backends with lifecycle and status awareness.

    Wraps a FileOperator with additional metadata about the backend's
    availability and connection state. This allows CompositeFileOperator
    to generate accurate context instructions and handle offline backends
    gracefully.

    For simple cases where the backend is always available (e.g., local
    filesystem), use Mount directly with a plain FileOperator -- Mount
    auto-wraps it in a LocalMountBackend.
    """

    @property
    @abstractmethod
    def file_operator(self) -> FileOperator:
        """Return the underlying FileOperator."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the backend is currently available for operations.

        Default: always True. Remote backends should override this
        to reflect connection state.
        """
        return True

    @property
    def status(self) -> str:
        """Human-readable status string for context instructions.

        Default: 'online'. Remote backends may return 'offline',
        'reconnecting', etc.
        """
        return "online"

    @property
    def status_detail(self) -> str | None:
        """Optional detail about current status (e.g., 'last seen 2 minutes ago')."""
        return None

    @property
    def cached_file_tree(self) -> str | None:
        """Cached file tree to show when backend is unavailable.

        Remote backends can cache the file tree on connect and serve
        it when offline, so the agent can still reason about structure.
        """
        return None


class LocalMountBackend(MountBackend):
    """Mount backend for always-available local FileOperators."""

    def __init__(self, file_op: FileOperator) -> None:
        self._file_op = file_op

    @property
    def file_operator(self) -> FileOperator:
        return self._file_op


@dataclass(frozen=True)
class Mount:
    """Maps a virtual path prefix to a mount backend.

    The backend can be either:
    - A plain FileOperator (auto-wrapped in LocalMountBackend, always available)
    - A MountBackend instance (with status/availability tracking)

    Attributes:
        virtual_path: Virtual path prefix presented to the agent. Must be absolute.
        backend: MountBackend or FileOperator that handles operations under this mount.
        label: Human-readable label for context instructions.
        read_only: If True, write/delete/move/copy-to operations raise errors.
    """

    virtual_path: Path | VirtualPath
    backend: MountBackend | FileOperator
    label: str = ""
    read_only: bool = False
    _resolved_backend: MountBackend = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        virtual_path = normalize_virtual_path(self.virtual_path)
        if not virtual_path.is_absolute():
            raise ValueError(f"virtual_path must be absolute, got: {self.virtual_path}")
        object.__setattr__(self, "virtual_path", virtual_path)
        # Auto-wrap plain FileOperator in LocalMountBackend
        if isinstance(self.backend, MountBackend):
            object.__setattr__(self, "_resolved_backend", self.backend)
        else:
            object.__setattr__(self, "_resolved_backend", LocalMountBackend(self.backend))

    @property
    def mount_backend(self) -> MountBackend:
        """Return the resolved MountBackend (auto-wrapped if needed)."""
        return self._resolved_backend

    @property
    def file_operator(self) -> FileOperator:
        """Shortcut to the underlying FileOperator."""
        return self._resolved_backend.file_operator


class CompositeFileOperator(FileOperator):
    """Routes file operations to multiple FileOperator backends based on path prefix.

    Supports:
    - Heterogeneous backends (local, remote, S3, etc.) in a unified path space
    - Read-only mounts
    - Cross-mount copy/move via streaming
    - Mount-aware context instructions with per-mount file trees and status
    - Backend availability tracking (online/offline for remote mounts)
    - Transparent tmp file handling (inherited from base FileOperator)
    """

    def __init__(
        self,
        mounts: list[Mount],
        default_mount: Path | VirtualPath | None = None,
        tmp_dir: Path | None = None,
        tmp_file_operator: TmpFileOperator | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = 3,
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize CompositeFileOperator.

        Args:
            mounts: List of mount mappings. At least one is required.
                Each maps a virtual path prefix to a FileOperator backend.
            default_mount: Default virtual path for resolving relative paths.
                If None, uses the first mount's virtual_path.
            tmp_dir: Directory for temporary files. Handled by base class.
            tmp_file_operator: Custom tmp file operator. Takes precedence over tmp_dir.
            instructions_skip_dirs: Directories to skip in file tree generation.
            instructions_max_depth: Maximum depth for file tree generation.
            default_chunk_size: Default chunk size for streaming operations.

        Raises:
            ValueError: If mounts is empty.
        """
        if not mounts:
            raise ValueError("At least one mount is required")

        self._mounts = mounts
        default_path = normalize_virtual_path(default_mount) if default_mount is not None else mounts[0].virtual_path

        super().__init__(
            default_path=default_path,
            allowed_paths=[m.virtual_path for m in mounts],
            instructions_skip_dirs=instructions_skip_dirs,
            instructions_max_depth=instructions_max_depth,
            tmp_dir=tmp_dir,
            tmp_file_operator=tmp_file_operator,
            skip_instructions=True,  # We override get_context_instructions
            default_chunk_size=default_chunk_size,
        )

    @property
    def mounts(self) -> list[Mount]:
        """Return the list of mounts."""
        return list(self._mounts)

    def _find_mount(self, normalized: VirtualPath) -> Mount:
        """Find mount with longest-prefix match.

        Args:
            normalized: Normalized absolute virtual path.

        Returns:
            The best-matching Mount.

        Raises:
            PathNotAllowedError: If no mount matches the path.
        """
        best: Mount | None = None
        best_depth = -1
        for mount in self._mounts:
            try:
                normalized.relative_to(mount.virtual_path)
                depth = len(mount.virtual_path.parts)
                if depth > best_depth:
                    best = mount
                    best_depth = depth
            except ValueError:
                continue
        if best is None:
            raise PathNotAllowedError(
                str(normalized),
                [str(m.virtual_path) for m in self._mounts],
            )
        return best

    def _resolve(self, path: str) -> tuple[Mount, str]:
        """Resolve a virtual path to (mount, relative_path_within_mount).

        Relative paths are resolved against default_path (the default mount's virtual_path).
        Absolute paths are matched against mounts directly.

        Args:
            path: Virtual path (relative or absolute).

        Returns:
            Tuple of (matched_mount, relative_path_string).
            The relative path is "." for the mount root.

        Raises:
            PathNotAllowedError: If the path is outside all mounts.
        """
        if self._default_path is None:
            raise PathNotAllowedError(path, [])

        target = as_virtual_path(path)
        if not target.is_absolute():
            target = as_virtual_path(self._default_path) / target
        normalized = normalize_virtual_path(target)

        mount = self._find_mount(normalized)
        rel = normalized.relative_to(mount.virtual_path)
        rel_str = str(rel) if str(rel) != "." else "."
        return mount, rel_str

    def _resolve_pair(self, src: str, dst: str) -> tuple[Mount, str, Mount, str]:
        """Resolve both src and dst paths.

        Returns:
            Tuple of (src_mount, src_rel, dst_mount, dst_rel).
        """
        src_mount, src_rel = self._resolve(src)
        dst_mount, dst_rel = self._resolve(dst)
        return src_mount, src_rel, dst_mount, dst_rel

    def _check_writable(self, mount: Mount, operation: str, path: str) -> None:
        """Raise FileOperationError if mount is read-only."""
        if mount.read_only:
            mount_name = mount.label or str(mount.virtual_path)
            raise FileOperationError(operation, path, f"mount '{mount_name}' is read-only")

    def _check_available(self, mount: Mount, operation: str, path: str) -> None:
        """Raise FileOperationError if mount backend is unavailable."""
        mb = mount.mount_backend
        if not mb.is_available:
            mount_name = mount.label or str(mount.virtual_path)
            detail = f" ({mb.status_detail})" if mb.status_detail else ""
            raise FileOperationError(operation, path, f"mount '{mount_name}' is {mb.status}{detail}")

    # --- _impl methods: resolve mount and delegate ---

    async def _read_file_impl(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        mount, rel = self._resolve(path)
        self._check_available(mount, "read", path)
        return await mount.file_operator.read_file(rel, encoding=encoding, offset=offset, length=length)

    async def _read_bytes_impl(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        mount, rel = self._resolve(path)
        self._check_available(mount, "read", path)
        return await mount.file_operator.read_bytes(rel, offset=offset, length=length)

    async def _write_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        mount, rel = self._resolve(path)
        self._check_available(mount, "write", path)
        self._check_writable(mount, "write", path)
        await mount.file_operator.write_file(rel, content, encoding=encoding)

    async def _append_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        mount, rel = self._resolve(path)
        self._check_available(mount, "append", path)
        self._check_writable(mount, "append", path)
        await mount.file_operator.append_file(rel, content, encoding=encoding)

    async def _delete_impl(self, path: str) -> None:
        mount, rel = self._resolve(path)
        self._check_available(mount, "delete", path)
        self._check_writable(mount, "delete", path)
        await mount.file_operator.delete(rel)

    async def _list_dir_impl(self, path: str) -> list[str]:
        mount, rel = self._resolve(path)
        self._check_available(mount, "list", path)
        return await mount.file_operator.list_dir(rel)

    async def _list_dir_with_types_impl(self, path: str) -> list[tuple[str, bool]]:
        mount, rel = self._resolve(path)
        self._check_available(mount, "list", path)
        return await mount.file_operator.list_dir_with_types(rel)

    async def _exists_impl(self, path: str) -> bool:
        mount, rel = self._resolve(path)
        self._check_available(mount, "exists", path)
        return await mount.file_operator.exists(rel)

    async def _is_file_impl(self, path: str) -> bool:
        mount, rel = self._resolve(path)
        self._check_available(mount, "is_file", path)
        return await mount.file_operator.is_file(rel)

    async def _is_dir_impl(self, path: str) -> bool:
        mount, rel = self._resolve(path)
        self._check_available(mount, "is_dir", path)
        return await mount.file_operator.is_dir(rel)

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        mount, rel = self._resolve(path)
        self._check_available(mount, "mkdir", path)
        self._check_writable(mount, "mkdir", path)
        await mount.file_operator.mkdir(rel, parents=parents)

    async def _stat_impl(self, path: str) -> FileStat:
        mount, rel = self._resolve(path)
        self._check_available(mount, "stat", path)
        return await mount.file_operator.stat(rel)

    async def _move_impl(self, src: str, dst: str) -> None:
        src_mount, src_rel, dst_mount, dst_rel = self._resolve_pair(src, dst)
        self._check_available(src_mount, "move (source)", src)
        self._check_available(dst_mount, "move (destination)", dst)
        self._check_writable(src_mount, "move (source)", src)
        self._check_writable(dst_mount, "move (destination)", dst)

        if src_mount is dst_mount:
            # Same mount: delegate directly (efficient)
            await src_mount.file_operator.move(src_rel, dst_rel)
        else:
            # Cross-mount: only files supported (streaming doesn't work for directories)
            if await src_mount.file_operator.is_dir(src_rel):
                raise FileOperationError(
                    "move", src, "cross-mount directory move is not supported; copy files individually"
                )
            stream = await src_mount.file_operator.read_bytes_stream(src_rel, chunk_size=self._default_chunk_size)
            await dst_mount.file_operator.write_bytes_stream(dst_rel, stream)
            await src_mount.file_operator.delete(src_rel)

    async def _copy_impl(self, src: str, dst: str) -> None:
        src_mount, src_rel, dst_mount, dst_rel = self._resolve_pair(src, dst)
        self._check_available(src_mount, "copy (source)", src)
        self._check_available(dst_mount, "copy (destination)", dst)
        self._check_writable(dst_mount, "copy (destination)", dst)

        if src_mount is dst_mount:
            # Same mount: delegate directly (efficient)
            await src_mount.file_operator.copy(src_rel, dst_rel)
        else:
            # Cross-mount: only files supported (streaming doesn't work for directories)
            if await src_mount.file_operator.is_dir(src_rel):
                raise FileOperationError(
                    "copy", src, "cross-mount directory copy is not supported; copy files individually"
                )
            stream = await src_mount.file_operator.read_bytes_stream(src_rel, chunk_size=self._default_chunk_size)
            await dst_mount.file_operator.write_bytes_stream(dst_rel, stream)

    async def _glob_impl(self, pattern: str) -> list[str]:
        """Find files matching glob pattern.

        Relative patterns are globbed against the default mount's backend.
        Absolute patterns are matched to the appropriate mount and globbed there.
        Results are returned as relative paths for relative patterns, or absolute
        virtual paths for absolute patterns.
        """
        if self._default_path is None:
            return []

        pattern_path = as_virtual_path(pattern)

        if pattern_path.is_absolute():
            # Find mount for absolute pattern
            normalized = normalize_virtual_path(pattern_path)
            try:
                mount = self._find_mount(normalized)
            except PathNotAllowedError:
                return []
            rel_pattern = str(normalized.relative_to(mount.virtual_path))
            results = await mount.file_operator.glob(rel_pattern)
            # Convert backend-relative results to absolute virtual paths
            return [str(mount.virtual_path / r) for r in results]
        else:
            # Relative pattern: glob against default mount only
            mount = self._find_mount(as_virtual_path(self._default_path))
            return await mount.file_operator.glob(pattern)

    # --- Streaming overrides ---

    async def _read_bytes_stream_impl(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        mount, rel = self._resolve(path)
        self._check_available(mount, "read_stream", path)
        stream = await mount.file_operator.read_bytes_stream(rel, chunk_size=chunk_size)
        async for chunk in stream:
            yield chunk

    async def _write_bytes_stream_impl(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        mount, rel = self._resolve(path)
        self._check_available(mount, "write_stream", path)
        self._check_writable(mount, "write_stream", path)
        await mount.file_operator.write_bytes_stream(rel, stream)

    # --- Context instructions ---

    async def get_context_instructions(self) -> str | None:
        """Generate mount-aware context instructions.

        Produces XML describing all mounts, their status, and file trees.
        If a mount's backend is unavailable (e.g., remote and offline),
        the error is caught and a note is shown instead of a file tree.
        """
        root = ET.Element("file-system")

        # Default directory
        if self._default_path is not None:
            default_dir = ET.SubElement(root, "default-directory")
            default_dir.text = str(self._default_path)

        # Tmp directory (if configured)
        self._build_tmp_instructions(root)

        # Mount information with file trees
        mounts_elem = ET.SubElement(root, "mounts")
        for mount in self._mounts:
            await self._build_mount_instructions(mounts_elem, mount)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    def _build_tmp_instructions(self, root: ET.Element) -> None:
        """Add tmp directory information to context instructions."""
        if self._tmp_file_operator:
            tmp_dir_info = self._tmp_file_operator.tmp_dir
            if tmp_dir_info:
                tmp_dir = ET.SubElement(root, "tmp-directory")
                tmp_dir.text = tmp_dir_info
                tmp_note = ET.SubElement(root, "tmp-directory-note")
                tmp_note.text = (
                    "This is an agent-only temporary directory for intermediate files. "
                    "Never write deliverables or user-facing files here. "
                    "Files the user needs to access must be written to the project directory. "
                    "Never mention this path to the user."
                )

    async def _build_mount_instructions(self, parent: ET.Element, mount: Mount) -> None:
        """Add a single mount's information to context instructions."""
        from ya_agent_environment.utils import generate_filetree

        mb = mount.mount_backend
        mount_elem = ET.SubElement(parent, "mount")
        mount_elem.set("path", str(mount.virtual_path))
        if mount.label:
            mount_elem.set("label", mount.label)
        mount_elem.set("status", mb.status)
        if mount.read_only:
            mount_elem.set("read-only", "true")
        if mb.status_detail:
            mount_elem.set("status-detail", mb.status_detail)

        if mb.is_available:
            # Backend is online: generate live file tree
            try:
                tree = await generate_filetree(
                    mount.file_operator,
                    root_path=".",
                    max_depth=self._instructions_max_depth,
                    skip_dirs=self._instructions_skip_dirs,
                )
                if tree and not tree.startswith("Directory not found"):
                    tree_elem = ET.SubElement(mount_elem, "file-tree")
                    tree_elem.text = "\n" + tree + "\n      "
            except Exception:
                note = ET.SubElement(mount_elem, "note")
                note.text = "File tree unavailable (backend error)"
        else:
            # Backend is offline: show cached tree if available
            cached = mb.cached_file_tree
            if cached:
                tree_elem = ET.SubElement(mount_elem, "cached-file-tree")
                tree_elem.text = "\n" + cached + "\n      "
            note = ET.SubElement(mount_elem, "note")
            note.text = f"This mount is currently {mb.status}. File operations on this path will fail."

    # --- Lifecycle ---

    async def close(self) -> None:
        """Close all backend operators and own resources."""
        for mount in self._mounts:
            with contextlib.suppress(Exception):
                await mount.file_operator.close()
        await super().close()
