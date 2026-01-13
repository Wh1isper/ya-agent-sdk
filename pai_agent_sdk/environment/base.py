"""Environment protocols for file operations and shell execution.

This module defines abstract base classes for environment abstractions,
allowing different implementations (local, remote, S3, SSH, etc.) to be
used interchangeably.

Architecture Overview:
    Environment (outer, long-lived)
      - Manages resource lifecycle (file_operator, shell, resources)
      - Optionally manages tmp_dir for temporary file storage
      - Subclasses implement _setup() and _teardown() hooks
      - async with environment as env:
          env.file_operator, env.shell, env.resources

        AgentContext (inner, short-lived)
          - Manages session state (run_id, timing, handoff)
          - Receives file_operator, shell, resources as parameters
          - async with AgentContext(...) as ctx:

Example:
    Using AsyncExitStack for flat structure (recommended for dependent contexts):

    ```python
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(tmp_base_dir=Path("/tmp"))
        )
        ctx = await stack.enter_async_context(
            AgentContext(env=env)
        )
        # Handle request
        ...
    # Resources cleaned up when stack exits
    ```

    Multiple sessions sharing environment:

    ```python
    async with LocalEnvironment(tmp_base_dir=Path("/tmp")) as env:
        # First session
        async with AgentContext(env=env) as ctx1:
            ...

        # Second session (reuses same environment)
        async with AgentContext(env=env) as ctx2:
            ...
    # tmp_dir and resources cleaned up when environment exits
    ```
"""

import asyncio
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar, runtime_checkable
from xml.etree import ElementTree as ET

import anyio
import pathspec
from pydantic_ai.toolsets import AbstractToolset

from pai_agent_sdk.environment.exceptions import EnvironmentNotEnteredError

if TYPE_CHECKING:
    from typing import Self

T = TypeVar("T")


# --- Type definitions ---


class FileStat(TypedDict):
    """File status information."""

    size: int
    """File size in bytes."""
    mtime: float
    """Modification time as Unix timestamp."""
    is_file: bool
    """True if path is a regular file."""
    is_dir: bool
    """True if path is a directory."""


class TruncatedResult(TypedDict):
    """Result from truncate_to_tmp operation."""

    content: str
    """The truncated content."""
    file_path: str
    """Path to the full content file in tmp directory."""
    message: str
    """Message indicating truncation occurred."""


# --- Resource Protocol and Registry ---


@runtime_checkable
class Resource(Protocol):
    """Protocol for resources managed by Environment.

    Resources must implement a close() method that can be either
    synchronous or asynchronous. The Environment will call close()
    during cleanup.

    Example:
        class DatabaseConnection:
            async def close(self) -> None:
                await self._pool.close()

        class FileHandle:
            def close(self) -> None:
                self._handle.close()
    """

    def close(self) -> Any:
        """Close the resource. Can be sync or async."""
        ...


class ResourceRegistry:
    """Type-safe resource container with protocol validation.

    Provides a registry for managing resources with:
    - Protocol validation on set()
    - Type-safe get operations
    - Unified cleanup via close_all()

    Example:
        registry = ResourceRegistry()
        registry.set("browser", browser_instance)  # Validates Resource protocol
        browser = registry.get_typed("browser", Browser)
        await registry.close_all()
    """

    def __init__(self) -> None:
        self._resources: dict[str, Resource] = {}

    def set(self, key: str, resource: Resource) -> None:
        """Register a resource with protocol validation.

        Args:
            key: Unique identifier for the resource.
            resource: Resource instance (must implement Resource protocol).

        Raises:
            TypeError: If resource doesn't implement Resource protocol.
        """
        if not isinstance(resource, Resource):
            raise TypeError(
                f"Resource must implement Resource protocol (have close() method), got {type(resource).__name__}"
            )
        self._resources[key] = resource

    def get(self, key: str) -> Resource | None:
        """Get a resource by key.

        Args:
            key: Resource identifier.

        Returns:
            Resource instance or None if not found.
        """
        return self._resources.get(key)

    def get_typed(self, key: str, resource_type: type[T]) -> T | None:
        """Get a resource with type casting.

        Provides better IDE support by returning the expected type.

        Args:
            key: Resource identifier.
            resource_type: Expected type of the resource.

        Returns:
            Resource cast to the expected type, or None if not found
            or type doesn't match.

        Example:
            browser = resources.get_typed("browser", Browser)
            if browser:
                await browser.screenshot(url)
        """
        resource = self._resources.get(key)
        if resource is not None and isinstance(resource, resource_type):
            return resource
        return None

    def remove(self, key: str) -> Resource | None:
        """Remove and return a resource.

        Args:
            key: Resource identifier.

        Returns:
            Removed resource or None if not found.
        """
        return self._resources.pop(key, None)

    def __contains__(self, key: str) -> bool:
        """Check if a resource exists."""
        return key in self._resources

    def __len__(self) -> int:
        """Return number of registered resources."""
        return len(self._resources)

    def keys(self) -> list[str]:
        """Return list of resource keys."""
        return list(self._resources.keys())

    async def close_all(self) -> None:
        """Close all resources in reverse registration order.

        Uses best-effort cleanup - continues even if individual
        resources fail to close. Handles both sync and async close().
        """
        for resource in reversed(list(self._resources.values())):
            try:
                result = resource.close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: S110
                pass  # Best effort cleanup
        self._resources.clear()


# Default directories to skip but mark in file tree
DEFAULT_INSTRUCTIONS_SKIP_DIRS: frozenset[str] = frozenset({"node_modules", ".git", ".venv", "__pycache__"})
DEFAULT_INSTRUCTIONS_MAX_DEPTH: int = 3


@runtime_checkable
class TmpFileOperator(Protocol):
    """Protocol for temporary file operations.

    Any FileOperator implementation can serve as a TmpFileOperator.
    This protocol enables composition: a FileOperator can delegate
    tmp path operations to another FileOperator instance.

    Example:
        ```python
        # Create a dedicated operator for tmp files
        tmp_operator = LocalFileOperator(
            allowed_paths=[tmp_dir],
            default_path=tmp_dir,
        )

        # Inject into main operator
        main_operator = S3FileOperator(
            bucket="my-bucket",
            tmp_dir=tmp_dir,
            tmp_file_operator=tmp_operator,
        )

        # Operations on tmp paths automatically use tmp_operator
        await main_operator.write_file("/tmp/pai_xxx/data.json", content)
        ```
    """

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str: ...

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes: ...

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None: ...

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None: ...

    async def delete(self, path: str) -> None: ...

    async def list_dir(self, path: str) -> list[str]: ...

    async def exists(self, path: str) -> bool: ...

    async def is_file(self, path: str) -> bool: ...

    async def is_dir(self, path: str) -> bool: ...

    async def mkdir(self, path: str, *, parents: bool = False) -> None: ...

    async def move(self, src: str, dst: str) -> None: ...

    async def copy(self, src: str, dst: str) -> None: ...

    async def stat(self, path: str) -> FileStat:
        """Get file/directory status information."""
        ...

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern."""
        ...

    async def truncate_to_tmp(
        self,
        content: str,
        filename: str,
        max_length: int = 60000,
    ) -> str | TruncatedResult:
        """Truncate content and save full version to tmp file if needed.

        Args:
            content: Content to potentially truncate.
            filename: Filename to use if saving to tmp.
            max_length: Maximum length before truncation.

        Returns:
            Original content if under max_length, or TruncatedResult with
            truncated content and path to full content file.
        """
        ...

    def is_managed_path(self, path: str, base_path: Path) -> tuple[bool, str]:
        """Check if path is managed by this operator.

        Args:
            path: Path to check (relative or absolute).
            base_path: Base path for resolving relative paths.

        Returns:
            Tuple of (is_managed, resolved_path).
            If is_managed is True, resolved_path is the path to use with this operator.
        """
        ...

    @property
    def tmp_dir(self) -> str | None:
        """Return tmp directory path as string, or None if not configured."""
        ...


class LocalTmpFileOperator:
    """Default local filesystem implementation of TmpFileOperator.

    Provides a simple local filesystem implementation used as the default
    tmp_file_operator when none is provided.
    """

    def __init__(self, tmp_dir: Path):
        self._tmp_dir = tmp_dir.resolve()

    def is_managed_path(self, path: str, base_path: Path) -> tuple[bool, str]:
        target = Path(path)
        resolved = target if target.is_absolute() else (base_path / target).resolve()
        try:
            rel_path = resolved.relative_to(self._tmp_dir)
            return True, str(rel_path) if str(rel_path) != "." else "."
        except ValueError:
            return False, path

    @property
    def tmp_dir(self) -> str | None:
        return str(self._tmp_dir)

    def _resolve(self, path: str) -> Path:
        target = Path(path)
        return target if target.is_absolute() else self._tmp_dir / target

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        resolved = self._resolve(path)
        content = await anyio.Path(resolved).read_text(encoding=encoding)
        if offset > 0 or length is not None:
            end = None if length is None else offset + length
            content = content[offset:end]
        return content

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        resolved = self._resolve(path)
        content = await anyio.Path(resolved).read_bytes()
        if offset > 0 or length is not None:
            end = None if length is None else offset + length
            content = content[offset:end]
        return content

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        resolved = self._resolve(path)
        apath = anyio.Path(resolved)
        if isinstance(content, bytes):
            await apath.write_bytes(content)
        else:
            await apath.write_text(content, encoding=encoding)

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        resolved = self._resolve(path)

        def _append() -> None:
            mode = "ab" if isinstance(content, bytes) else "a"
            with open(resolved, mode, encoding=None if isinstance(content, bytes) else encoding) as f:
                f.write(content)

        await anyio.to_thread.run_sync(_append)  # type: ignore[arg-type]

    async def delete(self, path: str) -> None:
        resolved = self._resolve(path)
        apath = anyio.Path(resolved)
        if await apath.is_dir():
            await apath.rmdir()
        else:
            await apath.unlink()

    async def list_dir(self, path: str) -> list[str]:
        resolved = self._resolve(path)
        entries = [entry.name async for entry in anyio.Path(resolved).iterdir()]
        return sorted(entries)

    async def exists(self, path: str) -> bool:
        return await anyio.Path(self._resolve(path)).exists()

    async def is_file(self, path: str) -> bool:
        return await anyio.Path(self._resolve(path)).is_file()

    async def is_dir(self, path: str) -> bool:
        return await anyio.Path(self._resolve(path)).is_dir()

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        await anyio.Path(self._resolve(path)).mkdir(parents=parents, exist_ok=True)

    async def move(self, src: str, dst: str) -> None:
        src_resolved, dst_resolved = self._resolve(src), self._resolve(dst)
        await anyio.to_thread.run_sync(shutil.move, src_resolved, dst_resolved)  # type: ignore[arg-type]

    async def copy(self, src: str, dst: str) -> None:
        src_resolved, dst_resolved = self._resolve(src), self._resolve(dst)
        if src_resolved.is_dir():
            await anyio.to_thread.run_sync(shutil.copytree, src_resolved, dst_resolved)  # type: ignore[arg-type]
        else:
            await anyio.to_thread.run_sync(shutil.copy2, src_resolved, dst_resolved)  # type: ignore[arg-type]

    async def stat(self, path: str) -> FileStat:
        resolved = self._resolve(path)
        st = await anyio.Path(resolved).stat()
        return FileStat(
            size=st.st_size,
            mtime=st.st_mtime,
            is_file=await anyio.Path(resolved).is_file(),
            is_dir=await anyio.Path(resolved).is_dir(),
        )

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern relative to tmp_dir."""
        matches = []
        for p in self._tmp_dir.glob(pattern):
            try:
                rel = p.relative_to(self._tmp_dir)
                matches.append(str(rel))
            except ValueError:
                matches.append(str(p))
        return sorted(matches)

    async def truncate_to_tmp(
        self,
        content: str,
        filename: str,
        max_length: int = 60000,
    ) -> str | TruncatedResult:
        """Truncate content and save full version to tmp file if needed."""
        if len(content) <= max_length:
            return content

        # Save full content to tmp file
        file_path = self._tmp_dir / filename
        await anyio.Path(file_path).write_text(content, encoding="utf-8")

        # Truncate content
        truncated = content[:max_length]
        if truncated and not truncated.endswith("\n"):
            # Try to truncate at last newline for cleaner output
            last_newline = truncated.rfind("\n")
            if last_newline > max_length * 0.8:  # Only if we don't lose too much
                truncated = truncated[: last_newline + 1]

        return TruncatedResult(
            content=truncated,
            file_path=str(file_path),
            message=f"Content truncated. Full content saved to: {file_path}",
        )


class FileOperator(ABC):
    """Abstract base class for file system operations.

    Provides common initialization logic for path validation,
    instructions configuration, and transparent tmp file handling.

    Tmp File Handling:
        When tmp_dir and tmp_file_operator are provided, operations on
        paths under tmp_dir are automatically delegated to tmp_file_operator.
        Subclasses only need to implement _xxx_impl methods and don't need
        to be aware of tmp handling.

    Example:
        ```python
        # Environment assembles the operators
        tmp_dir = Path("/tmp/pai_agent_xxx")
        tmp_operator = LocalTmpFileOperator(tmp_dir)

        main_operator = MyCustomOperator(
            default_path=Path("/data"),
            allowed_paths=[Path("/data"), tmp_dir],
            tmp_file_operator=tmp_operator,
        )

        # Tmp paths use local filesystem transparently
        await main_operator.write_file("/tmp/pai_agent_xxx/cache.json", data)

        # Non-tmp paths use subclass implementation
        await main_operator.write_file("/data/output.json", data)
        ```
    """

    def __init__(
        self,
        default_path: Path,
        allowed_paths: list[Path] | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = DEFAULT_INSTRUCTIONS_MAX_DEPTH,
        tmp_dir: Path | None = None,
        tmp_file_operator: TmpFileOperator | None = None,
    ):
        """Initialize FileOperator.

        Args:
            default_path: Default working directory for operations. Required.
            allowed_paths: Directories accessible for file operations.
                If None, defaults to [default_path].
                default_path is always included in allowed_paths.
            instructions_skip_dirs: Directories to skip in file tree generation.
            instructions_max_depth: Maximum depth for file tree generation.
            tmp_dir: Directory for temporary files.
            tmp_file_operator: Operator for tmp file operations.
        """
        self._default_path = default_path.resolve()

        if allowed_paths is None:
            self._allowed_paths = [self._default_path]
        else:
            resolved_paths = [p.resolve() for p in allowed_paths]
            if self._default_path not in resolved_paths:
                resolved_paths.append(self._default_path)
            self._allowed_paths = resolved_paths

        self._instructions_skip_dirs = (
            instructions_skip_dirs if instructions_skip_dirs is not None else DEFAULT_INSTRUCTIONS_SKIP_DIRS
        )
        self._instructions_max_depth = instructions_max_depth

        # Auto-create LocalTmpFileOperator with tmp_dir or a random temp directory
        self._owned_tmp_dir: Path | None = None  # Track tmp_dir we created (for cleanup)
        if tmp_file_operator is not None:
            self._tmp_file_operator: TmpFileOperator | None = tmp_file_operator
        else:
            if tmp_dir is None:
                tmp_dir = Path(tempfile.mkdtemp(prefix="pai_agent_"))
                self._owned_tmp_dir = tmp_dir  # We created it, we must clean it up
            self._tmp_file_operator = LocalTmpFileOperator(tmp_dir)

    def _is_tmp_path(self, path: str) -> tuple[bool, str]:
        """Delegate to tmp_file_operator to check if path is managed."""
        if self._tmp_file_operator is None:
            return False, path
        return self._tmp_file_operator.is_managed_path(path, self._default_path)

    def _is_tmp_path_pair(self, src: str, dst: str) -> tuple[bool, bool, str, str]:
        """Check if src and/or dst are under tmp_dir.

        Returns:
            Tuple of (src_is_tmp, dst_is_tmp, src_path, dst_path).
        """
        src_is_tmp, src_path = self._is_tmp_path(src)
        dst_is_tmp, dst_path = self._is_tmp_path(dst)
        return src_is_tmp, dst_is_tmp, src_path, dst_path

    # --- Abstract methods for subclass implementation ---
    # Subclasses implement these without worrying about tmp handling

    @abstractmethod
    async def _read_file_impl(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Read file content as string. Implement in subclass."""
        ...

    @abstractmethod
    async def _read_bytes_impl(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read file content as bytes. Implement in subclass."""
        ...

    @abstractmethod
    async def _write_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write content to file. Implement in subclass."""
        ...

    @abstractmethod
    async def _append_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Append content to file. Implement in subclass."""
        ...

    @abstractmethod
    async def _delete_impl(self, path: str) -> None:
        """Delete file or empty directory. Implement in subclass."""
        ...

    @abstractmethod
    async def _list_dir_impl(self, path: str) -> list[str]:
        """List directory contents. Implement in subclass."""
        ...

    @abstractmethod
    async def _exists_impl(self, path: str) -> bool:
        """Check if path exists. Implement in subclass."""
        ...

    @abstractmethod
    async def _is_file_impl(self, path: str) -> bool:
        """Check if path is a file. Implement in subclass."""
        ...

    @abstractmethod
    async def _is_dir_impl(self, path: str) -> bool:
        """Check if path is a directory. Implement in subclass."""
        ...

    @abstractmethod
    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        """Create directory. Implement in subclass."""
        ...

    @abstractmethod
    async def _move_impl(self, src: str, dst: str) -> None:
        """Move file or directory. Implement in subclass."""
        ...

    @abstractmethod
    async def _copy_impl(self, src: str, dst: str) -> None:
        """Copy file or directory. Implement in subclass."""
        ...

    @abstractmethod
    async def _stat_impl(self, path: str) -> FileStat:
        """Get file status. Implement in subclass."""
        ...

    @abstractmethod
    async def _glob_impl(self, pattern: str) -> list[str]:
        """Find files matching glob pattern. Implement in subclass."""
        ...

    # --- Public methods with tmp routing ---

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Read file content as string."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.read_file(  # type: ignore[union-attr]
                routed_path, encoding=encoding, offset=offset, length=length
            )
        return await self._read_file_impl(path, encoding=encoding, offset=offset, length=length)

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read file content as bytes."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.read_bytes(  # type: ignore[union-attr]
                routed_path, offset=offset, length=length
            )
        return await self._read_bytes_impl(path, offset=offset, length=length)

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write content to file."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            await self._tmp_file_operator.write_file(  # type: ignore[union-attr]
                routed_path, content, encoding=encoding
            )
            return
        await self._write_file_impl(path, content, encoding=encoding)

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Append content to file."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            await self._tmp_file_operator.append_file(  # type: ignore[union-attr]
                routed_path, content, encoding=encoding
            )
            return
        await self._append_file_impl(path, content, encoding=encoding)

    async def delete(self, path: str) -> None:
        """Delete file or empty directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            await self._tmp_file_operator.delete(routed_path)  # type: ignore[union-attr]
            return
        await self._delete_impl(path)

    async def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.list_dir(routed_path)  # type: ignore[union-attr]
        return await self._list_dir_impl(path)

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.exists(routed_path)  # type: ignore[union-attr]
        return await self._exists_impl(path)

    async def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.is_file(routed_path)  # type: ignore[union-attr]
        return await self._is_file_impl(path)

    async def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.is_dir(routed_path)  # type: ignore[union-attr]
        return await self._is_dir_impl(path)

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            await self._tmp_file_operator.mkdir(routed_path, parents=parents)  # type: ignore[union-attr]
            return
        await self._mkdir_impl(path, parents=parents)

    async def move(self, src: str, dst: str) -> None:
        """Move file or directory."""
        src_is_tmp, dst_is_tmp, src_path, dst_path = self._is_tmp_path_pair(src, dst)
        if src_is_tmp and dst_is_tmp:
            # Both in tmp: delegate to tmp_file_operator
            await self._tmp_file_operator.move(src_path, dst_path)  # type: ignore[union-attr]
        elif not src_is_tmp and not dst_is_tmp:
            # Neither in tmp: delegate to subclass
            await self._move_impl(src, dst)
        else:
            # Cross-boundary move: read from source, write to dest, delete source
            if src_is_tmp:
                content = await self._tmp_file_operator.read_bytes(src_path)  # type: ignore[union-attr]
                await self._write_file_impl(dst, content)
                await self._tmp_file_operator.delete(src_path)  # type: ignore[union-attr]
            else:
                content = await self._read_bytes_impl(src)
                await self._tmp_file_operator.write_file(dst_path, content)  # type: ignore[union-attr]
                await self._delete_impl(src)

    async def copy(self, src: str, dst: str) -> None:
        """Copy file or directory."""
        src_is_tmp, dst_is_tmp, src_path, dst_path = self._is_tmp_path_pair(src, dst)
        if src_is_tmp and dst_is_tmp:
            # Both in tmp: delegate to tmp_file_operator
            await self._tmp_file_operator.copy(src_path, dst_path)  # type: ignore[union-attr]
        elif not src_is_tmp and not dst_is_tmp:
            # Neither in tmp: delegate to subclass
            await self._copy_impl(src, dst)
        else:
            # Cross-boundary copy: read from source, write to dest
            if src_is_tmp:
                content = await self._tmp_file_operator.read_bytes(src_path)  # type: ignore[union-attr]
                await self._write_file_impl(dst, content)
            else:
                content = await self._read_bytes_impl(src)
                await self._tmp_file_operator.write_file(dst_path, content)  # type: ignore[union-attr]

    async def stat(self, path: str) -> FileStat:
        """Get file/directory status information."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:
            return await self._tmp_file_operator.stat(routed_path)  # type: ignore[union-attr]
        return await self._stat_impl(path)

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern."""
        # Note: glob doesn't support tmp routing as patterns are relative to default_path
        return await self._glob_impl(pattern)

    async def truncate_to_tmp(
        self,
        content: str,
        filename: str,
        max_length: int = 60000,
    ) -> str | TruncatedResult:
        """Truncate content and save full version to tmp file if needed.

        Args:
            content: Content to potentially truncate.
            filename: Filename to use if saving to tmp.
            max_length: Maximum length before truncation.

        Returns:
            Original content if under max_length, or TruncatedResult with
            truncated content and path to full content file.
        """
        if self._tmp_file_operator is None:
            # No tmp configured, just truncate without saving
            if len(content) <= max_length:
                return content
            return content[:max_length] + "\n... (truncated)"
        return await self._tmp_file_operator.truncate_to_tmp(content, filename, max_length)

    # --- Tmp-specific convenience methods ---

    async def read_tmp_file(self, path: str, *, encoding: str = "utf-8") -> str:
        """Read file from tmp directory.

        Args:
            path: Relative path within tmp_dir.
            encoding: Text encoding.

        Returns:
            File content as string.

        Raises:
            RuntimeError: If tmp_dir is not configured.
        """
        if self._tmp_file_operator is None:
            raise RuntimeError("tmp_dir is not configured")
        return await self._tmp_file_operator.read_file(path, encoding=encoding)

    async def write_tmp_file(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> str:
        """Write file to tmp directory.

        Args:
            path: Relative path within tmp_dir.
            content: Content to write.
            encoding: Text encoding for string content.

        Returns:
            Absolute path to the written file.

        Raises:
            RuntimeError: If tmp_dir is not configured.
        """
        if self._tmp_file_operator is None:
            raise RuntimeError("tmp_dir is not configured")
        await self._tmp_file_operator.write_file(path, content, encoding=encoding)
        tmp_dir = self._tmp_file_operator.tmp_dir
        return f"{tmp_dir}/{path}" if tmp_dir else path

    async def tmp_exists(self, path: str) -> bool:
        """Check if path exists in tmp directory.

        Args:
            path: Relative path within tmp_dir.

        Returns:
            True if path exists.

        Raises:
            RuntimeError: If tmp_dir is not configured.
        """
        if self._tmp_file_operator is None:
            raise RuntimeError("tmp_dir is not configured")
        return await self._tmp_file_operator.exists(path)

    async def delete_tmp_file(self, path: str) -> None:
        """Delete file from tmp directory.

        Args:
            path: Relative path within tmp_dir.

        Raises:
            RuntimeError: If tmp_dir is not configured.
        """
        if self._tmp_file_operator is None:
            raise RuntimeError("tmp_dir is not configured")
        await self._tmp_file_operator.delete(path)

    async def get_context_instructions(self) -> str | None:
        """Return file system context in XML format."""
        root = ET.Element("file-system")

        # Default directory
        default_dir = ET.SubElement(root, "default-directory")
        default_dir.text = str(self._default_path)

        # Tmp directory (if configured)
        if self._tmp_file_operator:
            tmp_dir_info = self._tmp_file_operator.tmp_dir
            if tmp_dir_info:
                tmp_dir = ET.SubElement(root, "tmp-directory")
                tmp_dir.text = tmp_dir_info

        # File trees for each allowed path
        file_trees = ET.SubElement(root, "file-trees")
        for allowed_path in self._allowed_paths:
            try:
                rel_path = str(allowed_path.relative_to(self._default_path))
                if rel_path == ".":
                    rel_path = "."
            except ValueError:
                # Path is not under default_path, use absolute path
                rel_path = str(allowed_path)

            tree = await generate_filetree(
                self,
                root_path=rel_path,
                max_depth=self._instructions_max_depth,
                skip_dirs=self._instructions_skip_dirs,
            )
            if tree and not tree.startswith("Directory not found"):
                directory = ET.SubElement(file_trees, "directory")
                directory.set("path", str(allowed_path))
                directory.text = "\n" + tree + "\n    "

        # Convert to string with indentation
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    async def close(self) -> None:
        """Clean up resources owned by this FileOperator.

        If the FileOperator created its own tmp_dir (when neither tmp_dir
        nor tmp_file_operator was provided), this method will remove it.

        Subclasses can override this to clean up additional resources.
        Always call super().close() when overriding.
        """
        if self._owned_tmp_dir is not None:
            await anyio.to_thread.run_sync(shutil.rmtree, self._owned_tmp_dir, True)  # type: ignore[reportAttributeAccessIssue]
            self._owned_tmp_dir = None
        self._tmp_file_operator = None


class Shell(ABC):
    """Abstract base class for shell command execution."""

    def __init__(
        self,
        default_cwd: Path,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
    ):
        """Initialize Shell.

        Args:
            default_cwd: Default working directory for command execution. Required.
                Always included in allowed_paths.
            allowed_paths: Directories allowed as working directories.
                If None, defaults to [default_cwd].
            default_timeout: Default timeout in seconds.
        """
        self._default_cwd = default_cwd.resolve()

        # Build allowed_paths, ensuring default_cwd is included
        if allowed_paths is None:
            self._allowed_paths = [self._default_cwd]
        else:
            resolved_paths = [p.resolve() for p in allowed_paths]
            if self._default_cwd not in resolved_paths:
                resolved_paths.append(self._default_cwd)
            self._allowed_paths = resolved_paths

        self._default_timeout = default_timeout

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command and return (exit_code, stdout, stderr).

        Args:
            command: Command string to execute via shell.
            timeout: Timeout in seconds (uses default if None).
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        ...

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about shell capabilities."""
        paths_str = "\n".join(f"    <path>{p}</path>" for p in self._allowed_paths)
        return f"""<shell-execution>
  <allowed-working-directories>
{paths_str}
  </allowed-working-directories>
  <default-working-directory>{self._default_cwd}</default-working-directory>
  <default-timeout>{self._default_timeout}s</default-timeout>
  <note>Commands will be executed with the working directory validated.</note>
</shell-execution>"""

    async def close(self) -> None:  # noqa: B027
        """Clean up resources owned by this Shell.

        Subclasses can override this to clean up additional resources
        (e.g., persistent shell sessions, SSH connections).
        Always call super().close() when overriding.
        """


class Environment(ABC):
    """Abstract base class for environment context manager.

    Environment manages the lifecycle of shared resources (file_operator, shell, resources)
    that can be reused across multiple AgentContext sessions.

    Subclasses should:
    - Call super().__init__() to initialize the resource registry
    - Implement _setup() to create file_operator, shell, and any custom resources
    - Implement _teardown() to clean up environment-specific resources
    - Optionally populate self._toolsets in _setup() to provide environment-specific tools
    - NOT override __aenter__ or __aexit__ (use _setup/_teardown instead)

    The base class handles:
    - Calling _setup() in __aenter__
    - Calling _teardown() then resources.close_all() in __aexit__

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
            # Optionally add environment toolsets to agent
            agent = Agent(..., toolsets=[*core_toolsets, *env.toolsets])
            ...
        # Resources cleaned up when stack exits
        ```
    """

    def __init__(self) -> None:
        """Initialize the resource registry."""
        self._resources = ResourceRegistry()
        self._file_operator: FileOperator | None = None
        self._shell: Shell | None = None
        self._toolsets: list[AbstractToolset[Any]] = []
        self._entered: bool = False
        self._enter_lock: asyncio.Lock = asyncio.Lock()

    @property
    def file_operator(self) -> FileOperator:
        """Return the file operator.

        Raises:
            RuntimeError: If environment has not been entered.
        """
        if self._file_operator is None:
            raise RuntimeError("Environment not entered. Use 'async with' to enter the environment first.")
        return self._file_operator

    @property
    def shell(self) -> Shell:
        """Return the shell.

        Raises:
            RuntimeError: If environment has not been entered.
        """
        if self._shell is None:
            raise RuntimeError("Environment not entered. Use 'async with' to enter the environment first.")
        return self._shell

    @property
    def resources(self) -> ResourceRegistry:
        """Return the resource registry for runtime resources.

        Resources can be accessed by AgentContext and tools.
        """
        return self._resources

    @property
    def toolsets(self) -> list[AbstractToolset[Any]]:
        """Return optional pydantic-ai toolsets provided by this environment.

        Subclasses can populate self._toolsets in _setup() to provide
        environment-specific tools. These can be injected into an Agent's
        toolsets list.

        Example:
            ```python
            async with MyEnvironment() as env:
                agent = Agent(
                    ...,
                    toolsets=[core_toolset, *env.toolsets],
                )
            ```
        """
        return self._toolsets

    # --- Subclass hooks ---

    @abstractmethod
    async def _setup(self) -> None:
        """Initialize environment resources.

        Subclasses must implement this to:
        - Create and assign self._file_operator
        - Create and assign self._shell
        - Optionally register custom resources via self._resources.set()

        This is called by __aenter__.
        """
        ...

    @abstractmethod
    async def _teardown(self) -> None:
        """Clean up environment-specific resources.

        Subclasses must implement this to:
        - Clean up tmp_dir, containers, connections, etc.
        - Set self._file_operator = None
        - Set self._shell = None

        Note: self._resources.close_all() is called automatically after _teardown().
        This is called by __aexit__.
        """
        ...

    # --- Fixed lifecycle management ---

    async def __aenter__(self) -> "Self":
        """Enter context and setup resources.

        Raises:
            RuntimeError: If the environment has already been entered.
        """
        async with self._enter_lock:
            if self._entered:
                raise RuntimeError(
                    f"{self.__class__.__name__} has already been entered. "
                    "Each Environment instance can only be entered once at a time."
                )
            self._entered = True
        await self._setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and cleanup resources."""
        try:
            await self._teardown()
        finally:
            # Close file_operator and shell, then close all registered resources
            if self._file_operator is not None:
                await self._file_operator.close()
            if self._shell is not None:
                await self._shell.close()
            await self._resources.close_all()
            async with self._enter_lock:
                self._entered = False

    async def get_context_instructions(self) -> str:
        """Return combined context instructions from file_operator and shell.

        Subclasses can override this to provide additional environment-specific
        instructions. The default implementation combines file_operator and shell
        instructions.

        Returns:
            Combined XML-formatted instructions string wrapped in <environment-context> tags.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered yet.
        """
        parts: list[str] = []

        try:
            file_instructions = await self.file_operator.get_context_instructions()
            if file_instructions:
                parts.append(file_instructions)
        except RuntimeError as e:
            raise EnvironmentNotEnteredError("file_operator") from e

        try:
            shell_instructions = await self.shell.get_context_instructions()
            if shell_instructions:
                parts.append(shell_instructions)
        except RuntimeError as e:
            raise EnvironmentNotEnteredError("shell") from e

        if not parts:
            return ""

        content = "\n\n".join(parts)
        return f"<environment-context>\n{content}\n</environment-context>"


# --- File tree generation utilities ---


def _should_skip_hidden_item(name: str, is_dir: bool, skip_dirs: frozenset[str]) -> tuple[bool, bool]:
    """Check if a hidden item should be skipped.

    Returns:
        (should_skip, should_mark_skipped)
    """
    if not name.startswith("."):
        return False, False
    if name == ".env":
        return False, False  # Always show .env
    if is_dir and name in skip_dirs:
        return True, True  # Skip but mark
    return True, False  # Skip completely


def _load_gitignore_spec(gitignore_content: str) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from content."""
    try:
        patterns = gitignore_content.splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except Exception:
        return None


async def generate_filetree(  # noqa: C901
    file_op: FileOperator,
    root_path: str = ".",
    *,
    max_depth: int = DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    skip_dirs: frozenset[str] | None = None,
) -> str:
    """Generate a flat file listing using FileOperator interface.

    This function works with any FileOperator implementation.
    Output format is flat paths like: src/main.py, src/cli.py

    Args:
        file_op: FileOperator instance to use for file operations.
        root_path: Root path to generate file listing for.
        max_depth: Maximum depth to traverse.
        skip_dirs: Set of directory names to skip but mark.

    Returns:
        Newline-separated flat file paths.
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_INSTRUCTIONS_SKIP_DIRS

    if not await file_op.exists(root_path) or not await file_op.is_dir(root_path):
        return f"Directory not found: {root_path}"

    # Try to load gitignore
    gitignore_spec: pathspec.PathSpec | None = None
    gitignore_path = f"{root_path}/.gitignore" if root_path != "." else ".gitignore"
    try:
        if await file_op.exists(gitignore_path):
            content = await file_op.read_file(gitignore_path)
            gitignore_spec = _load_gitignore_spec(content)
    except Exception:  # noqa: S110
        pass

    def _is_gitignored(rel_path: str, is_dir: bool) -> bool:
        if gitignore_spec is None:
            return False
        path = rel_path + "/" if is_dir else rel_path
        return gitignore_spec.match_file(path)

    async def _collect_paths(current_path: str, current_depth: int, path_prefix: str = "") -> list[str]:  # noqa: C901
        """Collect all file paths recursively, returning flat paths."""
        result: list[str] = []
        try:
            entries = await file_op.list_dir(current_path)
            # Sort: directories first, then files, alphabetically
            dir_entries = []
            file_entries = []
            for name in entries:
                entry_path = f"{current_path}/{name}" if current_path != "." else name
                if await file_op.is_dir(entry_path):
                    dir_entries.append(name)
                else:
                    file_entries.append(name)
            dir_entries.sort()
            file_entries.sort()

            # Process directories first
            for name in dir_entries:
                entry_path = f"{current_path}/{name}" if current_path != "." else name
                flat_path = f"{path_prefix}{name}" if path_prefix else name

                should_skip, should_mark = _should_skip_hidden_item(name, True, skip_dirs)
                if should_skip:
                    if should_mark:
                        result.append(f"{flat_path}/ (skipped)")
                    continue

                # Check gitignore
                if _is_gitignored(flat_path, True):
                    result.append(f"{flat_path}/ (gitignored)")
                    continue

                if current_depth < max_depth:
                    result.extend(await _collect_paths(entry_path, current_depth + 1, f"{flat_path}/"))

            # Then files
            for name in file_entries:
                flat_path = f"{path_prefix}{name}" if path_prefix else name

                should_skip, _ = _should_skip_hidden_item(name, False, skip_dirs)
                if should_skip:
                    continue

                suffix = " (gitignored)" if _is_gitignored(flat_path, False) else ""
                result.append(f"{flat_path}{suffix}")

        except Exception:  # noqa: S110
            pass
        return result

    all_paths = await _collect_paths(root_path, 1)
    return "\n".join(all_paths)
