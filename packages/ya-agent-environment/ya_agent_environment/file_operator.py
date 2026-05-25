"""File operator abstractions for environment module.

This module provides abstract base classes and implementations for file
system operations, supporting both local and remote backends.
"""

import shutil
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePath
from xml.etree import ElementTree as ET

import anyio

from ya_agent_environment.protocols import DEFAULT_CHUNK_SIZE, TmpFileOperator
from ya_agent_environment.types import FileStat, TruncatedResult

# Default directories to skip but mark in file tree
DEFAULT_INSTRUCTIONS_SKIP_DIRS: frozenset[str] = frozenset({"node_modules", ".git", ".venv", "__pycache__"})
DEFAULT_INSTRUCTIONS_MAX_DEPTH: int = 3


class LocalTmpFileOperator:
    """Default local filesystem implementation of TmpFileOperator.

    Provides a simple local filesystem implementation used as the default
    tmp_file_operator when none is provided.
    """

    def __init__(self, tmp_dir: Path):
        self._tmp_dir = tmp_dir.resolve()

    def is_managed_path(self, path: str, base_path: Path | PurePath) -> tuple[bool, str]:
        target = Path(path)
        resolved = target if target.is_absolute() else Path(base_path / target).resolve()
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
        # Optimize: use seek instead of reading entire file then slicing
        if offset == 0 and length is None:
            # Fast path: read entire file
            return await anyio.Path(resolved).read_bytes()

        # Use seek for partial reads
        async with await anyio.open_file(resolved, "rb") as f:
            if offset > 0:
                await f.seek(offset)
            if length is None:
                return await f.read()
            return await f.read(length)

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

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        """List directory contents with type information.

        More efficient than calling list_dir + is_dir for each entry.

        Args:
            path: Directory path.

        Returns:
            List of (name, is_dir) tuples, sorted alphabetically.
        """
        resolved = self._resolve(path)
        result: list[tuple[str, bool]] = []
        async for entry in anyio.Path(resolved).iterdir():
            is_dir = await entry.is_dir()
            result.append((entry.name, is_dir))
        return sorted(result, key=lambda x: x[0])

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

    async def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read file content as an async stream of bytes.

        Memory-efficient way to read large files.

        Args:
            path: Path to file.
            chunk_size: Size of each chunk in bytes (default: 64KB).

        Yields:
            Chunks of bytes from the file.
        """
        resolved = self._resolve(path)
        async with await anyio.open_file(resolved, "rb") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def write_bytes_stream(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        """Write bytes stream to file.

        Memory-efficient way to write large files.

        Args:
            path: Path to file.
            stream: Async iterator yielding bytes chunks.
        """
        resolved = self._resolve(path)
        async with await anyio.open_file(resolved, "wb") as f:
            async for chunk in stream:
                await f.write(chunk)

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

    This class has no local system dependencies - it's designed to work
    with both local and remote backends. Tmp file handling is optional
    and must be explicitly configured.

    Tmp File Handling:
        When tmp_dir or tmp_file_operator is provided, operations on
        paths under tmp_dir are automatically delegated to tmp_file_operator.
        Subclasses only need to implement _xxx_impl methods and don't need
        to be aware of tmp handling.

        If neither tmp_dir nor tmp_file_operator is provided, tmp handling
        is disabled and cross-boundary operations will not be available.

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

        # Tmp-only operator (no main filesystem)
        tmp_only_operator = MyCustomOperator(
            default_path=None,
            tmp_dir=tmp_dir,
        )
        # All operations go through tmp_file_operator
        await tmp_only_operator.write_file("cache.json", data)
        ```
    """

    def __init__(
        self,
        default_path: Path | PurePath | None = None,
        allowed_paths: Sequence[Path | PurePath] | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = DEFAULT_INSTRUCTIONS_MAX_DEPTH,
        tmp_dir: Path | None = None,
        tmp_file_operator: TmpFileOperator | None = None,
        skip_instructions: bool = False,
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize FileOperator.

        Args:
            default_path: Default working directory for operations.
                If None, only tmp file operations are available (requires
                tmp_dir or tmp_file_operator).
            allowed_paths: Directories accessible for file operations.
                If None, defaults to [default_path] when default_path is set,
                or [] when default_path is None.
                default_path is always included in allowed_paths when set.
            instructions_skip_dirs: Directories to skip in file tree generation.
            instructions_max_depth: Maximum depth for file tree generation.
            tmp_dir: Directory for temporary files. If provided without
                tmp_file_operator, a LocalTmpFileOperator will be created.
            tmp_file_operator: Operator for tmp file operations. Takes
                precedence over tmp_dir if both are provided.
            skip_instructions: If True, get_context_instructions returns None.
            default_chunk_size: Default chunk size for streaming operations (default: 64KB).

        Note:
            If neither tmp_dir nor tmp_file_operator is provided, tmp handling
            is disabled. Cross-boundary operations will not be available.
        """
        self._default_path: Path | PurePath | None = self._normalize_config_path(default_path)

        if allowed_paths is None:
            self._allowed_paths: list[Path | PurePath] = [self._default_path] if self._default_path is not None else []
        else:
            resolved_paths = [self._normalize_config_path(p) for p in allowed_paths]
            if self._default_path is not None and self._default_path not in resolved_paths:
                resolved_paths.append(self._default_path)
            self._allowed_paths = [p for p in resolved_paths if p is not None]

        self._instructions_skip_dirs = (
            instructions_skip_dirs if instructions_skip_dirs is not None else DEFAULT_INSTRUCTIONS_SKIP_DIRS
        )
        self._instructions_max_depth = instructions_max_depth
        self._skip_instructions = skip_instructions

        # Default chunk size for streaming operations
        self._default_chunk_size = default_chunk_size

        # Tmp file operator setup - no auto-creation to avoid local system dependency
        # Environment or subclass is responsible for providing tmp_file_operator if needed
        self._owned_tmp_dir: Path | None = None  # Track tmp_dir we created (for cleanup)
        if tmp_file_operator is not None:
            self._tmp_file_operator: TmpFileOperator | None = tmp_file_operator
        elif tmp_dir is not None:
            # Only create LocalTmpFileOperator if tmp_dir is explicitly provided
            self._tmp_file_operator = LocalTmpFileOperator(tmp_dir)
        else:
            # No tmp handling - cross-boundary operations will not be available
            self._tmp_file_operator = None

    @staticmethod
    def _normalize_config_path(path: Path | PurePath | None) -> Path | PurePath | None:
        if path is None:
            return None
        if type(path) is Path:
            return path.resolve()
        if isinstance(path, Path):
            return path.resolve()
        return path

    def _is_tmp_path(self, path: str) -> tuple[bool, str]:
        """Delegate to tmp_file_operator to check if path is managed."""
        if self._tmp_file_operator is None:
            return False, path
        if self._default_path is None:
            # No main filesystem -- route everything through tmp
            return True, path
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

    async def _list_dir_with_types_impl(self, path: str) -> list[tuple[str, bool]]:
        """List directory with type info. Override for efficiency.

        Default implementation calls list_dir + is_dir for each entry.
        Subclasses can override for more efficient implementation.

        Returns:
            List of (name, is_dir) tuples, sorted alphabetically.
        """
        entries = await self._list_dir_impl(path)
        result: list[tuple[str, bool]] = []
        for name in entries:
            entry_path = f"{path}/{name}" if path != "." else name
            is_dir = await self._is_dir_impl(entry_path)
            result.append((name, is_dir))
        return sorted(result, key=lambda x: x[0])

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

    # Streaming methods - optional to override (default uses read_bytes/write_file)

    async def _read_bytes_stream_impl(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read file content as an async stream. Override for efficiency.

        Default implementation loads entire file into memory and yields as single chunk.
        Subclasses should override this for true streaming with large files.

        Args:
            path: Path to file.
            chunk_size: Size of each chunk in bytes (default: 64KB).

        Yields:
            Chunks of bytes from the file.
        """
        # Default: read entire file and yield as single chunk
        content = await self._read_bytes_impl(path)
        yield content

    async def _write_bytes_stream_impl(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        """Write bytes stream to file. Override for efficiency.

        Default implementation collects all chunks and writes at once.
        Subclasses should override this for true streaming with large files.

        Args:
            path: Path to file.
            stream: Async iterator yielding bytes chunks.
        """
        # Default: collect all chunks and write at once
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        await self._write_file_impl(path, b"".join(chunks))

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
        if is_tmp:  # pragma: no cover
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
        if is_tmp:  # pragma: no cover
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
        if is_tmp:  # pragma: no cover
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
        if is_tmp:  # pragma: no cover
            await self._tmp_file_operator.append_file(  # type: ignore[union-attr]
                routed_path, content, encoding=encoding
            )
            return
        await self._append_file_impl(path, content, encoding=encoding)

    async def delete(self, path: str) -> None:
        """Delete file or empty directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            await self._tmp_file_operator.delete(routed_path)  # type: ignore[union-attr]
            return
        await self._delete_impl(path)

    async def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.list_dir(routed_path)  # type: ignore[union-attr]
        return await self._list_dir_impl(path)

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        """List directory contents with type information.

        More efficient than calling list_dir + is_dir for each entry.

        Args:
            path: Directory path.

        Returns:
            List of (name, is_dir) tuples, sorted alphabetically.
        """
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.list_dir_with_types(routed_path)  # type: ignore[union-attr]
        return await self._list_dir_with_types_impl(path)

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.exists(routed_path)  # type: ignore[union-attr]
        return await self._exists_impl(path)

    async def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.is_file(routed_path)  # type: ignore[union-attr]
        return await self._is_file_impl(path)

    async def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.is_dir(routed_path)  # type: ignore[union-attr]
        return await self._is_dir_impl(path)

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create directory."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            await self._tmp_file_operator.mkdir(routed_path, parents=parents)  # type: ignore[union-attr]
            return
        await self._mkdir_impl(path, parents=parents)

    async def move(self, src: str, dst: str) -> None:  # pragma: no cover
        """Move file or directory."""
        src_is_tmp, dst_is_tmp, src_path, dst_path = self._is_tmp_path_pair(src, dst)
        if src_is_tmp and dst_is_tmp:
            # Both in tmp: delegate to tmp_file_operator
            await self._tmp_file_operator.move(src_path, dst_path)  # type: ignore[union-attr]
        elif not src_is_tmp and not dst_is_tmp:
            # Neither in tmp: delegate to subclass
            await self._move_impl(src, dst)
        else:
            # Cross-boundary move: use streaming to avoid loading entire file into memory
            if src_is_tmp:
                stream = self._tmp_file_operator.read_bytes_stream(  # type: ignore[union-attr]
                    src_path, chunk_size=self._default_chunk_size
                )
                await self._write_bytes_stream_impl(dst, stream)
                await self._tmp_file_operator.delete(src_path)  # type: ignore[union-attr]
            else:
                stream = self._read_bytes_stream_impl(src, chunk_size=self._default_chunk_size)
                await self._tmp_file_operator.write_bytes_stream(dst_path, stream)  # type: ignore[union-attr]
                await self._delete_impl(src)

    async def copy(self, src: str, dst: str) -> None:  # pragma: no cover
        """Copy file or directory."""
        src_is_tmp, dst_is_tmp, src_path, dst_path = self._is_tmp_path_pair(src, dst)
        if src_is_tmp and dst_is_tmp:
            # Both in tmp: delegate to tmp_file_operator
            await self._tmp_file_operator.copy(src_path, dst_path)  # type: ignore[union-attr]
        elif not src_is_tmp and not dst_is_tmp:
            # Neither in tmp: delegate to subclass
            await self._copy_impl(src, dst)
        else:
            # Cross-boundary copy: use streaming to avoid loading entire file into memory
            if src_is_tmp:
                stream = self._tmp_file_operator.read_bytes_stream(  # type: ignore[union-attr]
                    src_path, chunk_size=self._default_chunk_size
                )
                await self._write_bytes_stream_impl(dst, stream)
            else:
                stream = self._read_bytes_stream_impl(src, chunk_size=self._default_chunk_size)
                await self._tmp_file_operator.write_bytes_stream(dst_path, stream)  # type: ignore[union-attr]

    async def stat(self, path: str) -> FileStat:
        """Get file/directory status information."""
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.stat(routed_path)  # type: ignore[union-attr]
        return await self._stat_impl(path)

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern."""
        # Note: glob doesn't support tmp routing as patterns are relative to default_path
        return await self._glob_impl(pattern)

    async def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read file content as an async stream of bytes.

        Memory-efficient way to read large files. This is used internally
        for cross-boundary copy/move operations.

        Args:
            path: Path to file.
            chunk_size: Size of each chunk in bytes (default: 64KB).

        Yields:
            Chunks of bytes from the file.
        """
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            return await self._tmp_file_operator.read_bytes_stream(  # type: ignore[union-attr]
                routed_path, chunk_size=chunk_size
            )
        return self._read_bytes_stream_impl(path, chunk_size=chunk_size)

    async def write_bytes_stream(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        """Write bytes stream to file.

        Memory-efficient way to write large files. This is used internally
        for cross-boundary copy/move operations.

        Args:
            path: Path to file.
            stream: Async iterator yielding bytes chunks.
        """
        is_tmp, routed_path = self._is_tmp_path(path)
        if is_tmp:  # pragma: no cover
            await self._tmp_file_operator.write_bytes_stream(  # type: ignore[union-attr]
                routed_path, stream
            )
            return
        await self._write_bytes_stream_impl(path, stream)

    async def truncate_to_tmp(
        self,
        content: str,
        filename: str,
        max_length: int = 60000,
    ) -> str | TruncatedResult:  # pragma: no cover
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

    async def read_tmp_file(self, path: str, *, encoding: str = "utf-8") -> str:  # pragma: no cover
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

    async def write_tmp_file(
        self, path: str, content: str | bytes, *, encoding: str = "utf-8"
    ) -> str:  # pragma: no cover
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

    async def tmp_exists(self, path: str) -> bool:  # pragma: no cover
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

    async def delete_tmp_file(self, path: str) -> None:  # pragma: no cover
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
        # Import here to avoid circular dependency
        from ya_agent_environment.utils import generate_filetree

        if self._skip_instructions:
            return None
        root = ET.Element("file-system")

        # Default directory
        if self._default_path is not None:
            default_dir = ET.SubElement(root, "default-directory")
            default_dir.text = str(self._default_path)

        # Tmp directory (if configured)
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

        # File trees for each allowed path
        file_trees = ET.SubElement(root, "file-trees")
        default_path = self._default_path
        for allowed_path in self._allowed_paths:
            if default_path is not None:
                base_path: Path | PurePath = default_path
                try:
                    rel_path = str(allowed_path.relative_to(base_path))
                    if rel_path == ".":
                        rel_path = "."
                except ValueError:
                    # Path is not under default_path, use absolute path
                    rel_path = str(allowed_path)
            else:
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
