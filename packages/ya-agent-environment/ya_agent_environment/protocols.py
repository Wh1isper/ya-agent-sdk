"""Protocol definitions for environment module.

This module defines runtime-checkable protocols for resources and operators.
"""

from collections.abc import AsyncIterator
from pathlib import Path, PurePath
from typing import Any, Protocol, runtime_checkable

from ya_agent_environment.types import FileEntry, FileStat, TruncatedResult

# Default chunk size for streaming operations (64KB)
DEFAULT_CHUNK_SIZE = 65536


@runtime_checkable
class Resource(Protocol):
    """Protocol for resources managed by Environment.

    Resources must implement a close() method that can be either
    synchronous or asynchronous. The Environment will call close()
    during cleanup.

    Resources can optionally provide toolsets via get_toolsets().
    The default implementation returns an empty list.

    Example:
        class DatabaseConnection:
            async def close(self) -> None:
                await self._pool.close()

            def get_toolsets(self) -> list[Any]:
                return [self._db_toolset]

        class FileHandle:
            def close(self) -> None:
                self._handle.close()

            def get_toolsets(self) -> list[Any]:
                return []  # No toolsets
    """

    def close(self) -> object:
        """Close the resource. Can be sync or async."""
        ...

    async def setup(self) -> None:
        """Initialize the resource after creation.

        Called by ResourceRegistry after factory creation, before restore_state().
        Use for async initialization like starting processes or establishing connections.
        """
        ...

    def get_toolsets(self) -> list[Any]:
        """Return toolsets provided by this resource.

        Toolsets are tool collections that will be collected by
        ResourceRegistry.get_toolsets() and can be injected into an Agent.

        Returns:
            List of toolset instances. Default implementation returns [].
        """
        ...


@runtime_checkable
class ResumableResource(Resource, Protocol):
    """Protocol for resources that support state export/restore.

    Resources implementing this protocol can have their state serialized
    and restored across process restarts. The factory pattern ensures
    resources are properly initialized before state restoration.

    Example:
        class BrowserSession:
            def __init__(self, browser: Browser):
                self._browser = browser
                self._cookies: list[dict] = []

            async def export_state(self) -> dict[str, Any]:
                # May need to fetch current state from browser
                self._cookies = await self._browser.get_cookies()
                return {"cookies": self._cookies}

            async def restore_state(self, state: dict[str, Any]) -> None:
                self._cookies = state.get("cookies", [])
                await self._browser.set_cookies(self._cookies)

            def close(self) -> None:
                self._browser.close()
    """

    async def export_state(self) -> dict[str, Any]:
        """Export resource state for serialization.

        Returns:
            Dictionary of JSON-serializable state data.
            Should NOT include sensitive data (passwords, tokens, API keys).
        """
        ...

    async def restore_state(self, state: dict[str, Any]) -> None:
        """Restore resource from serialized state.

        Called after the resource is created via factory.
        Should restore the resource to the state it was in when
        export_state() was called.

        Args:
            state: State dictionary from export_state().

        Raises:
            ValueError: If state is invalid or incompatible.
        """
        ...


@runtime_checkable
class InstructableResource(Resource, Protocol):
    """Protocol for resources that provide context instructions.

    Resources implementing this protocol can contribute instructions
    to the environment context, which will be included in the agent's
    system prompt.

    Example:
        class BrowserSession:
            async def get_context_instructions(self) -> str | None:
                return "Browser session is active. Use browser tools for web tasks."

            def close(self) -> None:
                self._browser.close()
    """

    async def get_context_instructions(self) -> str | None:
        """Return context instructions for this resource.

        Returns:
            Instructions string to include in environment context,
            or None if no instructions.
        """
        ...


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

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        """List directory contents with type information.

        More efficient than calling list_dir + is_dir for each entry.

        Args:
            path: Directory path.

        Returns:
            List of (name, is_dir) tuples, sorted alphabetically.
        """
        ...

    async def exists(self, path: str) -> bool: ...

    async def is_file(self, path: str) -> bool: ...

    async def is_dir(self, path: str) -> bool: ...

    async def mkdir(self, path: str, *, parents: bool = False) -> None: ...

    async def move(self, src: str, dst: str) -> None: ...

    async def copy(self, src: str, dst: str) -> None: ...

    async def stat(self, path: str) -> FileStat:
        """Get file/directory status information."""
        ...

    def walk_files(
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[FileEntry]:
        """Walk files and directories through the operator logical path space."""
        ...

    def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read file content as an async stream of bytes.

        Memory-efficient way to read large files.

        Note: This method returns an async iterator directly (not a coroutine).
        Call it without await: `stream = op.read_bytes_stream(path)`

        Args:
            path: Path to file.
            chunk_size: Size of each chunk in bytes (default: 64KB).

        Yields:
            Chunks of bytes from the file.
        """
        ...

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

    def is_managed_path(self, path: str, base_path: Path | PurePath) -> tuple[bool, str]:
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
