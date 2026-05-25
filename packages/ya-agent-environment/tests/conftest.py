"""Shared test fixtures and mock classes for ya_agent_environment tests."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ya_agent_environment import (
    BaseResource,
    Environment,
    FileEntry,
    FileOperator,
    FileStat,
    Shell,
)
from ya_agent_environment.shell import ExecutionHandle

# --- Test fixtures and helpers ---


class SimpleResource:
    """A simple resource that only has close()."""

    def __init__(self) -> None:
        self.closed = False
        self.setup_called = False

    def close(self) -> None:
        self.closed = True

    async def setup(self) -> None:
        self.setup_called = True

    def get_toolsets(self) -> list[Any]:
        return []


class ResumableMockResource:
    """A resumable resource for testing."""

    def __init__(self, initial_data: str = "") -> None:
        self.data = initial_data
        self.closed = False
        self.setup_called = False
        self._restored_state: dict[str, Any] | None = None

    async def setup(self) -> None:
        self.setup_called = True

    async def export_state(self) -> dict[str, Any]:
        return {"data": self.data}

    async def restore_state(self, state: dict[str, Any]) -> None:
        self.data = state.get("data", "")
        self._restored_state = state

    def close(self) -> None:
        self.closed = True

    def get_toolsets(self) -> list[Any]:
        return []


class MockBaseResource(BaseResource):
    """A BaseResource subclass for testing."""

    def __init__(self, value: str = "") -> None:
        self.value = value
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def export_state(self) -> dict[str, Any]:
        return {"value": self.value}

    async def restore_state(self, state: dict[str, Any]) -> None:
        self.value = state.get("value", "")


class MinimalBaseResource(BaseResource):
    """A minimal BaseResource subclass with default export/restore."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class ResourceWithInstructions(BaseResource):
    """A BaseResource subclass with context instructions."""

    def __init__(self, instructions: str) -> None:
        self._instructions = instructions
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def get_context_instructions(self) -> str | None:
        return self._instructions


class ResourceWithEnvAccess(BaseResource):
    """A resource that captures environment references during creation."""

    def __init__(
        self,
        file_operator: FileOperator,
        shell: Shell,
    ) -> None:
        self.file_operator = file_operator
        self.shell = shell
        self.closed = False

    async def close(self) -> None:
        self.closed = True


# --- Mock Environment for integration tests ---


class MockFileOperator(FileOperator):
    """Mock FileOperator for testing."""

    def __init__(self) -> None:
        super().__init__(
            default_path=Path("/tmp/mock"),
            allowed_paths=[Path("/tmp/mock")],
        )

    async def _read_file_impl(
        self, path: str, *, encoding: str = "utf-8", offset: int = 0, length: int | None = None
    ) -> str:
        return ""

    async def _read_bytes_impl(self, path: str, *, offset: int = 0, length: int | None = None) -> bytes:
        return b""

    async def _write_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        pass

    async def _append_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        pass

    async def _delete_impl(self, path: str) -> None:
        pass

    async def _list_dir_impl(self, path: str) -> list[str]:
        return []

    async def _exists_impl(self, path: str) -> bool:
        return False

    async def _is_file_impl(self, path: str) -> bool:
        return False

    async def _is_dir_impl(self, path: str) -> bool:
        return False

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        pass

    async def _move_impl(self, src: str, dst: str) -> None:
        pass

    async def _copy_impl(self, src: str, dst: str) -> None:
        pass

    async def _stat_impl(self, path: str) -> FileStat:
        return FileStat(size=0, mtime=0, is_file=False, is_dir=False)

    async def _walk_files_impl(
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[FileEntry]:
        if False:
            yield FileEntry(path=root, is_file=False, is_dir=False, size=0, mtime=0)


class MockShell(Shell):
    """Mock Shell for testing."""

    def __init__(self) -> None:
        super().__init__(default_cwd=Path("/tmp/mock"))

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        return (0, "", "")

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        import asyncio
        import contextlib

        stdout_stream = asyncio.StreamReader()
        stderr_stream = asyncio.StreamReader()

        async def _execute() -> int:
            exit_code, stdout, stderr = await self.execute(command, timeout=None, env=env, cwd=cwd)
            if stdout:
                stdout_stream.feed_data(stdout.encode("utf-8"))
            stdout_stream.feed_eof()
            if stderr:
                stderr_stream.feed_data(stderr.encode("utf-8"))
            stderr_stream.feed_eof()
            return exit_code

        exec_task = asyncio.create_task(_execute())

        async def _wait() -> int:
            return await exec_task

        async def _kill() -> None:
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await exec_task

        return ExecutionHandle(
            stdout=stdout_stream,
            stderr=stderr_stream,
            wait=_wait,
            kill=_kill,
        )


class MockEnvironment(Environment):
    """Mock Environment for testing."""

    async def _setup(self) -> None:
        self._file_operator = MockFileOperator()
        self._shell = MockShell()

    async def _teardown(self) -> None:
        self._file_operator = None
        self._shell = None
