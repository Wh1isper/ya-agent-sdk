"""Shared test fixtures and mock classes for ya_agent_environment tests."""

import shutil
from pathlib import Path
from typing import Any

from ya_agent_environment import (
    BaseResource,
    Environment,
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

    async def read_file(self, path: str, *, encoding: str = "utf-8", offset: int = 0, length: int | None = None) -> str:
        return ""

    async def read_bytes(self, path: str, *, offset: int = 0, length: int | None = None) -> bytes:
        return b""

    async def write_file(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        pass

    async def append_file(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        pass

    async def delete(self, path: str) -> None:
        pass

    async def list_dir(self, path: str) -> list[str]:
        return []

    async def exists(self, path: str) -> bool:
        return False

    async def is_file(self, path: str) -> bool:
        return False

    async def is_dir(self, path: str) -> bool:
        return False

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        pass

    async def move(self, src: str, dst: str) -> None:
        pass

    async def copy(self, src: str, dst: str) -> None:
        pass

    async def stat(self, path: str) -> FileStat:
        return FileStat(size=0, mtime=0, is_file=False, is_dir=False)


class LocalTestFileOperator(FileOperator):
    """Minimal local backend for testing backend-independent utilities."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        super().__init__(default_path=self.root, allowed_paths=[self.root])

    def _path(self, path: str) -> Path:
        return self.root if path in {"", "."} else self.root / path

    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        content = self._path(path).read_text(encoding=encoding)
        end = None if length is None else offset + length
        return content[offset:end]

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        content = self._path(path).read_bytes()
        end = None if length is None else offset + length
        return content[offset:end]

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        target = self._path(path)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding=encoding)

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        mode = "ab" if isinstance(content, bytes) else "a"
        with self._path(path).open(mode, encoding=None if isinstance(content, bytes) else encoding) as file:
            file.write(content)

    async def delete(self, path: str) -> None:
        target = self._path(path)
        target.rmdir() if target.is_dir() else target.unlink()

    async def list_dir(self, path: str) -> list[str]:
        return sorted(entry.name for entry in self._path(path).iterdir())

    async def exists(self, path: str) -> bool:
        return self._path(path).exists()

    async def is_file(self, path: str) -> bool:
        return self._path(path).is_file()

    async def is_dir(self, path: str) -> bool:
        return self._path(path).is_dir()

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        self._path(path).mkdir(parents=parents, exist_ok=True)

    async def move(self, src: str, dst: str) -> None:
        shutil.move(self._path(src), self._path(dst))

    async def copy(self, src: str, dst: str) -> None:
        source = self._path(src)
        target = self._path(dst)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    async def stat(self, path: str) -> FileStat:
        target = self._path(path)
        info = target.stat()
        return FileStat(
            size=info.st_size,
            mtime=info.st_mtime,
            is_file=target.is_file(),
            is_dir=target.is_dir(),
        )


class MockShell(Shell):
    """Mock Shell for testing."""

    def __init__(self) -> None:
        super().__init__(default_cwd=Path("/tmp/mock"))

    async def _simulate_execute(
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
            exit_code, stdout, stderr = await self._simulate_execute(command, timeout=None, env=env, cwd=cwd)
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
