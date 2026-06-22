"""Tests for ya_agent_sdk.toolsets.core.filesystem.ls module."""

import json
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from inline_snapshot import snapshot
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem import ls as ls_module
from ya_agent_sdk.toolsets.core.filesystem.ls import ListTool


async def test_list_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert ListTool.name == "ls"
    assert "List directory" in ListTool.description
    tool = ListTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = await tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_ls_uses_list_dir_with_types_for_unlimited_listing() -> None:
    """Should avoid separate per-entry is_dir checks in unlimited listings."""

    class RecordingFileOperator:
        def __init__(self) -> None:
            self.entry_is_dir_calls: list[str] = []
            self.stat_calls: list[str] = []

        async def exists(self, path: str) -> bool:
            return path == "."

        async def is_dir(self, path: str) -> bool:
            if path == ".":
                return True
            self.entry_is_dir_calls.append(path)
            return path == "subdir"

        async def list_dir(self, path: str) -> list[str]:
            raise AssertionError("list_dir should not be used for unlimited listings")

        async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
            assert path == "."
            return [("file.txt", False), ("subdir", True)]

        async def stat(self, path: str) -> dict[str, object]:
            self.stat_calls.append(path)
            return {"size": 7, "mtime": 10.0, "is_file": True, "is_dir": False}

    file_operator = RecordingFileOperator()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = SimpleNamespace(file_operator=file_operator)

    result = await ListTool().call(mock_run_ctx, path=".", max_results=-1)

    assert result["success"] is True
    assert result["count"] == 2
    assert file_operator.entry_is_dir_calls == []
    assert file_operator.stat_calls == ["file.txt"]


async def test_ls_bounds_concurrent_stat_tasks() -> None:
    """Should not create unbounded concurrent stat work for large directories."""

    class SlowStatFileOperator:
        def __init__(self) -> None:
            self.active_stat_calls = 0
            self.max_active_stat_calls = 0

        async def exists(self, path: str) -> bool:
            return path == "."

        async def is_dir(self, path: str) -> bool:
            return path == "."

        async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
            assert path == "."
            return [(f"file-{index}.txt", False) for index in range(ls_module._METADATA_CONCURRENCY_LIMIT + 8)]

        async def stat(self, path: str) -> dict[str, object]:
            self.active_stat_calls += 1
            self.max_active_stat_calls = max(self.max_active_stat_calls, self.active_stat_calls)
            try:
                await anyio.sleep(0.01)
                return {"size": 1, "mtime": 10.0, "is_file": True, "is_dir": False}
            finally:
                self.active_stat_calls -= 1

    file_operator = SlowStatFileOperator()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = SimpleNamespace(file_operator=file_operator)

    result = await ListTool().call(mock_run_ctx, path=".", max_results=-1)

    assert result["success"] is True
    assert result["count"] == ls_module._METADATA_CONCURRENCY_LIMIT + 8
    assert 1 < file_operator.max_active_stat_calls <= ls_module._METADATA_CONCURRENCY_LIMIT


async def test_ls_max_results_limits_stat_work() -> None:
    """Should only stat entries inside the returned result window."""

    class RecordingFileOperator:
        def __init__(self) -> None:
            self.stat_calls: list[str] = []

        async def exists(self, path: str) -> bool:
            return path == "."

        async def is_dir(self, path: str) -> bool:
            if path == ".":
                return True
            self.stat_calls.append(f"is_dir:{path}")
            return False

        async def list_dir(self, path: str) -> list[str]:
            assert path == "."
            return [f"file-{index}.txt" for index in range(5)]

        async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
            raise AssertionError("list_dir_with_types should not be used with finite max_results")

        async def stat(self, path: str) -> dict[str, object]:
            self.stat_calls.append(path)
            return {"size": 1, "mtime": 10.0, "is_file": True, "is_dir": False}

    file_operator = RecordingFileOperator()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = SimpleNamespace(file_operator=file_operator)

    result = await ListTool().call(mock_run_ctx, path=".", max_results=2)

    assert result["success"] is True
    assert result["count"] == 2
    assert result["truncated"] is True
    assert result["total_entries"] == 5
    assert result["showing"] == 2
    assert file_operator.stat_calls == ["is_dir:file-0.txt", "file-0.txt", "is_dir:file-1.txt", "file-1.txt"]


async def test_ls_writes_oversized_output_to_tmp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should spill oversized responses to tmp and return a bounded preview."""

    monkeypatch.setattr(ls_module, "OUTPUT_TRUNCATE_LIMIT", 700)

    class LargeOutputFileOperator:
        def __init__(self) -> None:
            self.saved_content: str | None = None

        async def exists(self, path: str) -> bool:
            return path == "."

        async def is_dir(self, path: str) -> bool:
            return path == "."

        async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
            assert path == "."
            suffix = "x" * 80
            return [(f"file-{index}-{suffix}.txt", False) for index in range(12)]

        async def stat(self, path: str) -> dict[str, object]:
            return {"size": 1, "mtime": 10.0, "is_file": True, "is_dir": False}

        async def write_tmp_file(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> str:
            assert isinstance(content, str)
            self.saved_content = content
            return f"tmp/{path}"

    file_operator = LargeOutputFileOperator()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = SimpleNamespace(file_operator=file_operator)

    result = await ListTool().call(mock_run_ctx, path=".", max_results=-1)

    assert result["success"] is True
    assert result["truncated"] is True
    assert "output_file_path" in result
    assert len(json.dumps(result, ensure_ascii=False)) <= ls_module.OUTPUT_TRUNCATE_LIMIT
    assert file_operator.saved_content is not None
    saved = json.loads(file_operator.saved_content)
    assert saved["count"] == 12
    assert len(saved["entries"]) == 12
    assert result["count"] < saved["count"]


async def test_ls_filters_ignored_names_before_type_checks() -> None:
    """Should avoid type and stat calls for ignored entries."""

    class RecordingFileOperator:
        def __init__(self) -> None:
            self.entry_is_dir_calls: list[str] = []
            self.stat_calls: list[str] = []

        async def exists(self, path: str) -> bool:
            return path == "."

        async def is_dir(self, path: str) -> bool:
            if path == ".":
                return True
            self.entry_is_dir_calls.append(path)
            return path == "subdir"

        async def list_dir(self, path: str) -> list[str]:
            assert path == "."
            return ["keep.txt", "skip.pyc", "subdir"]

        async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
            raise AssertionError("list_dir_with_types should not be used with ignore patterns")

        async def stat(self, path: str) -> dict[str, object]:
            self.stat_calls.append(path)
            return {"size": 4, "mtime": 10.0, "is_file": True, "is_dir": False}

    file_operator = RecordingFileOperator()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = SimpleNamespace(file_operator=file_operator)

    result = await ListTool().call(mock_run_ctx, path=".", ignore=["*.pyc"])

    assert result["success"] is True
    names = [entry["name"] for entry in result["entries"]]
    assert names == ["keep.txt", "subdir"]
    assert file_operator.entry_is_dir_calls == ["keep.txt", "subdir"]
    assert file_operator.stat_calls == ["keep.txt"]


async def test_ls_list_directory(tmp_path: Path) -> None:
    """Should list directory contents."""
    async with AsyncExitStack() as stack:
        # Create a clean subdirectory to avoid tmp_dir pollution
        test_dir = tmp_path / "test_list"
        test_dir.mkdir()

        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[test_dir], default_path=test_dir, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        # Create files and directories in the clean test directory
        (test_dir / "file1.txt").write_text("content")
        (test_dir / "file2.py").write_text("content")
        (test_dir / "subdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        assert result["success"] is True
        assert result["count"] == 3
        names = [e["name"] for e in result["entries"]]
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names


async def test_ls_file_info(tmp_path: Path) -> None:
    """Should include file info (size, modified) for files."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "test.txt").write_text("hello")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        file_entry = next(e for e in result["entries"] if e["name"] == "test.txt")
        assert file_entry["type"] == "file"
        assert "size" in file_entry
        assert file_entry["size"] == 5
        assert "modified" in file_entry


async def test_ls_directory_type(tmp_path: Path) -> None:
    """Should identify directories correctly."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "subdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        dir_entry = next(e for e in result["entries"] if e["name"] == "subdir")
        assert dir_entry["type"] == "directory"


async def test_ls_directory_not_found(tmp_path: Path) -> None:
    """Should return error when directory not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="nonexistent")
        assert result["success"] is False
        assert result["error"] == snapshot("Directory not found: nonexistent")


async def test_ls_path_is_file(tmp_path: Path) -> None:
    """Should return error when path is a file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "test.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="test.txt")
        assert result["success"] is False
        assert result["error"] == snapshot("Path is not a directory: test.txt")


async def test_ls_with_ignore_pattern(tmp_path: Path) -> None:
    """Should ignore files matching ignore patterns."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "keep.txt").write_text("content")
        (tmp_path / "ignore.pyc").write_text("content")
        (tmp_path / "__pycache__").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".", ignore=["*.pyc", "__pycache__"])
        names = [e["name"] for e in result["entries"]]
        assert "keep.txt" in names
        assert "ignore.pyc" not in names
        assert "__pycache__" not in names


async def test_ls_empty_directory(tmp_path: Path) -> None:
    """Should handle empty directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "empty").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="empty")
        assert result["success"] is True
        assert result["count"] == 0
        assert result["entries"] == []
