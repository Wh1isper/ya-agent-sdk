"""Tests for pai_agent_sdk.toolsets.core.filesystem.grep module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.grep import GrepTool


def test_grep_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert GrepTool.name == "grep_tool"
    assert "regex" in GrepTool.description
    tool = GrepTool(agent_context)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_grep_find_pattern(tmp_path: Path) -> None:
    """Should find lines matching pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        (tmp_path / "test.py").write_text("def hello():\n    print('hello')\n\ndef world():\n    pass")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="def \\w+")
        assert isinstance(result, dict)
        assert len(result) == 2


async def test_grep_invalid_regex(tmp_path: Path) -> None:
    """Should return error for invalid regex."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="[invalid")
        assert "Error: Invalid regex" in result


async def test_grep_with_include_filter(tmp_path: Path) -> None:
    """Should filter files by include pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        (tmp_path / "test.py").write_text("hello world")
        (tmp_path / "test.txt").write_text("hello universe")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="hello", include="*.py")
        assert isinstance(result, dict)
        assert all("test.py" in key for key in result)


async def test_grep_context_lines(tmp_path: Path) -> None:
    """Should include context lines around matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        content = "line1\nline2\nMATCH\nline4\nline5"
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="MATCH", context_lines=2)
        assert isinstance(result, dict)
        match_data = next(iter(result.values()))
        assert "line2" in match_data["context"]
        assert "line4" in match_data["context"]


async def test_grep_max_results_limit(tmp_path: Path) -> None:
    """Should limit total matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        # Create file with many matches
        content = "\n".join([f"match{i}" for i in range(20)])
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="match", max_results=5)
        assert isinstance(result, dict)
        # Should have 5 matches + possible system message
        match_count = len([k for k in result if k != "<system>"])
        assert match_count == 5


async def test_grep_max_matches_per_file(tmp_path: Path) -> None:
    """Should limit matches per file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        content = "\n".join([f"match{i}" for i in range(10)])
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="match", max_matches_per_file=3, max_results=100)
        assert isinstance(result, dict)
        match_count = len([k for k in result if k != "<system>"])
        assert match_count == 3


async def test_grep_no_matches(tmp_path: Path) -> None:
    """Should return empty dict when no matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        (tmp_path / "test.txt").write_text("no matches here")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="xyz123")
        assert result == {}


async def test_grep_match_data_structure(tmp_path: Path) -> None:
    """Should return correct match data structure."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool(ctx)

        (tmp_path / "test.txt").write_text("line with pattern")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="pattern")
        assert isinstance(result, dict)
        match_data = next(iter(result.values()))
        assert "file_path" in match_data
        assert "line_number" in match_data
        assert "matching_line" in match_data
        assert "context" in match_data
        assert "context_start_line" in match_data
