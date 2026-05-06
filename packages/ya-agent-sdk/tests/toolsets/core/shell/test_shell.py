"""Tests for shell tools."""

import os
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.shell import ShellTool
from ya_agent_sdk.toolsets.core.shell.shell import OUTPUT_TRUNCATE_LIMIT


async def test_shell_tool_basic_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    tool = ShellTool()
    assert tool.name == "shell_exec"
    assert "Execute" in tool.description


async def test_shell_tool_empty_command(tmp_path: Path) -> None:
    """Should return error for empty command."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "")
        assert result["return_code"] == 1
        assert "empty" in result.get("error", "").lower()


async def test_shell_tool_accepts_default_shell_executable(tmp_path: Path) -> None:
    """Should execute commands with LocalShell's default shell executable."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, shell_executable=None)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "echo hello")
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]


async def test_shell_tool_execute_success(tmp_path: Path) -> None:
    """Should execute command and return results."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "echo hello")
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]


@pytest.mark.skipif(os.name != "posix" or not Path("/bin/bash").exists(), reason="/bin/bash is required")
async def test_shell_tool_supports_bash_syntax(tmp_path: Path) -> None:
    """Should execute local POSIX shell commands with Bash by default."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        quoted = await tool.call(mock_run_ctx, "printf %s $'a\\nb'")
        assert quoted["return_code"] == 0
        assert quoted["stdout"] == "a\nb"

        conditional = await tool.call(mock_run_ctx, "[[ -n hello ]] && echo ok")
        assert conditional["return_code"] == 0
        assert conditional["stdout"].strip() == "ok"


async def test_shell_tool_execute_with_timeout(tmp_path: Path) -> None:
    """Should respect timeout parameter."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Should succeed with reasonable timeout
        result = await tool.call(mock_run_ctx, "echo test", timeout_seconds=60)
        assert result["return_code"] == 0


async def test_shell_tool_execute_with_cwd(tmp_path: Path) -> None:
    """Should execute command in specified working directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "pwd", cwd=str(subdir))
        assert result["return_code"] == 0
        assert "subdir" in result["stdout"]


async def test_shell_tool_execute_with_env(tmp_path: Path) -> None:
    """Should pass environment variables to command."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "echo $MY_VAR", environment={"MY_VAR": "test_value"})
        assert result["return_code"] == 0
        assert "test_value" in result["stdout"]


async def test_shell_tool_execute_failure(tmp_path: Path) -> None:
    """Should return non-zero exit code on command failure."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "exit 1")
        assert result["return_code"] == 1


async def test_shell_tool_captures_stderr(tmp_path: Path) -> None:
    """Should capture stderr output."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "ls nonexistent_file_xyz_123")
        assert result["return_code"] != 0
        assert result["stderr"]  # Should have stderr


async def test_shell_tool_stdout_truncation(tmp_path: Path) -> None:
    """Should truncate large stdout and save to tmp file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Generate large output
        large_count = OUTPUT_TRUNCATE_LIMIT + 1000
        result = await tool.call(mock_run_ctx, f"python3 -c \"print('x' * {large_count})\"")

        assert result["return_code"] == 0
        assert "truncated" in result["stdout"]
        assert "stdout_file_path" in result
        # Verify file exists
        assert Path(result["stdout_file_path"]).exists()


@pytest.mark.skipif(os.name != "posix" or not Path("/bin/bash").exists(), reason="/bin/bash is required")
async def test_shell_tool_background_supports_bash_syntax(tmp_path: Path) -> None:
    """Should execute background local POSIX shell commands with Bash by default."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ShellTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, "[[ -n hello ]] && echo ok", background=True)
        assert result["return_code"] == -1
        process_id = result["process_id"]
        stdout, stderr, is_running, exit_code = await ctx.shell.wait_process(process_id, timeout=5.0)

        assert stderr == ""
        assert is_running is False
        assert exit_code == 0
        assert stdout.strip() == "ok"


async def test_shell_tool_get_instruction(agent_context: AgentContext) -> None:
    """Should load instruction from prompts/shell.md."""
    tool = ShellTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = await tool.get_instruction(mock_run_ctx)
    assert "shell" in instruction.lower()
    assert "command" in instruction.lower()
