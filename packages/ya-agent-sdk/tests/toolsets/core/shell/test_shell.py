"""Tests for shell tools."""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.filters.background_shell import consume_background_results
from ya_agent_sdk.toolsets.core.shell import ShellStatusTool, ShellTool, ShellWaitTool
from ya_agent_sdk.toolsets.core.shell.shell import OUTPUT_TRUNCATE_LIMIT


async def test_shell_tool_basic_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description, and output budget."""
    tool = ShellTool()
    assert OUTPUT_TRUNCATE_LIMIT == 12_000
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


async def test_shell_tool_background_returns_short_process_handle(tmp_path: Path) -> None:
    """Background mode should expose the Shell's compact session-local handle."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await ShellTool().call(mock_run_ctx, "echo hello", background=True)

        assert result["process_id"] == "process-1"


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


async def test_shell_tool_rejects_foreground_execute_from_retired_session(tmp_path: Path) -> None:
    """Backend-defined foreground execute must honor the shared session lease."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        shell = ctx.shell
        assert shell is not None
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        with shell.session_access_scope():
            shell.revoke_session_access()
            result = await ShellTool().call(mock_run_ctx, "touch escaped.txt")

        assert result["return_code"] == 1
        assert "retired session" in result.get("error", "")
        assert not (tmp_path / "escaped.txt").exists()


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

        command = "echo $MY_VAR" if os.name == "posix" else "echo %MY_VAR%"
        result = await tool.call(mock_run_ctx, command, environment={"MY_VAR": "test_value"})
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

        # Exceed both the tool preview limit and asyncio's default readline limit.
        large_count = 70_000
        result = await tool.call(
            mock_run_ctx,
            f'''python3 -c "import sys; sys.stdout.write('HEAD-' + 'x' * {large_count} + '-TAIL')"''',
        )

        assert result["return_code"] == 0
        assert result["stdout"].startswith("HEAD-")
        assert result["stdout"].endswith("-TAIL")
        assert "truncated" in result["stdout"]
        assert "stdout_file_path" in result
        assert "output_file_path" in result
        assert len(json.dumps(result, ensure_ascii=False)) <= OUTPUT_TRUNCATE_LIMIT
        # Verify file exists
        assert Path(result["stdout_file_path"]).exists()
        assert Path(result["output_file_path"]).exists()


async def test_shell_wait_retains_boundaries_of_oversized_single_line(tmp_path: Path) -> None:
    """Background capture should preserve both ends beyond asyncio's readline limit."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        run_ctx = MagicMock(spec=RunContext)
        run_ctx.deps = ctx

        started = await ShellTool().call(
            run_ctx,
            '''python3 -c "import sys; sys.stdout.write('HEAD-' + 'x' * 70000 + '-TAIL')"''',
            background=True,
        )
        waited = await ShellWaitTool().call(run_ctx, started["process_id"], timeout_seconds=5)

        assert waited["return_code"] == 0
        assert waited["stdout"].startswith("HEAD-")
        assert waited["stdout"].endswith("-TAIL")
        assert "truncated" in waited["stdout"]


async def test_shell_status_spills_oversized_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized process summaries should use the shared bounded shell preview."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        assert ctx.shell is not None
        summary = '<process command="' + "x" * (OUTPUT_TRUNCATE_LIMIT * 2) + '"/>'
        monkeypatch.setattr(ctx.shell, "background_status_summary_with_retained_results", lambda: summary)
        run_ctx = MagicMock(spec=RunContext)
        run_ctx.deps = ctx

        result = await ShellStatusTool().call(run_ctx)

        assert isinstance(result, dict)
        assert result["truncated"] is True
        assert len(json.dumps(result, ensure_ascii=False)) <= OUTPUT_TRUNCATE_LIMIT
        assert Path(result["output_file_path"]).is_file()
        assert json.loads(Path(result["output_file_path"]).read_text())["stdout"] == summary


async def test_shell_wait_bounds_oversized_error_metadata(tmp_path: Path) -> None:
    """Oversized process handles in error paths must not bypass the shell limit."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        run_ctx = MagicMock(spec=RunContext)
        run_ctx.deps = ctx
        process_id = "x" * (OUTPUT_TRUNCATE_LIMIT + 1000)

        result = await ShellWaitTool().call(run_ctx, process_id, timeout_seconds=0)

        assert result["truncated"] is True
        assert len(json.dumps(result, ensure_ascii=False)) <= OUTPUT_TRUNCATE_LIMIT
        output_path = Path(result["output_file_path"])
        assert output_path.is_file()
        full_result = json.loads(output_path.read_text())
        assert full_result["process_id"] == process_id
        assert process_id in full_result["error"]


async def test_shell_wait_spills_oversized_result_to_tmp_files(tmp_path: Path) -> None:
    """Oversized background results should return a bounded preview with full temp files."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        run_ctx = MagicMock(spec=RunContext)
        run_ctx.deps = ctx

        line = "HEAD-" + "x" * 250 + "-TAIL\n"
        line_count = 100
        started = await ShellTool().call(
            run_ctx,
            f'''python3 -c "import sys; sys.stdout.write({line!r} * {line_count})"''',
            background=True,
        )
        waited = await ShellWaitTool().call(run_ctx, started["process_id"], timeout_seconds=5)

        assert waited["return_code"] == 0
        assert waited["truncated"] is True
        assert len(json.dumps(waited, ensure_ascii=False)) <= OUTPUT_TRUNCATE_LIMIT

        stdout_path = Path(waited["stdout_file_path"])
        output_path = Path(waited["output_file_path"])
        assert stdout_path.is_file()
        assert output_path.is_file()
        saved_stdout = json.loads(output_path.read_text())["stdout"]
        assert stdout_path.read_bytes() == saved_stdout.encode("utf-8")
        assert saved_stdout.splitlines() == [line.rstrip("\n")] * line_count


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
        assert ctx.shell is not None
        stdout, stderr, is_running, exit_code = await ctx.shell.wait_process(process_id, timeout=5.0)

        assert stderr == ""
        assert is_running is False
        assert exit_code == 0
        assert stdout.strip() == "ok"


@pytest.mark.skipif(os.name != "posix" or not Path("/bin/bash").exists(), reason="/bin/bash is required")
async def test_shell_wait_reads_result_after_completion_delivery(tmp_path: Path) -> None:
    """Completion delivery should retain stdout and stderr for a later shell_wait call."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        run_ctx = MagicMock(spec=RunContext)
        run_ctx.deps = ctx

        started = await ShellTool().call(
            run_ctx,
            "printf 'stdout-value'; printf 'stderr-value' >&2",
            background=True,
        )
        process_id = started["process_id"]
        assert ctx.shell is not None
        process_task = ctx.shell._background_tasks[process_id]
        await process_task
        ctx.shell._refresh_completed_tasks()

        completion = await consume_background_results(ctx)
        assert completion is not None
        assert "stdout-value" in completion
        assert "stderr-value" in completion

        status = await ShellStatusTool().call(run_ctx)
        assert process_id in status
        assert 'result="available"' in status

        waited = await ShellWaitTool().call(run_ctx, process_id, timeout_seconds=0)
        assert waited["stdout"] == "stdout-value"
        assert waited["stderr"] == "stderr-value"
        assert waited["is_running"] is False
        assert waited["return_code"] == 0

        waited_again = await ShellWaitTool().call(run_ctx, process_id, timeout_seconds=0)
        assert "No background process" in waited_again["error"]


async def test_shell_tool_get_instruction(agent_context: AgentContext) -> None:
    """Should not inject extra shell tool instructions."""
    tool = ShellTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert await tool.get_instruction(mock_run_ctx) is None
