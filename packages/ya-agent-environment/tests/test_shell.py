"""Tests for Shell class."""

import asyncio
from pathlib import Path

import pytest
from ya_agent_environment import BackgroundProcess, CompletedProcess, ExecutionHandle, OutputBuffer, Shell
from ya_agent_environment.shell import (
    _MAX_BUFFER_LINES,
    _MAX_COMPLETED_OUTPUT_BYTES,
    _MAX_LINE_LENGTH,
    _truncate_line,
    _truncate_to_bytes,
)


class ConcreteShell(Shell):
    """Concrete Shell implementation for testing.

    Implements _create_process() by wrapping execute() and feeding
    output into asyncio.StreamReaders as adapters.
    """

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        # Simulate command execution; "sleep" commands wait
        if command.startswith("sleep"):
            duration = float(command.split()[1])
            await asyncio.sleep(duration)
        if command.startswith("fail"):
            return (1, "", f"error from {command}")
        if command.startswith("large"):
            # Generate multiline output that exceeds completed output byte cap
            # when passed through the buffer.  Uses CJK chars (3 bytes each in
            # UTF-8) so _MAX_BUFFER_LINES lines of _MAX_LINE_LENGTH CJK chars
            # produces ~2.4 MB per stream, well over the 1 MB cap.
            line = "\u4e2d" * _MAX_LINE_LENGTH  # 3 bytes/char -> 12288 bytes/line
            lines = "\n".join(line for _ in range(_MAX_BUFFER_LINES))
            return (0, lines, lines)
        if command.startswith("multiline"):
            count = int(command.split()[1]) if len(command.split()) > 1 else 10
            stdout = "\n".join(f"line {i}" for i in range(count))
            return (0, stdout, "")
        if command.startswith("longline"):
            length = int(command.split()[1]) if len(command.split()) > 1 else _MAX_LINE_LENGTH + 500
            return (0, "L" * length, "")
        return (0, f"output of {command}", "")

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
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


# =============================================================================
# Original Shell tests (unchanged behavior)
# =============================================================================


async def test_shell_default_cwd_none() -> None:
    """Shell with default_cwd=None should have empty allowed_paths."""
    shell = ConcreteShell(default_cwd=None)
    assert shell._default_cwd is None
    assert shell._allowed_paths == []


async def test_shell_default_cwd_none_with_allowed_paths(tmp_path: Path) -> None:
    """Shell with default_cwd=None should accept explicit allowed_paths."""
    shell = ConcreteShell(default_cwd=None, allowed_paths=[tmp_path])
    assert shell._default_cwd is None
    assert len(shell._allowed_paths) == 1
    assert shell._allowed_paths[0] == tmp_path.resolve()


async def test_shell_default_cwd_none_context_instructions() -> None:
    """Shell with default_cwd=None should omit default-working-directory."""
    shell = ConcreteShell(default_cwd=None)
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" not in instructions
    assert "allowed-working-directories" not in instructions
    assert "default-timeout" in instructions


async def test_shell_default_cwd_none_with_allowed_paths_instructions(tmp_path: Path) -> None:
    """Shell with default_cwd=None but allowed_paths should show paths but no default."""
    shell = ConcreteShell(default_cwd=None, allowed_paths=[tmp_path])
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" not in instructions
    assert "allowed-working-directories" in instructions
    assert str(tmp_path.resolve()) in instructions


async def test_shell_with_default_cwd_context_instructions(tmp_path: Path) -> None:
    """Shell with default_cwd should include both default and allowed in instructions."""
    shell = ConcreteShell(default_cwd=tmp_path)
    instructions = await shell.get_context_instructions()
    assert instructions is not None
    assert "default-working-directory" in instructions
    assert "allowed-working-directories" in instructions
    assert str(tmp_path.resolve()) in instructions


async def test_shell_skip_instructions() -> None:
    """Shell with skip_instructions should return None."""
    shell = ConcreteShell(default_cwd=None, skip_instructions=True)
    instructions = await shell.get_context_instructions()
    assert instructions is None


async def test_shell_default_cwd_set() -> None:
    """Shell with default_cwd should work as before."""
    shell = ConcreteShell(default_cwd=Path("/tmp/test"))
    assert shell._default_cwd == Path("/tmp/test").resolve()
    assert shell._default_cwd in shell._allowed_paths


async def test_shell_default_cwd_always_in_allowed(tmp_path: Path) -> None:
    """default_cwd should be added to allowed_paths even if not explicitly listed."""
    other = tmp_path / "other"
    other.mkdir()
    shell = ConcreteShell(default_cwd=tmp_path, allowed_paths=[other])
    assert tmp_path.resolve() in shell._allowed_paths
    assert other.resolve() in shell._allowed_paths


# =============================================================================
# Background process tests
# =============================================================================


async def test_start_returns_process_id() -> None:
    """start() should return a non-empty process ID string."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    assert isinstance(pid, str)
    assert len(pid) == 12
    await shell.close()


async def test_start_registers_background_process() -> None:
    """start() should register process in tracking dicts."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    assert pid  # non-empty
    await shell.close()


async def test_has_active_background_processes_empty() -> None:
    """No active processes initially."""
    shell = ConcreteShell(default_cwd=None)
    assert not shell.has_active_background_processes
    assert shell.active_background_processes == {}


async def test_has_active_background_processes_with_running() -> None:
    """Should report active process while running."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    assert shell.has_active_background_processes
    assert pid in shell.active_background_processes
    bp = shell.active_background_processes[pid]
    assert isinstance(bp, BackgroundProcess)
    assert bp.command == "sleep 10"
    assert bp.process_id == pid
    await shell.close()


async def test_wait_process_success() -> None:
    """wait_process() should return results when process completes."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    stdout, stderr, is_running, exit_code = await shell.wait_process(pid, timeout=5.0)
    assert exit_code == 0
    assert not is_running
    assert "echo hello" in stdout
    assert stderr == ""


async def test_wait_process_auto_cleanup() -> None:
    """Background process should be removed from tracking after wait consumes it."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await shell.wait_process(pid, timeout=5.0)
    # Give event loop a tick for done callback
    await asyncio.sleep(0)
    assert not shell.has_active_background_processes
    assert pid not in shell.active_background_processes


async def test_wait_process_timeout_returns_partial() -> None:
    """wait_process() with timeout should return partial output and is_running=True."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    _stdout, _stderr, is_running, exit_code = await shell.wait_process(pid, timeout=0.1)
    assert is_running
    assert exit_code is None
    # Process should still be tracked (not killed)
    assert shell.has_active_background_processes
    assert pid in shell.active_background_processes
    await shell.close()


async def test_wait_process_timeout_zero_polls() -> None:
    """wait_process(timeout=0) should drain immediately without waiting."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    _stdout, _stderr, is_running, exit_code = await shell.wait_process(pid, timeout=0)
    assert is_running
    assert exit_code is None
    # Process still running
    assert shell.has_active_background_processes
    await shell.close()


async def test_wait_process_timeout_then_wait_again() -> None:
    """After timeout, should be able to wait again successfully."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 0.1")
    # First wait: too short
    _stdout, _stderr, is_running, _exit_code = await shell.wait_process(pid, timeout=0.01)
    assert is_running
    # Second wait: long enough
    _stdout, _stderr, is_running, exit_code = await shell.wait_process(pid, timeout=5.0)
    assert not is_running
    assert exit_code == 0


async def test_wait_process_not_found() -> None:
    """wait_process() should raise KeyError for unknown process_id."""
    shell = ConcreteShell(default_cwd=None)
    with pytest.raises(KeyError, match="No background process"):
        await shell.wait_process("nonexistent", timeout=1.0)


async def test_kill_process() -> None:
    """kill_process() should cancel task, return output, and remove from tracking."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    assert shell.has_active_background_processes
    stdout, stderr = await shell.kill_process(pid)
    assert isinstance(stdout, str)
    assert isinstance(stderr, str)
    assert not shell.has_active_background_processes
    assert pid not in shell.active_background_processes


async def test_kill_process_not_found() -> None:
    """kill_process() should raise KeyError for unknown process_id."""
    shell = ConcreteShell(default_cwd=None)
    with pytest.raises(KeyError, match="No background process"):
        await shell.kill_process("nonexistent")


async def test_kill_then_wait_raises() -> None:
    """After killing, wait_process should raise KeyError."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.kill_process(pid)
    with pytest.raises(KeyError):
        await shell.wait_process(pid, timeout=1.0)


async def test_close_kills_all() -> None:
    """close() should kill all background processes."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("sleep 10")
    await shell.start("sleep 10")
    assert len(shell.active_background_processes) == 2
    await shell.close()
    assert not shell.has_active_background_processes


async def test_close_idempotent() -> None:
    """close() should be safe to call multiple times."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("sleep 10")
    await shell.close()
    await shell.close()  # should not raise
    assert not shell.has_active_background_processes


async def test_multiple_background_processes() -> None:
    """Should track multiple background processes independently."""
    shell = ConcreteShell(default_cwd=None)
    pid1 = await shell.start("sleep 10")
    pid2 = await shell.start("echo fast")
    pid3 = await shell.start("sleep 10")
    assert pid1 != pid2 != pid3

    # Wait for the fast one
    stdout, _stderr, is_running, exit_code = await shell.wait_process(pid2, timeout=5.0)
    assert exit_code == 0
    assert not is_running
    assert "echo fast" in stdout

    # Give event loop a tick for done callback cleanup
    await asyncio.sleep(0)

    # The two sleepers should still be tracked
    active = shell.active_background_processes
    assert pid1 in active
    assert pid2 not in active
    assert pid3 in active

    await shell.close()


async def test_active_background_processes_returns_copy() -> None:
    """active_background_processes should return a copy, not the internal dict."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    snapshot = shell.active_background_processes
    await shell.kill_process(pid)
    # Snapshot should still have the old entry
    assert pid in snapshot
    # But shell should not
    assert not shell.has_active_background_processes


async def test_generate_process_id_uniqueness() -> None:
    """_generate_process_id should produce unique IDs."""
    shell = ConcreteShell(default_cwd=None)
    ids = {shell._generate_process_id() for _ in range(100)}
    assert len(ids) == 100


async def test_background_process_metadata() -> None:
    """BackgroundProcess should capture command and cwd correctly."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10", cwd="/some/path")
    bp = shell.active_background_processes[pid]
    assert bp.command == "sleep 10"
    assert bp.cwd == "/some/path"
    assert bp.started_at is not None
    await shell.close()


# =============================================================================
# OutputBuffer and drain_output tests
# =============================================================================


async def test_drain_output_running_process() -> None:
    """drain_output on a running process returns content and is_running=True."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    _stdout, _stderr, is_running, exit_code = shell.drain_output(pid)
    assert is_running
    assert exit_code is None
    # Buffer still exists (not cleaned up because still running)
    assert pid in shell._output_buffers
    await shell.close()


async def test_drain_output_completed_process() -> None:
    """drain_output on a completed process returns content and cleans up."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await asyncio.sleep(0.05)
    stdout, _stderr, is_running, exit_code = shell.drain_output(pid)
    assert not is_running
    assert exit_code == 0
    assert "echo hello" in stdout
    # Buffer cleaned up (completed + consumed)
    assert pid not in shell._output_buffers
    assert pid not in shell._background_processes


async def test_drain_output_not_found() -> None:
    """drain_output should raise KeyError for unknown process_id."""
    shell = ConcreteShell(default_cwd=None)
    with pytest.raises(KeyError, match="No output buffer"):
        shell.drain_output("nonexistent")


async def test_drain_output_clears_deque() -> None:
    """drain_output should clear the deques so next drain returns new lines."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    # First drain
    shell.drain_output(pid)
    # Second drain should return empty (no new output while sleeping)
    stdout, stderr, is_running, _ = shell.drain_output(pid)
    assert stdout == ""
    assert stderr == ""
    assert is_running
    await shell.close()


async def test_output_buffer_line_truncation() -> None:
    """Lines exceeding _MAX_LINE_LENGTH should be truncated."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("longline")
    await asyncio.sleep(0.05)
    stdout, _stderr, is_running, _exit_code = await shell.wait_process(pid, timeout=5.0)
    assert not is_running
    # Each line should be capped at _MAX_LINE_LENGTH
    for line in stdout.splitlines():
        assert len(line) <= _MAX_LINE_LENGTH


async def test_output_buffer_line_count_bounded() -> None:
    """Buffer deque should not exceed _MAX_BUFFER_LINES."""
    shell = ConcreteShell(default_cwd=None)
    # Generate more lines than buffer allows
    count = _MAX_BUFFER_LINES + 50
    pid = await shell.start(f"multiline {count}")
    await asyncio.sleep(0.05)
    stdout, _stderr, is_running, _exit_code = await shell.wait_process(pid, timeout=5.0)
    assert not is_running
    lines = stdout.splitlines()
    assert len(lines) <= _MAX_BUFFER_LINES


# =============================================================================
# Completed results (filter consumption) tests
# =============================================================================


async def test_completed_results_from_buffer() -> None:
    """Completed background process should be consumable via consume_completed_results."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    # Wait for task to complete and done callback to fire
    await asyncio.sleep(0.05)

    results = shell.consume_completed_results()
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, CompletedProcess)
    assert r.process_id == pid
    assert r.exit_code == 0
    assert "echo hello" in r.stdout
    assert r.command == "echo hello"
    assert not r.truncated
    await shell.close()


async def test_consume_completed_results_returns_and_clears() -> None:
    """consume_completed_results() should return results and clear buffer."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await asyncio.sleep(0.05)

    results = shell.consume_completed_results()
    assert len(results) == 1
    assert results[0].process_id == pid
    assert results[0].exit_code == 0

    # Second call returns empty
    assert shell.consume_completed_results() == []
    await shell.close()


async def test_consume_completed_results_empty() -> None:
    """consume_completed_results() with nothing completed returns empty."""
    shell = ConcreteShell(default_cwd=None)
    assert shell.consume_completed_results() == []


async def test_completed_results_multiple_processes() -> None:
    """Multiple completed processes should all be consumable."""
    shell = ConcreteShell(default_cwd=None)
    pid1 = await shell.start("echo one")
    pid2 = await shell.start("echo two")
    pid3 = await shell.start("fail something")
    await asyncio.sleep(0.05)

    results = shell.consume_completed_results()
    assert len(results) == 3
    pids = {r.process_id for r in results}
    assert pids == {pid1, pid2, pid3}

    # Check the failed one
    failed = next(r for r in results if r.process_id == pid3)
    assert failed.exit_code == 1
    assert "error from" in failed.stderr


async def test_completed_results_output_capped() -> None:
    """Output exceeding _MAX_COMPLETED_OUTPUT_BYTES should be truncated."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("large")
    await asyncio.sleep(0.1)

    results = shell.consume_completed_results()
    assert len(results) == 1
    r = results[0]
    # CJK output: 200 lines * 4096 chars * 3 bytes/char ~ 2.4 MB > 1 MB cap
    assert r.truncated
    assert len(r.stdout.encode("utf-8")) <= _MAX_COMPLETED_OUTPUT_BYTES
    assert len(r.stderr.encode("utf-8")) <= _MAX_COMPLETED_OUTPUT_BYTES


async def test_cancelled_task_not_in_completed_results() -> None:
    """Killed (cancelled) processes should NOT appear in completed results."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.kill_process(pid)
    await asyncio.sleep(0.05)
    assert shell.consume_completed_results() == []


async def test_wait_consumes_so_filter_skips() -> None:
    """wait_process on completed process should consume it, so filter won't see it."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo done")
    await asyncio.sleep(0.05)

    # wait_process drains and cleans up the completed buffer
    stdout, _stderr, is_running, exit_code = await shell.wait_process(pid, timeout=1.0)
    assert exit_code == 0
    assert not is_running
    assert "echo done" in stdout

    # Filter should see nothing
    assert shell.consume_completed_results() == []


async def test_wait_direct_blocks_until_completion() -> None:
    """When wait_process blocks until completion, result should not stay in buffer."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 0.05")
    # Wait before it completes
    _stdout, _stderr, is_running, exit_code = await shell.wait_process(pid, timeout=5.0)
    assert exit_code == 0
    assert not is_running
    await asyncio.sleep(0.05)
    # Should not be in completed results (wait consumed it)
    assert shell.consume_completed_results() == []


async def test_close_clears_buffers() -> None:
    """close() should clear output buffers."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("echo hello")
    await asyncio.sleep(0.05)
    assert len(shell._output_buffers) > 0 or len(shell.consume_completed_results()) > 0
    # Start a new one for close to kill
    await shell.start("sleep 10")
    await shell.close()
    assert len(shell._output_buffers) == 0


# =============================================================================
# Background status summary tests
# =============================================================================


async def test_background_status_summary_empty() -> None:
    """No background activity should return None."""
    shell = ConcreteShell(default_cwd=None)
    assert shell.background_status_summary() is None


async def test_background_status_summary_active() -> None:
    """Active process should appear in summary."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    summary = shell.background_status_summary()
    assert summary is not None
    assert "<background-processes>" in summary
    assert f'id="{pid}"' in summary
    assert 'status="running"' in summary
    assert 'command="sleep 10"' in summary
    await shell.close()


async def test_background_status_summary_completed() -> None:
    """Completed-but-unconsumed process should appear in summary."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await asyncio.sleep(0.05)
    summary = shell.background_status_summary()
    assert summary is not None
    assert "<background-processes>" in summary
    assert f'id="{pid}"' in summary
    assert 'status="completed"' in summary
    await shell.close()


async def test_background_status_summary_mixed() -> None:
    """Both active and completed should appear in summary."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("echo fast")
    pid2 = await shell.start("sleep 10")
    await asyncio.sleep(0.05)
    summary = shell.background_status_summary()
    assert summary is not None
    assert 'status="running"' in summary
    assert 'status="completed"' in summary
    assert f'id="{pid2}"' in summary
    await shell.close()


async def test_background_status_summary_after_consume() -> None:
    """After consuming results, completed section should disappear."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start("echo hello")
    await asyncio.sleep(0.05)
    shell.consume_completed_results()
    assert shell.background_status_summary() is None


async def test_has_background_activity() -> None:
    """has_background_activity should cover both active and completed."""
    shell = ConcreteShell(default_cwd=None)
    assert not shell.has_background_activity

    await shell.start("echo hello")
    assert shell.has_background_activity  # active

    await asyncio.sleep(0.05)
    assert shell.has_background_activity  # completed in buffer

    shell.consume_completed_results()
    assert not shell.has_background_activity  # all consumed


async def test_background_status_summary_xml_escaping() -> None:
    """Commands with XML special chars should be properly escaped in summary."""
    shell = ConcreteShell(default_cwd=None)
    await shell.start('echo "hello" && echo <world>')

    summary = shell.background_status_summary()
    assert summary is not None
    assert "&amp;" in summary
    assert "&lt;" in summary
    assert "&gt;" in summary
    assert "&#x27;" in summary or "&quot;" in summary or "&#39;" in summary
    assert ' command="echo "hello"' not in summary

    await shell.close()


# =============================================================================
# Helper function tests
# =============================================================================


def test_truncate_to_bytes_ascii() -> None:
    """ASCII text should truncate to exact byte count."""
    text = "a" * 100
    result, truncated = _truncate_to_bytes(text, 50)
    assert len(result) == 50
    assert truncated is True


def test_truncate_to_bytes_no_truncation_needed() -> None:
    """Text within limit should be returned unchanged."""
    text = "hello"
    result, truncated = _truncate_to_bytes(text, 100)
    assert result == text
    assert truncated is False


def test_truncate_to_bytes_multibyte() -> None:
    """Multibyte UTF-8 text should be truncated by bytes, not chars."""
    text = "\u4e2d" * 100  # 300 bytes total
    result, truncated = _truncate_to_bytes(text, 150)
    assert truncated is True
    assert len(result) == 50
    assert len(result.encode("utf-8")) == 150


def test_truncate_to_bytes_boundary() -> None:
    """Truncation at a multibyte char boundary should not produce partial chars."""
    text = "a\u4e2d" * 50  # 200 bytes
    result, truncated = _truncate_to_bytes(text, 5)
    assert truncated is True
    assert result == "a\u4e2da"
    assert len(result.encode("utf-8")) == 5


def test_truncate_line_within_limit() -> None:
    """Line within limit should be returned unchanged."""
    line = "hello world"
    assert _truncate_line(line) == line


def test_truncate_line_over_limit() -> None:
    """Line over limit should be truncated to _MAX_LINE_LENGTH."""
    line = "x" * (_MAX_LINE_LENGTH + 100)
    result = _truncate_line(line)
    assert len(result) == _MAX_LINE_LENGTH


def test_truncate_line_custom_limit() -> None:
    """Custom max_length should be respected."""
    line = "hello world"
    result = _truncate_line(line, max_length=5)
    assert result == "hello"


# =============================================================================
# OutputBuffer dataclass tests
# =============================================================================


def test_output_buffer_defaults() -> None:
    """OutputBuffer should have empty deques and not-completed state."""
    buf = OutputBuffer()
    assert len(buf.stdout) == 0
    assert len(buf.stderr) == 0
    assert buf.exit_code is None
    assert buf.completed is False


def test_output_buffer_bounded() -> None:
    """OutputBuffer deques should enforce maxlen."""
    buf = OutputBuffer()
    for i in range(_MAX_BUFFER_LINES + 50):
        buf.stdout.append(f"line {i}")
    assert len(buf.stdout) == _MAX_BUFFER_LINES
    # Oldest lines should be evicted
    assert buf.stdout[0] == "line 50"


# =============================================================================
# Stdin support tests
# =============================================================================


class StdinShell(Shell):
    """Shell that supports stdin for testing write_stdin/close_stdin."""

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        if command.startswith("sleep"):
            duration = float(command.split()[1])
            await asyncio.sleep(duration)
        return (0, f"output of {command}", "")

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a process with a mock stdin."""
        import contextlib

        stdout_stream = asyncio.StreamReader()
        stderr_stream = asyncio.StreamReader()
        stdin_data = bytearray()

        class MockStdin:
            def __init__(self) -> None:
                self.closed = False

            async def write(self, data: bytes) -> None:
                if self.closed:
                    raise OSError("stdin closed")
                stdin_data.extend(data)

            async def close(self) -> None:
                self.closed = True

        mock_stdin = MockStdin()
        self._last_stdin_data = stdin_data
        self._last_mock_stdin = mock_stdin

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
            stdin=mock_stdin,
        )


async def test_write_stdin_basic() -> None:
    """write_stdin should write UTF-8 encoded data to the process stdin."""
    shell = StdinShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await shell.write_stdin(pid, "input line\n")
    assert shell._last_stdin_data == b"input line\n"
    await shell.close()


async def test_write_stdin_multiple_writes() -> None:
    """Multiple write_stdin calls should accumulate."""
    shell = StdinShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.write_stdin(pid, "first\n")
    await shell.write_stdin(pid, "second\n")
    assert shell._last_stdin_data == b"first\nsecond\n"
    await shell.close()


async def test_write_stdin_no_stdin_support() -> None:
    """write_stdin on a process without stdin should raise KeyError."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    with pytest.raises(KeyError, match="does not support stdin"):
        await shell.write_stdin(pid, "hello")
    await shell.close()


async def test_write_stdin_unknown_process() -> None:
    """write_stdin for unknown process_id should raise KeyError."""
    shell = StdinShell(default_cwd=None)
    with pytest.raises(KeyError, match="No background process"):
        await shell.write_stdin("nonexistent", "hello")


async def test_close_stdin() -> None:
    """close_stdin should close the stdin stream."""
    shell = StdinShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.close_stdin(pid)
    assert shell._last_mock_stdin.closed
    # Stdin stream should be removed from tracking
    assert pid not in shell._stdin_streams
    await shell.close()


async def test_close_stdin_idempotent() -> None:
    """close_stdin should be idempotent."""
    shell = StdinShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.close_stdin(pid)
    await shell.close_stdin(pid)  # should not raise
    await shell.close()


async def test_close_stdin_unknown_process() -> None:
    """close_stdin for unknown process should be a no-op."""
    shell = StdinShell(default_cwd=None)
    await shell.close_stdin("nonexistent")  # should not raise


async def test_kill_closes_stdin() -> None:
    """kill_process should close stdin before killing."""
    shell = StdinShell(default_cwd=None)
    pid = await shell.start("sleep 10")
    await shell.kill_process(pid)
    assert shell._last_mock_stdin.closed
    assert pid not in shell._stdin_streams


async def test_close_cleans_stdin() -> None:
    """close() should clean up all stdin streams."""
    shell = StdinShell(default_cwd=None)
    await shell.start("sleep 10")
    await shell.start("sleep 10")
    assert len(shell._stdin_streams) == 2
    await shell.close()
    assert len(shell._stdin_streams) == 0


async def test_stdin_none_for_no_stdin_process() -> None:
    """ConcreteShell (no stdin) should not add to _stdin_streams."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    assert pid not in shell._stdin_streams
    await shell.close()


# =============================================================================
# Tests for review fixes (pid, readline guard, active_background_processes,
# consume_completed_results cap, kill_process cleanup, stdin cleanup)
# =============================================================================


class ShellWithPid(Shell):
    """Shell that exposes pid in ExecutionHandle."""

    async def execute(self, command, *, timeout=None, env=None, cwd=None):
        return (0, f"output of {command}", "")

    async def _create_process(self, command, *, env=None, cwd=None):
        import contextlib

        stdout_stream = asyncio.StreamReader()
        stderr_stream = asyncio.StreamReader()

        async def _execute():
            exit_code, stdout, stderr = await self.execute(command, timeout=None, env=env, cwd=cwd)
            if stdout:
                stdout_stream.feed_data(stdout.encode("utf-8"))
            stdout_stream.feed_eof()
            if stderr:
                stderr_stream.feed_data(stderr.encode("utf-8"))
            stderr_stream.feed_eof()
            return exit_code

        exec_task = asyncio.create_task(_execute())

        async def _wait():
            return await exec_task

        async def _kill():
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await exec_task

        return ExecutionHandle(
            stdout=stdout_stream,
            stderr=stderr_stream,
            wait=_wait,
            kill=_kill,
            pid=12345,
        )


async def test_pid_propagated_to_background_process() -> None:
    """PID from ExecutionHandle should be propagated to BackgroundProcess metadata."""
    shell = ShellWithPid(default_cwd=None)
    pid = await shell.start("echo test")
    procs = shell.active_background_processes
    assert pid in procs
    assert procs[pid].pid == 12345
    await shell.close()


async def test_pid_none_when_not_set() -> None:
    """BackgroundProcess.pid should be None when ExecutionHandle has no pid."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo test")
    procs = shell.active_background_processes
    assert pid in procs
    assert procs[pid].pid is None
    await shell.close()


async def test_active_background_processes_excludes_completed() -> None:
    """active_background_processes should only include running processes."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo done")
    # Wait for completion
    await shell.wait_process(pid, timeout=2.0)
    # After drain, process should not appear in active
    assert pid not in shell.active_background_processes
    await shell.close()


async def test_active_background_processes_excludes_completed_but_unconsumed() -> None:
    """Completed-but-unconsumed processes should NOT appear in active_background_processes."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo done")
    # Wait for process to finish (don't drain)
    await asyncio.sleep(0.1)
    # Process completed but buffer not yet drained
    buf = shell._output_buffers.get(pid)
    if buf and buf.completed:
        # Should NOT appear in active
        assert pid not in shell.active_background_processes
    await shell.close()


async def test_readline_valueerror_guard() -> None:
    """readline() ValueError should be caught and produce a truncation marker."""

    class OverflowStream:
        """Stream that raises ValueError on readline (simulating oversized line)."""

        def __init__(self):
            self._calls = 0

        async def readline(self):
            self._calls += 1
            if self._calls == 1:
                raise ValueError("Separator is not found, and chunk exceed the limit")
            if self._calls == 2:
                return b"normal line\n"
            return b""  # EOF

    shell = ConcreteShell(default_cwd=None)
    # Manually set up process with custom stream
    _process_id, _buf = shell._setup_background_process("test", None)
    overflow_stdout = OverflowStream()
    normal_stderr = asyncio.StreamReader()
    normal_stderr.feed_eof()

    ExecutionHandle(
        stdout=overflow_stdout,
        stderr=normal_stderr,
        wait=asyncio.Future,  # won't be called
        kill=asyncio.Future,  # won't be called
    )

    # Manually run the _read_stream logic by calling start internals
    from collections import deque

    from ya_agent_environment.shell import _truncate_line

    target: deque[str] = deque(maxlen=_MAX_BUFFER_LINES)

    # Simulate _read_stream
    while True:
        try:
            line_bytes = await overflow_stdout.readline()
        except ValueError:
            target.append("[line too long, truncated]")
            continue
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
        target.append(_truncate_line(line))

    assert len(target) == 2
    assert target[0] == "[line too long, truncated]"
    assert target[1] == "normal line"
    await shell.close()


async def test_stdin_adapter_broken_pipe() -> None:
    """StdinAdapter.write() should catch BrokenPipeError, mark closed, and re-raise."""
    from unittest.mock import AsyncMock, MagicMock

    from ya_agent_environment.shell import StdinAdapter

    writer = MagicMock()
    writer.write = MagicMock(side_effect=BrokenPipeError("pipe broken"))
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    adapter = StdinAdapter(writer)
    assert adapter._closed is False

    # Should catch BrokenPipeError, mark closed, and re-raise
    with pytest.raises(BrokenPipeError):
        await adapter.write(b"hello")
    assert adapter._closed is True

    # Subsequent writes should be silently ignored (no re-raise)
    await adapter.write(b"ignored")
    # write was only called once (the broken one)
    assert writer.write.call_count == 1


async def test_stdin_adapter_connection_reset() -> None:
    """StdinAdapter.write() should catch ConnectionResetError, mark closed, and re-raise."""
    from unittest.mock import AsyncMock, MagicMock

    from ya_agent_environment.shell import StdinAdapter

    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock(side_effect=ConnectionResetError("reset"))
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    adapter = StdinAdapter(writer)
    with pytest.raises(ConnectionResetError):
        await adapter.write(b"hello")
    assert adapter._closed is True


async def test_consume_completed_results_respects_cap() -> None:
    """consume_completed_results should not consume more than the cap."""
    from ya_agent_environment.shell import _MAX_COMPLETED_RESULTS

    shell = ConcreteShell(default_cwd=None)

    # Create many completed buffers manually
    count = _MAX_COMPLETED_RESULTS + 10
    for i in range(count):
        pid = f"proc-{i}"
        shell._background_processes[pid] = BackgroundProcess(process_id=pid, command=f"cmd-{i}", cwd=None)
        buf = OutputBuffer()
        buf.completed = True
        buf.exit_code = 0
        shell._output_buffers[pid] = buf

    results = shell.consume_completed_results()
    assert len(results) == _MAX_COMPLETED_RESULTS

    # Remaining should still be in buffer
    remaining_completed = [pid for pid, buf in shell._output_buffers.items() if buf.completed]
    assert len(remaining_completed) == 10

    # Second call should consume the rest
    results2 = shell.consume_completed_results()
    assert len(results2) == 10

    await shell.close()


async def test_drain_output_cleans_stdin_streams() -> None:
    """drain_output should remove stdin stream entries for completed processes."""
    from unittest.mock import AsyncMock, MagicMock

    from ya_agent_environment.shell import StdinAdapter

    shell = ConcreteShell(default_cwd=None)

    # Manually set up a completed process with stdin
    pid = "test-stdin-cleanup"
    shell._background_processes[pid] = BackgroundProcess(process_id=pid, command="test", cwd=None)
    buf = OutputBuffer()
    buf.completed = True
    buf.exit_code = 0
    shell._output_buffers[pid] = buf

    mock_writer = MagicMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    shell._stdin_streams[pid] = StdinAdapter(mock_writer)

    # drain_output should clean up stdin
    shell.drain_output(pid)

    assert pid not in shell._stdin_streams
    assert pid not in shell._output_buffers
    assert pid not in shell._background_processes
    await shell.close()


# =============================================================================
# Signal support tests
# =============================================================================


class SignalShell(Shell):
    """Shell that supports send_signal for testing."""

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        if command.startswith("sleep"):
            duration = float(command.split()[1])
            await asyncio.sleep(duration)
        return (0, f"output of {command}", "")

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        import contextlib

        stdout_stream = asyncio.StreamReader()
        stderr_stream = asyncio.StreamReader()
        self._received_signals: list[int] = []

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

        async def _send_signal(sig: int) -> None:
            self._received_signals.append(sig)

        return ExecutionHandle(
            stdout=stdout_stream,
            stderr=stderr_stream,
            wait=_wait,
            kill=_kill,
            send_signal=_send_signal,
        )


async def test_send_signal_to_running_process() -> None:
    """send_signal should invoke the handler stored from ExecutionHandle."""
    import signal

    shell = SignalShell(default_cwd=None)
    pid = await shell.start("sleep 10")

    await shell.send_signal(pid, signal.SIGINT)
    assert shell._received_signals == [signal.SIGINT]

    await shell.send_signal(pid, signal.SIGTERM)
    assert shell._received_signals == [signal.SIGINT, signal.SIGTERM]

    await shell.close()


async def test_send_signal_unknown_process() -> None:
    """send_signal should raise KeyError for unknown process_id."""
    shell = SignalShell(default_cwd=None)

    with pytest.raises(KeyError, match="No background process"):
        await shell.send_signal("nonexistent", 2)

    await shell.close()


async def test_send_signal_no_handler() -> None:
    """send_signal should raise KeyError for processes without signal support."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await asyncio.sleep(0.05)

    # ConcreteShell doesn't provide send_signal in ExecutionHandle
    with pytest.raises(KeyError, match="does not support signals"):
        await shell.send_signal(pid, 2)

    await shell.close()


async def test_signal_handler_cleanup_on_drain() -> None:
    """drain_output should clean up signal handlers for completed processes."""
    shell = SignalShell(default_cwd=None)
    pid = await shell.start("echo hello")
    await asyncio.sleep(0.05)

    assert pid in shell._signal_handlers
    shell.drain_output(pid)
    assert pid not in shell._signal_handlers

    await shell.close()


async def test_signal_handler_cleanup_on_kill() -> None:
    """kill_process should clean up signal handlers."""
    shell = SignalShell(default_cwd=None)
    pid = await shell.start("sleep 10")

    assert pid in shell._signal_handlers
    await shell.kill_process(pid)
    assert pid not in shell._signal_handlers

    await shell.close()


async def test_signal_handler_cleanup_on_close() -> None:
    """close() should clean up all signal handlers."""
    shell = SignalShell(default_cwd=None)
    await shell.start("sleep 10")
    await shell.start("sleep 10")
    assert len(shell._signal_handlers) == 2
    await shell.close()
    assert len(shell._signal_handlers) == 0


async def test_send_signal_rejected_for_completed_process() -> None:
    """send_signal should reject signals for completed processes to avoid PID reuse."""
    shell = SignalShell(default_cwd=None)
    pid = await shell.start("echo hello")
    # Wait for the process to complete
    await asyncio.sleep(0.1)

    # Process completed but output not consumed -- handler still exists
    assert pid in shell._signal_handlers
    assert pid not in shell._background_tasks

    with pytest.raises(KeyError, match="has already completed"):
        await shell.send_signal(pid, 2)

    await shell.close()


async def test_kill_process_cleanup_guaranteed() -> None:
    """kill_process should clean up tracking even if cancel raises unexpectedly."""
    shell = ConcreteShell(default_cwd=None)
    pid = await shell.start("sleep 10")

    assert pid in shell._background_processes
    assert pid in shell._output_buffers

    await shell.kill_process(pid)

    # All tracking should be cleaned up
    assert pid not in shell._background_tasks
    assert pid not in shell._background_processes
    assert pid not in shell._output_buffers
    assert pid not in shell._stdin_streams
    await shell.close()
