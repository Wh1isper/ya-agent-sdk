"""Tests for canonical background-shell completion formatting."""

from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock

from ya_agent_environment import CompletedProcess, Shell
from ya_agent_environment.shell import ExecutionHandle
from ya_agent_sdk.filters.background_shell import consume_background_results


def _make_context(shell: Shell | None = None, file_operator=None) -> MagicMock:
    context = MagicMock()
    context.shell = shell
    context.file_operator = file_operator
    context.tmp_dir = PurePosixPath("/agent-tmp") if file_operator is not None else None
    context.resolve_tmp_path.side_effect = lambda name: PurePosixPath("/agent-tmp") / name
    context.run_id = "test-run-12345678"
    context.emit_event = AsyncMock()
    return context


class MockShell(Shell):
    """Shell with controllable completed results and summary."""

    def __init__(self, completed: list[CompletedProcess] | None = None, summary: str | None = None):
        super().__init__(default_cwd=None)
        self._mock_completed = completed or []
        self._mock_summary = summary

    async def _create_process(self, command, *, env=None, cwd=None) -> ExecutionHandle:
        raise NotImplementedError("MockShell._create_process not used in formatter tests")

    def consume_completed_results(self) -> list[CompletedProcess]:
        results = self._mock_completed
        self._mock_completed = []
        return results

    def background_status_summary(self) -> str | None:
        return self._mock_summary


async def test_no_shell_returns_none() -> None:
    assert await consume_background_results(_make_context()) is None


async def test_no_activity_returns_none() -> None:
    assert await consume_background_results(_make_context(MockShell())) is None


async def test_completed_result_is_formatted_and_emitted() -> None:
    completed = CompletedProcess(
        process_id="abc123",
        command="make test",
        cwd="/workspace",
        exit_code=0,
        stdout="All tests passed",
        stderr="",
        truncated=False,
    )
    context = _make_context(MockShell(completed=[completed]))

    result = await consume_background_results(context)

    assert result is not None
    assert "abc123" in result
    assert "make test" in result
    assert "All tests passed" in result
    context.emit_event.assert_awaited_once()


async def test_failed_result_includes_exit_code_and_stderr() -> None:
    completed = CompletedProcess(
        process_id="def456",
        command="make build",
        cwd=None,
        exit_code=1,
        stdout="",
        stderr="compilation error",
        truncated=False,
    )

    result = await consume_background_results(_make_context(MockShell(completed=[completed])))

    assert result is not None
    assert 'exit-code="1"' in result
    assert "compilation error" in result


async def test_completed_result_and_running_summary_are_combined() -> None:
    completed = CompletedProcess(
        process_id="abc123",
        command="echo done",
        cwd=None,
        exit_code=0,
        stdout="done",
        stderr="",
        truncated=False,
    )
    summary = '<background-processes><process id="xyz" status="running" /></background-processes>'

    result = await consume_background_results(_make_context(MockShell([completed], summary)))

    assert result is not None
    assert "background-result" in result
    assert "background-processes" in result


async def test_one_time_consumption() -> None:
    completed = CompletedProcess(
        process_id="abc123",
        command="echo hello",
        cwd=None,
        exit_code=0,
        stdout="hello",
        stderr="",
        truncated=False,
    )
    context = _make_context(MockShell(completed=[completed]))

    assert await consume_background_results(context) is not None
    assert await consume_background_results(context) is None


async def test_large_output_retains_head_and_tail_and_writes_file() -> None:
    completed = CompletedProcess(
        process_id="big1",
        command="big output",
        cwd=None,
        exit_code=0,
        stdout="HEAD-" + "x" * 30000 + "-TAIL",
        stderr="",
        truncated=False,
    )
    file_operator = AsyncMock()
    context = _make_context(MockShell(completed=[completed]), file_operator)

    result = await consume_background_results(context)

    assert result is not None
    assert "truncated" in result.lower()
    assert "HEAD-" in result
    assert "-TAIL" in result
    assert "full output" in result.lower()
    assert "Full stdout:" in result
    file_operator.write_file.assert_awaited_once()


async def test_source_capped_output_is_not_labeled_full() -> None:
    completed = CompletedProcess(
        process_id="capped1",
        command="large output",
        cwd=None,
        exit_code=0,
        stdout="x" * 30000,
        stderr="",
        truncated=True,
    )
    file_operator = AsyncMock()
    context = _make_context(MockShell(completed=[completed]), file_operator)

    result = await consume_background_results(context)

    assert result is not None
    assert "stored output" in result.lower()
    assert "Stored stdout:" in result
    assert "shell_wait can retrieve the retained terminal output" in result
    assert "full output" not in result.lower()
