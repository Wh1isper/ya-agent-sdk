"""Tests for background_shell filter."""

from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.tools import RunContext
from ya_agent_environment import CompletedProcess, Shell
from ya_agent_environment.shell import ExecutionHandle
from ya_agent_sdk.filters.background_shell import inject_background_results


def _make_ctx(shell: Shell | None = None, file_operator=None) -> RunContext:
    """Create a minimal RunContext with mocked AgentContext."""
    deps = MagicMock()
    deps.shell = shell
    deps.file_operator = file_operator
    deps.run_id = "test-run-12345678"
    deps.emit_event = AsyncMock()

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


def _make_messages_with_user_prompt(text: str = "hello") -> list:
    """Create a message list ending with a ModelRequest."""
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


class MockShell(Shell):
    """Shell with controllable completed results and summary."""

    def __init__(self, completed: list[CompletedProcess] | None = None, summary: str | None = None):
        super().__init__(default_cwd=None)
        self._mock_completed = completed or []
        self._mock_summary = summary

    async def execute(self, command, *, timeout=None, env=None, cwd=None):
        return (0, "", "")

    async def _create_process(self, command, *, env=None, cwd=None) -> ExecutionHandle:
        raise NotImplementedError("MockShell._create_process not used in filter tests")

    def consume_completed_results(self) -> list[CompletedProcess]:
        results = self._mock_completed
        self._mock_completed = []
        return results

    def background_status_summary(self) -> str | None:
        return self._mock_summary


async def test_no_shell_returns_unchanged() -> None:
    """When shell is None, messages should be unchanged."""
    ctx = _make_ctx(shell=None)
    messages = _make_messages_with_user_prompt()
    result = await inject_background_results(ctx, messages)
    assert len(result[-1].parts) == 1


async def test_no_activity_returns_unchanged() -> None:
    """When no completed results and no summary, unchanged."""
    shell = MockShell(completed=[], summary=None)
    ctx = _make_ctx(shell=shell)
    messages = _make_messages_with_user_prompt()
    result = await inject_background_results(ctx, messages)
    assert len(result[-1].parts) == 1


async def test_completed_result_injected() -> None:
    """Completed result should be injected as UserPromptPart."""
    completed = CompletedProcess(
        process_id="abc123",
        command="make test",
        cwd="/workspace",
        exit_code=0,
        stdout="All tests passed",
        stderr="",
        truncated=False,
    )
    shell = MockShell(completed=[completed], summary=None)
    ctx = _make_ctx(shell=shell)
    messages = _make_messages_with_user_prompt()

    result = await inject_background_results(ctx, messages)
    assert len(result[-1].parts) == 2
    injected = result[-1].parts[1]
    assert isinstance(injected, UserPromptPart)
    assert "abc123" in injected.content
    assert "make test" in injected.content
    assert "All tests passed" in injected.content


async def test_failed_result_injected() -> None:
    """Failed process result should show exit code."""
    completed = CompletedProcess(
        process_id="def456",
        command="make build",
        cwd=None,
        exit_code=1,
        stdout="",
        stderr="compilation error",
        truncated=False,
    )
    shell = MockShell(completed=[completed], summary=None)
    ctx = _make_ctx(shell=shell)
    messages = _make_messages_with_user_prompt()

    result = await inject_background_results(ctx, messages)
    injected = result[-1].parts[1]
    assert 'exit-code="1"' in injected.content
    assert "compilation error" in injected.content


async def test_summary_only_injected() -> None:
    """Summary without completed results should still inject."""
    summary_xml = '<background-processes>\n  <process id="p1" status="running" command="sleep 100" elapsed="30s" />\n</background-processes>'
    shell = MockShell(completed=[], summary=summary_xml)
    ctx = _make_ctx(shell=shell)
    messages = _make_messages_with_user_prompt()

    result = await inject_background_results(ctx, messages)
    assert len(result[-1].parts) == 2
    injected = result[-1].parts[1]
    assert "background-processes" in injected.content
    assert "running" in injected.content


async def test_completed_plus_summary() -> None:
    """Both completed results and summary should be injected together."""
    completed = CompletedProcess(
        process_id="abc123",
        command="echo done",
        cwd=None,
        exit_code=0,
        stdout="done",
        stderr="",
        truncated=False,
    )
    summary_xml = '<background-processes>\n  <process id="xyz" status="running" command="sleep 100" elapsed="5s" />\n</background-processes>'
    shell = MockShell(completed=[completed], summary=summary_xml)
    ctx = _make_ctx(shell=shell)
    messages = _make_messages_with_user_prompt()

    result = await inject_background_results(ctx, messages)
    injected = result[-1].parts[1]
    assert "background-result" in injected.content
    assert "background-processes" in injected.content


async def test_one_time_consumption() -> None:
    """Second call should not inject again (results consumed)."""
    completed = CompletedProcess(
        process_id="abc123",
        command="echo hello",
        cwd=None,
        exit_code=0,
        stdout="hello",
        stderr="",
        truncated=False,
    )
    shell = MockShell(completed=[completed], summary=None)
    ctx = _make_ctx(shell=shell)

    messages = _make_messages_with_user_prompt()
    await inject_background_results(ctx, messages)
    assert len(messages[-1].parts) == 2

    # Second call: no more results
    messages2 = _make_messages_with_user_prompt()
    await inject_background_results(ctx, messages2)
    assert len(messages2[-1].parts) == 1


async def test_truncated_output_with_file_op() -> None:
    """Large output should be truncated and full content written to tmp."""
    large_stdout = "x" * 30000
    completed = CompletedProcess(
        process_id="big1",
        command="big output",
        cwd=None,
        exit_code=0,
        stdout=large_stdout,
        stderr="",
        truncated=False,
    )
    shell = MockShell(completed=[completed], summary=None)

    file_op = AsyncMock()
    file_op.write_tmp_file = AsyncMock(return_value="/tmp/bg-stdout-big1.log")  # noqa: S108
    ctx = _make_ctx(shell=shell, file_operator=file_op)
    messages = _make_messages_with_user_prompt()

    await inject_background_results(ctx, messages)
    injected = messages[-1].parts[1]
    assert "truncated" in injected.content.lower()
    file_op.write_tmp_file.assert_called_once()


async def test_no_model_request_returns_unchanged() -> None:
    """If last message is not ModelRequest, should return unchanged."""
    shell = MockShell(
        completed=[
            CompletedProcess(
                process_id="x",
                command="echo",
                cwd=None,
                exit_code=0,
                stdout="out",
                stderr="",
                truncated=False,
            )
        ],
        summary=None,
    )
    ctx = _make_ctx(shell=shell)
    messages = [ModelResponse(parts=[TextPart(content="response")])]
    result = await inject_background_results(ctx, messages)
    assert result == messages
