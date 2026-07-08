"""Tests for background monitor and spawn delegate tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.context.agent import AgentInfo
from ya_agent_sdk.context.bus import BusMessage, MessageBus
from ya_agent_sdk.toolsets.core.base import BaseTool
from yaacli.app.tui import TUIApp, TUIState
from yaacli.background import (
    BACKGROUND_MONITOR_KEY,
    DELEGATE_BACKEND_TOOL_NAME,
    BackgroundMonitor,
    BackgroundTaskResult,
)
from yaacli.environment import TUIEnvironment
from yaacli.toolsets.background import AsyncDelegateTool, SpawnDelegateTool, SteerSubagentTool, WaitSubagentTool

# =============================================================================
# BackgroundMonitor Tests (subagent task tracking)
# =============================================================================


@pytest.mark.asyncio
async def test_monitor_register_task() -> None:
    """Registering a task should track it in active_tasks."""
    monitor = BackgroundMonitor()

    async def noop() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(noop())
    monitor.register_task("test-agent", task)

    assert "test-agent" in monitor.active_tasks
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_monitor_task_auto_removed_on_completion() -> None:
    """Task should be auto-removed from active_tasks when it completes."""
    monitor = BackgroundMonitor()

    async def quick() -> None:
        pass

    task = asyncio.create_task(quick())
    monitor.register_task("test-agent", task)
    await task
    # Allow done callback to fire
    await asyncio.sleep(0)

    assert "test-agent" not in monitor.active_tasks


@pytest.mark.asyncio
async def test_monitor_completion_callback() -> None:
    """notify_completion should invoke the registered callback."""
    monitor = BackgroundMonitor()
    callback_calls: list[str] = []

    def callback(agent_id: str) -> None:
        callback_calls.append(agent_id)

    monitor.set_completion_callback(callback)
    monitor.notify_completion("test-agent-1")
    monitor.notify_completion("test-agent-2")

    assert callback_calls == ["test-agent-1", "test-agent-2"]

    # Clear callback
    monitor.set_completion_callback(None)
    monitor.notify_completion("test-agent-3")
    assert len(callback_calls) == 2  # No new calls


@pytest.mark.asyncio
async def test_monitor_has_active_tasks() -> None:
    """has_active_tasks should reflect running tasks."""
    monitor = BackgroundMonitor()
    assert not monitor.has_active_tasks

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    monitor.register_task("test-agent", task)
    assert monitor.has_active_tasks

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # Allow done callback
    assert not monitor.has_active_tasks


@pytest.mark.asyncio
async def test_monitor_close_cancels_tasks() -> None:
    """close() should cancel all registered tasks."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task1 = asyncio.create_task(sleeper())
    task2 = asyncio.create_task(sleeper())
    monitor.register_task("agent-1", task1)
    monitor.register_task("agent-2", task2)

    assert len(monitor.active_tasks) == 2

    await monitor.close()

    assert task1.cancelled()
    assert task2.cancelled()
    assert len(monitor.active_tasks) == 0


def test_monitor_get_context_instruction_empty() -> None:
    """get_context_instruction should return None with no tasks."""
    monitor = BackgroundMonitor()
    assert monitor.get_context_instruction() is None


@pytest.mark.asyncio
async def test_monitor_get_context_instruction_with_tasks() -> None:
    """get_context_instruction should return XML with running tasks."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    monitor.register_task("explorer-bg-a1b2", task)

    result = monitor.get_context_instruction()
    assert result is not None
    assert "explorer-bg-a1b2" in result
    assert "background-tasks" in result

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_monitor_set_core_toolset_and_get_delegate() -> None:
    """set_core_toolset should enable get_delegate_tool."""
    monitor = BackgroundMonitor()
    assert monitor.get_delegate_tool() is None
    assert not monitor.has_delegate_tool

    # Mock a toolset with a delegate tool
    mock_tool = MagicMock(spec=BaseTool)
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_tool

    monitor.set_core_toolset(mock_toolset)
    assert monitor.has_delegate_tool
    assert monitor.get_delegate_tool() is mock_tool


def test_monitor_get_delegate_tool_not_found() -> None:
    """get_delegate_tool should return None if delegate tool doesn't exist."""
    monitor = BackgroundMonitor()

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.side_effect = Exception("not found")

    monitor.set_core_toolset(mock_toolset)
    assert monitor.get_delegate_tool() is None


# =============================================================================
# Shell Monitor Tests
# =============================================================================


def _make_mock_shell(active_pids: dict[str, str] | None = None) -> MagicMock:
    """Create a mock Shell with active_background_processes.

    Args:
        active_pids: Mapping of process_id -> command for active processes.
    """
    mock_shell = MagicMock()
    processes = {}
    if active_pids:
        for pid, cmd in active_pids.items():
            proc = MagicMock()
            proc.command = cmd
            proc.process_id = pid
            processes[pid] = proc
    mock_shell.active_background_processes = processes
    # Also set up _background_processes for command lookup
    mock_shell._background_processes = processes
    mock_shell._output_buffers = {}
    return mock_shell


@pytest.mark.asyncio
async def test_shell_monitor_start() -> None:
    """start_shell_monitor should snapshot active processes and start polling."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell({"pid-1": "npm run dev"})
    bus = MessageBus()

    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.1)

    assert monitor.is_shell_monitor_running
    assert monitor._known_active == {"pid-1"}

    await monitor.close()
    assert not monitor.is_shell_monitor_running


@pytest.mark.asyncio
async def test_shell_monitor_detects_completion() -> None:
    """Shell monitor should detect when a process leaves active set."""
    monitor = BackgroundMonitor()
    callback_calls: list[str] = []
    monitor.set_completion_callback(lambda pid: callback_calls.append(pid))

    # Start with one active process
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "npm run dev"},
        buffers={"pid-1": (["ready"], [], True)},
    )
    bus = MessageBus()
    bus.subscribe("main")
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    # Wait for at least one poll cycle
    await asyncio.sleep(0.1)

    # Simulate process completion: remove from active set
    shell.active_background_processes = {}

    # Wait for next poll cycle to detect
    await asyncio.sleep(0.15)

    # Callback should have been invoked
    assert "pid-1" in callback_calls

    # Notification should be queued for TUI redelivery.
    assert monitor.has_pending_messages is True
    assert monitor.deliver_pending_messages(bus, "main") == 1
    messages = bus.consume("main")
    assert len(messages) >= 1
    shell_msg = [m for m in messages if m.source == "shell-monitor"]
    assert len(shell_msg) == 1
    assert "pid-1" in shell_msg[0].content
    assert "npm run dev" in shell_msg[0].content

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_drops_completion_after_result_drained() -> None:
    """Completion wakeup should be dropped after the shell result buffer is consumed."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "npm run dev"},
        buffers={"pid-1": (["ready"], [], True)},
    )
    bus = MessageBus()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    await asyncio.sleep(0.1)
    shell.active_background_processes = {}
    await asyncio.sleep(0.15)

    assert monitor.has_pending_messages is True

    # Simulate shell_wait() or inject_background_results() consuming the result.
    shell._output_buffers.pop("pid-1")
    assert monitor.deliver_pending_messages(bus, "main") == 0
    assert bus.has_pending("main") is False
    assert monitor.has_pending_messages is False

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_delivers_completion_while_result_buffer_exists() -> None:
    """Completion wakeup should be delivered while completed result is still buffered."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "npm run dev"},
        buffers={"pid-1": (["ready"], [], True)},
    )
    bus = MessageBus()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    await asyncio.sleep(0.1)
    shell.active_background_processes = {}
    await asyncio.sleep(0.15)

    assert monitor.deliver_pending_messages(bus, "main") == 1
    messages = bus.consume("main")
    assert len(messages) == 1
    assert "pid-1" in messages[0].content

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_queues_completion_for_redelivery() -> None:
    """Shell completion notifications should be queued until the TUI redelivers them."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "npm run dev"},
        buffers={"pid-1": (["ready"], [], True)},
    )
    bus = MessageBus()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    await asyncio.sleep(0.1)
    shell.active_background_processes = {}
    await asyncio.sleep(0.15)

    assert bus.has_pending("main") is False
    assert monitor.has_pending_messages is True
    assert monitor.deliver_pending_messages(bus, "main") == 1
    assert bus.has_pending("main") is True
    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].target == "main"
    assert messages[0].source == "shell-monitor"
    assert "pid-1" in messages[0].content

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_detects_new_process() -> None:
    """Shell monitor should track new processes that appear."""
    monitor = BackgroundMonitor()

    # Start with no processes
    shell = _make_mock_shell()
    bus = MessageBus()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    assert monitor._known_active == set()

    # Simulate a new process appearing
    new_proc = MagicMock()
    new_proc.command = "make build"
    new_proc.process_id = "pid-2"
    shell.active_background_processes = {"pid-2": new_proc}
    shell._background_processes = {"pid-2": new_proc}

    # Wait for poll
    await asyncio.sleep(0.15)

    assert "pid-2" in monitor._known_active

    # Now simulate it completing
    callback_calls: list[str] = []
    monitor.set_completion_callback(lambda pid: callback_calls.append(pid))
    shell.active_background_processes = {}

    await asyncio.sleep(0.15)

    assert "pid-2" in callback_calls

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_multiple_completions() -> None:
    """Shell monitor should handle multiple processes completing."""
    monitor = BackgroundMonitor()
    callback_calls: list[str] = []
    monitor.set_completion_callback(lambda pid: callback_calls.append(pid))

    shell = _make_mock_shell({"pid-1": "cmd1", "pid-2": "cmd2", "pid-3": "cmd3"})
    bus = MessageBus()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)

    await asyncio.sleep(0.1)

    # Two processes complete, one stays
    remaining = MagicMock()
    remaining.command = "cmd3"
    shell.active_background_processes = {"pid-3": remaining}

    await asyncio.sleep(0.15)

    assert "pid-1" in callback_calls
    assert "pid-2" in callback_calls
    assert "pid-3" not in callback_calls

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_no_double_start() -> None:
    """start_shell_monitor should ignore if already running."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell()
    bus = MessageBus()

    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.1)
    first_task = monitor._poll_task

    # Second start should be ignored
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.1)
    assert monitor._poll_task is first_task

    await monitor.close()


@pytest.mark.asyncio
async def test_shell_monitor_close_stops_polling() -> None:
    """close() should stop the shell monitor polling loop."""
    monitor = BackgroundMonitor()
    shell = _make_mock_shell()
    bus = MessageBus()

    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    assert monitor.is_shell_monitor_running

    await monitor.close()

    assert not monitor.is_shell_monitor_running
    assert monitor._poll_task is None
    assert monitor._shell is None
    assert monitor._bus is None


@pytest.mark.asyncio
async def test_shell_monitor_bus_message_target() -> None:
    """Bus messages from shell monitor should target the configured agent_id."""
    monitor = BackgroundMonitor()

    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "echo hello"},
        buffers={"pid-1": (["hello"], [], True)},
    )
    bus = MessageBus()
    bus.subscribe("custom-agent")
    bus.subscribe("main")
    monitor.start_shell_monitor(shell, bus, "custom-agent", poll_interval=0.05)

    await asyncio.sleep(0.1)

    # Complete the process
    shell.active_background_processes = {}
    await asyncio.sleep(0.15)

    # Message should target "custom-agent"
    assert monitor.deliver_pending_messages(bus, "custom-agent") == 1
    messages = bus.consume("custom-agent")
    assert len(messages) >= 1
    assert messages[0].target == "custom-agent"
    assert messages[0].source == "shell-monitor"

    # No messages for "main" (targeted to custom-agent)
    main_msgs = bus.consume("main")
    assert len(main_msgs) == 0

    await monitor.close()


# =============================================================================
# SpawnDelegateTool Tests
# =============================================================================


class _AvailableDelegateBackend(BaseTool):
    name = DELEGATE_BACKEND_TOOL_NAME
    description = "Hidden delegate backend"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return False

    @staticmethod
    def _get_roster_instruction(ctx: RunContext[AgentContext]) -> str | None:
        return '<subagent name="helper">\nHelper subagent\n</subagent>'

    @staticmethod
    def _can_delegate(ctx: RunContext[AgentContext]) -> bool:
        return True

    async def call(self, ctx: RunContext[AgentContext], **kwargs: object) -> str:
        return "ok"


class _UnavailableDelegateBackend(_AvailableDelegateBackend):
    @staticmethod
    def _get_roster_instruction(ctx: RunContext[AgentContext]) -> str | None:
        return None

    @staticmethod
    def _can_delegate(ctx: RunContext[AgentContext]) -> bool:
        return False


def _make_run_ctx(
    *,
    monitor: BackgroundMonitor | None = None,
    agent_id: str = "main",
) -> RunContext[AgentContext]:
    """Create a mock RunContext with optional BackgroundMonitor."""
    mock_resources = MagicMock()
    if monitor is not None:
        mock_resources.get.return_value = monitor
    else:
        mock_resources.get.return_value = None

    mock_ctx = MagicMock()
    mock_ctx.resources = mock_resources
    mock_ctx.agent_id = agent_id

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_ctx
    return run_ctx


def test_tool_not_available_without_monitor() -> None:
    """SpawnDelegateTool should be unavailable without BackgroundMonitor."""
    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=None)
    assert not tool.is_available(ctx)


def test_tool_not_available_without_delegate() -> None:
    """SpawnDelegateTool should be unavailable if delegate tool doesn't exist."""
    monitor = BackgroundMonitor()
    # No core_toolset set -> no delegate tool
    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    assert not tool.is_available(ctx)


def test_tool_not_available_for_subagent() -> None:
    """SpawnDelegateTool should be unavailable for subagents."""
    monitor = BackgroundMonitor()

    mock_delegate = MagicMock(spec=BaseTool)
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    tool = SpawnDelegateTool()
    # Using a subagent id instead of "main"
    ctx = _make_run_ctx(monitor=monitor, agent_id="explorer-1234")
    assert not tool.is_available(ctx)


def test_tool_available_with_delegate() -> None:
    """SpawnDelegateTool should be available when delegate tool exists and agent is main."""
    monitor = BackgroundMonitor()

    mock_delegate = MagicMock(spec=BaseTool)
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    assert tool.is_available(ctx)


def test_spawn_delegate_checks_backend_can_delegate() -> None:
    """SpawnDelegateTool should respect backend target availability."""
    monitor = BackgroundMonitor()
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = _UnavailableDelegateBackend()
    monitor.set_core_toolset(mock_toolset)

    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    assert not tool.is_available(ctx)


@pytest.mark.asyncio
async def test_async_delegate_uses_hidden_backend_roster() -> None:
    """Async delegate should expose backend roster while keeping async wording."""
    monitor = BackgroundMonitor()
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = _AvailableDelegateBackend()
    monitor.set_core_toolset(mock_toolset)

    tool = AsyncDelegateTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    assert tool.is_available(ctx)
    instruction = await tool.get_instruction(ctx)
    assert instruction is not None
    assert "delegate is asynchronous" in instruction
    assert "returns an agent ID immediately" in instruction
    assert "do not manually poll or loop" in instruction
    assert "wait_subagent once with a bounded timeout" in instruction
    assert "finish the current response" in instruction
    assert "let the CLI notify you" in instruction
    assert "steer_subagent" in instruction
    assert '<subagent name="helper">' in instruction
    assert "Helper subagent" in instruction
    assert "Delegate calls are blocking" not in instruction


def test_background_tools_export_keeps_legacy_spawn_name() -> None:
    """Legacy background_tools export should not include both delegate variants."""
    from yaacli.toolsets.background import background_tools

    assert SpawnDelegateTool in background_tools
    assert WaitSubagentTool in background_tools
    assert AsyncDelegateTool not in background_tools


@pytest.mark.asyncio
async def test_wait_subagent_unavailable_without_work() -> None:
    """WaitSubagentTool should be unavailable when no background work exists."""
    monitor = BackgroundMonitor()
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    assert not tool.is_available(ctx)


@pytest.mark.asyncio
async def test_wait_subagent_unavailable_for_subagent() -> None:
    """WaitSubagentTool should be unavailable outside the main agent."""
    monitor = BackgroundMonitor()
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="helper-bg-a1b2",
            subagent_name="helper",
            status="completed",
            content="done",
        )
    )
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="helper-bg-a1b2")

    assert not tool.is_available(ctx)


@pytest.mark.asyncio
async def test_wait_subagent_returns_cached_result() -> None:
    """WaitSubagentTool should return completed cached results immediately."""
    monitor = BackgroundMonitor()
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="helper-bg-a1b2",
            subagent_name="helper",
            status="completed",
            content="done",
        )
    )
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    assert tool.is_available(ctx)
    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=0)

    assert result["status"] == "completed"
    assert result["agent_id"] == "helper-bg-a1b2"
    assert result["subagent_name"] == "helper"
    assert result["result"] == "done"
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_wait_subagent_unknown_agent() -> None:
    """WaitSubagentTool should report unknown ids without waiting."""
    monitor = BackgroundMonitor()
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="helper-bg-a1b2",
            subagent_name="helper",
            status="completed",
            content="done",
        )
    )
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    result = await tool.call(ctx, agent_id="missing-bg-z9", timeout_seconds=0)

    assert result["status"] == "not_found"
    assert result["agent_id"] == "missing-bg-z9"
    assert result["known_agent_ids"] == ["helper-bg-a1b2"]


@pytest.mark.asyncio
async def test_wait_subagent_timeout_does_not_cancel_task() -> None:
    """Timeout should return running without cancelling the background task."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(1)

    task = asyncio.create_task(sleeper())
    monitor.register_task("helper-bg-a1b2", task, subagent_name="helper", prompt="work")
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=0.01)

    assert result["status"] == "running"
    assert result["timed_out"] is True
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_wait_subagent_waits_for_active_task_result() -> None:
    """WaitSubagentTool should wait until a running task records its result."""
    monitor = BackgroundMonitor()

    async def complete() -> None:
        await asyncio.sleep(0.01)
        monitor.record_task_result(
            BackgroundTaskResult(
                agent_id="helper-bg-a1b2",
                subagent_name="helper",
                status="completed",
                content="done",
            )
        )

    task = asyncio.create_task(complete())
    monitor.register_task("helper-bg-a1b2", task, subagent_name="helper", prompt="work")
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=1)

    assert result["status"] == "completed"
    assert result["result"] == "done"


@pytest.mark.asyncio
async def test_wait_subagent_waits_for_all_known_agents() -> None:
    """Omitting agent_id should wait for all known background agents."""
    monitor = BackgroundMonitor()
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="cached-bg-a1b2",
            subagent_name="cached",
            status="completed",
            content="cached done",
        )
    )

    async def complete() -> None:
        await asyncio.sleep(0.01)
        monitor.record_task_result(
            BackgroundTaskResult(
                agent_id="active-bg-c3d4",
                subagent_name="active",
                status="completed",
                content="active done",
            )
        )

    task = asyncio.create_task(complete())
    monitor.register_task("active-bg-c3d4", task, subagent_name="active", prompt="work")
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    result = await tool.call(ctx, timeout_seconds=1)

    assert result["status"] == "completed"
    assert result["timed_out"] is False
    results_by_id = {item["agent_id"]: item for item in result["results"]}
    assert results_by_id["cached-bg-a1b2"]["result"] == "cached done"
    assert results_by_id["active-bg-c3d4"]["result"] == "active done"


@pytest.mark.asyncio
async def test_wait_subagent_formats_failed_and_cancelled_results() -> None:
    """WaitSubagentTool should format failed and cancelled terminal results."""
    monitor = BackgroundMonitor()
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="failed-bg-a1b2",
            subagent_name="failed",
            status="failed",
            error="boom",
        )
    )
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="cancelled-bg-c3d4",
            subagent_name="cancelled",
            status="cancelled",
            error="stopped",
        )
    )
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    failed = await tool.call(ctx, agent_id="failed-bg-a1b2", timeout_seconds=0)
    cancelled = await tool.call(ctx, agent_id="cancelled-bg-c3d4", timeout_seconds=0)

    assert failed["status"] == "failed"
    assert failed["error"] == "boom"
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] == "stopped"


@pytest.mark.asyncio
async def test_wait_subagent_done_without_result_reports_failed() -> None:
    """A done task without a cached terminal result should not be reported as running."""
    monitor = BackgroundMonitor()

    async def complete_without_result() -> None:
        return None

    task = asyncio.create_task(complete_without_result())
    monitor.register_task("helper-bg-a1b2", task, subagent_name="helper", prompt="work")
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=1)

    assert result["status"] == "failed"
    assert result["agent_id"] == "helper-bg-a1b2"
    assert result["subagent_name"] == "helper"
    assert result["error"] == "Background task finished without recording a result."


@pytest.mark.asyncio
async def test_wait_subagent_marks_active_bus_result_consumed() -> None:
    """wait_subagent should suppress duplicate delivery from the active message bus."""
    monitor = BackgroundMonitor()
    bus = MessageBus()
    bus.subscribe("main")
    message_id = monitor.get_task_result_message_id("helper-bg-a1b2")
    bus.send(BusMessage(id=message_id, content="done", source="helper-bg-a1b2", target="main"))
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="helper-bg-a1b2",
            subagent_name="helper",
            status="completed",
            content="done",
        )
    )
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    ctx.deps.message_bus = bus

    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=0)

    assert result["status"] == "completed"
    assert result["result"] == "done"
    assert bus.consume("main") == []


@pytest.mark.asyncio
async def test_wait_subagent_drops_queued_fallback_result() -> None:
    """wait_subagent should remove matching queued fallback notifications."""
    monitor = BackgroundMonitor()
    message_id = monitor.get_task_result_message_id("helper-bg-a1b2")
    monitor.enqueue_message(BusMessage(id=message_id, content="done", source="helper-bg-a1b2", target="main"))
    monitor.record_task_result(
        BackgroundTaskResult(
            agent_id="helper-bg-a1b2",
            subagent_name="helper",
            status="completed",
            content="done",
        )
    )
    bus = MessageBus()
    tool = WaitSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    ctx.deps.message_bus = bus

    result = await tool.call(ctx, agent_id="helper-bg-a1b2", timeout_seconds=0)

    assert result["status"] == "completed"
    assert monitor.deliver_pending_messages(bus, "main") == 0


@pytest.mark.asyncio
async def test_wait_subagent_waiting_task_suppresses_completion_bus_message() -> None:
    """A task completing while wait_subagent is active should not send a duplicate bus message."""
    monitor = BackgroundMonitor()
    mock_delegate = AsyncMock(spec=BaseTool)
    release = asyncio.Event()

    async def delegate_call(*args: object, **kwargs: object) -> str:
        await release.wait()
        return "Subagent result"

    mock_delegate.call = AsyncMock(side_effect=delegate_call)
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    bus = MessageBus()
    bus.subscribe("main")
    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {}
    mock_deps.agent_id = "main"
    mock_deps.message_bus = bus
    mock_deps.send_message.side_effect = bus.send

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    spawn_result = await SpawnDelegateTool().call(
        run_ctx,
        subagent_name="helper",
        prompt="work",
        agent_id="helper-bg-a1b2",
    )
    assert "helper-bg-a1b2" in spawn_result

    wait_task = asyncio.create_task(WaitSubagentTool().call(run_ctx, agent_id="helper-bg-a1b2", timeout_seconds=1))
    await asyncio.sleep(0)
    release.set()
    wait_result = await wait_task

    assert wait_result["status"] == "completed"
    assert wait_result["result"] == "Subagent result"
    assert mock_deps.send_message.call_count == 0
    assert bus.consume("main") == []


@pytest.mark.parametrize(
    ("input_timeout", "expected"),
    [
        (-1.0, 0.0),
        (999.0, 300.0),
        (float("inf"), 300.0),
        (float("nan"), 300.0),
    ],
)
def test_wait_subagent_normalizes_timeout(input_timeout: float, expected: float) -> None:
    """Timeout normalization should clamp unsafe float values."""
    assert WaitSubagentTool._normalize_timeout(input_timeout) == expected


@pytest.mark.asyncio
async def test_tool_call_launches_background_task() -> None:
    """Calling SpawnDelegateTool should launch a background task."""
    monitor = BackgroundMonitor()

    # Create a mock delegate tool that returns a result
    mock_delegate = AsyncMock(spec=BaseTool)
    mock_delegate.call = AsyncMock(return_value="Subagent result")

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    # Create mock context
    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {}
    mock_deps.agent_id = "main"
    mock_deps.send_message = MagicMock()

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SpawnDelegateTool()
    result = await tool.call(run_ctx, subagent_name="explorer", prompt="Find stuff")

    # Should return immediately with a status message that prevents polling loops.
    assert "Spawned delegate" in result
    assert "explorer" in result
    assert "Do not manually poll or loop" in result
    assert "wait_subagent once with a bounded timeout" in result
    assert "finish your current response now" in result
    assert "automatically notify you" in result
    assert "steer_subagent" in result

    # A background task should be registered
    assert len(monitor.active_tasks) == 1

    # Wait for the background task to complete
    tasks = list(monitor.active_tasks.values())
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)  # Allow done callbacks to fire

    # Completion callback should have been called (via notify_completion)
    # We don't have a callback set, but the task should complete
    assert not monitor.has_active_tasks

    # Message is delivered immediately for the active run guard.
    assert mock_deps.send_message.call_count == 1
    assert monitor.has_pending_messages is False


@pytest.mark.asyncio
async def test_tool_call_queues_result_for_redelivery() -> None:
    """Background delegate results should be delivered immediately when main is subscribed."""
    monitor = BackgroundMonitor()

    mock_delegate = AsyncMock(spec=BaseTool)
    mock_delegate.call = AsyncMock(return_value="Subagent result")

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    bus = MessageBus()
    bus.subscribe("main")
    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {}
    mock_deps.agent_id = "main"
    mock_deps.message_bus = bus
    mock_deps.send_message.side_effect = bus.send

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    result = await SpawnDelegateTool().call(run_ctx, subagent_name="explorer", prompt="Find stuff")

    assert "Spawned delegate" in result
    tasks = list(monitor.active_tasks.values())
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert bus.has_pending("main") is True
    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].source.startswith("explorer-bg-")
    assert messages[0].target == "main"
    assert messages[0].content == "Subagent result"

    # No fallback is queued while the main agent is still subscribed; the
    # active run consumes the directly delivered message instead.
    assert monitor.has_pending_messages is False
    assert monitor.deliver_pending_messages(bus, "main") == 0
    assert bus.has_pending("main") is False


@pytest.mark.asyncio
async def test_tool_call_redelivery_works_when_main_not_subscribed() -> None:
    """Fallback redelivery should work if the active main context already exited."""
    monitor = BackgroundMonitor()

    mock_delegate = AsyncMock(spec=BaseTool)
    mock_delegate.call = AsyncMock(return_value="Late subagent result")

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    bus = MessageBus()
    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {}
    mock_deps.agent_id = "main"
    mock_deps.message_bus = bus
    mock_deps.send_message.side_effect = bus.send

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    result = await SpawnDelegateTool().call(run_ctx, subagent_name="explorer", prompt="Find stuff")

    assert "Spawned delegate" in result
    tasks = list(monitor.active_tasks.values())
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert mock_deps.send_message.call_count == 0
    assert bus.has_pending("main") is False
    assert monitor.deliver_pending_messages(bus, "main") == 1
    assert bus.has_pending("main") is True
    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].content == "Late subagent result"


@pytest.mark.asyncio
async def test_tool_call_no_monitor() -> None:
    """Calling SpawnDelegateTool without monitor should return error."""
    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=None)
    result = await tool.call(ctx, subagent_name="explorer", prompt="Find stuff")
    assert "Error" in result


@pytest.mark.asyncio
async def test_tool_call_no_delegate() -> None:
    """Calling SpawnDelegateTool without delegate tool should return error."""
    monitor = BackgroundMonitor()
    tool = SpawnDelegateTool()
    ctx = _make_run_ctx(monitor=monitor)
    result = await tool.call(ctx, subagent_name="explorer", prompt="Find stuff")
    assert "Error" in result


@pytest.mark.asyncio
async def test_tui_redelivers_background_message_after_running_turn_exits() -> None:
    """TUI should redeliver queued background results after an active main-agent turn exits."""
    monitor = BackgroundMonitor()
    bus = MessageBus()

    env = MagicMock()
    env.resources = MagicMock()
    env.resources.get.return_value = monitor

    ctx = MagicMock()
    ctx.agent_id = "main"
    ctx.message_bus = bus

    runtime = MagicMock()
    runtime.env = env
    runtime.ctx = ctx

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()

    app = TUIApp(config=config, config_manager=config_manager)
    app._runtime = runtime
    app._state = TUIState.RUNNING
    app._pending_bus_check_needed = False
    app._append_system_output = MagicMock()

    async def noop_run_agent(user_input: str) -> None:
        return None

    app._run_agent = noop_run_agent  # type: ignore[method-assign]

    monitor.enqueue_message(BusMessage(content="Subagent result", source="executor-bg-123", target="main"))
    app._on_background_task_complete("executor-bg-123")

    app._append_system_output.assert_called_once_with("Background task completed: executor-bg-123")
    assert app._pending_bus_check_needed is True
    assert bus.has_pending("main") is False

    app._state = TUIState.IDLE
    app._check_pending_bus_messages()

    assert bus.has_pending("main") is True
    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].content == "Subagent result"
    assert app._state == TUIState.RUNNING
    assert app._agent_task is not None
    await app._agent_task


# =============================================================================
# TUIEnvironment Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_env_background_monitor_registered(tmp_path: Path) -> None:
    """BackgroundMonitor should be registered as a resource."""
    async with TUIEnvironment(default_path=tmp_path) as env:
        monitor = env.resources.get_typed(BACKGROUND_MONITOR_KEY, BackgroundMonitor)
        assert isinstance(monitor, BackgroundMonitor)


@pytest.mark.asyncio
async def test_env_background_monitor_property(tmp_path: Path) -> None:
    """background_monitor property should return the BackgroundMonitor."""
    async with TUIEnvironment(default_path=tmp_path) as env:
        monitor = env.background_monitor
        assert isinstance(monitor, BackgroundMonitor)
        # Same instance from resources
        assert monitor is env.resources.get_typed(BACKGROUND_MONITOR_KEY, BackgroundMonitor)


def test_env_background_monitor_not_available_before_enter(tmp_path: Path) -> None:
    """background_monitor should raise before entering."""
    env = TUIEnvironment(default_path=tmp_path)
    with pytest.raises(RuntimeError, match="not entered"):
        _ = env.background_monitor


@pytest.mark.asyncio
async def test_env_background_tasks_cleaned_on_exit(tmp_path: Path) -> None:
    """Background tasks should be cancelled when environment exits."""

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task_ref: asyncio.Task[None] | None = None

    async with TUIEnvironment(default_path=tmp_path) as env:
        monitor = env.background_monitor
        task_ref = asyncio.create_task(sleeper())
        monitor.register_task("test-bg", task_ref)
        assert len(monitor.active_tasks) == 1

    # After exit, task should be cancelled (monitor.close() called by resource registry)
    assert task_ref is not None
    assert task_ref.cancelled() or task_ref.done()


# =============================================================================
# SteerSubagentTool Tests
# =============================================================================


def test_steer_not_available_without_monitor() -> None:
    """SteerSubagentTool should be unavailable without BackgroundMonitor."""
    tool = SteerSubagentTool()
    ctx = _make_run_ctx(monitor=None)
    assert not tool.is_available(ctx)


def test_steer_not_available_without_active_tasks() -> None:
    """SteerSubagentTool should be unavailable when no background tasks are running."""
    monitor = BackgroundMonitor()
    tool = SteerSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    assert not tool.is_available(ctx)


def test_steer_not_available_for_subagent() -> None:
    """SteerSubagentTool should be unavailable for subagents."""
    monitor = BackgroundMonitor()
    tool = SteerSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="explorer-1234")
    assert not tool.is_available(ctx)


@pytest.mark.asyncio
async def test_steer_available_with_active_tasks() -> None:
    """SteerSubagentTool should be available when background tasks are running."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    monitor.register_task("searcher-bg-a1b2", task)

    tool = SteerSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    assert tool.is_available(ctx)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_steer_sends_bus_message() -> None:
    """Steering a running subagent should send a targeted BusMessage."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    monitor.register_task("searcher-bg-a1b2", task)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.send_message = MagicMock()

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SteerSubagentTool()
    result = await tool.call(run_ctx, agent_id="searcher-bg-a1b2", message="also check docs folder")

    assert "Steering message sent" in result
    assert "searcher-bg-a1b2" in result

    # Verify BusMessage was sent with correct target
    mock_deps.send_message.assert_called_once()
    sent_msg = mock_deps.send_message.call_args[0][0]
    assert sent_msg.target == "searcher-bg-a1b2"
    assert sent_msg.source == "main"
    assert sent_msg.content == "also check docs folder"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_steer_finished_agent_suggests_resume() -> None:
    """Steering a finished subagent should suggest delegate resume."""
    monitor = BackgroundMonitor()

    # Create and immediately complete a task
    async def quick() -> None:
        pass

    task = asyncio.create_task(quick())
    monitor.register_task("searcher-bg-a1b2", task)
    await task
    await asyncio.sleep(0)  # Allow done callback to fire

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {}

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SteerSubagentTool()
    result = await tool.call(run_ctx, agent_id="searcher-bg-a1b2", message="dig deeper")

    assert "already completed" in result
    assert "spawn_delegate" not in result
    assert "agent_id" in result
    assert "searcher-bg-a1b2" in result
    assert "delegate" in result


@pytest.mark.asyncio
async def test_steer_unknown_agent_suggests_resume() -> None:
    """Steering an unknown agent_id should suggest resume."""
    monitor = BackgroundMonitor()

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {}

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SteerSubagentTool()
    result = await tool.call(run_ctx, agent_id="nonexistent-bg-0000", message="hello")

    assert "already completed" in result
    assert "spawn_delegate" not in result
    assert "delegate" in result


@pytest.mark.asyncio
async def test_steer_resume_suggests_delegate_in_async_alias_mode() -> None:
    """Async-only topology should suggest delegate for resume."""
    monitor = BackgroundMonitor()
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = AsyncDelegateTool()
    monitor.set_core_toolset(mock_toolset)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {}

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    result = await SteerSubagentTool().call(run_ctx, agent_id="searcher-bg-a1b2", message="dig deeper")

    assert "already completed" in result
    assert "spawn_delegate" not in result
    assert 'delegate(subagent_name="searcher"' in result


@pytest.mark.asyncio
async def test_steer_resume_suggests_spawn_delegate_in_dual_mode() -> None:
    """Dual topology should suggest spawn_delegate for async resume."""
    monitor = BackgroundMonitor()
    mock_delegate = MagicMock(spec=BaseTool)
    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {}

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    result = await SteerSubagentTool().call(run_ctx, agent_id="searcher-bg-a1b2", message="dig deeper")

    assert "already completed" in result
    assert 'spawn_delegate(subagent_name="searcher"' in result


@pytest.mark.asyncio
async def test_steer_uses_agent_registry_for_name() -> None:
    """Resume suggestion should look up agent_name from agent_registry."""
    monitor = BackgroundMonitor()

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {
        "searcher-bg-a1b2": AgentInfo(agent_id="searcher-bg-a1b2", agent_name="searcher", parent_agent_id="main"),
    }

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SteerSubagentTool()
    result = await tool.call(run_ctx, agent_id="searcher-bg-a1b2", message="check more")

    assert 'subagent_name="searcher"' in result


@pytest.mark.asyncio
async def test_steer_shows_active_tasks_hint() -> None:
    """Resume suggestion should mention other active tasks if any."""
    monitor = BackgroundMonitor()

    async def sleeper() -> None:
        await asyncio.sleep(100)

    active_task = asyncio.create_task(sleeper())
    monitor.register_task("debugger-bg-c3d1", active_task)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.agent_id = "main"
    mock_deps.agent_registry = {}

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SteerSubagentTool()
    # Try to steer a non-existent agent while another is active
    result = await tool.call(run_ctx, agent_id="searcher-bg-a1b2", message="hello")

    assert "Active tasks: debugger-bg-c3d1" in result

    active_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active_task


# =============================================================================
# SpawnDelegateTool Resume Tests
# =============================================================================


@pytest.mark.asyncio
async def test_spawn_delegate_with_agent_id_resume() -> None:
    """SpawnDelegateTool should pass agent_id through for resume."""
    monitor = BackgroundMonitor()

    mock_delegate = AsyncMock(spec=BaseTool)
    mock_delegate.call = AsyncMock(return_value="Resumed result")

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {"searcher-bg-a1b2": []}  # Existing history
    mock_deps.agent_id = "main"
    mock_deps.send_message = MagicMock()

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SpawnDelegateTool()
    result = await tool.call(run_ctx, subagent_name="searcher", prompt="dig deeper", agent_id="searcher-bg-a1b2")

    # Should indicate resume
    assert "Resumed" in result
    assert "searcher-bg-a1b2" in result

    # Wait for background task
    tasks = list(monitor.active_tasks.values())
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)

    # Delegate should have been called with the provided agent_id
    mock_delegate.call.assert_called_once()
    call_kwargs = mock_delegate.call.call_args
    assert call_kwargs[1]["agent_id"] == "searcher-bg-a1b2"


@pytest.mark.asyncio
async def test_spawn_delegate_without_agent_id_generates_new() -> None:
    """SpawnDelegateTool without agent_id should generate a new one."""
    monitor = BackgroundMonitor()

    mock_delegate = AsyncMock(spec=BaseTool)
    mock_delegate.call = AsyncMock(return_value="New result")

    mock_toolset = MagicMock()
    mock_toolset._get_tool_instance.return_value = mock_delegate
    monitor.set_core_toolset(mock_toolset)

    mock_deps = MagicMock()
    mock_deps.resources = MagicMock()
    mock_deps.resources.get.return_value = monitor
    mock_deps.subagent_history = {}
    mock_deps.agent_id = "main"
    mock_deps.send_message = MagicMock()

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_deps

    tool = SpawnDelegateTool()
    result = await tool.call(run_ctx, subagent_name="explorer", prompt="Find stuff")

    # Should indicate spawned (not resumed)
    assert "Spawned" in result
    assert "explorer" in result

    # Wait for background task
    tasks = list(monitor.active_tasks.values())
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_steer_instruction_only_with_active_tasks() -> None:
    """get_instruction should return None when no active tasks."""
    monitor = BackgroundMonitor()
    tool = SteerSubagentTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")

    instruction = await tool.get_instruction(ctx)
    assert instruction is None

    # Add an active task
    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    monitor.register_task("searcher-bg-a1b2", task)

    instruction = await tool.get_instruction(ctx)
    assert instruction is not None
    assert "Steer a running background subagent only" in instruction
    assert "Do not poll after steering" in instruction
    assert "CLI will notify you" in instruction

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# =============================================================================
# Output Monitoring Tests (BackgroundMonitor)
# =============================================================================


def _make_mock_shell_with_buffers(
    active_pids: dict[str, str] | None = None,
    buffers: dict[str, tuple[list[str], list[str]] | tuple[list[str], list[str], bool]] | None = None,
) -> MagicMock:
    """Create a mock Shell with active_background_processes and _output_buffers.

    Args:
        active_pids: Mapping of process_id -> command for active processes.
        buffers: Mapping of process_id -> (stdout_lines, stderr_lines).
    """
    from collections import deque

    mock_shell = _make_mock_shell(active_pids)

    # Set up output buffers
    output_buffers: dict[str, MagicMock] = {}
    if buffers:
        for pid, buffer_data in buffers.items():
            if len(buffer_data) == 2:
                stdout_lines, stderr_lines = buffer_data
                completed = False
            else:
                stdout_lines, stderr_lines, completed = buffer_data
            buf = MagicMock()
            buf.stdout = deque(stdout_lines)
            buf.stderr = deque(stderr_lines)
            buf.completed = completed
            buf.exit_code = 0 if completed else None
            output_buffers[pid] = buf

    mock_shell._output_buffers = output_buffers
    return mock_shell


@pytest.mark.asyncio
async def test_register_monitored_process() -> None:
    """register_monitored_process should add pid to _monitored_processes."""
    monitor = BackgroundMonitor()
    monitor.register_monitored_process("pid-1")

    assert "pid-1" in monitor._monitored_processes


@pytest.mark.asyncio
async def test_output_monitoring_notifies_on_new_output() -> None:
    """Should send bus message when monitored process has new output."""
    bus = MessageBus()
    bus.subscribe("main")
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "tail -f log"},
        buffers={"pid-1": (["line1", "line2"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    # Wait for poll cycle
    await asyncio.sleep(0.1)

    assert monitor.deliver_pending_messages(bus, "main") >= 1
    messages = bus.consume("main")
    # Should have at least one output notification
    output_msgs = [m for m in messages if "new output" in m.content]
    assert len(output_msgs) >= 1
    assert "pid-1" in output_msgs[0].content
    assert output_msgs[0].source == "shell-monitor"

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_no_duplicate_notifications() -> None:
    """Should not send duplicate notification if output hasn't been drained."""
    bus = MessageBus()
    bus.subscribe("main")
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "tail -f log"},
        buffers={"pid-1": (["line1"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    # Wait for multiple poll cycles
    await asyncio.sleep(0.2)

    assert monitor.deliver_pending_messages(bus, "main") >= 1
    messages = bus.consume("main")
    output_msgs = [m for m in messages if "new output" in m.content]
    # Only one notification despite multiple polls (output not drained)
    assert len(output_msgs) == 1

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_re_notifies_after_drain() -> None:
    """Should notify again after output is drained and new output appears."""
    from collections import deque

    bus = MessageBus()
    bus.subscribe("main")
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "build"},
        buffers={"pid-1": (["initial output"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    # Wait for first notification
    await asyncio.sleep(0.1)
    monitor.deliver_pending_messages(bus, "main")
    bus.consume("main")  # drain bus

    # Simulate drain: clear the output buffer (as shell_wait would)
    shell._output_buffers["pid-1"].stdout = deque()
    shell._output_buffers["pid-1"].stderr = deque()
    await asyncio.sleep(0.1)

    # Now add new output
    shell._output_buffers["pid-1"].stdout = deque(["new output"])
    await asyncio.sleep(0.1)

    assert monitor.deliver_pending_messages(bus, "main") >= 1
    messages = bus.consume("main")
    output_msgs = [m for m in messages if "new output" in m.content]
    assert len(output_msgs) >= 1

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_drops_wakeup_after_output_drained() -> None:
    """Output wakeup should be dropped if shell_wait drains output before delivery."""
    from collections import deque

    bus = MessageBus()
    bus.subscribe("main")
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "build"},
        buffers={"pid-1": (["initial output"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    await asyncio.sleep(0.1)
    assert monitor.has_pending_messages is True

    # Simulate shell_wait(timeout_seconds=0) draining the output before delivery.
    shell._output_buffers["pid-1"].stdout = deque()
    shell._output_buffers["pid-1"].stderr = deque()

    assert monitor.deliver_pending_messages(bus, "main") == 0
    assert bus.consume("main") == []
    assert monitor.has_pending_messages is False

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_stderr_triggers_notification() -> None:
    """Notification should trigger on stderr output too."""
    bus = MessageBus()
    bus.subscribe("main")
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "make"},
        buffers={"pid-1": ([], ["error: something"])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    await asyncio.sleep(0.1)

    assert monitor.deliver_pending_messages(bus, "main") >= 1
    messages = bus.consume("main")
    output_msgs = [m for m in messages if "new output" in m.content]
    assert len(output_msgs) >= 1

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_buffer_removed_stops_monitoring() -> None:
    """Should stop monitoring if output buffer is removed (killed/consumed)."""
    bus = MessageBus()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "cmd"},
        buffers={"pid-1": (["output"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    await asyncio.sleep(0.1)

    # Remove buffer (simulates kill_process consuming it)
    del shell._output_buffers["pid-1"]
    await asyncio.sleep(0.1)

    assert "pid-1" not in monitor._monitored_processes

    await monitor.close()


@pytest.mark.asyncio
async def test_output_monitoring_callback_invoked() -> None:
    """Completion callback should be invoked when output is detected."""
    bus = MessageBus()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "cmd"},
        buffers={"pid-1": (["output"], [])},
    )

    callback_calls: list[str] = []
    monitor = BackgroundMonitor()
    monitor.set_completion_callback(lambda pid: callback_calls.append(pid))
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    await asyncio.sleep(0.1)

    assert "pid-1" in callback_calls

    await monitor.close()


@pytest.mark.asyncio
async def test_close_clears_monitored_state() -> None:
    """close() should clear monitored processes and notified pending sets."""
    bus = MessageBus()
    shell = _make_mock_shell_with_buffers(
        active_pids={"pid-1": "cmd"},
        buffers={"pid-1": (["output"], [])},
    )

    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=0.05)
    monitor.register_monitored_process("pid-1")

    await asyncio.sleep(0.1)
    assert len(monitor._monitored_processes) > 0

    await monitor.close()

    assert len(monitor._monitored_processes) == 0
    assert len(monitor._notified_pending) == 0


# =============================================================================
# MonitoredShellTool Tests
# =============================================================================


def _make_run_ctx_with_shell(
    *,
    monitor: BackgroundMonitor | None = None,
    shell: MagicMock | None = None,
    agent_id: str = "main",
) -> RunContext[AgentContext]:
    """Create a mock RunContext with BackgroundMonitor and Shell."""
    mock_resources = MagicMock()
    if monitor is not None:
        mock_resources.get.return_value = monitor
    else:
        mock_resources.get.return_value = None

    mock_ctx = MagicMock()
    mock_ctx.resources = mock_resources
    mock_ctx.agent_id = agent_id
    mock_ctx.shell = shell
    mock_ctx.shell_env = {}
    mock_ctx.emit_event = AsyncMock()

    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_ctx
    return run_ctx


@pytest.mark.asyncio
async def test_monitored_shell_tool_not_available_without_monitor() -> None:
    """MonitoredShellTool should not be available without BackgroundMonitor."""
    from yaacli.toolsets.background import MonitoredShellTool

    tool = MonitoredShellTool()
    ctx = _make_run_ctx(monitor=None, agent_id="main")
    ctx.deps.shell = MagicMock()
    assert tool.is_available(ctx) is False


@pytest.mark.asyncio
async def test_monitored_shell_tool_not_available_without_shell() -> None:
    """MonitoredShellTool should not be available without shell."""
    from yaacli.toolsets.background import MonitoredShellTool

    monitor = BackgroundMonitor()
    tool = MonitoredShellTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    ctx.deps.shell = None
    assert tool.is_available(ctx) is False


@pytest.mark.asyncio
async def test_monitored_shell_tool_not_available_without_running_monitor() -> None:
    """MonitoredShellTool should not be available if shell monitor isn't running."""
    from yaacli.toolsets.background import MonitoredShellTool

    monitor = BackgroundMonitor()
    tool = MonitoredShellTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    ctx.deps.shell = MagicMock()
    # Monitor exists but start_shell_monitor not called
    assert tool.is_available(ctx) is False


@pytest.mark.asyncio
async def test_monitored_shell_tool_available_with_running_monitor() -> None:
    """MonitoredShellTool should be available with shell and running monitor."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    shell = _make_mock_shell({"pid-x": "existing"})
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx(monitor=monitor, agent_id="main")
    ctx.deps.shell = shell
    assert tool.is_available(ctx) is True

    await monitor.close()


@pytest.mark.asyncio
async def test_monitored_shell_tool_empty_command() -> None:
    """Should return error for empty command."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    shell = _make_mock_shell()
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx_with_shell(monitor=monitor, shell=shell)

    result = await tool.call(ctx, command="")
    assert "error" in result
    assert "empty" in result["error"].lower()

    await monitor.close()


@pytest.mark.asyncio
async def test_monitored_shell_tool_starts_and_registers() -> None:
    """Should start background process and register for output monitoring."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    mock_shell = _make_mock_shell()
    mock_shell.start = AsyncMock(return_value="proc-123")
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(mock_shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx_with_shell(monitor=monitor, shell=mock_shell)

    result = await tool.call(ctx, command="tail -f /var/log/syslog")

    assert result["process_id"] == "proc-123"
    assert "Monitored" in result["hint"]
    assert "proc-123" in monitor._monitored_processes
    mock_shell.start.assert_called_once()

    await monitor.close()


@pytest.mark.asyncio
async def test_monitored_shell_tool_start_failure() -> None:
    """Should return error if shell.start() fails."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    mock_shell = _make_mock_shell()
    mock_shell.start = AsyncMock(side_effect=RuntimeError("permission denied"))
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(mock_shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx_with_shell(monitor=monitor, shell=mock_shell)

    result = await tool.call(ctx, command="sudo reboot")

    assert "error" in result
    assert "permission denied" in result["error"]
    # Should NOT register for monitoring
    assert len(monitor._monitored_processes) == 0

    await monitor.close()


@pytest.mark.asyncio
async def test_monitored_shell_tool_merges_env() -> None:
    """Should merge shell_env with per-call environment."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    mock_shell = _make_mock_shell()
    mock_shell.start = AsyncMock(return_value="proc-456")
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(mock_shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx_with_shell(monitor=monitor, shell=mock_shell)
    ctx.deps.shell_env = {"BASE_KEY": "base_value"}

    result = await tool.call(ctx, command="echo test", environment={"EXTRA": "extra_value"})

    assert result["process_id"] == "proc-456"
    # Check that merged env was passed
    call_args = mock_shell.start.call_args
    assert call_args.kwargs["env"]["BASE_KEY"] == "base_value"
    assert call_args.kwargs["env"]["EXTRA"] == "extra_value"

    await monitor.close()


@pytest.mark.asyncio
async def test_monitored_shell_tool_emits_event() -> None:
    """Should emit BackgroundShellStartEvent on successful start."""
    from yaacli.toolsets.background import MonitoredShellTool

    bus = MessageBus()
    mock_shell = _make_mock_shell()
    mock_shell.start = AsyncMock(return_value="proc-789")
    monitor = BackgroundMonitor()
    monitor.start_shell_monitor(mock_shell, bus, "main", poll_interval=1.0)

    tool = MonitoredShellTool()
    ctx = _make_run_ctx_with_shell(monitor=monitor, shell=mock_shell)

    await tool.call(ctx, command="npm run dev")

    ctx.deps.emit_event.assert_called_once()
    event = ctx.deps.emit_event.call_args[0][0]
    assert event.process_id == "proc-789"
    assert event.command == "npm run dev"

    await monitor.close()
