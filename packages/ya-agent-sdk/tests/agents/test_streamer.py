"""Tests for AgentStreamer class-based async iterator."""

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelResponse, NativeToolCallPart, TextPart, ToolCallPart
from ya_agent_sdk.agents.main import AgentStreamer, _has_tool_call_parts
from ya_agent_sdk.context import StreamEvent


def _make_event(name: str = "test") -> StreamEvent:
    """Create a minimal StreamEvent for testing."""
    return StreamEvent(agent_id="main", agent_name=name, event=MagicMock())


def test_has_tool_call_parts_matches_tool_call_parts() -> None:
    text_response = ModelResponse(parts=[TextPart(content="hello")])
    tool_response = ModelResponse(parts=[ToolCallPart(tool_name="shell_exec", args={"command": "pwd"})])
    builtin_tool_response = ModelResponse(parts=[NativeToolCallPart(tool_name="output", args={"value": "done"})])

    assert _has_tool_call_parts(text_response.parts) is False
    assert _has_tool_call_parts(tool_response.parts) is True
    assert _has_tool_call_parts(builtin_tool_response.parts) is True


def _make_streamer(
    events: list[StreamEvent] | None = None,
    main_task_exc: BaseException | None = None,
    poll_done_immediately: bool = False,
) -> AgentStreamer:
    """Create an AgentStreamer with controllable internals for unit testing.

    Args:
        events: Events to pre-populate in the output queue.
        main_task_exc: Exception to set on the mock main_task.
        poll_done_immediately: Whether poll_done should be set immediately.
    """
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    if events:
        for e in events:
            queue.put_nowait(e)

    poll_done = asyncio.Event()
    if poll_done_immediately:
        poll_done.set()

    # Create a finished main_task mock
    main_task = MagicMock(spec=asyncio.Task)
    if main_task_exc is not None:
        main_task.done.return_value = True
        main_task.cancelled.return_value = False
        main_task.exception.return_value = main_task_exc
    else:
        main_task.done.return_value = False
        main_task.cancelled.return_value = False
        main_task.exception.return_value = None

    poll_task = MagicMock(spec=asyncio.Task)
    poll_task.done.return_value = False
    poll_task.cancelled.return_value = False
    poll_task.exception.return_value = None

    return AgentStreamer(
        _output_queue=queue,
        _main_task=main_task,
        _poll_done=poll_done,
        _tasks=[main_task, poll_task],
    )


async def test_streamer_consumes_all_events() -> None:
    """Normal iteration: streamer yields all queued events then stops."""
    events = [_make_event("e1"), _make_event("e2"), _make_event("e3")]
    streamer = _make_streamer(events=events, poll_done_immediately=True)

    # main_task should appear done (finished normally) for clean exit
    streamer._main_task.done.return_value = True
    streamer._main_task.cancelled.return_value = False
    streamer._main_task.exception.return_value = None

    collected = []
    async for event in streamer:
        collected.append(event)

    assert len(collected) == 3
    assert [e.agent_name for e in collected] == ["e1", "e2", "e3"]


async def test_streamer_propagates_main_task_exception() -> None:
    """When main_task fails, __anext__ raises the exception."""
    error = RuntimeError("main_task failed")
    streamer = _make_streamer(main_task_exc=error)

    with pytest.raises(RuntimeError, match="main_task failed"):
        await streamer.__anext__()


async def test_streamer_propagates_poll_task_exception() -> None:
    """When poll_task fails, __anext__ raises the exception (via _check_task_exceptions)."""
    streamer = _make_streamer()

    # Make poll_task (the second task in _tasks) fail
    poll_task = streamer._tasks[1]
    poll_task.done.return_value = True
    poll_task.cancelled.return_value = False
    poll_task.exception.return_value = RuntimeError("poll_task failed")

    with pytest.raises(RuntimeError, match="poll_task failed"):
        await streamer.__anext__()


async def test_streamer_interrupt_cancels_tasks() -> None:
    """interrupt() sets _interrupted and cancels all non-done tasks."""
    streamer = _make_streamer()

    assert not streamer._interrupted
    streamer.interrupt()

    assert streamer._interrupted
    for task in streamer._tasks:
        task.cancel.assert_called_once()


async def test_streamer_raise_if_exception_with_stored() -> None:
    """raise_if_exception re-raises stored exception."""
    streamer = _make_streamer()
    streamer.exception = ValueError("stored error")

    with pytest.raises(ValueError, match="stored error"):
        streamer.raise_if_exception()


async def test_streamer_raise_if_exception_from_task() -> None:
    """raise_if_exception detects task exceptions even without stored exception."""
    streamer = _make_streamer()
    streamer._tasks[0].done.return_value = True
    streamer._tasks[0].cancelled.return_value = False
    streamer._tasks[0].exception.return_value = TypeError("task error")

    with pytest.raises(TypeError, match="task error"):
        streamer.raise_if_exception()


async def test_streamer_raise_if_exception_no_error() -> None:
    """raise_if_exception does nothing when there are no errors."""
    streamer = _make_streamer()
    # Should not raise
    streamer.raise_if_exception()


async def test_streamer_aiter_returns_self() -> None:
    """__aiter__ returns self (class-based async iterator protocol)."""
    streamer = _make_streamer()
    assert streamer.__aiter__() is streamer


async def test_streamer_stops_when_poll_done_and_queue_empty() -> None:
    """StopAsyncIteration is raised when poll_done is set and queue is empty."""
    streamer = _make_streamer(poll_done_immediately=True)

    # main_task finished normally
    streamer._main_task.done.return_value = True
    streamer._main_task.cancelled.return_value = False
    streamer._main_task.exception.return_value = None

    with pytest.raises(StopAsyncIteration):
        await streamer.__anext__()


async def test_streamer_events_before_stop() -> None:
    """Streamer yields events even when poll_done is already set, then stops."""
    events = [_make_event("a")]
    streamer = _make_streamer(events=events, poll_done_immediately=True)

    # main_task finished normally
    streamer._main_task.done.return_value = True
    streamer._main_task.cancelled.return_value = False
    streamer._main_task.exception.return_value = None

    # First call should return the event
    event = await streamer.__anext__()
    assert event.agent_name == "a"

    # Second call should stop
    with pytest.raises(StopAsyncIteration):
        await streamer.__anext__()


async def test_streamer_stops_when_all_tasks_done_without_poll_done() -> None:
    """Critical #2 fix: __anext__ terminates when all tasks are done and queue is empty,
    even if _poll_done was never set (e.g. tasks cancelled before finally block ran)."""
    streamer = _make_streamer(poll_done_immediately=False)

    # Simulate both tasks cancelled before poll_done was set
    for task in streamer._tasks:
        task.done.return_value = True
        task.cancelled.return_value = True
        task.exception.return_value = None

    # Should NOT hang -- must raise StopAsyncIteration
    with pytest.raises(StopAsyncIteration):
        await streamer.__anext__()


async def test_streamer_drains_queue_before_stopping_on_all_tasks_done() -> None:
    """Fallback exit still drains remaining events before stopping."""
    events = [_make_event("x"), _make_event("y")]
    streamer = _make_streamer(events=events, poll_done_immediately=False)

    # All tasks done (cancelled), but poll_done NOT set
    for task in streamer._tasks:
        task.done.return_value = True
        task.cancelled.return_value = True
        task.exception.return_value = None

    # Should yield the two queued events first
    e1 = await streamer.__anext__()
    assert e1.agent_name == "x"
    e2 = await streamer.__anext__()
    assert e2.agent_name == "y"

    # Then stop
    with pytest.raises(StopAsyncIteration):
        await streamer.__anext__()


async def test_streamer_all_tasks_done_propagates_exception() -> None:
    """Fallback exit path still propagates task exceptions."""
    streamer = _make_streamer(poll_done_immediately=False)

    # main_task failed, poll_task cancelled, poll_done never set
    streamer._tasks[0].done.return_value = True
    streamer._tasks[0].cancelled.return_value = False
    streamer._tasks[0].exception.return_value = RuntimeError("boom")
    streamer._tasks[1].done.return_value = True
    streamer._tasks[1].cancelled.return_value = True
    streamer._tasks[1].exception.return_value = None

    with pytest.raises(RuntimeError, match="boom"):
        await streamer.__anext__()
