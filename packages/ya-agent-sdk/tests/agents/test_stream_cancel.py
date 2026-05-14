"""Tests for stream_agent cancellation behavior.

Verifies that:
1. Cancellation actually stops agent execution (no token waste)
2. A fresh run after cancellation works without ContextVar errors
3. The cleanup does not re-cancel tasks (allowing pydantic-ai's internal
   ContextVar cleanup to complete)
"""

import asyncio
import contextlib
import contextvars
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, PartStartEvent, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from ya_agent_sdk.agents.main import (
    AgentInterrupted,
    _restore_task_cancellation,
    _suspend_current_task_cancellation,
    create_agent,
    stream_agent,
)
from ya_agent_sdk.environment.local import LocalEnvironment


def _make_runtime(tmp_path: Path, model="test", **kwargs):
    """Create a simple runtime with test model for cancel tests."""
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
    return create_agent(model=model, env=env, **kwargs)


async def test_cancel_stops_agent_execution(tmp_path: Path) -> None:
    """Cancelling the stream task stops agent execution promptly.

    After interrupt(), the streamer should stop yielding events and report
    the interruption. This ensures we don't waste tokens on a cancelled run.
    """
    runtime = _make_runtime(tmp_path)

    async with runtime:
        async with stream_agent(runtime, "Hello") as streamer:
            events_before_cancel = 0
            async for _event in streamer:
                events_before_cancel += 1
                # Cancel after receiving the first event
                if events_before_cancel == 1:
                    streamer.interrupt()

            # _interrupted flag is set immediately by interrupt()
            assert streamer._interrupted

        # exception is set in the finally block after async with exits
        assert isinstance(streamer.exception, AgentInterrupted)


async def test_fresh_context_per_run(tmp_path: Path) -> None:
    """Each stream_agent run gets a fresh contextvars.Context copy.

    This prevents stale ContextVar state from a previous cancelled run
    from leaking into subsequent runs.
    """
    runtime = _make_runtime(tmp_path)

    # Track which contexts are used for main_task creation
    contexts_used: list[contextvars.Context | None] = []
    original_create_task = asyncio.create_task

    def tracking_create_task(coro, *, name=None, context=None):
        # Only track tasks that explicitly pass a context (our main_task)
        if context is not None:
            contexts_used.append(context)
        return original_create_task(coro, name=name, context=context)

    async with runtime:
        with patch("ya_agent_sdk.agents.main.asyncio.create_task", side_effect=tracking_create_task):
            # Run 1
            async with stream_agent(runtime, "Hello run 1") as streamer:
                async for _event in streamer:
                    pass

            # Run 2
            async with stream_agent(runtime, "Hello run 2") as streamer:
                async for _event in streamer:
                    pass

    # Both runs should have created main_task with an explicit context
    assert len(contexts_used) >= 2, f"Expected at least 2 context-aware tasks, got {len(contexts_used)}"

    # The contexts should be distinct objects (fresh copy each time)
    assert contexts_used[0] is not contexts_used[1], "Each run should use a distinct context copy"


async def test_cancel_then_rerun_succeeds(tmp_path: Path) -> None:
    """After cancelling a run, starting a new run should succeed without errors.

    This is the core regression test for the ContextVar "was created in a
    different Context" error that occurred when pydantic-ai's internal
    wrap_task cleanup was interrupted by re-cancellation.
    """
    runtime = _make_runtime(tmp_path)

    async with runtime:
        # Run 1: cancel mid-stream
        async with stream_agent(runtime, "Hello run 1") as streamer:
            async for _event in streamer:
                streamer.interrupt()
                break

        # Run 2: should succeed without ContextVar errors
        async with stream_agent(runtime, "Hello run 2") as streamer:
            events = []
            async for event in streamer:
                events.append(event)

            # Should have completed successfully
            assert streamer.exception is None
            assert len(events) > 0


async def test_cleanup_does_not_recancel_tasks(tmp_path: Path) -> None:
    """The cleanup loop should not re-cancel tasks after the initial cancel.

    Re-cancelling interrupts pydantic-ai's internal ContextVar cleanup
    (set_current_run_context's finally block), causing ValueError on
    subsequent runs.
    """
    runtime = _make_runtime(tmp_path)
    cancel_counts: dict[str, int] = {"main": 0, "poll": 0}

    async with runtime:
        async with stream_agent(runtime, "Hello") as streamer:
            # Consume all events normally
            async for _event in streamer:
                pass

            # Patch cancel on both tasks to count calls
            main_task = streamer._tasks[0]
            poll_task = streamer._tasks[1]

            original_main_cancel = main_task.cancel
            original_poll_cancel = poll_task.cancel

            def counting_main_cancel(msg=None):
                cancel_counts["main"] += 1
                return original_main_cancel(msg) if msg else original_main_cancel()

            def counting_poll_cancel(msg=None):
                cancel_counts["poll"] += 1
                return original_poll_cancel(msg) if msg else original_poll_cancel()

            main_task.cancel = counting_main_cancel
            poll_task.cancel = counting_poll_cancel

        # After the async with exits, cleanup runs.
        # For a normal (non-cancelled) exit, tasks may already be done,
        # so cancel might be called 0 or 1 times (the initial cancel
        # in finally checks `not task.done()`). It should NEVER be > 1.
        assert cancel_counts["main"] <= 1, f"main_task cancelled {cancel_counts['main']} times, expected <= 1"
        assert cancel_counts["poll"] <= 1, f"poll_task cancelled {cancel_counts['poll']} times, expected <= 1"


async def test_cancel_with_external_cancellation(tmp_path: Path) -> None:
    """Simulate external cancellation (like Ctrl+C) during stream_agent.

    The agent should handle CancelledError gracefully and not leak
    ContextVar state to subsequent runs.
    """
    runtime = _make_runtime(tmp_path)

    async with runtime:
        # Run 1: simulate external cancel via task cancellation
        async def run_and_cancel():
            async with stream_agent(runtime, "Hello") as streamer:
                async for _event in streamer:
                    # Cancel our own task to simulate Ctrl+C
                    raise asyncio.CancelledError()

        task = asyncio.create_task(run_and_cancel())
        # Wait for task; it should end with CancelledError
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Brief pause to let any orphaned task cleanup run
        await asyncio.sleep(0.05)

        # Run 2: should succeed despite previous cancellation
        async with stream_agent(runtime, "Hello after cancel") as streamer:
            events = []
            async for event in streamer:
                events.append(event)

            assert streamer.exception is None
            assert len(events) > 0


async def test_interrupted_stream_exposes_text_only_recoverable_messages(tmp_path: Path) -> None:
    """Interrupted streams expose emitted text as partial recoverable history."""

    async def stream_function(_messages, _agent_info: AgentInfo):
        yield "hello "
        await asyncio.sleep(10)
        yield "world"

    runtime = _make_runtime(tmp_path, FunctionModel(stream_function=stream_function))

    async with stream_agent(runtime, "say something") as streamer:
        async for event in streamer:
            if isinstance(event.event, PartStartEvent) and isinstance(event.event.part, TextPart):
                streamer.interrupt()
                break

    messages = streamer.recoverable_messages()
    assert len(messages) >= 2
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.parts == [TextPart(content="hello ")]
    assert response.metadata == {"ya_agent_sdk": {"partial": True, "reason": "stream_interrupted"}}


async def test_interrupted_stream_skips_partial_history_after_tool_call_part(tmp_path: Path) -> None:
    """Partial history skips streams that emitted tool call structure."""

    async def stream_function(_messages, _agent_info: AgentInfo):
        yield ModelResponse(parts=[ToolCallPart(tool_name="some_tool", args={}, tool_call_id="call-1")])
        await asyncio.sleep(10)

    runtime = _make_runtime(tmp_path, FunctionModel(stream_function=stream_function))

    async with stream_agent(runtime, "call tool") as streamer:
        async for _event in streamer:
            streamer.interrupt()
            break

    messages = streamer.recoverable_messages()
    assert not (messages and isinstance(messages[-1], ModelResponse))


async def test_interrupted_stream_appends_current_text_after_completed_tool_history(tmp_path: Path) -> None:
    """Current partial text is appended after prior tool/text history."""

    history = [
        ModelResponse(parts=[ToolCallPart(tool_name="first_tool", args={}, tool_call_id="call-1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="first_tool", content="first result", tool_call_id="call-1")]),
        ModelResponse(
            parts=[
                TextPart(content="Text after first tool."),
                ToolCallPart(tool_name="second_tool", args={}, tool_call_id="call-2"),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="second_tool", content="second result", tool_call_id="call-2")]),
    ]

    async def stream_function(_messages, _agent_info: AgentInfo):
        yield "final partial "
        await asyncio.sleep(10)
        yield "tail"

    runtime = _make_runtime(tmp_path, FunctionModel(stream_function=stream_function))

    async with stream_agent(runtime, "continue", message_history=history) as streamer:
        async for event in streamer:
            if isinstance(event.event, PartStartEvent) and event.event.part == TextPart(content="final partial "):
                streamer.interrupt()
                break

    messages = streamer.recoverable_messages()
    assert messages[: len(history)] == history
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.parts == [TextPart(content="final partial ")]
    assert response.metadata == {"ya_agent_sdk": {"partial": True, "reason": "stream_interrupted"}}


async def test_suspend_current_task_cancellation_allows_cleanup_waits() -> None:
    """Temporarily clearing cancellation should let cleanup awaits finish.

    This mirrors stream_agent's Ctrl+C path: the current task is already
    cancelling, cleanup needs to await inner tasks, and the cancellation
    request must still be restored afterward.
    """
    current_task = asyncio.current_task()
    assert current_task is not None

    worker_finished = False

    async def worker() -> None:
        nonlocal worker_finished
        await asyncio.sleep(0.01)
        worker_finished = True

    current_task.cancel()

    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        task, cleared = _suspend_current_task_cancellation()
        assert task is current_task
        assert cleared >= 1
        try:
            await asyncio.gather(worker())
        finally:
            _restore_task_cancellation(task, cleared)

    assert worker_finished is True

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(0)

    _task, cleared_after_restore = _suspend_current_task_cancellation()
    assert _task is current_task
    assert cleared_after_restore >= 1
