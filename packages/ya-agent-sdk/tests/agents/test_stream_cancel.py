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

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from ya_agent_sdk.agents.main import (
    AgentInterrupted,
    AgentStreamer,
    PartialTextAccumulator,
    _restore_task_cancellation,
    _suspend_current_task_cancellation,
    create_agent,
    stream_agent,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.base import BaseTool


def _make_runtime(tmp_path: Path, model="test", **kwargs):
    """Create a simple runtime with test model for cancel tests."""
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
    return create_agent(model=model, env=env, **kwargs)


class ReviewedTool(BaseTool):
    """Tool used to exercise HITL approval deferral in stream recovery tests."""

    name = "reviewed_tool"
    description = "A reviewed tool for HITL stream tests"

    async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
        return f"reviewed {path}"


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


async def test_completed_stream_uses_final_run_history_without_partial_duplicate(tmp_path: Path) -> None:
    """Completed streams expose final run history exactly once."""

    async def stream_function(_messages, _agent_info: AgentInfo):
        yield "hello "
        yield "world"

    runtime = _make_runtime(tmp_path, FunctionModel(stream_function=stream_function))

    async with stream_agent(runtime, "say something") as streamer:
        async for _event in streamer:
            pass

    assert streamer.run is not None
    messages = streamer.recoverable_messages()
    assert messages == streamer.run.all_messages()
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.parts == [TextPart(content="hello world")]
    assert response.metadata is None


async def test_recoverable_messages_preserve_completed_hitl_tool_call_history() -> None:
    """Completed HITL tool-call history takes precedence over accumulated partial parts."""

    formal_response = ModelResponse(
        parts=[ToolCallPart(tool_name="reviewed_tool", args={"path": "file.py"}, tool_call_id="call-1")]
    )
    formal_history = [ModelRequest(parts=[]), formal_response]

    class CompletedHitlRun:
        def all_messages(self) -> list[ModelRequest | ModelResponse]:
            return formal_history

    streamer = AgentStreamer(
        _output_queue=asyncio.Queue(),
        _main_task=asyncio.create_task(asyncio.sleep(0)),
        _poll_done=asyncio.Event(),
        _tasks=[],
    )
    streamer.run = CompletedHitlRun()  # type: ignore[assignment]
    streamer._partial_text.observe(PartStartEvent(index=2, part=TextPart(content="duplicate partial")))

    assert streamer.recoverable_messages() == formal_history


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


def test_recoverable_messages_keep_complete_thinking_before_partial_text() -> None:
    """Completed thinking is preserved as a whole block before partial text."""

    accumulator = PartialTextAccumulator()
    accumulator.observe(PartStartEvent(index=0, part=ThinkingPart(content="Plan")))
    accumulator.observe(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" carefully")))
    accumulator.observe(PartEndEvent(index=0, part=ThinkingPart(content="Plan carefully", signature="sig")))
    accumulator.observe(PartStartEvent(index=1, part=TextPart(content="Answer")))
    accumulator.observe(PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" partial")))

    response = accumulator.build_response()
    assert response is not None
    assert response.parts == [
        ThinkingPart(content="Plan carefully", signature="sig"),
        TextPart(content="Answer partial"),
    ]
    assert response.metadata == {"ya_agent_sdk": {"partial": True, "reason": "stream_interrupted"}}


def test_recoverable_messages_keep_complete_tool_call_args() -> None:
    """Completed tool calls are preserved as whole arguments for the next run."""

    accumulator = PartialTextAccumulator()
    accumulator.observe(
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(tool_name_delta="shell", tool_call_id="call-1"))
    )
    accumulator.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"command": "echo')))
    accumulator.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=' hello"}')))
    accumulator.observe(
        PartEndEvent(
            index=0,
            part=ToolCallPart(tool_name="shell", args='{"command": "echo hello"}', tool_call_id="call-1"),
        )
    )

    response = accumulator.build_response()
    assert response is not None
    assert response.parts == [ToolCallPart(tool_name="shell", args='{"command": "echo hello"}', tool_call_id="call-1")]


def test_recoverable_messages_wait_for_whole_thinking_and_tool_call_parts() -> None:
    """In-progress thinking and tool-call deltas enter history after PartEndEvent."""

    accumulator = PartialTextAccumulator()
    accumulator.observe(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hidden")))
    accumulator.observe(
        PartDeltaEvent(index=1, delta=ToolCallPartDelta(tool_name_delta="shell", args_delta='{"command"'))
    )
    assert accumulator.build_response() is None

    accumulator.observe(PartEndEvent(index=0, part=ThinkingPart(content="hidden")))
    accumulator.observe(
        PartEndEvent(index=1, part=ToolCallPart(tool_name="shell", args='{"command": "pwd"}', tool_call_id="call-2"))
    )

    response = accumulator.build_response()
    assert response is not None
    assert response.parts == [
        ThinkingPart(content="hidden"),
        ToolCallPart(tool_name="shell", args='{"command": "pwd"}', tool_call_id="call-2"),
    ]


def test_recoverable_messages_preserve_interleaved_thinking_tool_and_text_order() -> None:
    """Interleaved thinking and tool calls are preserved in stream order."""

    accumulator = PartialTextAccumulator()
    accumulator.observe(PartStartEvent(index=0, part=ThinkingPart(content="Think before tool")))
    accumulator.observe(PartEndEvent(index=0, part=ThinkingPart(content="Think before tool", signature="sig-1")))
    accumulator.observe(
        PartEndEvent(
            index=1,
            part=ToolCallPart(tool_name="shell", args={"command": "pwd"}, tool_call_id="call-1"),
        )
    )
    accumulator.observe(PartStartEvent(index=2, part=ThinkingPart(content="Think after tool")))
    accumulator.observe(PartEndEvent(index=2, part=ThinkingPart(content="Think after tool", signature="sig-2")))
    accumulator.observe(PartStartEvent(index=3, part=TextPart(content="Final")))
    accumulator.observe(PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=" partial")))

    response = accumulator.build_response()
    assert response is not None
    assert response.parts == [
        ThinkingPart(content="Think before tool", signature="sig-1"),
        ToolCallPart(tool_name="shell", args={"command": "pwd"}, tool_call_id="call-1"),
        ThinkingPart(content="Think after tool", signature="sig-2"),
        TextPart(content="Final partial"),
    ]


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
