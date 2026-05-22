"""Tests for yaacli output guards."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.exceptions import ModelRetry
from yaacli.events import GoalCompleteEvent, GoalCompleteReason, GoalIterationEvent
from yaacli.guards import GOAL_COMPLETE_MARKER, _build_goal_check_prompt, _has_completion_marker, goal_guard
from yaacli.session import TUIContext


def _make_ctx(deps: TUIContext) -> MagicMock:
    """Create a mock RunContext with the given deps."""
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


@pytest.fixture
def tui_ctx() -> TUIContext:
    """Create a minimal TUIContext for testing (not entered)."""
    ctx = TUIContext.model_construct()
    ctx.goal_task = None
    ctx.goal_iteration = 0
    ctx.goal_max_iterations = 10
    ctx._stream_queue_enabled = False
    return ctx


@pytest.mark.asyncio
async def test_goal_guard_passthrough_when_inactive(tui_ctx: TUIContext) -> None:
    """Guard should pass through output when goal mode is inactive."""
    ctx = _make_ctx(tui_ctx)
    result = await goal_guard(ctx, "Hello world")
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_goal_guard_passthrough_deferred_tool_requests(tui_ctx: TUIContext) -> None:
    """Guard should pass through DeferredToolRequests even when goal mode is active."""
    tui_ctx.goal_task = "fix tests"
    ctx = _make_ctx(tui_ctx)

    mock_deferred = MagicMock()  # Simulate DeferredToolRequests (not a str)
    result = await goal_guard(ctx, mock_deferred)
    assert result is mock_deferred


@pytest.mark.asyncio
async def test_goal_guard_verified_complete(tui_ctx: TUIContext) -> None:
    """Guard should pass through and reset when output contains GOAL_COMPLETE_MARKER."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 3
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        output = f"All done.\n{GOAL_COMPLETE_MARKER}"
        result = await goal_guard(ctx, output)

        assert result == output
        assert tui_ctx.goal_task is None
        assert tui_ctx.goal_iteration == 0

        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert isinstance(event, GoalCompleteEvent)
        assert event.reason == GoalCompleteReason.verified
        assert event.iteration == 3


@pytest.mark.asyncio
async def test_goal_guard_continues_iteration(tui_ctx: TUIContext) -> None:
    """Guard should raise ModelRetry to continue when task is not verified."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 0
    tui_ctx.goal_max_iterations = 10
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        with pytest.raises(ModelRetry) as exc_info:
            await goal_guard(ctx, "I think it's done")

        message = str(exc_info.value)
        assert "goal-check" in message
        assert "Completion audit" in message
        assert "Work from evidence" in message
        assert GOAL_COMPLETE_MARKER in message
        assert tui_ctx.goal_iteration == 1

        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert isinstance(event, GoalIterationEvent)
        assert event.iteration == 1
        assert event.max_iterations == 10


@pytest.mark.asyncio
async def test_goal_guard_max_iterations_reached(tui_ctx: TUIContext) -> None:
    """Guard should stop and pass through when max iterations are exceeded."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 10  # Already at max
    tui_ctx.goal_max_iterations = 10
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        result = await goal_guard(ctx, "Still working...")

        assert result == "Still working..."
        assert tui_ctx.goal_task is None
        assert tui_ctx.goal_iteration == 0

        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert isinstance(event, GoalCompleteEvent)
        assert event.reason == GoalCompleteReason.max_iterations


@pytest.mark.asyncio
async def test_goal_guard_multiple_iterations(tui_ctx: TUIContext) -> None:
    """Guard should increment iteration each time it retries."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 0
    tui_ctx.goal_max_iterations = 3
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        with pytest.raises(ModelRetry):
            await goal_guard(ctx, "working...")
        assert tui_ctx.goal_iteration == 1

        with pytest.raises(ModelRetry):
            await goal_guard(ctx, "still working...")
        assert tui_ctx.goal_iteration == 2

        with pytest.raises(ModelRetry):
            await goal_guard(ctx, "almost done...")
        assert tui_ctx.goal_iteration == 3

        result = await goal_guard(ctx, "giving up...")
        assert result == "giving up..."
        assert tui_ctx.goal_task is None

        assert mock_emit.call_count == 4


@pytest.mark.asyncio
async def test_goal_guard_marker_in_sentence_does_not_complete(tui_ctx: TUIContext) -> None:
    """Guard should keep iterating when marker is embedded in a sentence."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 0
    tui_ctx.goal_max_iterations = 10
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock):
        with pytest.raises(ModelRetry):
            await goal_guard(ctx, f"I have more work before {GOAL_COMPLETE_MARKER} can be used")

        assert tui_ctx.goal_iteration == 1


def test_has_completion_marker_standalone_line() -> None:
    """Marker on its own line should be detected."""
    assert _has_completion_marker(f"Done.\n{GOAL_COMPLETE_MARKER}") is True
    assert _has_completion_marker(f"{GOAL_COMPLETE_MARKER}\nExtra text") is True
    assert _has_completion_marker(f"  {GOAL_COMPLETE_MARKER}  ") is True
    assert _has_completion_marker(GOAL_COMPLETE_MARKER) is True


def test_has_completion_marker_embedded_text() -> None:
    """Marker embedded in a sentence should not be detected."""
    assert _has_completion_marker(f"I will use {GOAL_COMPLETE_MARKER} later") is False
    assert _has_completion_marker(f"Output: {GOAL_COMPLETE_MARKER} is the marker") is False
    assert _has_completion_marker("No marker here") is False


def test_build_goal_check_prompt_escapes_objective() -> None:
    """Goal prompt should escape XML-sensitive objective text."""
    prompt = _build_goal_check_prompt("fix <tests> & docs")
    assert "fix &lt;tests&gt; &amp; docs" in prompt
    assert "Completion audit" in prompt
    assert "Work from evidence" in prompt


def test_tui_context_goal_active_property() -> None:
    """TUIContext.goal_active property should reflect goal_task state."""
    ctx = TUIContext.model_construct()
    ctx.goal_task = None
    assert ctx.goal_active is False

    ctx.goal_task = "some task"
    assert ctx.goal_active is True


def test_tui_context_reset_goal() -> None:
    """TUIContext.reset_goal should clear all goal state."""
    ctx = TUIContext.model_construct()
    ctx.goal_task = "some task"
    ctx.goal_iteration = 5

    ctx.reset_goal()

    assert ctx.goal_task is None
    assert ctx.goal_iteration == 0
