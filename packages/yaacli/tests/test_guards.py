"""Tests for yaacli output guards."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.exceptions import ModelRetry
from yaacli.events import GoalCompleteEvent, GoalCompleteReason, GoalIterationEvent
from yaacli.guards import (
    GOAL_COMPLETE_MARKER,
    _build_goal_check_prompt,
    _build_post_restore_goal_audit_prompt,
    _has_completion_marker,
    goal_guard,
)
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
    ctx.goal_needs_post_restore_audit = False
    ctx.goal_last_context_handoff_source = None
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
        assert tui_ctx.goal_needs_post_restore_audit is False
        assert tui_ctx.goal_last_context_handoff_source is None

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


def test_build_post_restore_goal_audit_prompt_requires_fresh_evidence() -> None:
    """Post-restore audit prompt should distinguish handoff from completion."""
    prompt = _build_post_restore_goal_audit_prompt("fix <tests> & docs", "summarize_tool")
    assert "goal-post-restore-audit" in prompt
    assert "fix &lt;tests&gt; &amp; docs" in prompt
    assert "summarize_tool" in prompt
    assert "handoff/compact" in prompt
    assert "not proof" in prompt
    assert "fresh audit" in prompt
    assert GOAL_COMPLETE_MARKER in prompt


@pytest.mark.asyncio
async def test_goal_guard_rejects_completion_marker_after_context_restore(tui_ctx: TUIContext) -> None:
    """The first completion marker after context restore should trigger a fresh audit."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 2
    tui_ctx.goal_max_iterations = 10
    tui_ctx.mark_goal_context_restored("compact")
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        with pytest.raises(ModelRetry) as exc_info:
            await goal_guard(ctx, f"Done.\n{GOAL_COMPLETE_MARKER}")

        message = str(exc_info.value)
        assert "goal-post-restore-audit" in message
        assert "handoff/compact" in message
        assert "not proof" in message
        assert "fresh audit" in message
        assert "compact" in message
        assert tui_ctx.goal_task == "fix tests"
        assert tui_ctx.goal_iteration == 3
        assert tui_ctx.goal_needs_post_restore_audit is False
        assert tui_ctx.goal_last_context_handoff_source is None

        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert isinstance(event, GoalIterationEvent)
        assert event.iteration == 3
        assert event.max_iterations == 10


@pytest.mark.asyncio
async def test_goal_guard_context_restore_marker_can_stop_at_max_iterations(tui_ctx: TUIContext) -> None:
    """Post-restore audit should still honor the configured max-iteration stop."""
    tui_ctx.goal_task = "fix tests"
    tui_ctx.goal_iteration = 10
    tui_ctx.goal_max_iterations = 10
    tui_ctx.mark_goal_context_restored("summarize_tool")
    ctx = _make_ctx(tui_ctx)

    with patch.object(TUIContext, "emit_event", new_callable=AsyncMock) as mock_emit:
        result = await goal_guard(ctx, f"Done.\n{GOAL_COMPLETE_MARKER}")

        assert result == f"Done.\n{GOAL_COMPLETE_MARKER}"
        assert tui_ctx.goal_task is None
        assert tui_ctx.goal_iteration == 0
        assert tui_ctx.goal_needs_post_restore_audit is False
        assert tui_ctx.goal_last_context_handoff_source is None

        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert isinstance(event, GoalCompleteEvent)
        assert event.reason == GoalCompleteReason.max_iterations
        assert event.iteration == 10


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
    ctx.goal_needs_post_restore_audit = True
    ctx.goal_last_context_handoff_source = "compact"

    ctx.reset_goal()

    assert ctx.goal_task is None
    assert ctx.goal_iteration == 0
    assert ctx.goal_needs_post_restore_audit is False
    assert ctx.goal_last_context_handoff_source is None


def test_tui_context_mark_goal_context_restored_only_when_active() -> None:
    """Context restore markers should only apply while goal mode is active."""
    ctx = TUIContext.model_construct()
    ctx.goal_task = None
    ctx.goal_needs_post_restore_audit = False
    ctx.goal_last_context_handoff_source = None

    ctx.mark_goal_context_restored("compact")
    assert ctx.goal_needs_post_restore_audit is False
    assert ctx.goal_last_context_handoff_source is None

    ctx.goal_task = "some task"
    ctx.mark_goal_context_restored("summarize_tool")
    assert ctx.goal_needs_post_restore_audit is True
    assert ctx.goal_last_context_handoff_source == "summarize_tool"


def test_tui_context_consume_goal_context_restore_audit() -> None:
    """Consuming post-restore audit state should be one-shot."""
    ctx = TUIContext.model_construct()
    ctx.goal_task = "some task"
    ctx.goal_needs_post_restore_audit = True
    ctx.goal_last_context_handoff_source = "compact"

    assert ctx.consume_goal_context_restore_audit() == (True, "compact")
    assert ctx.goal_needs_post_restore_audit is False
    assert ctx.goal_last_context_handoff_source is None
    assert ctx.consume_goal_context_restore_audit() == (False, None)
