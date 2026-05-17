"""Output guards for yaacli.

This module provides output validators (guards) that are attached to the
TUI agent. Guards use ModelRetry to continue agent execution when certain
conditions are met (e.g., loop mode not yet verified complete).

Guards read state from TUIContext (accessed via ctx.deps) and emit events
via ctx.deps.emit_event() for TUI rendering.
"""

from __future__ import annotations

import uuid
from typing import TypeVar

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

from yaacli.events import LoopCompleteEvent, LoopCompleteReason, LoopIterationEvent
from yaacli.logging import get_logger
from yaacli.session import TUIContext

logger = get_logger(__name__)

LOOP_COMPLETE_MARKER = "[LOOP_COMPLETE]"


def _has_completion_marker(output: str) -> bool:
    """Check if output contains the completion marker as a standalone line.

    The marker must appear on its own line (ignoring surrounding whitespace)
    to avoid false positives when the model mentions the marker in
    explanatory text (e.g. "I can't output [LOOP_COMPLETE] yet").
    """
    return any(line.strip() == LOOP_COMPLETE_MARKER for line in output.splitlines())


async def loop_guard(ctx: RunContext[TUIContext], output: OutputT) -> OutputT:
    """Output guard that drives loop mode via ModelRetry.

    When loop mode is active (ctx.deps.loop_task is not None), this guard
    checks whether the agent has verified task completion. If not, it
    raises ModelRetry with a loop-check prompt to make the agent continue.

    The guard is a no-op when loop mode is inactive.

    Args:
        ctx: Run context containing TUIContext with loop state.
        output: The output from the agent (str or DeferredToolRequests).

    Returns:
        The output unchanged if loop is inactive or complete.

    Raises:
        ModelRetry: If the agent should continue working on the loop task.
    """
    deps = ctx.deps

    # Not in loop mode - pass through immediately
    if not deps.loop_active:
        return output

    # DeferredToolRequests (HITL approval) should not trigger loop check
    if not isinstance(output, str):
        return output

    task = deps.loop_task or ""

    # Agent verified completion (marker must be on its own line)
    if _has_completion_marker(output):
        iteration = deps.loop_iteration
        await deps.emit_event(
            LoopCompleteEvent(
                event_id=f"loop-{uuid.uuid4().hex[:8]}",
                iteration=iteration,
                reason=LoopCompleteReason.verified,
                task=task,
            )
        )
        deps.reset_loop()
        logger.info("Loop completed: task verified after %d iteration(s)", iteration)
        return output

    # Increment iteration
    deps.loop_iteration += 1
    iteration = deps.loop_iteration

    # Hit max iterations - stop gracefully
    if iteration > deps.loop_max_iterations:
        await deps.emit_event(
            LoopCompleteEvent(
                event_id=f"loop-{uuid.uuid4().hex[:8]}",
                iteration=iteration - 1,
                reason=LoopCompleteReason.max_iterations,
                task=task,
            )
        )
        deps.reset_loop()
        logger.info("Loop stopped: reached max iterations")
        return output

    # Emit iteration event and continue
    await deps.emit_event(
        LoopIterationEvent(
            event_id=f"loop-{uuid.uuid4().hex[:8]}",
            iteration=iteration,
            max_iterations=deps.loop_max_iterations,
            task=task,
        )
    )
    logger.debug("Loop iteration %d/%d", iteration, deps.loop_max_iterations)

    raise ModelRetry(
        f"<loop-check>\n"
        f"Original task: {task}\n"
        f"Review your work against the original task. Have you actually verified the results "
        f"(e.g., ran tests, checked output)? Do NOT assume success without verification.\n"
        f"If you have verified the task is fully complete, respond with {LOOP_COMPLETE_MARKER} on its own line.\n"
        f"Otherwise, continue working on what remains.\n"
        f"</loop-check>"
    )


OutputT = TypeVar("OutputT")


def attach_loop_guard(agent: Agent[TUIContext, OutputT]) -> None:
    """Attach loop guard to an agent as an output validator.

    This function adds the loop_guard as an output validator to the given
    agent. It should be called after agent creation, before execution.

    Args:
        agent: The agent to attach the guard to.
    """

    @agent.output_validator
    async def _guard(ctx: RunContext[TUIContext], output: OutputT) -> OutputT:
        return await loop_guard(ctx, output)
