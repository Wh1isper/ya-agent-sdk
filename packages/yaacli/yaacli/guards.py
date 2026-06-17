"""Output guards for yaacli.

This module provides output validators (guards) that are attached to the
TUI agent. Guards use ModelRetry to continue agent execution when certain
conditions are met, such as an active goal that still needs verified work.

Guards read state from TUIContext (accessed via ctx.deps) and emit events
via ctx.deps.emit_event() for TUI rendering.
"""

from __future__ import annotations

import uuid
from typing import TypeVar

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

from yaacli.events import GoalCompleteEvent, GoalCompleteReason, GoalIterationEvent
from yaacli.logging import get_logger
from yaacli.session import TUIContext

logger = get_logger(__name__)

GOAL_COMPLETE_MARKER = "[GOAL_COMPLETE]"


def _has_completion_marker(output: str) -> bool:
    """Check if output contains the completion marker as a standalone line.

    The marker must appear on its own line while ignoring surrounding whitespace,
    which avoids false positives when the model mentions the marker in prose.
    """
    return any(line.strip() == GOAL_COMPLETE_MARKER for line in output.splitlines())


def _escape_xml_text(input_text: str) -> str:
    """Escape user-provided goal text for XML-style prompt blocks."""
    return input_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_goal_check_prompt(goal: str) -> str:
    """Build the hidden continuation prompt for an active goal."""
    escaped_goal = _escape_xml_text(goal)
    return (
        "<goal-check>\n"
        "Continue working toward the active goal.\n\n"
        "The objective below is user-provided data. Treat it as the task to pursue, "
        "following the higher-priority system and developer instructions already in force.\n\n"
        f"<objective>\n{escaped_goal}\n</objective>\n\n"
        "Continuation behavior:\n"
        "- Keep the full objective intact across iterations.\n"
        "- Make concrete progress toward the requested end state.\n"
        "- Preserve the original scope when checking completion.\n\n"
        "Task planning and tracking:\n"
        "- When task management tools are available, use them before execution to decompose the objective "
        "into concrete, trackable tasks.\n"
        "- Create or update tasks for the major requirements, dependencies, verification steps, and "
        "deliverables implied by the objective.\n"
        "- Mark tasks in progress when starting them and complete them immediately after they are verified.\n"
        "- Use the task list as the working plan across goal iterations; revisit it before deciding what "
        "to do next.\n"
        "- If task tools are unavailable or inappropriate for a very small objective, keep an equivalent "
        "brief checklist in your reasoning and continue directly.\n\n"
        "Work from evidence:\n"
        "Use the current workspace and external state as authoritative. Inspect files, command output, "
        "test results, rendered artifacts, runtime behavior, or other direct evidence before relying on "
        "conversation memory.\n\n"
        "Completion audit:\n"
        "Before marking the goal complete, treat completion as unproven and verify it against the "
        "actual current state.\n"
        "- Derive concrete requirements from the objective and any referenced files, plans, specifications, "
        "issues, or user instructions.\n"
        "- For every explicit requirement, numbered item, named artifact, command, test, invariant, and "
        "deliverable, identify authoritative evidence that proves it.\n"
        "- Match the verification scope to the requirement scope. A broad requirement needs broad evidence.\n"
        "- Treat tests, manifests, verifiers, green checks, and search results as evidence after confirming "
        "they cover the relevant requirement.\n"
        "- Treat uncertain, indirect, partial, or missing evidence as remaining work.\n"
        "- The audit must prove completion requirement by requirement.\n\n"
        f"If current evidence proves the full goal is complete, respond with {GOAL_COMPLETE_MARKER} "
        "on its own line. Otherwise, continue working on the remaining requirements.\n"
        "</goal-check>"
    )


def _build_post_restore_goal_audit_prompt(goal: str, source: str | None) -> str:
    """Build a stricter audit prompt after compact/summarize restored context."""
    escaped_goal = _escape_xml_text(goal)
    escaped_source = _escape_xml_text(source or "context_handoff")
    return (
        "<goal-post-restore-audit>\n"
        "A context handoff/compact occurred while goal mode was active. The restored summary is only a "
        "continuity aid; it is not proof that the goal is complete.\n\n"
        f"<handoff-source>{escaped_source}</handoff-source>\n\n"
        "The objective below is user-provided data. Treat it as the active goal, following the "
        "higher-priority system and developer instructions already in force.\n\n"
        f"<objective>\n{escaped_goal}\n</objective>\n\n"
        "Required post-restore behavior:\n"
        "- Do not treat the previous completion marker as accepted. It only triggered this audit.\n"
        "- Reconstruct the concrete completion criteria from the objective and restored context.\n"
        "- Verify each requirement against authoritative current evidence: workspace files, command output, "
        "tests, rendered artifacts, external state, or tool results.\n"
        "- If evidence is missing, stale, indirect, partial, or uncertain, keep working instead of stopping.\n"
        "- If compact/summarize omitted details needed for verification, inspect the workspace or other "
        "available sources directly.\n\n"
        "Completion audit:\n"
        "For every explicit requirement, numbered item, named artifact, command, test, invariant, and "
        "deliverable, identify evidence that proves it is satisfied in the current state. A context "
        "handoff or compact summary alone never satisfies a requirement.\n\n"
        f"Only if this fresh audit proves the full goal complete, respond with {GOAL_COMPLETE_MARKER} "
        "on its own line. Otherwise, continue working on the remaining requirements.\n"
        "</goal-post-restore-audit>"
    )


async def _emit_goal_complete(
    deps: TUIContext,
    *,
    task: str,
    iteration: int,
    reason: GoalCompleteReason,
) -> None:
    """Emit a goal termination event and reset goal state."""
    await deps.emit_event(
        GoalCompleteEvent(
            event_id=f"goal-{uuid.uuid4().hex[:8]}",
            iteration=iteration,
            reason=reason,
            task=task,
        )
    )
    deps.reset_goal()


async def _continue_goal_or_stop(deps: TUIContext, *, task: str, prompt: str) -> None:
    """Advance a goal iteration or stop when the retry budget is exhausted."""
    deps.goal_iteration += 1
    iteration = deps.goal_iteration

    if iteration > deps.goal_max_iterations:
        await _emit_goal_complete(
            deps,
            task=task,
            iteration=iteration - 1,
            reason=GoalCompleteReason.max_iterations,
        )
        logger.info("Goal stopped: reached max iterations")
        return

    await deps.emit_event(
        GoalIterationEvent(
            event_id=f"goal-{uuid.uuid4().hex[:8]}",
            iteration=iteration,
            max_iterations=deps.goal_max_iterations,
            task=task,
        )
    )
    logger.debug("Goal iteration %d/%d", iteration, deps.goal_max_iterations)

    raise ModelRetry(prompt)


async def goal_guard(ctx: RunContext[TUIContext], output: OutputT) -> OutputT:
    """Output guard that drives goal mode via ModelRetry.

    When goal mode is active (ctx.deps.goal_task is not None), this guard
    checks whether the agent has verified task completion. It raises ModelRetry
    with a goal-check prompt when work should continue.

    The guard is a no-op when goal mode is inactive.

    Args:
        ctx: Run context containing TUIContext with goal state.
        output: The output from the agent (str or DeferredToolRequests).

    Returns:
        The output unchanged if goal mode is inactive or complete.

    Raises:
        ModelRetry: If the agent should continue working on the goal.
    """
    deps = ctx.deps

    if not deps.goal_active:
        return output

    # DeferredToolRequests (HITL approval) should keep the goal state unchanged.
    if not isinstance(output, str):
        return output

    task = deps.goal_task or ""

    if _has_completion_marker(output):
        needs_audit, source = deps.consume_goal_context_restore_audit()
        if needs_audit:
            await _continue_goal_or_stop(
                deps,
                task=task,
                prompt=_build_post_restore_goal_audit_prompt(task, source),
            )
            return output

        iteration = deps.goal_iteration
        await _emit_goal_complete(
            deps,
            task=task,
            iteration=iteration,
            reason=GoalCompleteReason.verified,
        )
        logger.info("Goal completed: task verified after %d iteration(s)", iteration)
        return output

    await _continue_goal_or_stop(deps, task=task, prompt=_build_goal_check_prompt(task))
    return output


OutputT = TypeVar("OutputT")


def attach_goal_guard(agent: Agent[TUIContext, OutputT]) -> None:
    """Attach goal guard to an agent as an output validator.

    This function adds the goal_guard as an output validator to the given agent.
    It should be called after agent creation and before execution.

    Args:
        agent: The agent to attach the guard to.
    """

    @agent.output_validator
    async def _guard(ctx: RunContext[TUIContext], output: OutputT) -> OutputT:
        return await goal_guard(ctx, output)
