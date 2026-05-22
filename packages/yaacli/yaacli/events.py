"""TUI-specific event types for yaacli.

All TUI events extend ya_agent_sdk.events.AgentEvent to integrate with
the SDK's agent_stream_queues mechanism. This allows TUI events to flow
through the same channel as SDK events (compact, handoff, etc.).

Events are emitted via AgentContext.emit_event() and consumed by stream_agent().

Note: Steering functionality now uses SDK's MessageReceivedEvent from
ya_agent_sdk.events. The TUI sends steering messages via ctx.send_message()
and receives MessageReceivedEvent when they are injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ya_agent_sdk.events import AgentEvent


@dataclass
class ContextUpdateEvent(AgentEvent):
    """Real-time context usage update for status bar.

    Contains the current total tokens from message history for context window calculation.

    Attributes:
        total_tokens: Current total tokens used.
        context_window_size: Maximum context window size.
    """

    total_tokens: int = 0
    context_window_size: int = 0


# =============================================================================
# Goal Events
# =============================================================================


class GoalCompleteReason(StrEnum):
    """Enumerated reasons for goal completion."""

    verified = "verified"
    """Agent verified the goal is complete."""

    max_iterations = "max_iterations"
    """Reached the maximum iteration limit."""


@dataclass
class GoalIterationEvent(AgentEvent):
    """Emitted when the goal guard triggers a new iteration.

    Attributes:
        iteration: Current iteration number (1-based).
        max_iterations: Maximum iterations allowed.
        task: Original task description.
    """

    iteration: int = 0
    max_iterations: int = 0
    task: str = ""


@dataclass
class GoalCompleteEvent(AgentEvent):
    """Emitted when goal mode ends.

    Attributes:
        iteration: Final iteration count.
        reason: Why the goal ended (enumerated).
        task: Original task description.
    """

    iteration: int = 0
    reason: GoalCompleteReason = GoalCompleteReason.verified
    task: str = ""
