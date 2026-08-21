"""TUI-specific typed events for durable YAACLI projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ya_agent_sdk.events import AgentEvent

# =============================================================================
# Goal Events
# =============================================================================


class GoalCompleteReason(StrEnum):
    """Enumerated reasons for goal-mode termination."""

    verified = "verified"
    """Agent verified the goal is complete."""

    max_iterations = "max_iterations"
    """Reached the maximum iteration limit."""

    cancelled = "cancelled"
    """User cancelled the active goal."""

    error = "error"
    """Goal stopped because agent execution failed."""

    unverified_stop = "unverified_stop"
    """Goal mode ended without an accepted verification marker."""


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
