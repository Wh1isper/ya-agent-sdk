"""TUI context for durable input routing, usage, and goal state."""

from __future__ import annotations

from ya_agent_sdk.context import AgentContext, ResumableState
from ya_agent_sdk.usage import UsageSnapshot


class TUIResumableState(ResumableState):
    """YAACLI session state with a compact cumulative usage snapshot."""

    session_usage_snapshot: UsageSnapshot | None = None


class TUIContext(AgentContext):
    """TUI context extending AgentContext with goal mode support.

    Goal mode fields are set by the /goal command and read by the
    goal output guard to enqueue autonomous task iterations.

    Attributes:
        goal_task: Original task description when goal mode is active. None when inactive.
        goal_iteration: Current iteration count (0-based, incremented by guard).
        goal_max_iterations: Maximum iterations allowed before stopping.
        goal_needs_post_restore_audit: Whether a context restore happened during the active goal.
        goal_last_context_handoff_source: Source of the most recent goal-time context restore.
    """

    model_profile_instructions: str | None = None
    """Static instructions for the currently active main-agent model profile."""

    durable_binding_ref: str | None = None
    """Worker-local authority reference used by persisted inbox capabilities."""

    durable_logical_run_id: str | None = None
    """Current product logical run selected by the execution coordinator."""

    goal_task: str | None = None
    goal_iteration: int = 0
    goal_max_iterations: int = 10
    goal_needs_post_restore_audit: bool = False
    goal_last_context_handoff_source: str | None = None

    def __init__(self, **data: object) -> None:
        """Initialize TUIContext."""
        super().__init__(**data)

    @property
    def goal_active(self) -> bool:
        """Whether goal mode is currently active."""
        return self.goal_task is not None

    def reset_goal(self) -> None:
        """Reset all goal state."""
        self.goal_task = None
        self.goal_iteration = 0
        self.goal_needs_post_restore_audit = False
        self.goal_last_context_handoff_source = None

    def mark_goal_context_restored(self, source: str) -> None:
        """Record that active goal state crossed a context handoff boundary.

        The SDK owns compact/handoff mechanics, while YAACLI owns goal-mode
        completion. When history is replaced during an active goal, YAACLI
        requires one explicit post-restore audit before accepting a completion
        marker.
        """
        if not self.goal_active:
            return
        self.goal_needs_post_restore_audit = True
        self.goal_last_context_handoff_source = source

    def consume_goal_context_restore_audit(self) -> tuple[bool, str | None]:
        """Return and clear the pending post-restore goal audit flag."""
        if not self.goal_needs_post_restore_audit:
            return False, None

        source = self.goal_last_context_handoff_source
        self.goal_needs_post_restore_audit = False
        self.goal_last_context_handoff_source = None
        return True, source
