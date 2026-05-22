"""TUI session management.

TUIContext extends AgentContext with TUI-specific state such as goal mode.

The message bus (inherited from AgentContext) is used for injecting user
guidance during agent execution:
- User sends messages via ctx.send_message("guidance", source="user")
- SDK's inject_bus_messages filter handles injection
- SDK's message_bus_guard ensures messages are processed before completion
"""

from __future__ import annotations

from ya_agent_sdk.context import AgentContext


class TUIContext(AgentContext):
    """TUI context extending AgentContext with goal mode support.

    Goal mode fields are set by the /goal command and read by the
    goal output guard to drive autonomous task iteration.

    Attributes:
        goal_task: Original task description when goal mode is active. None when inactive.
        goal_iteration: Current iteration count (0-based, incremented by guard).
        goal_max_iterations: Maximum iterations allowed before stopping.
    """

    goal_task: str | None = None
    goal_iteration: int = 0
    goal_max_iterations: int = 10

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
