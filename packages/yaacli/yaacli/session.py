"""TUI session management.

TUIContext extends AgentContext with TUI-specific state such as loop mode.

The message bus (inherited from AgentContext) is used for injecting user
guidance during agent execution:
- User sends messages via ctx.send_message("guidance", source="user")
- SDK's inject_bus_messages filter handles injection
- SDK's message_bus_guard ensures messages are processed before completion
"""

from __future__ import annotations

from ya_agent_sdk.context import AgentContext


class TUIContext(AgentContext):
    """TUI context extending AgentContext with loop mode support.

    Loop mode fields are set by the /loop command and read by the
    loop output guard to drive autonomous task iteration.

    Attributes:
        loop_task: Original task description when loop is active. None when inactive.
        loop_iteration: Current iteration count (0-based, incremented by guard).
        loop_max_iterations: Maximum iterations allowed before stopping.
    """

    loop_task: str | None = None
    loop_iteration: int = 0
    loop_max_iterations: int = 10

    def __init__(self, **data: object) -> None:
        """Initialize TUIContext."""
        super().__init__(**data)

    @property
    def loop_active(self) -> bool:
        """Whether loop mode is currently active."""
        return self.loop_task is not None

    def reset_loop(self) -> None:
        """Reset all loop state."""
        self.loop_task = None
        self.loop_iteration = 0
