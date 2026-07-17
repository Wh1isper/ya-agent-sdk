"""TUI state management.

Provides explicit state machine for TUI application state.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto


class TUIPhase(Enum):
    """TUI execution phase.

    Represents the current phase of agent execution:
    - IDLE: Waiting for user input
    - THINKING: Model is generating response
    - TOOL_CALLING: Executing tool calls
    - AWAITING_APPROVAL: Waiting for HITL approval
    - STREAMING_OUTPUT: Streaming text output
    - SHELL_RUNNING: Running a direct foreground shell command
    - COMMAND_RUNNING: Dispatching a foreground slash command
    - SAVING: Persisting a session snapshot
    - CANCELLING: Cancelling foreground work
    - BACKGROUND_RESULT_READY: Background results are ready for integration
    """

    IDLE = auto()
    THINKING = auto()
    TOOL_CALLING = auto()
    AWAITING_APPROVAL = auto()
    STREAMING_OUTPUT = auto()
    SHELL_RUNNING = auto()
    COMMAND_RUNNING = auto()
    SAVING = auto()
    CANCELLING = auto()
    BACKGROUND_RESULT_READY = auto()


# Valid state transitions
VALID_TRANSITIONS: dict[TUIPhase, set[TUIPhase]] = {
    TUIPhase.IDLE: {
        TUIPhase.THINKING,
        TUIPhase.SHELL_RUNNING,
        TUIPhase.COMMAND_RUNNING,
        TUIPhase.SAVING,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.THINKING: {
        TUIPhase.TOOL_CALLING,
        TUIPhase.STREAMING_OUTPUT,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.CANCELLING,
        TUIPhase.SAVING,
        TUIPhase.IDLE,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.TOOL_CALLING: {
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.THINKING,
        TUIPhase.STREAMING_OUTPUT,
        TUIPhase.CANCELLING,
        TUIPhase.SAVING,
        TUIPhase.IDLE,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.AWAITING_APPROVAL: {
        TUIPhase.TOOL_CALLING,
        TUIPhase.CANCELLING,
        TUIPhase.SAVING,
        TUIPhase.IDLE,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.STREAMING_OUTPUT: {
        TUIPhase.THINKING,
        TUIPhase.TOOL_CALLING,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.CANCELLING,
        TUIPhase.SAVING,
        TUIPhase.IDLE,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.SHELL_RUNNING: {TUIPhase.CANCELLING, TUIPhase.IDLE, TUIPhase.BACKGROUND_RESULT_READY},
    TUIPhase.COMMAND_RUNNING: {
        TUIPhase.THINKING,
        TUIPhase.SAVING,
        TUIPhase.CANCELLING,
        TUIPhase.IDLE,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.SAVING: {
        TUIPhase.IDLE,
        TUIPhase.THINKING,
        TUIPhase.TOOL_CALLING,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.STREAMING_OUTPUT,
        TUIPhase.COMMAND_RUNNING,
        TUIPhase.CANCELLING,
        TUIPhase.BACKGROUND_RESULT_READY,
    },
    TUIPhase.CANCELLING: {TUIPhase.SAVING, TUIPhase.IDLE, TUIPhase.BACKGROUND_RESULT_READY},
    TUIPhase.BACKGROUND_RESULT_READY: {
        TUIPhase.THINKING,
        TUIPhase.SHELL_RUNNING,
        TUIPhase.COMMAND_RUNNING,
        TUIPhase.IDLE,
    },
}


class TUIStateMachine:
    """Explicit state machine for TUI application.

    Manages state transitions and notifies observers when state changes.
    Invalid transitions are rejected without mutating authoritative state.
    """

    def __init__(self) -> None:
        """Initialize the state machine in the idle phase."""
        self._phase = TUIPhase.IDLE
        self._observers: list[Callable[[TUIPhase, TUIPhase], None]] = []

    @property
    def phase(self) -> TUIPhase:
        """Get current execution phase."""
        return self._phase

    @property
    def is_idle(self) -> bool:
        """Check if in idle state."""
        return self._phase == TUIPhase.IDLE

    @property
    def is_running(self) -> bool:
        """Check if foreground work is active."""
        return self._phase not in {TUIPhase.IDLE, TUIPhase.BACKGROUND_RESULT_READY}

    @property
    def is_agent_running(self) -> bool:
        """Check if an agent turn is active or awaiting approval."""
        return self._phase in {
            TUIPhase.THINKING,
            TUIPhase.TOOL_CALLING,
            TUIPhase.AWAITING_APPROVAL,
            TUIPhase.STREAMING_OUTPUT,
            TUIPhase.CANCELLING,
        }

    @property
    def is_awaiting_approval(self) -> bool:
        """Check if waiting for HITL approval."""
        return self._phase == TUIPhase.AWAITING_APPROVAL

    def add_observer(self, callback: Callable[[TUIPhase, TUIPhase], None]) -> None:
        """Add phase change observer.

        Args:
            callback: Function called with (old_phase, new_phase).
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[TUIPhase, TUIPhase], None]) -> None:
        """Remove phase change observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def transition(self, new_phase: TUIPhase) -> bool:
        """Transition to a new phase.

        Args:
            new_phase: Target phase.

        Returns:
            True if transition was valid, False otherwise.
        """
        if self._phase == new_phase:
            return True

        if not self._is_valid_transition(new_phase):
            return False

        old_phase = self._phase
        self._phase = new_phase

        # Notify observers
        for observer in self._observers:
            observer(old_phase, new_phase)

        return True

    def reset(self) -> None:
        """Reset to idle state."""
        if self._phase != TUIPhase.IDLE:
            self.transition(TUIPhase.IDLE)

    def _is_valid_transition(self, new_phase: TUIPhase) -> bool:
        """Check if transition is valid.

        Args:
            new_phase: Target phase.

        Returns:
            True if transition is allowed.
        """
        valid_targets = VALID_TRANSITIONS.get(self._phase, set())
        return new_phase in valid_targets

    # Convenience methods for common transitions

    def start_thinking(self) -> bool:
        """Transition to thinking phase."""
        return self.transition(TUIPhase.THINKING)

    def start_tools(self) -> bool:
        """Transition to tool calling phase."""
        return self.transition(TUIPhase.TOOL_CALLING)

    def start_approval(self) -> bool:
        """Transition to awaiting approval phase."""
        return self.transition(TUIPhase.AWAITING_APPROVAL)

    def start_streaming(self) -> bool:
        """Transition to streaming output phase."""
        return self.transition(TUIPhase.STREAMING_OUTPUT)

    def finish(self) -> bool:
        """Transition back to idle."""
        return self.transition(TUIPhase.IDLE)

    def get_status_text(self) -> str:
        """Get human-readable status text for current phase."""
        status_map = {
            TUIPhase.IDLE: "Idle",
            TUIPhase.THINKING: "Thinking...",
            TUIPhase.TOOL_CALLING: "Running tools...",
            TUIPhase.AWAITING_APPROVAL: "Awaiting approval...",
            TUIPhase.STREAMING_OUTPUT: "Generating...",
            TUIPhase.SHELL_RUNNING: "Running shell...",
            TUIPhase.COMMAND_RUNNING: "Running command...",
            TUIPhase.SAVING: "Saving session...",
            TUIPhase.CANCELLING: "Cancelling...",
            TUIPhase.BACKGROUND_RESULT_READY: "Background result ready",
        }
        return status_map.get(self._phase, "Unknown")
