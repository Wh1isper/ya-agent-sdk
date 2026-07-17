"""Tests for yaacli.app.state module."""

from __future__ import annotations

from yaacli.app import (
    VALID_TRANSITIONS,
    TUIPhase,
    TUIStateMachine,
)

# =============================================================================
# TUIPhase Tests
# =============================================================================


def test_tui_phase_values():
    """Test TUIPhase enum has every foreground and ready-state value."""
    assert list(TUIPhase) == [
        TUIPhase.IDLE,
        TUIPhase.THINKING,
        TUIPhase.TOOL_CALLING,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.STREAMING_OUTPUT,
        TUIPhase.SHELL_RUNNING,
        TUIPhase.COMMAND_RUNNING,
        TUIPhase.SAVING,
        TUIPhase.CANCELLING,
        TUIPhase.BACKGROUND_RESULT_READY,
    ]
    assert set(VALID_TRANSITIONS) == set(TUIPhase)


# =============================================================================
# VALID_TRANSITIONS Tests
# =============================================================================


def test_valid_transitions_from_idle():
    """Test valid transitions from IDLE."""
    assert TUIPhase.THINKING in VALID_TRANSITIONS[TUIPhase.IDLE]
    assert TUIPhase.TOOL_CALLING not in VALID_TRANSITIONS[TUIPhase.IDLE]


def test_valid_transitions_from_thinking():
    """Test valid transitions from THINKING."""
    valid = VALID_TRANSITIONS[TUIPhase.THINKING]
    assert TUIPhase.TOOL_CALLING in valid
    assert TUIPhase.STREAMING_OUTPUT in valid
    assert TUIPhase.IDLE in valid


def test_valid_transitions_from_tool_calling():
    """Test valid transitions from TOOL_CALLING."""
    valid = VALID_TRANSITIONS[TUIPhase.TOOL_CALLING]
    assert TUIPhase.AWAITING_APPROVAL in valid
    assert TUIPhase.THINKING in valid
    assert TUIPhase.IDLE in valid


def test_valid_transitions_from_awaiting_approval():
    """Test valid transitions from AWAITING_APPROVAL."""
    valid = VALID_TRANSITIONS[TUIPhase.AWAITING_APPROVAL]
    assert TUIPhase.TOOL_CALLING in valid
    assert TUIPhase.IDLE in valid


def test_valid_transitions_from_streaming():
    """Test valid transitions from STREAMING_OUTPUT."""
    valid = VALID_TRANSITIONS[TUIPhase.STREAMING_OUTPUT]
    assert TUIPhase.THINKING in valid
    assert TUIPhase.TOOL_CALLING in valid
    assert TUIPhase.IDLE in valid


def test_valid_transition_graph_covers_every_phase_exactly():
    """Keep the enforced graph complete for every foreground and ready phase."""
    assert {
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
        TUIPhase.SHELL_RUNNING: {
            TUIPhase.CANCELLING,
            TUIPhase.IDLE,
            TUIPhase.BACKGROUND_RESULT_READY,
        },
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
        TUIPhase.CANCELLING: {
            TUIPhase.SAVING,
            TUIPhase.IDLE,
            TUIPhase.BACKGROUND_RESULT_READY,
        },
        TUIPhase.BACKGROUND_RESULT_READY: {
            TUIPhase.THINKING,
            TUIPhase.SHELL_RUNNING,
            TUIPhase.COMMAND_RUNNING,
            TUIPhase.IDLE,
        },
    } == VALID_TRANSITIONS


# =============================================================================
# TUIStateMachine Tests
# =============================================================================


def test_state_machine_init():
    """Test TUIStateMachine initialization."""
    sm = TUIStateMachine()

    assert sm.phase == TUIPhase.IDLE
    assert sm.is_idle
    assert not sm.is_running


def test_state_machine_transition_valid():
    """Test valid state transition."""
    sm = TUIStateMachine()

    result = sm.transition(TUIPhase.THINKING)

    assert result is True
    assert sm.phase == TUIPhase.THINKING
    assert sm.is_running


def test_state_machine_transition_invalid():
    """An invalid transition must not corrupt authoritative phase state."""
    sm = TUIStateMachine()
    transitions: list[tuple[TUIPhase, TUIPhase]] = []
    sm.add_observer(lambda old, new: transitions.append((old, new)))

    # IDLE -> TOOL_CALLING is not valid.
    result = sm.transition(TUIPhase.TOOL_CALLING)

    assert result is False
    assert sm.phase == TUIPhase.IDLE
    assert transitions == []


def test_state_machine_transition_same_state():
    """Test transitioning to same state."""
    sm = TUIStateMachine()

    result = sm.transition(TUIPhase.IDLE)

    assert result is True
    assert sm.phase == TUIPhase.IDLE


def test_state_machine_observer():
    """Test phase change observer."""
    sm = TUIStateMachine()
    transitions = []

    def observer(old: TUIPhase, new: TUIPhase):
        transitions.append((old, new))

    sm.add_observer(observer)
    sm.transition(TUIPhase.THINKING)
    sm.transition(TUIPhase.TOOL_CALLING)

    assert len(transitions) == 2
    assert transitions[0] == (TUIPhase.IDLE, TUIPhase.THINKING)
    assert transitions[1] == (TUIPhase.THINKING, TUIPhase.TOOL_CALLING)


def test_state_machine_remove_observer():
    """Test removing observer."""
    sm = TUIStateMachine()
    calls = []

    def observer(old: TUIPhase, new: TUIPhase):
        calls.append(1)

    sm.add_observer(observer)
    sm.transition(TUIPhase.THINKING)
    assert len(calls) == 1

    sm.remove_observer(observer)
    sm.transition(TUIPhase.IDLE)
    assert len(calls) == 1  # No new call


def test_state_machine_reset():
    """Test resetting to idle."""
    sm = TUIStateMachine()
    sm.transition(TUIPhase.THINKING)
    sm.transition(TUIPhase.TOOL_CALLING)

    sm.reset()

    assert sm.phase == TUIPhase.IDLE
    assert sm.is_idle


def test_state_machine_reset_when_idle():
    """Test reset when already idle (no-op)."""
    sm = TUIStateMachine()
    transitions = []

    sm.add_observer(lambda o, n: transitions.append((o, n)))
    sm.reset()

    assert len(transitions) == 0  # No transition occurred


def test_state_machine_is_awaiting_approval():
    """Test is_awaiting_approval property."""
    sm = TUIStateMachine()

    assert not sm.is_awaiting_approval

    sm.transition(TUIPhase.THINKING)
    sm.transition(TUIPhase.TOOL_CALLING)
    sm.transition(TUIPhase.AWAITING_APPROVAL)

    assert sm.is_awaiting_approval


# =============================================================================
# Convenience Methods Tests
# =============================================================================


def test_start_thinking():
    """Test start_thinking convenience method."""
    sm = TUIStateMachine()
    result = sm.start_thinking()

    assert result is True
    assert sm.phase == TUIPhase.THINKING


def test_start_tools():
    """Test start_tools convenience method."""
    sm = TUIStateMachine()
    sm.start_thinking()
    result = sm.start_tools()

    assert result is True
    assert sm.phase == TUIPhase.TOOL_CALLING


def test_start_approval():
    """Test start_approval convenience method."""
    sm = TUIStateMachine()
    sm.start_thinking()
    sm.start_tools()
    result = sm.start_approval()

    assert result is True
    assert sm.phase == TUIPhase.AWAITING_APPROVAL


def test_start_streaming():
    """Test start_streaming convenience method."""
    sm = TUIStateMachine()
    sm.start_thinking()
    result = sm.start_streaming()

    assert result is True
    assert sm.phase == TUIPhase.STREAMING_OUTPUT


def test_finish():
    """Test finish convenience method."""
    sm = TUIStateMachine()
    sm.start_thinking()
    result = sm.finish()

    assert result is True
    assert sm.is_idle


def test_get_status_text():
    """Test get_status_text method."""
    sm = TUIStateMachine()

    assert sm.get_status_text() == "Idle"

    sm.start_thinking()
    assert sm.get_status_text() == "Thinking..."

    sm.start_tools()
    assert sm.get_status_text() == "Running tools..."

    sm.start_approval()
    assert sm.get_status_text() == "Awaiting approval..."

    sm.transition(TUIPhase.TOOL_CALLING)
    sm.transition(TUIPhase.THINKING)
    sm.start_streaming()
    assert sm.get_status_text() == "Generating..."

    assert sm.transition(TUIPhase.IDLE)
    assert sm.transition(TUIPhase.COMMAND_RUNNING)
    assert sm.get_status_text() == "Running command..."


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_execution_flow():
    """Test a complete execution flow through state machine."""
    sm = TUIStateMachine()
    history = []

    sm.add_observer(lambda o, n: history.append(n))

    # Start execution
    sm.start_thinking()

    # Stream some output
    sm.start_streaming()

    # Back to thinking
    sm.transition(TUIPhase.THINKING)

    # Tool call
    sm.start_tools()

    # Need approval
    sm.start_approval()

    # Approved, continue
    sm.start_tools()

    # Back to thinking
    sm.transition(TUIPhase.THINKING)

    # Complete
    sm.finish()

    expected = [
        TUIPhase.THINKING,
        TUIPhase.STREAMING_OUTPUT,
        TUIPhase.THINKING,
        TUIPhase.TOOL_CALLING,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.TOOL_CALLING,
        TUIPhase.THINKING,
        TUIPhase.IDLE,
    ]

    assert history == expected
    assert sm.is_idle
