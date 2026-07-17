"""TUI application module.

This module provides the core TUI application components:
- TUIApp: Main TUI application class
- TUIState: Application state (IDLE/RUNNING)
- TUIPhase: Execution phase (for state machine)
- TUIStateMachine: State management
- CommandRegistry: Slash command handling
"""

from __future__ import annotations

from yaacli.app.commands import (
    BUILTIN_COMMANDS,
    BUSY_CONTROL_COMMANDS,
    Command,
    CommandContext,
    CommandRegistry,
    create_default_registry,
)
from yaacli.app.state import (
    VALID_TRANSITIONS,
    TUIPhase,
    TUIStateMachine,
)
from yaacli.app.tui import TUIApp, TUIState

__all__ = [
    "BUILTIN_COMMANDS",
    "BUSY_CONTROL_COMMANDS",
    "VALID_TRANSITIONS",
    "Command",
    "CommandContext",
    "CommandRegistry",
    "TUIApp",
    "TUIPhase",
    "TUIState",
    "TUIStateMachine",
    "create_default_registry",
]
