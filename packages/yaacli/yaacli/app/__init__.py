"""Public YAACLI TUI application exports, loaded lazily for fast startup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yaacli.app.commands import (
        BUILTIN_COMMANDS,
        BUSY_CONTROL_COMMANDS,
        Command,
        CommandContext,
        CommandRegistry,
        create_default_registry,
    )
    from yaacli.app.state import VALID_TRANSITIONS, TUIPhase, TUIStateMachine
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

_COMMAND_EXPORTS = frozenset({
    "BUILTIN_COMMANDS",
    "BUSY_CONTROL_COMMANDS",
    "Command",
    "CommandContext",
    "CommandRegistry",
    "create_default_registry",
})
_STATE_EXPORTS = frozenset({"VALID_TRANSITIONS", "TUIPhase", "TUIStateMachine"})
_TUI_EXPORTS = frozenset({"TUIApp", "TUIState"})


def __getattr__(name: str) -> Any:
    """Resolve and cache public exports without loading the runtime eagerly."""
    if name in _COMMAND_EXPORTS:
        from yaacli.app import commands

        value = getattr(commands, name)
    elif name in _STATE_EXPORTS:
        from yaacli.app import state

        value = getattr(state, name)
    elif name in _TUI_EXPORTS:
        from yaacli.app import tui

        value = getattr(tui, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
