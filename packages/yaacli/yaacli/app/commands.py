"""Command handling for TUI slash commands.

Provides a registry-based command system for slash commands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    pass


class CommandContext(Protocol):
    """Protocol for command execution context."""

    def output(self, text: str) -> None:
        """Output text to display."""
        ...

    def output_system(self, text: str) -> None:
        """Output system message."""
        ...

    def get_config(self) -> object:
        """Get configuration."""
        ...


@dataclass
class Command:
    """A registered slash command."""

    name: str
    handler: Callable[[CommandContext, str], Awaitable[None] | None]
    description: str = ""
    aliases: list[str] = field(default_factory=list)


class SlashCommandCompleter(Completer):
    """Complete slash commands and saved session IDs without taking over Tab."""

    def __init__(
        self,
        command_provider: Callable[[], Iterable[str]],
        session_provider: Callable[[], Iterable[str]],
    ) -> None:
        self._command_provider = command_provider
        self._session_provider = session_provider

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/session "):
            fragment = text.removeprefix("/session ")
            if " " in fragment:
                return
            for session_id in self._session_provider():
                if session_id.startswith(fragment):
                    yield Completion(
                        session_id,
                        start_position=-len(fragment),
                        display_meta="saved session",
                    )
            return

        if not text.startswith("/") or " " in text:
            return
        for command in self._command_provider():
            if command.startswith(text):
                yield Completion(command, start_position=-len(text), display_meta="command")


class CommandRegistry:
    """Registry for slash commands.

    Supports:
    - Built-in commands (cannot be overridden)
    - Custom commands from config
    - Command aliases
    """

    def __init__(self) -> None:
        """Initialize command registry."""
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[CommandContext, str], Awaitable[None] | None],
        description: str = "",
        aliases: list[str] | None = None,
    ) -> None:
        """Register a command.

        Args:
            name: Command name (without leading /).
            handler: Async or sync function taking (ctx, args).
            description: Help text for the command.
            aliases: Alternative names for the command.
        """
        cmd = Command(
            name=name,
            handler=handler,
            description=description,
            aliases=aliases or [],
        )
        self._commands[name] = cmd

        # Register aliases
        for alias in cmd.aliases:
            self._aliases[alias] = name

    def get(self, name: str) -> Command | None:
        """Get a command by name or alias.

        Args:
            name: Command name (without leading /).

        Returns:
            Command if found, None otherwise.
        """
        # Check direct name
        if name in self._commands:
            return self._commands[name]

        # Check aliases
        if name in self._aliases:
            return self._commands.get(self._aliases[name])

        return None

    def has(self, name: str) -> bool:
        """Check if command exists."""
        return self.get(name) is not None

    def list_commands(self) -> list[Command]:
        """Get all registered commands."""
        return list(self._commands.values())

    async def execute(self, command_str: str, ctx: CommandContext) -> bool:
        """Execute a command.

        Args:
            command_str: Full command string (e.g., "/help" or "/dump folder").
            ctx: Command context.

        Returns:
            True if command was found and executed, False otherwise.
        """
        # Parse command
        parts = command_str.lstrip("/").split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Find command
        cmd = self.get(name)
        if not cmd:
            return False

        # Execute
        result = cmd.handler(ctx, args)
        if result is not None:
            await result

        return True


# Single source of truth for built-in command discovery and help.
BUILTIN_COMMAND_HELP: dict[str, str] = {
    "help": "Show available commands",
    "clear": "Clear the visible transcript only",
    "new": "Start a new conversation and session",
    "cancel": "Cancel the foreground agent or shell command",
    "integrate": "Integrate ready background results into the active or next agent turn",
    "cost": "Show token usage and cost summary",
    "perf": "Show performance stats when enabled",
    "model": "Select a model profile",
    "agents": "Show background subagents",
    "process": "Show background shell processes",
    "attachments": "List images queued for the next prompt",
    "paste-image": "Attach an image from the clipboard",
    "remove-image": "Remove a queued image by number, or all",
    "tool": "Show the complete stored result for a tool call ID",
    "session": "List sessions or restore a session ID",
    "dump": "Export the current session to a folder",
    "load": "Load a session export from a folder",
    "goal": "Run toward a task until verified complete",
    "exit": "Exit YAACLI",
}
BUILTIN_COMMANDS = frozenset(BUILTIN_COMMAND_HELP)

# Commands that are safe and useful while another foreground activity owns the
# TUI. Every other recognized command remains on the control plane but waits
# for idle instead of being reinterpreted as agent steering.
BUSY_CONTROL_COMMANDS = frozenset({
    "/agents",
    "/attachments",
    "/cancel",
    "/cost",
    "/help",
    "/integrate",
    "/paste-image",
    "/perf",
    "/process",
    "/remove-image",
    "/tool",
})


def create_default_registry() -> CommandRegistry:
    """Create a registry with placeholder for built-in commands.

    Note: Actual handlers are registered by TUIApp since they need
    access to app state.
    """
    return CommandRegistry()
