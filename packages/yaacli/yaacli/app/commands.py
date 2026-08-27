"""Command handling for TUI slash commands.

Provides a registry-based command system for slash commands.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from ya_agent_sdk.context import AvailableSkill

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


@dataclass(frozen=True)
class SkillInvocation:
    """Explicit skill prefixes parsed from one user prompt."""

    names: tuple[str, ...]
    prompt: str


def parse_skill_invocation(
    text: str,
    available_skills: Mapping[str, AvailableSkill],
    *,
    command_names: Iterable[str] = (),
) -> SkillInvocation | None:
    """Parse consecutive ``/skill-name`` prefixes from the start of input.

    Names supplied through ``command_names`` are reserved control commands and
    take precedence when the first token is also a skill. Unknown slash tokens
    are left in the remaining prompt.
    """
    commands = {name.removeprefix("/").lower() for name in command_names}
    names: list[str] = []
    cursor = 0

    for match in re.finditer(r"\S+", text):
        if match.start() < cursor:
            continue
        token = match.group(0)
        if not token.startswith("/") or len(token) == 1:
            break
        name = token[1:]
        if not names and name.lower() in commands:
            return None
        if name not in available_skills:
            break
        names.append(name)
        cursor = match.end()

    if not names:
        return None
    return SkillInvocation(names=tuple(names), prompt=text[cursor:].lstrip())


def format_skill_invocation(
    invocation: SkillInvocation,
    available_skills: Mapping[str, AvailableSkill],
) -> str:
    """Render selected skills as a direct user prompt."""
    lines = [f"Use this skill: {available_skills[name].name}" for name in invocation.names]
    if invocation.prompt:
        lines.extend(["", f"User instructions: {invocation.prompt}"])
    return "\n".join(lines)


class SlashCommandCompleter(Completer):
    """Complete slash commands and saved session IDs without taking over Tab."""

    def __init__(
        self,
        command_provider: Callable[[], Iterable[str]],
        session_provider: Callable[[], Iterable[str]],
        skill_provider: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        self._command_provider = command_provider
        self._session_provider = session_provider
        self._skill_provider = skill_provider or (lambda: ())

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

        if not text.startswith("/"):
            return

        commands = list(self._command_provider())
        skills = list(self._skill_provider())
        if " " not in text:
            seen: set[str] = set()
            for candidate, kind in [
                *((skill, "skill") for skill in skills),
                *((command, "command") for command in commands),
            ]:
                if candidate in seen or not candidate.startswith(text):
                    continue
                seen.add(candidate)
                yield Completion(candidate, start_position=-len(text), display_meta=kind)
            return

        prefix, separator, fragment = text.rpartition(" ")
        if not separator or not fragment.startswith("/"):
            return
        selected = prefix.split()
        skill_set = set(skills)
        if not selected or any(token not in skill_set for token in selected):
            return
        for skill in skills:
            if skill not in selected and skill.startswith(fragment):
                yield Completion(skill, start_position=-len(fragment), display_meta="skill")


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
    "cost": "Show token usage and cost summary",
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
    "/paste-image",
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
