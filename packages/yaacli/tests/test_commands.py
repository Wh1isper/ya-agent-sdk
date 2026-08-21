"""Tests for yaacli.app.commands module."""

from __future__ import annotations

from typing import Any

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from ya_agent_sdk.context import AvailableSkill
from yaacli.app import (
    BUILTIN_COMMANDS,
    BUSY_CONTROL_COMMANDS,
    Command,
    CommandRegistry,
    create_default_registry,
)
from yaacli.app.commands import (
    SlashCommandCompleter,
    format_skill_invocation,
    parse_skill_invocation,
)

# =============================================================================
# Mock Context
# =============================================================================


class MockCommandContext:
    """Mock command context for testing."""

    def __init__(self) -> None:
        self.outputs: list[str] = []
        self.system_outputs: list[str] = []
        self.config: dict[str, Any] = {}

    def output(self, text: str) -> None:
        self.outputs.append(text)

    def output_system(self, text: str) -> None:
        self.system_outputs.append(text)

    def get_config(self) -> dict[str, Any]:
        return self.config


# =============================================================================
# Command Tests
# =============================================================================


def test_command_creation():
    """Test Command dataclass creation."""

    def handler(ctx, args):
        pass

    cmd = Command(
        name="test",
        handler=handler,
        description="Test command",
        aliases=["t", "tst"],
    )

    assert cmd.name == "test"
    assert cmd.description == "Test command"
    assert cmd.aliases == ["t", "tst"]


def test_command_default_values():
    """Test Command default values."""

    def handler(ctx, args):
        pass

    cmd = Command(name="test", handler=handler)

    assert cmd.description == ""
    assert cmd.aliases == []


# =============================================================================
# CommandRegistry Tests
# =============================================================================


def test_registry_init():
    """Test CommandRegistry initialization."""
    registry = CommandRegistry()
    assert registry.list_commands() == []


def test_registry_register():
    """Test registering a command."""
    registry = CommandRegistry()

    def handler(ctx, args):
        pass

    registry.register("test", handler, "Test command")

    assert registry.has("test")
    cmd = registry.get("test")
    assert cmd is not None
    assert cmd.name == "test"
    assert cmd.description == "Test command"


def test_registry_register_with_aliases():
    """Test registering a command with aliases."""
    registry = CommandRegistry()

    def handler(ctx, args):
        pass

    registry.register("help", handler, "Show help", aliases=["h", "?"])

    assert registry.has("help")
    assert registry.has("h")
    assert registry.has("?")

    # All should return the same command
    assert registry.get("help") == registry.get("h")
    assert registry.get("help") == registry.get("?")


def test_registry_get_not_found():
    """Test getting non-existent command."""
    registry = CommandRegistry()

    assert registry.get("nonexistent") is None
    assert not registry.has("nonexistent")


def test_registry_list_commands():
    """Test listing all commands."""
    registry = CommandRegistry()

    def handler1(ctx, args):
        pass

    def handler2(ctx, args):
        pass

    registry.register("cmd1", handler1)
    registry.register("cmd2", handler2)

    commands = registry.list_commands()
    assert len(commands) == 2
    names = {c.name for c in commands}
    assert names == {"cmd1", "cmd2"}


@pytest.mark.asyncio
async def test_registry_execute_sync():
    """Test executing a sync command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(args)
        context.output(f"Got: {args}")

    registry.register("echo", handler)

    result = await registry.execute("/echo hello world", ctx)

    assert result is True
    assert executed == ["hello world"]
    assert ctx.outputs == ["Got: hello world"]


@pytest.mark.asyncio
async def test_registry_execute_async():
    """Test executing an async command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    async def handler(context, args):
        executed.append(args)
        context.output(f"Async: {args}")

    registry.register("async_cmd", handler)

    result = await registry.execute("/async_cmd test", ctx)

    assert result is True
    assert executed == ["test"]
    assert ctx.outputs == ["Async: test"]


@pytest.mark.asyncio
async def test_registry_execute_not_found():
    """Test executing non-existent command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()

    result = await registry.execute("/nonexistent", ctx)

    assert result is False


@pytest.mark.asyncio
async def test_registry_execute_no_args():
    """Test executing command without args."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    received_args = []

    def handler(context, args):
        received_args.append(args)

    registry.register("noargs", handler)

    result = await registry.execute("/noargs", ctx)

    assert result is True
    assert received_args == [""]


@pytest.mark.asyncio
async def test_registry_execute_case_insensitive():
    """Test command name is case insensitive."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(True)

    registry.register("test", handler)

    await registry.execute("/TEST", ctx)
    await registry.execute("/Test", ctx)
    await registry.execute("/test", ctx)

    assert len(executed) == 3


@pytest.mark.asyncio
async def test_registry_execute_via_alias():
    """Test executing via alias."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(args)

    registry.register("help", handler, aliases=["h"])

    await registry.execute("/h info", ctx)

    assert executed == ["info"]


# =============================================================================
# BUILTIN_COMMANDS Tests
# =============================================================================


def test_builtin_commands():
    """Test BUILTIN_COMMANDS contains expected commands."""
    expected = {
        "help",
        "clear",
        "new",
        "cancel",
        "cost",
        "model",
        "agents",
        "process",
        "attachments",
        "paste-image",
        "remove-image",
        "tool",
        "dump",
        "load",
        "session",
        "exit",
        "goal",
    }
    assert expected == BUILTIN_COMMANDS
    assert "act" not in BUILTIN_COMMANDS
    assert "background" not in BUILTIN_COMMANDS
    assert "plan" not in BUILTIN_COMMANDS
    assert "tasks" not in BUILTIN_COMMANDS
    assert "loop" not in BUILTIN_COMMANDS


def test_busy_control_commands_cover_the_concurrent_control_surface() -> None:
    assert {
        "/agents",
        "/attachments",
        "/cancel",
        "/cost",
        "/help",
        "/paste-image",
        "/process",
        "/remove-image",
        "/tool",
    } == BUSY_CONTROL_COMMANDS
    assert {name.removeprefix("/") for name in BUSY_CONTROL_COMMANDS} <= BUILTIN_COMMANDS


def test_builtin_commands_is_frozen():
    """Test BUILTIN_COMMANDS is immutable."""
    assert isinstance(BUILTIN_COMMANDS, frozenset)


# =============================================================================
# create_default_registry Tests
# =============================================================================


def test_create_default_registry():
    """Test create_default_registry returns empty registry."""
    registry = create_default_registry()
    assert isinstance(registry, CommandRegistry)
    # Empty by default (handlers registered by TUIApp)
    assert len(registry.list_commands()) == 0


def test_slash_command_completer_completes_commands() -> None:
    completer = SlashCommandCompleter(
        command_provider=lambda: ["/help", "/session"],
        session_provider=lambda: ["session-123"],
    )

    completions = list(completer.get_completions(Document("/he"), CompleteEvent()))

    assert [item.text for item in completions] == ["/help"]
    assert completions[0].start_position == -3


def test_slash_command_completer_completes_session_ids_contextually() -> None:
    completer = SlashCommandCompleter(
        command_provider=lambda: ["/help", "/session"],
        session_provider=lambda: ["abc123", "abc999", "other"],
    )

    completions = list(completer.get_completions(Document("/session abc"), CompleteEvent()))

    assert [item.text for item in completions] == ["abc123", "abc999"]
    assert all(item.start_position == -3 for item in completions)


def test_parse_skill_invocation_supports_multiple_prefixes() -> None:
    skills = {
        "lark-cli": AvailableSkill(name="lark-cli", description="Lark", path="/skills/lark-cli"),
        "agent-builder": AvailableSkill(
            name="agent-builder",
            description="Agents",
            path="/skills/agent-builder",
        ),
    }

    invocation = parse_skill_invocation(
        "/lark-cli /agent-builder Build an agent",
        skills,
        command_names=["/help"],
    )

    assert invocation is not None
    assert invocation.names == ("lark-cli", "agent-builder")
    assert invocation.prompt == "Build an agent"
    formatted = format_skill_invocation(invocation, skills)
    assert '<skill name="lark-cli" path="/skills/lark-cli" />' in formatted
    assert formatted.endswith("Build an agent")


def test_parse_skill_invocation_preserves_command_precedence() -> None:
    skills = {"help": AvailableSkill(name="help", description="Help", path="/skills/help")}

    assert parse_skill_invocation("/help me", skills, command_names=["/help"]) is None


def test_format_skill_invocation_escapes_catalog_values() -> None:
    skills = {
        'unsafe"name': AvailableSkill(
            name='unsafe"name',
            description="Unsafe",
            path="/skills/a&b",
        )
    }
    invocation = parse_skill_invocation('/unsafe"name task', skills)

    assert invocation is not None
    formatted = format_skill_invocation(invocation, skills)
    assert 'name="unsafe&quot;name"' in formatted
    assert 'path="/skills/a&amp;b"' in formatted


def test_slash_command_completer_completes_multiple_skills() -> None:
    completer = SlashCommandCompleter(
        command_provider=lambda: ["/help"],
        session_provider=lambda: [],
        skill_provider=lambda: ["/agent-builder", "/lark-cli"],
    )

    first = list(completer.get_completions(Document("/la"), CompleteEvent()))
    second = list(completer.get_completions(Document("/lark-cli /ag"), CompleteEvent()))

    assert [(item.text, item.display_meta_text) for item in first] == [("/lark-cli", "skill")]
    assert [(item.text, item.display_meta_text) for item in second] == [("/agent-builder", "skill")]
