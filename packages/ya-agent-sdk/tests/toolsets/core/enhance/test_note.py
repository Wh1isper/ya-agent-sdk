"""Tests for note tools."""

from unittest.mock import MagicMock

from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.enhance.note import NoteGetTool, NoteTool


async def test_note_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert NoteTool.name == "note"
    assert "bounded note-key index" in NoteTool.description
    assert "use note_get to discover omitted notes or read values" in NoteTool.description

    tool = NoteTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = await tool.get_instruction(mock_run_ctx)
    assert instruction is not None
    assert "<note-guidelines>" in instruction.content
    assert "note_get" in instruction.content


async def test_note_get_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert NoteGetTool.name == "note_get"
    assert NoteGetTool.description == "Read note entries by key. Omit key to list all notes with values."

    tool = NoteGetTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = await tool.get_instruction(mock_run_ctx)
    assert instruction is not None
    assert "<note-guidelines>" in instruction.content


async def test_note_tool_set_update_delete(agent_context: AgentContext) -> None:
    """Should set, update, and delete note entries."""
    tool = NoteTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    assert await tool.call(mock_run_ctx, key="project", value="ya-mono") == "Note entry 'project' stored."
    assert agent_context.note_manager.get("project") == "ya-mono"

    assert await tool.call(mock_run_ctx, key="project", value="ya-agent-sdk") == "Note entry 'project' stored."
    assert agent_context.note_manager.get("project") == "ya-agent-sdk"

    assert await tool.call(mock_run_ctx, key="project") == "Note entry 'project' deleted."
    assert agent_context.note_manager.get("project") is None

    assert await tool.call(mock_run_ctx, key="project") == "Note entry 'project' not found."


async def test_note_get_tool_reads_by_key(agent_context: AgentContext) -> None:
    """Should read a note by key."""
    agent_context.note_manager.set("language", "Chinese")
    tool = NoteGetTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    assert await tool.call(mock_run_ctx, key="language") == "language: Chinese"
    assert await tool.call(mock_run_ctx, key="missing") == "Note entry 'missing' not found."


async def test_note_get_tool_lists_all(agent_context: AgentContext) -> None:
    """Should list all note values when key is omitted."""
    agent_context.note_manager.set("b", "second")
    agent_context.note_manager.set("a", "first")
    tool = NoteGetTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    assert await tool.call(mock_run_ctx) == "a: first\nb: second"


async def test_note_get_tool_empty(agent_context: AgentContext) -> None:
    """Should report empty note store."""
    tool = NoteGetTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    assert await tool.call(mock_run_ctx) == "No note entries found."
