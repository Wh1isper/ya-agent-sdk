"""Note tools for persistent key-value storage.

These tools allow the agent to store, update, delete, and read note entries
that persist across conversation turns. Runtime instructions expose note keys;
note values are read on demand through note_get.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import NoteEvent
from ya_agent_sdk.toolsets.base import Instruction
from ya_agent_sdk.toolsets.core.base import BaseTool

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _build_note_event(ctx: RunContext[AgentContext]) -> NoteEvent:
    """Build a NoteEvent with full snapshot of all entries."""
    return NoteEvent(
        event_id=f"note-{ctx.deps.run_id[:8]}",
        entries=dict(ctx.deps.note_manager.entries),
    )


class NoteTool(BaseTool):
    """Tool for storing, updating, or deleting note entries."""

    name = "note"
    description = (
        "Update or delete a note entry. "
        "Notes persist across turns. Runtime context shows note keys; use note_get to read values. "
        "Omit value to delete the entry."
    )

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool."""
        instruction_file = _PROMPTS_DIR / "note.md"
        if instruction_file.exists():
            return Instruction(group="note", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        key: Annotated[str, Field(description="Unique key for the note entry.")],
        value: Annotated[
            str | None,
            Field(description="Content to store. Omit or set to null to delete the entry."),
        ] = None,
    ) -> str:
        if value is None:
            if ctx.deps.note_manager.delete(key):
                await ctx.deps.emit_event(_build_note_event(ctx))
                return f"Note entry '{key}' deleted."
            return f"Note entry '{key}' not found."

        ctx.deps.note_manager.set(key, value)
        await ctx.deps.emit_event(_build_note_event(ctx))
        return f"Note entry '{key}' stored."


class NoteGetTool(BaseTool):
    """Tool for reading note entries."""

    name = "note_get"
    description = "Read note entries by key. Omit key to list all notes with values."

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool."""
        instruction_file = _PROMPTS_DIR / "note.md"
        if instruction_file.exists():
            return Instruction(group="note", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        key: Annotated[str | None, Field(description="The note key to retrieve. Omit to list all notes.")] = None,
    ) -> str:
        if key is not None:
            value = ctx.deps.note_manager.get(key)
            if value is None:
                return f"Note entry '{key}' not found."
            return f"{key}: {value}"

        entries = ctx.deps.note_manager.list_all()
        if not entries:
            return "No note entries found."

        return "\n".join(f"{entry_key}: {entry_value}" for entry_key, entry_value in entries)
