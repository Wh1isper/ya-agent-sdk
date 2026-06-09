"""Helpers for asserting Pydantic AI toolset instruction results."""

from collections.abc import Sequence

from pydantic_ai.messages import InstructionPart


def instruction_text(instructions: str | InstructionPart | Sequence[str | InstructionPart] | None) -> str:
    """Return text content from a Pydantic AI get_instructions result."""
    if instructions is None:
        return ""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, InstructionPart):
        return instructions.content
    return "\n".join(part if isinstance(part, str) else part.content for part in instructions)
