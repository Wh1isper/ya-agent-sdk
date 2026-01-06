"""Thinking tool for agent reasoning.

This tool allows the agent to think about something without obtaining new information
or making changes. Useful for complex reasoning or caching memory.
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load thinking instruction from prompts/thinking.md."""
    prompt_file = _PROMPTS_DIR / "thinking.md"
    return prompt_file.read_text()


class ThinkingTool(BaseTool):
    """Tool for agent to think and reason."""

    name = "thinking"
    description = "Think about something without obtaining new information or making changes."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/thinking.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        thought: Annotated[
            str,
            Field(description="A thought in markdown format."),
        ],
    ) -> dict[str, Any]:
        return {"thought": thought}
