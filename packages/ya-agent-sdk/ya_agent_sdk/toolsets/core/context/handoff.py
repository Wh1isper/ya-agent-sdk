"""Handoff tool for context management.

This tool allows the agent to summarize current work and clear context
to start fresh while preserving essential information.

Note:
    This tool must be used together with `ya_agent_sdk.filters.handoff.process_handoff_message`.
    The tool stores the handoff summary in context, and the history processor injects it
    into the message history on the next model request.

Example::

    from ya_agent_sdk.context import AgentContext
    from ya_agent_sdk.toolsets.core.base import Toolset
    from ya_agent_sdk.toolsets.core.context.handoff import HandoffTool
    from ya_agent_sdk.filters.handoff import process_handoff_message

    async with AgentContext() as ctx:
        toolset = Toolset(tools=[HandoffTool])
        # HandoffTool.is_context_manage_tool = True, so create_agent()
        # will automatically set ctx.context_manage_tool_name = 'summarize'
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            toolsets=[toolset],
            history_processors=[process_handoff_message],  # Required for handoff to work
        )
        result = await agent.run('prompt', deps=ctx)
"""

from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_FILE_INSPECTION_PATHS_DESCRIPTION = """File paths the resumed agent may need to inspect after summary.
Only the paths are added to a reminder; file contents are not loaded into context.
Use for: key config files, source code being edited, important references.
"""


@cache
def _load_instruction() -> str:
    """Load handoff instruction from prompts/handoff.md."""
    prompt_file = _PROMPTS_DIR / "handoff.md"
    return prompt_file.read_text()


class HandoffMessage(BaseModel):
    """Structured summary for context handoff between conversation sessions.

    Note: The class name 'HandoffMessage' is an internal implementation detail.
    The user-facing tool name is 'summarize'.
    """

    content: str = Field(
        ...,
        description="""Context summary to preserve across context reset. Should include:
1. User's primary request and intent
2. Current state and completed work
3. Key technical decisions
4. Pending tasks
5. Next step (if any)
6. Past Interactions: A concise bullet list of key interactions (both sides) that already occurred, to prevent repetition. Include your actions/proposals and user's responses, approaches tried and outcomes, explanations already given.
""",
    )

    auto_load_files: list[str] = Field(
        default_factory=list,
        description=_FILE_INSPECTION_PATHS_DESCRIPTION,
    )

    def render(self) -> str:
        """Render handoff message as Markdown for context injection."""
        return f"# Context Summary\n\n{self.content}"


class HandoffTool(BaseTool):
    """Tool for context handoff between sessions."""

    name = "summarize"
    description = """Summarize current work and clear context to start fresh.

Use this tool when context is getting large and you need to preserve essential information
before resetting, or when switching focus to a different topic or task.
The summary will be injected into the new context automatically.
"""
    auto_inherit = True
    is_context_manage_tool = True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/handoff.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        content: Annotated[
            str,
            Field(
                description="""Context summary to preserve across context reset. Should include:
1. User's primary request and intent
2. Current state and completed work
3. Key technical decisions
4. Pending tasks
5. Next step (if any)
6. Past Interactions: A concise bullet list of key interactions (both sides) that already occurred, to prevent repetition. Include your actions/proposals and user's responses, approaches tried and outcomes, explanations already given.
"""
            ),
        ],
        auto_load_files: Annotated[
            list[str] | None,
            Field(description=_FILE_INSPECTION_PATHS_DESCRIPTION),
        ] = None,
    ) -> str:
        message = HandoffMessage(content=content, auto_load_files=auto_load_files or [])
        # Store rendered message for history processor to pick up
        rendered = message.render()
        ctx.deps.handoff_message = rendered
        # Preserve the compatibility field for the prompt-only inspection reminder.
        # Use extend instead of assignment to keep paths set by external callers.
        ctx.deps.auto_load_files.extend(message.auto_load_files)
        return f"Summary complete. Context refreshed.\n\n{rendered}"
