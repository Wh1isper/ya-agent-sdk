"""Base classes for toolsets and tools.

This module provides the foundational abstractions for building toolsets:
- BaseTool: Abstract base class for individual tools
- BaseToolset: Abstract base class for toolsets with instruction support
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import InstructionPart
from pydantic_ai.toolsets import AbstractToolset
from typing_extensions import TypeVar

from ya_agent_sdk.context import AgentContext

if TYPE_CHECKING:
    pass

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)
ToolCallDynamic: TypeAlias = Any


class UserInputPreprocessResult(BaseModel):
    """Result from processing user input in HITL scenarios."""

    override_args: dict[str, Any] | None = None
    """Override arguments for the tool call."""

    metadata: dict[str, Any] | None = None
    """Additional metadata from user input processing."""


class Instruction(BaseModel):
    """Tool instruction with optional group for deduplication.

    When multiple tools return Instructions with the same group,
    only the first one is included in the system prompt.

    Example:
        # Multiple task tools sharing one instruction
        class TaskCreateTool(BaseTool):
            async def get_instruction(self, ctx):
                return Instruction(
                    group="task-manager",
                    content="Task manager guidelines..."
                )
    """

    group: str
    """Group identifier for deduplication. Same group = only first instruction kept."""

    content: str
    """The instruction content to inject into system prompt."""


@runtime_checkable
class InstructableToolset(Protocol):
    """Protocol for toolsets that provide instructions.

    This enables duck typing for any toolset that has a get_instructions method.
    """

    async def get_instructions(
        self, ctx: RunContext[Any]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        """Get instructions to inject into the system prompt."""
        ...


def collect_instruction_parts(
    instructions: str | InstructionPart | Sequence[str | InstructionPart],
    parts: list[str],
) -> None:
    """Normalize instruction results into a flat list of strings.

    Handles the various return types of ``get_instructions`` (plain ``str``,
    ``InstructionPart``, or a sequence of either) and appends string content
    to *parts* in place.
    """
    if isinstance(instructions, str):
        parts.append(instructions)
    elif isinstance(instructions, InstructionPart):
        parts.append(instructions.content)
    else:
        for item in instructions:
            if isinstance(item, InstructionPart):
                parts.append(item.content)
            else:
                parts.append(item)


class BaseTool(ABC):
    """Abstract base class for tools.

    Subclasses define name, description as class attributes, implement
    the `call` method, and optionally override `get_instruction()` for
    dynamic instruction generation.

    Example:
        class ReadFileTool(BaseTool):
            name = "read_file"
            description = "Read contents of a file"

            async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
                return "Use this tool to read file contents from the filesystem."

            async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
                return Path(path).read_text()
    """

    name: str
    """The name of the tool, used for invocation."""

    description: str
    """Description of what the tool does, shown to the model."""

    tags: frozenset[str] = frozenset()
    """Capability tags this tool provides.

    Tags represent capabilities that this tool makes available.
    Other tools can declare themselves superseded by these tags.
    The active tags are collected by Toolset and exposed via AgentContext.tool_tags.

    Example::

        class ShellTool(BaseTool):
            tags = frozenset({"shell"})
    """

    superseded_by_tags: frozenset[str] = frozenset()
    """Tags that make this tool redundant.

    If any of these tags are active (provided by an available tool),
    this tool will be automatically hidden from the agent.
    This allows tools to gracefully yield to more capable alternatives.

    Example::

        class MkdirTool(BaseTool):
            superseded_by_tags = frozenset({"shell"})  # shell can do mkdir better
    """

    auto_inherit: bool = False
    """Whether this tool is automatically inherited by subagents.

    When True, this tool will be automatically included in subagent toolsets
    without being explicitly listed in the subagent's tools or optional_tools.
    Useful for management/utility tools like task_*, summarize, etc.
    """

    is_context_manage_tool: bool = False
    """Whether this tool is a context management tool (e.g., summarize/handoff).

    When True, create_agent() will automatically detect this tool and set
    ctx.context_manage_tool_name, enabling proactive context management
    reminders when the context window usage exceeds the configured threshold.

    Only one context management tool should be active at a time.
    """

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available in current context.

        Override this method to check runtime conditions like model capabilities,
        optional dependencies, or configuration settings.
        Tools that return False will be excluded when skip_unavailable=True.

        Args:
            ctx: The run context containing runtime information.

        Returns:
            True if tool can be used, False otherwise.
        """
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | Instruction | None:
        """Get instruction for this tool.

        Override this method to provide dynamic instructions based on context.
        Default implementation returns None (no instruction).

        Returns:
            - str: Plain instruction text (uses tool name as group)
            - Instruction: Instruction with explicit group for deduplication
            - None: No instruction for this tool

        Example:
            # Plain string (no deduplication)
            async def get_instruction(self, ctx):
                return "Use this tool to read files."

            # Instruction with group (deduplicated with same group)
            async def get_instruction(self, ctx):
                return Instruction(
                    group="file-tools",
                    content="File operation guidelines..."
                )
        """
        return None

    def get_approval_metadata(self) -> dict[str, Any] | None:
        return None

    @abstractmethod
    async def call(
        self,
        ctx: RunContext[AgentContext],
        /,
        *args: ToolCallDynamic,
        **kwargs: ToolCallDynamic,
    ) -> ToolCallDynamic:
        """Execute the tool logic.

        Subclasses should override this method with their specific parameter signature.
        The base signature uses *args/**kwargs to allow any parameter combination.

        Args:
            ctx: The run context containing runtime information.
            *args: Tool-specific positional arguments.
            **kwargs: Tool-specific keyword arguments.

        Returns:
            The tool's result.
        """
        ...

    async def process_user_input(
        self,
        ctx: AgentContext,
        user_input: object,
    ) -> UserInputPreprocessResult | None:
        """Process user input for HITL scenarios.

        Override this method to handle user-provided input when a tool call
        requires approval. You can use this to:
        - Validate user input
        - Transform user input into tool arguments
        - Store metadata for later use

        Args:
            ctx: The agent context.
            user_input: The user-provided input data.

        Returns:
            A UserInputPreprocessResult with override_args and/or metadata,
            or None if no processing is needed.
        """
        return None

    def get_deferred_metadata(self, ctx: RunContext[AgentContext]) -> object:
        """Get HITL metadata for the tool call, if applicable."""
        return ctx.tool_call_metadata


class BaseToolset(AbstractToolset[AgentDepsT], ABC):
    """Base class for toolsets with instruction support.

    Subclasses can override get_instructions() to provide custom instructions.

    Example:
        class MyToolset(BaseToolset):
            async def get_instructions(self, ctx: RunContext[AgentContext]) -> str | list[str] | None:
                content = await self._load_instructions(ctx)
                return content
    """

    @property
    def description(self) -> str | None:
        """Human-readable description of this toolset.

        Used by ToolSearchToolSet for namespace description resolution.
        Override in subclasses to provide a description.

        Returns:
            Description string or None.
        """
        return None

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | list[str] | None:
        """Get instructions to inject into the system prompt.

        Override this method to provide tool-specific instructions.

        Args:
            ctx: The run context containing runtime information.

        Returns:
            Instruction string, list of strings, or None.
        """
        return None
