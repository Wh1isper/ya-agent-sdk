"""Core BaseTool adapter with HITL support.

Base classes (BaseTool, BaseToolset) are in ya_agent_sdk.toolsets.base. Cross-cutting
execution policy belongs in native Pydantic AI capabilities rather than this adapter.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import ApprovalRequired, CallDeferred, ModelRetry, RunContext, Tool, UserError
from pydantic_ai.messages import InstructionPart, ModelMessage
from pydantic_ai.tools import (
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.toolsets.abstract import ToolsetTool
from typing_extensions import TypeVar

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.base import (
    ACTIVE_TOOL_SUPERSESSION_TAGS,
    TOOL_SUPERSEDED_BY_TAGS_METADATA_KEY,
    TOOL_TAGS_METADATA_KEY,
    BaseTool,
    BaseToolset,
    Instruction,
)
from ya_agent_sdk.utils import get_tool_name_from_id

logger = get_logger(__name__)

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)


class UserInteraction(BaseModel):
    """Represents a user's interaction with a deferred tool call."""

    tool_call_id: str = Field(..., description="The ID of the tool interaction.")
    approved: bool = Field(
        ...,
        description="Whether the user approved the previous action. "
        "If false, the 'reason' field may provide additional context. "
        "If true, 'user_input' may contain additional data provided by the user.",
    )
    reason: str | None = Field(None, description="The reason for rejection, if any.")
    user_input: object = Field(
        None,
        description="Additional user input data. Structure depends on tool implementation.",
    )


@dataclass(kw_only=True)
class _BaseToolsetTool(ToolsetTool[AgentDepsT]):
    """Internal Pydantic AI tool wrapper for one ``BaseTool``."""

    call_func: Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]
    """The function to call when the tool is invoked."""

    is_async: bool
    """Whether the underlying function is async."""

    timeout: float | None = None
    """Timeout in seconds for tool execution."""

    tool_instance: BaseTool | None = None
    """Reference to the BaseTool instance for HITL processing."""


class Toolset(BaseToolset[AgentDepsT]):
    """Adapt ``BaseTool`` classes to a native Pydantic AI toolset."""

    def __init__(
        self,
        tools: Sequence[type[BaseTool]],
        *,
        max_retries: int | None = None,
        timeout: float | None = None,
        toolset_id: str | None = None,
        skip_unavailable: bool = True,
        description: str | None = None,
    ) -> None:
        """Initialize the toolset.

        Args:
            tools: Sequence of BaseTool classes to include in this toolset.
            max_retries: Explicit maximum retries for tool execution. When omitted,
                uses ``ctx.deps.retry_config.toolset`` (default 5).
            timeout: Default timeout for tool execution.
            toolset_id: Optional unique ID for the toolset.
            skip_unavailable: If True, skip tools where is_available() returns False in get_tools().
            description: Optional human-readable description for the toolset.
        """
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries must be non-negative or None")
        self._max_retries = max_retries
        self.timeout = timeout
        self._id = toolset_id
        self._skip_unavailable = skip_unavailable
        self._description = description

        # Store tool classes, instances created lazily in get_tools
        self._tool_classes: dict[str, type[BaseTool]] = {}
        self._tool_instances: dict[str, BaseTool] = {}

        logger.debug(f"Initializing Toolset with {len(tools)} tool classes")
        for tool_cls in tools:
            name = tool_cls.name
            if name in self._tool_classes:
                msg = f"Duplicate tool name: {name!r}"
                raise UserError(msg)
            self._tool_classes[name] = tool_cls
            logger.debug(f"Registered tool class: {name!r}")

        self._pydantic_tools: dict[str, Tool[AgentDepsT]] = {}
        logger.debug(f"Toolset initialized with tools: {list(self._tool_classes.keys())}")

    @property
    def max_retries(self) -> int:
        """Return the explicit retry limit or the SDK default for inspection."""
        return self._max_retries if self._max_retries is not None else 5

    @max_retries.setter
    def max_retries(self, value: int | None) -> None:
        """Set or clear the local retry override."""
        if value is not None and value < 0:
            raise ValueError("max_retries must be non-negative or None")
        self._max_retries = value

    def _resolve_max_retries(self, ctx: RunContext[AgentDepsT]) -> int:
        """Resolve a local override or the SDK-wide retry policy."""
        if self._max_retries is not None:
            return self._max_retries
        return ctx.deps.retry_config.toolset

    @property
    def id(self) -> str | None:
        """Get the toolset ID."""
        return self._id

    @property
    def description(self) -> str | None:
        """Get the toolset description."""
        return self._description

    @property
    def tool_names(self) -> list[str]:
        """Get list of tool names in this toolset."""
        return list(self._tool_classes.keys())

    def _get_tool_instance(self, name: str) -> BaseTool:
        """Get or create a tool instance by name."""
        if name not in self._tool_instances:
            if name not in self._tool_classes:
                msg = f"Tool {name!r} not found in toolset"
                raise UserError(msg)
            self._tool_instances[name] = self._tool_classes[name]()
        return self._tool_instances[name]

    @staticmethod
    def _is_tool_allowed_in_context(
        tool_instance: BaseTool,
        ctx: RunContext[AgentDepsT],
    ) -> bool:
        """Enforce policies that cannot be disabled by toolset configuration."""
        if not tool_instance.main_agent_only:
            return True
        return ctx.deps.agent_id == "main" and ctx.deps.parent_run_id is None

    def is_tool_available(
        self,
        tool_name: str,
        ctx: RunContext[AgentDepsT],
    ) -> bool:
        """Check if a tool is available.

        Args:
            tool_name: The name of the tool to check.
            ctx: The run context for checking runtime availability.

        Returns:
            True if the tool exists and is available, False otherwise.
        """
        if tool_name not in self._tool_classes:
            return False
        tool_instance = self._get_tool_instance(tool_name)
        return self._is_tool_allowed_in_context(tool_instance, ctx) and tool_instance.is_available(ctx)

    def _create_pydantic_tool(self, name: str, tool_instance: BaseTool) -> Tool[AgentDepsT]:
        """Create a pydantic_ai Tool wrapper for a BaseTool instance."""

        @functools.wraps(tool_instance.call)
        async def _call(ctx: RunContext[AgentDepsT], **kwargs: object) -> object:
            return await tool_instance.call(ctx, **kwargs)

        return Tool(
            function=_call,
            name=name,
            description=tool_instance.description,
            max_retries=self._max_retries,
            takes_ctx=True,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return all tools in this toolset.

        Uses two-phase filtering:
        1. Check basic availability and collect capability tags
        2. Filter out tools superseded by active tags
        """
        logger.debug(f"get_tools called, preparing {len(self._tool_classes)} tools")

        # Phase 1: determine basic availability and collect tags
        # Use a list to preserve registration order for deterministic tool output.
        available_names: list[str] = []
        collected_tags: set[str] = set()

        for name in self._tool_classes:
            tool_instance = self._get_tool_instance(name)
            if not self._is_tool_allowed_in_context(tool_instance, ctx):
                logger.debug(f"Skipping context-forbidden tool {name!r}")
                continue
            # Check availability at get_tools time (when env is entered)
            if self._skip_unavailable and not tool_instance.is_available(ctx):
                logger.debug(f"Skipping unavailable tool {name!r}")
                continue
            available_names.append(name)
            collected_tags.update(tool_instance.tags)

        # Set collected tags on context (recomputed fresh each call)
        ctx.deps.tool_tags = collected_tags

        # Phase 2: build final tools, filtering superseded ones
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for name in available_names:
            tool_instance = self._get_tool_instance(name)

            # Check if superseded by any active tag from THIS toolset's available tools
            # Use local collected_tags (not ctx.deps.tool_tags) to avoid stale/inherited tags
            if tool_instance.superseded_by_tags and (tool_instance.superseded_by_tags & collected_tags):
                superseding = tool_instance.superseded_by_tags & collected_tags
                logger.debug(f"Skipping tool {name!r}: superseded by tags {superseding}")
                continue

            # Get or create pydantic_ai Tool wrapper
            if name not in self._pydantic_tools:
                self._pydantic_tools[name] = self._create_pydantic_tool(name, tool_instance)

            pydantic_tool = self._pydantic_tools[name]
            tool_def = await pydantic_tool.prepare_tool_def(ctx)
            if not tool_def:
                continue
            tool_def = replace(
                tool_def,
                metadata={
                    **(tool_def.metadata or {}),
                    "codeact": tool_instance.codeact,
                    TOOL_TAGS_METADATA_KEY: tuple(sorted(tool_instance.tags)),
                    TOOL_SUPERSEDED_BY_TAGS_METADATA_KEY: tuple(sorted(tool_instance.superseded_by_tags)),
                },
            )

            tools[name] = _BaseToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self._resolve_max_retries(ctx),
                args_validator=pydantic_tool.function_schema.validator,
                call_func=pydantic_tool.function_schema.call,
                is_async=pydantic_tool.function_schema.is_async,
                timeout=self.timeout,
                tool_instance=tool_instance,
            )

        return tools

    async def _call_tool_func(
        self,
        args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: _BaseToolsetTool[AgentDepsT],
    ) -> object:
        """Execute the tool function and capture exceptions.

        Subclasses can override this method to customize tool execution,
        e.g., adding timeout handling, retry logic, or custom error handling.

        Note: ApprovalRequired, CallDeferred, and ModelRetry are pydantic-ai
        control flow exceptions that must propagate. They are NOT caught here.

        Args:
            args: The validated tool arguments.
            ctx: The run context.
            tool: The tool to execute.

        Returns:
            The tool result, or an Exception if execution failed.
        """
        try:
            return await tool.call_func(args, ctx)
        except (ApprovalRequired, CallDeferred, ModelRetry):
            raise
        except Exception as e:
            return e

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> object:
        """Execute one ``BaseTool`` through the native toolset boundary.

        Cross-cutting visibility, approval, timeout, retry, and observation behavior is
        composed as Pydantic AI capabilities. Pydantic AI control-flow exceptions are
        therefore propagated unchanged from this adapter.
        """
        logger.debug(f"call_tool: {name!r} with args keys: {list(tool_args.keys())}")

        if not isinstance(tool, _BaseToolsetTool):
            msg = f"Expected _BaseToolsetTool, got {type(tool)}"
            raise UserError(msg)

        if tool.tool_instance is not None and not self._is_tool_allowed_in_context(tool.tool_instance, ctx):
            msg = f"Tool {name!r} is not available in this agent context"
            raise UserError(msg)

        if name in ctx.deps.need_user_approve_tools and not ctx.tool_call_approved:
            approval_metadata = tool.tool_instance.get_approval_metadata() if tool.tool_instance else None
            logger.debug(f"call_tool: {name!r} requires user approval")
            raise ApprovalRequired(metadata=approval_metadata)

        result = await self._call_tool_func(tool_args, ctx, tool)
        if isinstance(result, BaseException):
            logger.debug(f"call_tool: {name!r} returned error: {type(result).__name__}")
            return f"Error calling tool {name}: {result}"

        logger.debug(f"call_tool: {name!r} completed successfully")
        return result

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> list[InstructionPart] | None:
        """Collect static instructions from all tools with group-based deduplication.

        When multiple tools return Instructions with the same group,
        only the first one is included. Tools returning plain strings use their
        tool name as the implicit group.

        Uses the same two-phase filtering as get_tools() to ensure only
        available and non-superseded tools contribute instructions.

        Returns Pydantic AI ``InstructionPart`` objects with ``dynamic=False``
        so tool instructions can participate in prompt caching.
        """
        # Phase 1: determine available tools and collect tags (same as get_tools)
        # Use a list to preserve registration order for deterministic deduplication
        available_names: list[str] = []
        collected_tags: set[str] = set()

        for name in self._tool_classes:
            tool_instance = self._get_tool_instance(name)
            if not self._is_tool_allowed_in_context(tool_instance, ctx):
                continue
            if self._skip_unavailable and not tool_instance.is_available(ctx):
                continue
            available_names.append(name)
            collected_tags.update(tool_instance.tags)

        # A composition-level supersession wrapper publishes one run-step-local tag
        # snapshot. Keep local tags as the fallback for standalone Toolset usage.
        active_tags = collected_tags | ACTIVE_TOOL_SUPERSESSION_TAGS.get()

        # Phase 2: collect instructions, filtering superseded tools
        instructions: dict[str, InstructionPart] = {}  # group -> part

        for name in available_names:
            tool_instance = self._get_tool_instance(name)

            # Skip tools superseded by local or composition-level active tags.
            if tool_instance.superseded_by_tags and (tool_instance.superseded_by_tags & active_tags):
                continue

            result = await tool_instance.get_instruction(ctx)

            if result is None:
                continue

            if isinstance(result, Instruction):
                group, content = result.group, result.content
            else:  # str - use tool name as implicit group and keep static by default
                group, content = tool_instance.name, result

            if not content.strip():
                continue

            # First instruction for this group wins
            if group not in instructions:
                instructions[group] = InstructionPart(
                    content=f'<tool-instruction name="{group}">{content}</tool-instruction>',
                    dynamic=False,
                )

        return list(instructions.values()) or None

    def _get_tool_impl_by_name(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        if name not in self._tool_classes:
            return None
        return self._get_tool_instance(name)

    async def process_hitl_call(
        self,
        ctx: AgentContext,
        user_interactions: list[UserInteraction] | None,
        message_history: list[ModelMessage],
    ) -> DeferredToolResults | None:
        """Process HITL interactions and return deferred tool results.

        Args:
            ctx: The agent context for processing user input.
            user_interactions: List of user interactions for deferred tool calls.
            message_history: The message history to look up tool names from call IDs.

        Returns:
            DeferredToolResults with approvals, or None if no interactions.
        """
        if not user_interactions:
            return None

        results = DeferredToolResults()
        for interaction in user_interactions:
            if interaction.approved:
                override_args = None
                metadata = None

                if interaction.user_input is not None:
                    tool_name = get_tool_name_from_id(interaction.tool_call_id, message_history)
                    if tool_name and (tool_impl := self._get_tool_impl_by_name(tool_name)):
                        try:
                            process_result = await tool_impl.process_user_input(
                                ctx,
                                user_input=interaction.user_input,
                            )
                            if process_result:
                                override_args = process_result.override_args
                                metadata = process_result.metadata
                        except Exception:
                            logger.exception(f"Failed to process user input for tool '{tool_name}'")
                            results.approvals[interaction.tool_call_id] = ToolDenied(
                                message="Failed to process user input"
                            )
                            continue

                results.approvals[interaction.tool_call_id] = ToolApproved(override_args=override_args)
                if metadata:
                    results.metadata[interaction.tool_call_id] = metadata
                logger.info(f"User approved tool call: {interaction.tool_call_id}")
            else:
                reason = interaction.reason or "User rejected the tool call."
                results.approvals[interaction.tool_call_id] = ToolDenied(message=reason)
                logger.info(f"User rejected tool call: {interaction.tool_call_id}, reason: {reason}")

        return results
