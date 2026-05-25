"""Factory functions for creating subagent tools.

This module provides:
- create_subagent_tool: Create BaseTool from a call function
- create_subagent_call_func: Create a BaseTool.call compatible function from a pydantic-ai Agent
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Container
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import replace
from inspect import isawaitable
from typing import Annotated, Any, cast
from uuid import uuid4

from pydantic import Field
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits
from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
)
from pydantic_ai.models import Model

from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.events import SubagentCompleteEvent, SubagentStartEvent, UsageSnapshotEvent
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.usage import coerce_run_usage

# Type alias for instruction functions
InstructionFunc = Callable[[RunContext[AgentContext]], str | None]

# Type alias for availability check functions
AvailabilityCheckFunc = Callable[[RunContext[AgentContext]], bool]

# Type alias for BaseTool.call compatible function
SubagentCallFunc = Callable[..., Awaitable[str]]
InitialMessageHistoryFactory = Callable[[RunContext[AgentContext], str], list[ModelMessage] | None]


def create_subagent_tool(
    name: str,
    description: str,
    call_func: SubagentCallFunc,
    *,
    instruction: str | InstructionFunc | None = None,
    availability_check: AvailabilityCheckFunc | None = None,
) -> type[BaseTool]:
    """Create a BaseTool subclass that wraps a subagent call function.

    This factory function creates a tool class that uses the provided call_func
    directly as the tool's call method. The call_func should have a signature
    compatible with BaseTool.call: (ctx: RunContext[AgentContext], **kwargs) -> str

    Use create_subagent_call_func() to create a compatible call_func from a
    pydantic-ai Agent.

    Args:
        name: Tool name used for invocation.
        description: Tool description shown to the model.
        call_func: Async function with signature (ctx: RunContext[AgentContext], **kwargs) -> str.
                   Use create_subagent_call_func() to create this from an Agent.
        instruction: Optional instruction for system prompt. Can be a string or
                     a callable that takes RunContext and returns a string.
        availability_check: Optional callable that returns True if the tool is available.
                            Called dynamically each time is_available() is invoked.

    Returns:
        A BaseTool subclass that can be used with Toolset.

    Example::

        from pydantic_ai import Agent

        # Create an agent
        search_agent: Agent[AgentContext, str] = Agent(...)

        # Create the call function using create_subagent_call_func
        search_call = create_subagent_call_func(search_agent)

        # Create the tool
        SearchTool = create_subagent_tool(
            name="search",
            description="Search the web for information",
            call_func=search_call,
            instruction="Use this tool to search for current information.",
        )
    """

    class DynamicSubagentTool(BaseTool):
        """Dynamically created subagent tool."""

        # These will be set by the closure
        name = ""  # Placeholder, will be overwritten
        description = ""  # Placeholder, will be overwritten

        def is_available(self, ctx: RunContext[AgentContext]) -> bool:
            if availability_check is None:
                return True
            return availability_check(ctx)

        async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            if instruction is None:
                return None
            if callable(instruction):
                return instruction(ctx)
            return instruction

        async def call(self, ctx: RunContext[AgentContext], /, **kwargs: object) -> str:
            # Placeholder - will be replaced by actual call_func
            raise NotImplementedError  # pragma: no cover

    # Set class attributes from closure variables
    DynamicSubagentTool.name = name
    DynamicSubagentTool.description = description

    # Use call_func directly as the call method
    # call_func should already have the correct signature from create_subagent_call_func
    DynamicSubagentTool.call = call_func  # type: ignore[method-assign]

    # Set a meaningful class name for debugging
    DynamicSubagentTool.__name__ = f"{_to_pascal_case(name)}Tool"
    DynamicSubagentTool.__qualname__ = DynamicSubagentTool.__name__

    return DynamicSubagentTool


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts)


async def _run_subagent_iter(
    agent: Agent[AgentContext, Any],
    parent_ctx: AgentContext,
    sub_ctx: AgentContext,
    prompt: str,
    message_history: list[Any] | None,
    *,
    model: Model,
    model_id: str,
    agent_name: str,
    usage_id: str | None = None,
) -> AgentRunResult:
    """Run subagent iteration and stream events to subagent's queue.

    Events are emitted to sub_ctx (subagent context) so they go to the
    subagent's queue keyed by agent_id, not the parent's queue.

    Args:
        agent: The subagent to run.
        sub_ctx: Subagent's context (events emitted here).
        prompt: The prompt to send to the subagent.
        message_history: Optional conversation history for resume.

    Returns:
        AgentRunResult from the subagent execution.
    """
    model_request_index = 0
    async with agent.iter(
        prompt,
        deps=sub_ctx,
        usage_limits=UsageLimits(request_limit=1000),
        message_history=message_history,
        model=model,
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue

            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        await sub_ctx.emit_event(event)
                if Agent.is_model_request_node(node):
                    parent_ctx.update_usage_snapshot_entry(
                        ledger_key=sub_ctx.agent_id,
                        agent_id=sub_ctx.agent_id,
                        agent_name=agent_name,
                        model_id=model_id,
                        usage=coerce_run_usage(run.usage),
                        usage_id=usage_id,
                        source="subagent_model_request",
                    )
                    snapshot = parent_ctx.build_usage_snapshot()
                    await parent_ctx.emit_event(
                        UsageSnapshotEvent(
                            event_id=(
                                f"{snapshot.run_id}:usage_snapshot:{sub_ctx.agent_id}:"
                                f"model_request_complete:{model_request_index}"
                            ),
                            snapshot=snapshot,
                            source="model_request_complete",
                        )
                    )
                    model_request_index += 1

    return cast(AgentRunResult, run.result)


def generate_unique_id(existing: Container[str], *, prefix: str = "", max_retries: int = 10) -> str:
    """Generate a unique 4-character ID with collision detection.

    First tries using the last 4 characters of run_id. If that collides
    with existing IDs, generates random UUIDs until a unique one is found.

    Args:
        run_id: The current run ID to derive initial ID from.
        existing: Container of existing IDs to check against.
        max_retries: Maximum number of UUID generation attempts (default 10).

    Returns:
        A unique 4-character ID string.

    Raises:
        RuntimeError: If unable to generate unique ID within max_retries.
    """
    for _ in range(max_retries):
        agent_id = uuid4().hex[:4]
        if f"{prefix}{agent_id}" not in existing:
            return agent_id

    raise RuntimeError(f"Failed to generate unique agent_id after {max_retries} retries")


def _unresolved_tool_call_ids(history: list[ModelMessage]) -> set[str]:
    """Return tool call IDs that have no matching return in the given history."""
    unresolved: set[str] = set()
    for message in history:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, BaseToolCallPart):
                    unresolved.add(part.tool_call_id)
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, BaseToolReturnPart | RetryPromptPart):
                    unresolved.discard(part.tool_call_id)
    return unresolved


def _history_without_unresolved_tool_calls(history: list[ModelMessage]) -> list[ModelMessage]:
    """Return history with pending tool calls removed.

    Self forks start from the parent conversation state and append their own
    prompt. During parallel tool execution, sibling tool calls may still be
    pending and therefore have no matching tool return. Removing all unresolved
    tool-call parts keeps the fork history valid and avoids showing the fork its
    own invocation.
    """
    unresolved_ids = _unresolved_tool_call_ids(history)
    if not unresolved_ids:
        return list(history)

    cleaned: list[ModelMessage] = []
    for message in history:
        if not isinstance(message, ModelResponse):
            cleaned.append(message)
            continue

        parts = [
            part
            for part in message.parts
            if not (isinstance(part, BaseToolCallPart) and part.tool_call_id in unresolved_ids)
        ]
        if parts:
            cleaned.append(replace(message, parts=parts))

    return cleaned


def _get_self_fork_history(ctx: RunContext[AgentContext], agent_id: str) -> list[ModelMessage] | None:
    resumed_history = ctx.deps.subagent_history.get(agent_id)
    if resumed_history is not None:
        return resumed_history
    return _history_without_unresolved_tool_calls(list(ctx.messages))


def create_self_fork_call_func(
    *,
    model_cfg: ModelConfig | None = None,
) -> SubagentCallFunc:
    """Create a call function for delegate(subagent_name="self")."""
    agent_name = "self"

    async def call_func(
        self: BaseTool,
        ctx: RunContext[AgentContext],
        prompt: Annotated[str, Field(description="The prompt to send to the fork")],
        agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
    ) -> str:
        deps = ctx.deps
        agent = deps.self_fork_agent
        if agent is None:
            return "Error: Self fork is not available in this runtime."
        return await _call_agent_as_subagent(
            agent,
            agent_name,
            self,
            ctx,
            prompt,
            agent_id,
            model_cfg=model_cfg,
            initial_message_history_factory=_get_self_fork_history,
        )

    return call_func  # type: ignore[return-value]


async def _call_agent_as_subagent(
    agent: Agent[AgentContext, Any],
    agent_name: str,
    self: BaseTool,
    ctx: RunContext[AgentContext],
    prompt: str,
    agent_id: str | None,
    *,
    model_cfg: ModelConfig | None = None,
    initial_message_history_factory: InitialMessageHistoryFactory | None = None,
) -> str:
    """Execute an agent using the common subagent lifecycle and history model."""
    deps = ctx.deps

    async with deps._subagent_state_lock:
        if agent_id is None:
            existing_agent_ids = set(deps.agent_registry) | set(deps.subagent_history)
            short_id = generate_unique_id(existing_agent_ids, prefix=f"{agent_name}-")
            agent_id = f"{agent_name}-{short_id}"

        # Track whether this is a new agent (not a resume) for cleanup on failure
        is_new_agent = agent_id not in deps.agent_registry
        deps.reserve_subagent_id(agent_name, agent_id)

    # Create subagent context (handles registration in agent_registry)
    override_kwargs: dict[str, Any] = {}
    if model_cfg is not None:
        override_kwargs["model_cfg"] = model_cfg

    error_msg = ""
    success = True
    result_output = ""
    request_count = 0

    async with deps.create_subagent_context(agent_name, agent_id=agent_id, **override_kwargs) as sub_ctx:
        # Set the subagent's initial prompt for compact
        sub_ctx.user_prompts = prompt

        # Emit start event to subagent's queue (inside context so sub_ctx.start_at is set)
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        await sub_ctx.emit_event(
            SubagentStartEvent(
                event_id=agent_id,  # Use agent_id as event_id to correlate Start/Complete
                agent_id=agent_id,
                agent_name=agent_name,
                prompt_preview=prompt_preview,
            )
        )

        # Wrap subagent execution with observability wrapper
        subagent_ctx: AbstractAsyncContextManager[None] = nullcontext()
        if deps.subagent_wrapper is not None:
            wrapper_metadata = deps.get_wrapper_metadata()
            subagent_ctx = deps.subagent_wrapper(agent_name, agent_id, wrapper_metadata)

        try:
            async with subagent_ctx:
                run_model = cast(Model, agent.model)
                if deps.model_wrapper is not None:
                    wrapper_metadata = sub_ctx.get_wrapper_metadata()
                    wrapped = deps.model_wrapper(run_model, agent_name, wrapper_metadata)
                    run_model = await wrapped if isawaitable(wrapped) else wrapped

                model_id = run_model.model_name
                usage_id = ctx.tool_call_id or uuid4().hex
                message_history = (
                    initial_message_history_factory(ctx, agent_id)
                    if initial_message_history_factory is not None
                    else deps.subagent_history.get(agent_id)
                )
                result = await _run_subagent_iter(
                    agent,
                    deps,
                    sub_ctx,
                    prompt,
                    message_history,
                    model=run_model,
                    model_id=model_id,
                    agent_name=agent_name,
                    usage_id=usage_id,
                )
                result_output = result.output
                result_usage = coerce_run_usage(result.usage)
                request_count = result_usage.requests

                # Store message history for future resume
                deps.subagent_history[agent_id] = result.all_messages()

                # Ensure final provider usage is reflected in the unified ledger.
                deps.update_usage_snapshot_entry(
                    ledger_key=agent_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    model_id=model_id,
                    usage=result_usage,
                    usage_id=usage_id,
                    source="subagent",
                )

        except Exception as e:
            success = False
            error_msg = str(e)
            # Clean up agent_registry for new agents that failed before
            # producing any history, to avoid ghost entries that show up
            # in known-subagents and SubagentInfoTool with no history.
            if is_new_agent and agent_id not in deps.subagent_history:
                deps.agent_registry.pop(agent_id, None)
            raise

        finally:
            # Emit complete event to subagent's queue (use sub_ctx.elapsed_time for duration)
            elapsed = sub_ctx.elapsed_time
            duration = elapsed.total_seconds() if elapsed else 0.0
            result_preview = result_output[:500] + "..." if len(result_output) > 500 else result_output
            await sub_ctx.emit_event(
                SubagentCompleteEvent(
                    event_id=agent_id,  # Same event_id as Start for correlation
                    agent_id=agent_id,
                    agent_name=agent_name,
                    success=success,
                    request_count=request_count,
                    result_preview=result_preview,
                    error=error_msg,
                    duration_seconds=duration,
                )
            )

    # Return formatted result
    return f"""<id>{agent_id}</id>
<response>{result.output}</response>
"""


def create_subagent_call_func(
    agent: Agent[AgentContext, Any],
    *,
    model_cfg: ModelConfig | None = None,
) -> SubagentCallFunc:
    """Create a BaseTool.call compatible function from a pydantic-ai Agent.

    This function creates a call method that:
    - Has the correct signature for BaseTool.call: (ctx: RunContext[AgentContext], **kwargs) -> str
    - Generates stable agent_id in format {agent.name}-{short_id}
    - Registers the agent in parent's agent_registry
    - Manages subagent_history for conversation continuity
    - Records usage in the unified usage ledger
    - Streams events to parent context
    """
    agent_name = agent.name or "subagent"

    async def call_func(
        self: BaseTool,
        ctx: RunContext[AgentContext],
        prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
        agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
    ) -> str:
        return await _call_agent_as_subagent(agent, agent_name, self, ctx, prompt, agent_id, model_cfg=model_cfg)

    return call_func  # type: ignore[return-value]
