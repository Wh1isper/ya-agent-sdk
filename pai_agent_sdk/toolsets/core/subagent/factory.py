"""Factory function for creating subagent tools.

This module provides the create_subagent_tool function which dynamically
creates BaseTool subclasses that wrap subagent call functions.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

# Type alias for subagent call functions
# First parameter is AgentContext (the subagent context), rest are user-defined
# Returns (output, RunUsage) tuple
SubagentCallFunc = Callable[..., Awaitable[tuple[Any, RunUsage]]]

# Type alias for instruction functions
InstructionFunc = Callable[[RunContext[AgentContext]], str | None]


def create_subagent_tool(
    name: str,
    description: str,
    call_func: SubagentCallFunc,
    *,
    instruction: str | InstructionFunc | None = None,
) -> type[BaseTool]:
    """Create a BaseTool subclass that wraps a subagent call function.

    This factory function creates a tool class that:
    - Uses the call_func's parameter signature (excluding ctx) as tool parameters
    - Automatically records RunUsage to ctx.deps.extra_usage
    - Converts the output to string for LLM consumption

    For streaming, use ctx.deps.subagent_stream_queues[tool_call_id] to send events.

    Args:
        name: Tool name used for invocation.
        description: Tool description shown to the model.
        call_func: Async function with signature (ctx, **kwargs) -> (output, RunUsage).
                   The function's parameters (after ctx) define the tool's input schema.
        instruction: Optional instruction for system prompt. Can be a string or
                     a callable that takes RunContext and returns a string.

    Returns:
        A BaseTool subclass that can be used with Toolset.

    Example::

        async def search(
            ctx: AgentContext,  # This is the subagent context (auto-created)
            query: str,
            max_results: int = 10,
        ) -> tuple[str, RunUsage]:
            agent = get_search_agent()
            result = await agent.run(f"Search: {query}, max: {max_results}", deps=ctx)
            return str(result.output), result.usage()

        SearchTool = create_subagent_tool(
            name="search",
            description="Search the web for information",
            call_func=search,
            instruction="Use this tool to search for current information.",
        )

        # For streaming, access the parent context's stream queue:
        async def search_with_stream(
            ctx: AgentContext,
            query: str,
        ) -> tuple[str, RunUsage]:
            # ctx.parent_run_id is the tool_call_id
            # Access parent's stream queue via the shared reference
            agent = get_search_agent()
            async for event in agent.run_stream_events(query, deps=ctx):
                # Forward events as needed
                pass
            result = await agent.run(query, deps=ctx)
            return str(result.output), result.usage()
    """

    class DynamicSubagentTool(BaseTool):
        """Dynamically created subagent tool."""

        # These will be set by the closure
        name = ""  # Placeholder, will be overwritten
        description = ""  # Placeholder, will be overwritten

        def __init__(self, ctx: AgentContext) -> None:
            super().__init__(ctx)

        def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            if instruction is None:
                return None
            if callable(instruction):
                return instruction(ctx)
            return instruction

        async def call(self, ctx: RunContext[AgentContext], /, **kwargs: Any) -> str:
            """Execute the subagent call and record usage."""
            output, usage = await call_func(ctx, **kwargs)

            # Record usage in extra_usages
            if ctx.tool_call_id:
                ctx.deps.add_extra_usage(agent=name, usage=usage, uuid=ctx.tool_call_id)

            # Convert output to string for LLM
            return str(output)

    # Set class attributes from closure variables
    DynamicSubagentTool.name = name
    DynamicSubagentTool.description = description

    # Copy the call signature from call_func to DynamicSubagentTool.call
    # This allows pydantic-ai to inspect the correct parameters
    DynamicSubagentTool.call = _create_call_method(call_func)  # type: ignore[method-assign]

    # Set a meaningful class name for debugging
    DynamicSubagentTool.__name__ = f"{_to_pascal_case(name)}Tool"
    DynamicSubagentTool.__qualname__ = DynamicSubagentTool.__name__

    return DynamicSubagentTool


def _create_call_method(
    call_func: SubagentCallFunc,
) -> Callable[..., Awaitable[str]]:
    """Create a call method with the correct signature from call_func.

    The first parameter (ctx: AgentContext) is replaced with RunContext[AgentContext]
    for pydantic-ai compatibility. The actual call_func receives the subagent context.
    """

    async def call(self: BaseTool, ctx: RunContext[AgentContext], /, **kwargs: Any) -> str:
        """Execute the subagent call and record usage."""
        async with ctx.deps.enter_subagent(self.name, agent_id=ctx.tool_call_id) as sub_ctx:
            output, usage = await call_func(sub_ctx, **kwargs)

        # Record usage in extra_usages
        if ctx.tool_call_id:
            ctx.deps.add_extra_usage(agent=self.name, usage=usage, uuid=ctx.tool_call_id)

        # Convert output to string for LLM
        return str(output)

    # Copy the signature from call_func, but replace ctx type
    original_sig = inspect.signature(call_func)
    params = list(original_sig.parameters.values())

    # Replace first param (ctx: AgentContext) with (ctx: RunContext[AgentContext])
    if params:
        first_param = params[0]
        new_first_param = first_param.replace(annotation=RunContext[AgentContext])
        params[0] = new_first_param

    # Create new signature with RunContext and str return type
    new_sig = original_sig.replace(parameters=params, return_annotation=str)
    call.__signature__ = new_sig  # type: ignore[attr-defined]

    # Copy docstring and name
    call.__doc__ = call_func.__doc__ or "Execute the subagent call and record usage."
    call.__name__ = "call"
    call.__qualname__ = "call"

    # Build annotations with RunContext[AgentContext] for the first param
    original_annotations = getattr(call_func, "__annotations__", {})
    new_annotations: dict[str, Any] = {}
    first_param_name = params[0].name if params else "ctx"
    for key, value in original_annotations.items():
        if key == first_param_name:
            # Replace ctx: AgentContext with ctx: RunContext[AgentContext]
            new_annotations[key] = RunContext[AgentContext]
        elif key != "return":
            new_annotations[key] = value
    new_annotations["return"] = str
    call.__annotations__ = new_annotations

    return call


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts)
