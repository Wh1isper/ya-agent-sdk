"""Environment instructions history processor factory.

This module provides a factory function that creates a history processor
for injecting environment context instructions (file system, shell configuration)
into the message history before each model request.

Example::

    from contextlib import AsyncExitStack
    from pydantic_ai import Agent

    from ya_agent_sdk.context import AgentContext
    from ya_agent_sdk.environment.local import LocalEnvironment
    from ya_agent_sdk.filters.environment_instructions import create_environment_instructions_filter

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment())
        ctx = await stack.enter_async_context(
            AgentContext(env=env)
        )
        env_filter = create_environment_instructions_filter(env)
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            history_processors=[env_filter],
        )
        result = await agent.run('Your prompt here', deps=ctx)
"""

from collections.abc import Awaitable, Callable

from pydantic_ai import RetryPromptPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import RunContext
from ya_agent_environment import Environment

from ya_agent_sdk.context import AgentContext


def create_environment_instructions_filter(
    env: Environment,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a history processor that injects environment instructions.

    This factory function creates a pydantic-ai history_processor that inserts
    environment context instructions (file system paths, shell configuration)
    into the last ModelRequest before ordinary user prompt content, while keeping
    tool response and retry prompt parts before injected instructions.

    Normally skips injection when the last request contains ToolReturnPart
    (intermediate tool responses). However, when ``ctx.deps.force_inject_instructions``
    is True (set by handoff/compact after context reset), injection is forced
    regardless of message content.

    Args:
        env: Environment instance to get context instructions from.

    Returns:
        A history processor function compatible with pydantic-ai Agent.

    Example:
        env_filter = create_environment_instructions_filter(env)
        agent = Agent(
            'openai-chat:gpt-4',
            history_processors=[env_filter],
        )
    """

    async def inject_environment_instructions(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Inject environment instructions into the last ModelRequest.

        Args:
            ctx: Runtime context (not used, but required by history_processor signature).
            message_history: Current message history to process.

        Returns:
            Processed message history with environment instructions injected into
            the last ModelRequest, or unchanged history if no ModelRequest found.
        """
        # Find the last ModelRequest in message history
        last_request: ModelRequest | None = None
        for msg in reversed(message_history):
            if isinstance(msg, ModelRequest):
                last_request = msg
                break

        if not last_request:
            return message_history

        # Skip injection if last_request contains ToolReturnPart (tool response)
        # We only inject environment instructions on user input, not tool responses or retry prompts
        # Exception: force_inject_instructions overrides this check after context reset (handoff/compact)
        has_tool_return = any(isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in last_request.parts)
        if has_tool_return and not ctx.deps.force_inject_instructions:
            return message_history

        # Get environment instructions
        instructions = await env.get_context_instructions()

        if not instructions:
            return message_history

        # Insert before ordinary user prompt content, while keeping any tool responses
        # and retry prompts before injected instructions for protocol-required ordering.
        env_part = UserPromptPart(content=instructions)
        insert_at = 0
        for index, part in enumerate(last_request.parts):
            if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                insert_at = index + 1
        last_request.parts = [*last_request.parts[:insert_at], env_part, *last_request.parts[insert_at:]]

        return message_history

    return inject_environment_instructions
