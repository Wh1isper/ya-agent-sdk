"""System prompt filter for message history.

This module provides a history processor factory that fixes system prompts
in message history by removing existing system prompts and injecting a new one.

Example::

    from pydantic_ai import Agent

    from pai_agent_sdk.context import AgentContext
    from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter

    agent = Agent(
        'openai:gpt-4',
        deps_type=AgentContext,
        history_processors=[create_system_prompt_filter("You are a helpful assistant.")],
    )
    result = await agent.run('Hello', deps=ctx)
"""

from collections.abc import Callable

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import RunContext

from pai_agent_sdk.context import AgentContext


def fix_system_prompt(message_history: list[ModelMessage], system_prompt: str) -> list[ModelMessage]:
    """Fix system prompt in message history.

    Removes all existing system prompts from the message history and injects
    the provided system prompt at the beginning of the first ModelRequest.

    Args:
        message_history: List of messages to process.
        system_prompt: The system prompt to inject.

    Returns:
        The modified message history with the new system prompt.
    """
    if not message_history:
        return message_history

    message_history_without_system: list[ModelMessage] = []
    for msg in message_history:
        # Filter out system prompts
        if not isinstance(msg, ModelRequest):
            message_history_without_system.append(msg)
            continue
        message_history_without_system.append(
            ModelRequest(
                parts=[part for part in msg.parts if not isinstance(part, SystemPromptPart)],
                instructions=msg.instructions,
            )
        )

    if message_history_without_system and isinstance(message_history_without_system[0], ModelRequest):
        # Inject system prompt at the beginning
        message_history_without_system[0].parts = [
            SystemPromptPart(content=system_prompt),
            *message_history_without_system[0].parts,
        ]

    return message_history_without_system


def create_system_prompt_filter(
    system_prompt: str,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], list[ModelMessage]]:
    """Create a history processor that fixes system prompts.

    This factory function creates a pydantic-ai history_processor that removes
    all existing system prompts from the message history and injects the
    provided system prompt at the beginning.

    This is useful when resuming from old message history where the system
    prompt may have changed.

    Note:
        This filter only affects SystemPromptPart in message history. It does NOT
        affect Agent.instructions, which is added separately by pydantic-ai after
        history processors run.

    Args:
        system_prompt: The system prompt to inject into the message history.

    Returns:
        A history processor function compatible with pydantic-ai Agent.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[create_system_prompt_filter("You are a helpful assistant.")],
        )
    """

    def processor(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        return fix_system_prompt(message_history, system_prompt)

    return processor
