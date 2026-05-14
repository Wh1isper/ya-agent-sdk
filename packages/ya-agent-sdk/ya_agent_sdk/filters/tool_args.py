"""Tool args validation filter for message history.

This module provides a history processor that fixes truncated or invalid
JSON tool arguments from model responses.

Example::

    from pydantic_ai import Agent

    from ya_agent_sdk.context import AgentContext
    from pydantic_ai.capabilities import ProcessHistory
    from ya_agent_sdk.filters.tool_args import fix_truncated_tool_args

    agent = Agent(
        'openai-chat:gpt-4',
        deps_type=AgentContext,
        capabilities=[ProcessHistory(fix_truncated_tool_args)],
    )
"""

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolCallPart,
    ToolSearchCallPart,
)
from pydantic_ai.tools import RunContext

from ya_agent_sdk._logger import logger
from ya_agent_sdk.context import AgentContext


async def fix_truncated_tool_args(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Fix truncated or invalid JSON tool arguments in model responses.

    This is a Pydantic AI history processor that validates tool call arguments
    and replaces invalid JSON with a placeholder that instructs the model to retry.

    Args:
        ctx: Runtime context containing AgentContext.
        message_history: List of messages to process.

    Returns:
        The modified message history with invalid tool args fixed.

    Example:
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            capabilities=[ProcessHistory(fix_truncated_tool_args)],
        )
    """
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if (
                isinstance(part, ToolCallPart)
                and not isinstance(part, ToolSearchCallPart)
                and isinstance(part.args, str)
                and part.args
            ):
                try:
                    json.loads(part.args)
                except json.JSONDecodeError:
                    logger.warning(f"({msg.model_name})Dropping unparseable tool args: {part}")
                    part.args = {
                        "system": "This tool's args is not a valid JSON. "
                        "Please refer the return value of the tool to try again."
                    }
    return message_history
