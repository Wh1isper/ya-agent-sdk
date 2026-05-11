"""Agent context management.

This module provides the AgentContext class and related components for managing
session state during agent execution.

Example:
    Using create_agent and stream_agent (recommended)::

        from ya_agent_sdk.agents.main import create_agent, stream_agent

        runtime = create_agent("openai:gpt-4")
        async with stream_agent(runtime, "Hello") as streamer:
            async for event in streamer:
                print(event)

    Manual Environment and AgentContext setup (advanced)::

        from ya_agent_sdk.environment.local import LocalEnvironment
        from ya_agent_sdk.context import AgentContext

        async with LocalEnvironment() as env:
            async with AgentContext(env=env) as ctx:
                await ctx.file_operator.read_file("test.txt")
"""

from .agent import (
    ENVIRONMENT_CONTEXT_TAG,
    PROJECT_GUIDANCE_TAG,
    RUNTIME_CONTEXT_TAG,
    USER_RULES_TAG,
    AgentContext,
    AgentInfo,
    AgentStreamEvent,
    MediaToUrlHook,
    ModelCapability,
    ModelConfig,
    ModelWrapper,
    ResumableState,
    SecurityConfig,
    ShellReviewAction,
    ShellReviewConfig,
    ShellReviewRiskLevel,
    StreamEvent,
    SubagentWrapper,
    ToolConfig,
    ToolIdWrapper,
    ToolSettings,
)
from .bus import BusMessage, MessageBus, content_as_text
from .note import NoteManager
from .tasks import Task, TaskManager, TaskStatus

__all__ = [
    "ENVIRONMENT_CONTEXT_TAG",
    "PROJECT_GUIDANCE_TAG",
    "RUNTIME_CONTEXT_TAG",
    "USER_RULES_TAG",
    "AgentContext",
    "AgentInfo",
    "AgentStreamEvent",
    "BusMessage",
    "MediaToUrlHook",
    "MessageBus",
    "ModelCapability",
    "ModelConfig",
    "ModelWrapper",
    "NoteManager",
    "ResumableState",
    "SecurityConfig",
    "ShellReviewAction",
    "ShellReviewConfig",
    "ShellReviewRiskLevel",
    "StreamEvent",
    "SubagentWrapper",
    "Task",
    "TaskManager",
    "TaskStatus",
    "ToolConfig",
    "ToolIdWrapper",
    "ToolSettings",
    "content_as_text",
]
