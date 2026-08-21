"""Enhancement tools for agent capabilities.

Tools for thinking, task management, and other enhancements.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.enhance.note import NoteGetTool, NoteTool
from ya_agent_sdk.toolsets.core.enhance.task import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
)
from ya_agent_sdk.toolsets.core.enhance.thinking import ThinkingTool

thinking_tools: list[type[BaseTool]] = [ThinkingTool]
note_tools: list[type[BaseTool]] = [NoteTool, NoteGetTool]
task_tools: list[type[BaseTool]] = [
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
]

tools: list[type[BaseTool]] = [
    # ThinkingTool,  # Disable by default via interleaved thinking
    TaskCreateTool,
    TaskGetTool,
    TaskUpdateTool,
    TaskListTool,
    NoteTool,
    NoteGetTool,
]

__all__ = [
    "NoteGetTool",
    "NoteTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "ThinkingTool",
    "note_tools",
    "tools",
]
