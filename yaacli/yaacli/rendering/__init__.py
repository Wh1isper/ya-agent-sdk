"""Rendering module for yaacli.

This module provides all rendering components for the TUI:
- Types and enums (ToolCallState, RenderDirective, ToolCallInfo)
- ToolCallTracker for tracking tool execution lifecycle
- RichRenderer for Rich-to-ANSI conversion
- ToolMessage for tool result formatting
- EventRenderer for agent event rendering

Example:
    from yaacli.rendering import EventRenderer, ToolMessage

    renderer = EventRenderer(width=120)

    # Render tool start
    output = renderer.render_tool_call_start("grep", "call-1")

    # Render tool completion
    msg = ToolMessage(tool_call_id="call-1", name="grep", content="Found 5 matches")
    output = renderer.render_tool_call_complete(msg, duration=0.5)
"""

from __future__ import annotations

from yaacli.rendering.event_renderer import EventRenderer
from yaacli.rendering.renderer import CachedRichRenderer, RichRenderer
from yaacli.rendering.tool_message import ToolMessage
from yaacli.rendering.tracker import ToolCallTracker
from yaacli.rendering.types import RenderDirective, ToolCallInfo, ToolCallState

__all__ = [
    "CachedRichRenderer",
    "EventRenderer",
    "RenderDirective",
    "RichRenderer",
    "ToolCallInfo",
    "ToolCallState",
    "ToolCallTracker",
    "ToolMessage",
]
