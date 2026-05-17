"""Type definitions for rendering module.

Contains enums and data classes used across the rendering layer.
"""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel

from yaacli.json_types import JsonObject, JsonValue


class ToolCallState(StrEnum):
    """Tool call execution state."""

    CALLING = "calling"
    COMPLETE = "complete"
    RENDERED = "rendered"


class RenderDirective(StrEnum):
    """Render directive for display updates."""

    CALLING = "calling"
    COMPLETE = "complete"
    TEXT = "text"
    RESULT = "result"


class ToolCallInfo(BaseModel):
    """Information about a single tool call."""

    tool_call_id: str
    name: str
    args: str | JsonObject | None = None
    state: ToolCallState
    start_time: float
    end_time: float | None = None
    result: JsonValue = None

    # Special tools that need detailed panel rendering
    SPECIAL_TOOLS: frozenset[str] = frozenset({
        "edit",
        "thinking",
        "to_do_read",
        "to_do_write",
        "multi_edit",
    })

    def duration(self) -> float:
        """Calculate execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def is_special_tool(self) -> bool:
        """Check if this is a special tool requiring detailed panel."""
        return self.name in self.SPECIAL_TOOLS
