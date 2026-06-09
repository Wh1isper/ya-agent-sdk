"""Tool call tracking for rendering.

Tracks tool call states and manages rendering lifecycle.
"""

from __future__ import annotations

import time

from yaacli.json_types import JsonObject, JsonValue
from yaacli.rendering.types import ToolCallInfo, ToolCallState


class ToolCallTracker:
    """Track tool call states and manage rendering lifecycle."""

    def __init__(self) -> None:
        self.tool_calls: dict[str, ToolCallInfo] = {}
        self.call_order: list[str] = []

    def start_call(
        self,
        tool_call_id: str,
        name: str,
        args: str | JsonObject | None = None,
        *,
        start_time: float | None = None,
    ) -> None:
        """Register a new tool call."""
        if tool_call_id in self.tool_calls:
            info = self.tool_calls[tool_call_id]
            info.name = name
            info.args = args
            info.state = ToolCallState.CALLING
            if start_time is not None:
                info.start_time = start_time
            info.end_time = None
            info.result = None
            return

        self.tool_calls[tool_call_id] = ToolCallInfo(
            tool_call_id=tool_call_id,
            name=name,
            args=args,
            state=ToolCallState.CALLING,
            start_time=start_time if start_time is not None else time.time(),
        )
        self.call_order.append(tool_call_id)

    def complete_call(
        self,
        tool_call_id: str,
        result: JsonValue = None,
        *,
        end_time: float | None = None,
    ) -> None:
        """Mark tool call as complete."""
        if tool_call_id in self.tool_calls:
            info = self.tool_calls[tool_call_id]
            info.state = ToolCallState.COMPLETE
            if end_time is not None or info.end_time is None:
                info.end_time = end_time if end_time is not None else time.time()
            info.result = result

    def mark_rendered(self, tool_call_id: str) -> None:
        """Mark tool call as rendered."""
        if tool_call_id in self.tool_calls:
            self.tool_calls[tool_call_id].state = ToolCallState.RENDERED

    def get_calling_tools(self) -> list[ToolCallInfo]:
        """Get tools in CALLING state."""
        return [
            self.tool_calls[tid]
            for tid in self.call_order
            if tid in self.tool_calls and self.tool_calls[tid].state == ToolCallState.CALLING
        ]

    def get_completed_tools(self) -> list[ToolCallInfo]:
        """Get tools in COMPLETE state (ready to render)."""
        return [
            self.tool_calls[tid]
            for tid in self.call_order
            if tid in self.tool_calls and self.tool_calls[tid].state == ToolCallState.COMPLETE
        ]

    def has_active_calls(self) -> bool:
        """Check if there are any active tool calls."""
        return len(self.get_calling_tools()) > 0 or len(self.get_completed_tools()) > 0

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        self.tool_calls.clear()
        self.call_order.clear()
