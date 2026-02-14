"""Display components for TUI rendering.

This module re-exports all rendering components from the refactored
yaacli.rendering package for backward compatibility.

The actual implementations are now in:
- yaacli.rendering.types - Enums and data types
- yaacli.rendering.tracker - ToolCallTracker
- yaacli.rendering.renderer - RichRenderer, CachedRichRenderer
- yaacli.rendering.tool_message - ToolMessage
- yaacli.rendering.event_renderer - EventRenderer
- yaacli.rendering.tool_panels - Special tool panel rendering

Example:
    # Both of these work:
    from yaacli.display import EventRenderer, ToolMessage
    from yaacli.rendering import EventRenderer, ToolMessage
"""

from __future__ import annotations

# Re-export everything from the rendering module for backward compatibility
from yaacli.rendering import (
    CachedRichRenderer,
    EventRenderer,
    RenderDirective,
    RichRenderer,
    ToolCallInfo,
    ToolCallState,
    ToolCallTracker,
    ToolMessage,
)

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
