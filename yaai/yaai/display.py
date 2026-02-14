"""Display components for TUI rendering.

This module re-exports all rendering components from the refactored
yaai.rendering package for backward compatibility.

The actual implementations are now in:
- yaai.rendering.types - Enums and data types
- yaai.rendering.tracker - ToolCallTracker
- yaai.rendering.renderer - RichRenderer, CachedRichRenderer
- yaai.rendering.tool_message - ToolMessage
- yaai.rendering.event_renderer - EventRenderer
- yaai.rendering.tool_panels - Special tool panel rendering

Example:
    # Both of these work:
    from yaai.display import EventRenderer, ToolMessage
    from yaai.rendering import EventRenderer, ToolMessage
"""

from __future__ import annotations

# Re-export everything from the rendering module for backward compatibility
from yaai.rendering import (
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
