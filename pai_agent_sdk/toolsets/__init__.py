"""Toolsets for pai-agent-sdk.

This module provides:
- BaseTool: Abstract base class for individual tools
- Toolset: Container for tools with hooks and HITL support
- InstructableToolset: Protocol for toolsets that provide instructions
- BrowserUseToolset: Browser automation via Chrome DevTools Protocol
"""

from pai_agent_sdk.toolsets.browser_use import BrowserUseSettings, BrowserUseToolset
from pai_agent_sdk.toolsets.core.base import (
    BaseTool,
    BaseToolset,
    GlobalHooks,
    HookableToolsetTool,
    InstructableToolset,
    Toolset,
    UserInputPreprocessResult,
    UserInteraction,
)

__all__ = [
    "BaseTool",
    "BaseToolset",
    "BrowserUseSettings",
    "BrowserUseToolset",
    "GlobalHooks",
    "HookableToolsetTool",
    "InstructableToolset",
    "Toolset",
    "UserInputPreprocessResult",
    "UserInteraction",
]
