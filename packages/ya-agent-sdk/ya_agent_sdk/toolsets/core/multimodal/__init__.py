"""Multimodal content processing tools.

Tools for processing images, videos, and audio when native model support is unavailable.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.multimodal.audio import ReadAudioTool
from ya_agent_sdk.toolsets.core.multimodal.image import ReadImageTool
from ya_agent_sdk.toolsets.core.multimodal.video import ReadVideoTool

tools: list[type[BaseTool]] = []
"""Deprecated: use the view tool directly."""

__all__ = [
    "ReadAudioTool",
    "ReadImageTool",
    "ReadVideoTool",
    "tools",
]
