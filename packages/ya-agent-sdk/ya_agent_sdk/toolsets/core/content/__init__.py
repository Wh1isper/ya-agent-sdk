"""Content loading tools.

Tools for loading multimedia content from URLs and external sources.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.content.load_media_url import LoadMediaUrlTool
from ya_agent_sdk.toolsets.core.content.read_media import ReadMediaTool

tools: list[type[BaseTool]] = [
    ReadMediaTool,
    # LoadMediaUrlTool,  # Disabled by default since not all models support it
]

__all__ = [
    "LoadMediaUrlTool",
    "ReadMediaTool",
    "tools",
]
