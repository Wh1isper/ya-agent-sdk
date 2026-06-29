"""Document processing tools.

Tools for converting PDF and Office documents to markdown format.
Optional conversion dependencies are imported lazily by each tool when used.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.document.office import OfficeConvertTool
from ya_agent_sdk.toolsets.core.document.pdf import PdfConvertTool

tools: list[type[BaseTool]] = [
    PdfConvertTool,
    OfficeConvertTool,
]

__all__ = [
    "OfficeConvertTool",
    "PdfConvertTool",
    "tools",
]
