"""Web toolset for web search, scraping, and file operations.

This module requires optional dependencies. Install with:
    pip install pai-agent-sdk[web]
"""

try:
    import firecrawl  # noqa: F401
    import tavily  # noqa: F401
except ImportError as e:
    raise ImportError("Web toolset requires optional dependencies. Install with: pip install pai-agent-sdk[web]") from e

from pai_agent_sdk.toolsets.core.web.download import DownloadTool
from pai_agent_sdk.toolsets.core.web.fetch import FetchTool
from pai_agent_sdk.toolsets.core.web.scrape import ScrapeTool
from pai_agent_sdk.toolsets.core.web.search import SearchImageTool, SearchStockImageTool, SearchTool

__all__ = [
    "DownloadTool",
    "FetchTool",
    "ScrapeTool",
    "SearchImageTool",
    "SearchStockImageTool",
    "SearchTool",
]
