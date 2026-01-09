"""Web toolset for web search, scraping, and file operations.

Optional dependencies are lazily imported when needed:
- tavily: required only when using SearchTool with tavily_api_key
- firecrawl: required only when using ScrapeTool with firecrawl_api_key

Tools use is_available() to check API key availability:
- SearchTool: requires google_search_api_key+cx OR tavily_api_key
- SearchStockImageTool: requires pixabay_api_key
- SearchImageTool: requires rapidapi_api_key
- DownloadTool: requires file_operator
- ScrapeTool: always available (falls back to MarkItDown if no firecrawl)
- FetchTool: always available (uses HTTP only)
"""

from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.web.download import DownloadTool
from pai_agent_sdk.toolsets.core.web.fetch import FetchTool
from pai_agent_sdk.toolsets.core.web.scrape import ScrapeTool
from pai_agent_sdk.toolsets.core.web.search import SearchImageTool, SearchStockImageTool, SearchTool

tools: list[type[BaseTool]] = [
    SearchTool,
    SearchStockImageTool,
    SearchImageTool,
    ScrapeTool,
    FetchTool,
    DownloadTool,
]

__all__ = [
    "DownloadTool",
    "FetchTool",
    "ScrapeTool",
    "SearchImageTool",
    "SearchStockImageTool",
    "SearchTool",
    "tools",
]
