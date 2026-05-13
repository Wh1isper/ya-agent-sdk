"""Tool search toolset for dynamic tool loading.

Provides ToolSearchToolSet, a wrapper over multiple AbstractToolsets that
enables dynamic tool discovery via a model-callable ``tool_search`` tool.

Toolsets with ``id`` are treated as namespaces (atomic loading);
toolsets without ``id`` provide loose tools (individual loading).
State is stored in AgentContext for automatic session restore.

Usage::

    from ya_agent_sdk.toolsets.tool_search import ToolSearchToolSet

    search_toolset = ToolSearchToolSet(
        toolsets=[arxiv_toolset, github_toolset, misc_toolset],
        namespace_descriptions={"arxiv": "Search academic papers"},
    )
"""

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata, extract_metadata_from_schema
from ya_agent_sdk.toolsets.tool_search.strategies import SearchStrategy
from ya_agent_sdk.toolsets.tool_search.strategies.bm25 import BM25SearchStrategy
from ya_agent_sdk.toolsets.tool_search.strategies.keyword import KeywordSearchStrategy
from ya_agent_sdk.toolsets.tool_search.toolset import ToolSearchToolSet

logger = get_logger(__name__)

__all__ = [
    "BM25SearchStrategy",
    "KeywordSearchStrategy",
    "SearchStrategy",
    "ToolMetadata",
    "ToolSearchToolSet",
    "create_best_strategy",
    "extract_metadata_from_schema",
]


def create_best_strategy(**kwargs) -> SearchStrategy:
    """Create the preferred search strategy.

    Uses BM25 ranking when ``rank-bm25`` is installed, with keyword matching as
    the dependency-free fallback.

    Args:
        **kwargs: Passed to BM25SearchStrategy.

    Returns:
        A SearchStrategy instance.
    """
    try:
        strategy = BM25SearchStrategy(**kwargs)
        # Eagerly verify rank-bm25 is importable so callers get the fallback at
        # construction time rather than during the first search index build.
        strategy._import_bm25()
        logger.debug("Using BM25SearchStrategy")
        return strategy
    except Exception as exc:
        logger.debug("BM25SearchStrategy not available (%s), falling back to KeywordSearchStrategy", exc)
        return KeywordSearchStrategy()
