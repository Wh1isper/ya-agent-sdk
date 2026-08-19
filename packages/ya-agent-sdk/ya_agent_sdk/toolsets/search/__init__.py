"""Search metadata and ranking strategies used by Tool Proxy."""

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.toolsets.search.metadata import ToolMetadata, extract_metadata_from_schema
from ya_agent_sdk.toolsets.search.strategies import SearchStrategy
from ya_agent_sdk.toolsets.search.strategies.bm25 import BM25SearchStrategy
from ya_agent_sdk.toolsets.search.strategies.keyword import KeywordSearchStrategy

logger = get_logger(__name__)

__all__ = [
    "BM25SearchStrategy",
    "KeywordSearchStrategy",
    "SearchStrategy",
    "ToolMetadata",
    "create_best_strategy",
    "extract_metadata_from_schema",
]


def create_best_strategy(**kwargs) -> SearchStrategy:
    """Create the best available Tool Proxy search strategy."""
    try:
        strategy = BM25SearchStrategy(**kwargs)
        strategy._import_bm25()
        logger.debug("Using BM25SearchStrategy")
        return strategy
    except Exception as exc:
        logger.debug("BM25SearchStrategy not available (%s), falling back to KeywordSearchStrategy", exc)
        return KeywordSearchStrategy()
