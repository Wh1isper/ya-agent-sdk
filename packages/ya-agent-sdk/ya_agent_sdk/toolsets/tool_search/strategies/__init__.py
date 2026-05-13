"""Search strategies for tool discovery.

Provides pluggable search algorithms for ToolSearchToolSet:
- KeywordSearchStrategy: Zero-dependency regex/keyword matching
- BM25SearchStrategy: Lightweight ranked lexical search via rank-bm25
"""

from __future__ import annotations

from typing import Protocol

from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata


class SearchStrategy(Protocol):
    """Protocol for pluggable tool search strategies.

    Implementations must provide async ``search`` and ``build_index`` methods,
    plus a ``get_search_hint`` method that returns a brief usage hint for the model.

    Optionally implement ``build_index`` for strategies that need pre-computation
    (e.g., ranked lexical indexes).
    """

    async def build_index(self, tools: list[ToolMetadata]) -> None:
        """Build or rebuild the search index from tool metadata.

        Called when the tool registry changes. Strategies that need
        pre-computation (e.g., ranked lexical indexes) should implement this.

        Strategies that don't need pre-computation can implement this as ``pass``.
        """
        ...

    async def search(
        self,
        query: str,
        candidates: list[ToolMetadata],
        max_results: int = 5,
    ) -> list[ToolMetadata]:
        """Search candidate tools and return ranked results.

        Args:
            query: The search query (natural language or regex pattern).
            candidates: The pool of tools to search through.
            max_results: Maximum number of results to return.

        Returns:
            List of matching ToolMetadata, ranked by relevance.
        """
        ...

    def get_search_hint(self) -> str:
        """Return a brief hint for the model on how to formulate search queries.

        Included in the tool_search instruction to guide query style.
        """
        ...


__all__ = [
    "SearchStrategy",
]
