"""Keyword search strategy with regex support.

Zero external dependencies. Searches tool names, descriptions,
and parameter names/descriptions using case-insensitive matching.
Supports Python regex patterns.
"""

from __future__ import annotations

import re

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.toolsets.search.metadata import ToolMetadata

logger = get_logger(__name__)


class KeywordSearchStrategy:
    """Simple keyword matching with regex support.

    Searches tool names, descriptions, and parameter info using
    case-insensitive regex matching. Each match is scored by the
    number of fields that match.

    This is the default strategy with zero external dependencies.

    Example::

        strategy = KeywordSearchStrategy()
        results = await strategy.search("weather", candidates)
        # Matches tools with "weather" in name, description, or params
    """

    async def build_index(self, tools: list[ToolMetadata]) -> None:
        """No-op for keyword strategy (no pre-computation needed)."""
        pass

    def get_search_hint(self) -> str:
        """Hint for keyword search usage."""
        return (
            "Search uses keyword matching. Use specific tool names, action verbs, or "
            'parameter names as query terms. Regex patterns are supported (e.g., "get_.*data").'
        )

    async def search(
        self,
        query: str,
        candidates: list[ToolMetadata],
        max_results: int = 5,
    ) -> list[ToolMetadata]:
        """Search tools using keyword/regex matching.

        Scoring:
        - Name match: 3 points
        - Description match: 2 points
        - Parameter name match: 1 point per match
        - Parameter description match: 1 point per match
        - Namespace match: 2 points

        Args:
            query: Regex pattern or keywords to search for.
            candidates: Tools to search through.
            max_results: Maximum results to return.

        Returns:
            List of matching tools, sorted by relevance score (descending).
        """
        if not query or not candidates:
            return []

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            logger.debug(f"Invalid regex pattern: {query!r}, falling back to literal match")
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        scored: list[tuple[float, ToolMetadata]] = []

        for tool in candidates:
            score = self._score_tool(pattern, tool)
            if score > 0:
                scored.append((score, tool))

        # Sort by score descending, then by name for stability
        scored.sort(key=lambda x: (-x[0], x[1].name))

        return [tool for _, tool in scored[:max_results]]

    @staticmethod
    def _score_tool(pattern: re.Pattern[str], tool: ToolMetadata) -> float:
        """Compute relevance score for a single tool against the search pattern.

        Scoring:
        - Name match: 3 points
        - Description match: 2 points
        - Namespace match: 2 points
        - Parameter name match: 1 point per match
        - Parameter description match: 1 point per match
        """
        score = 0.0

        if pattern.search(tool.name):
            score += 3.0
        if pattern.search(tool.description):
            score += 2.0
        if tool.namespace and pattern.search(tool.namespace):
            score += 2.0
        for pname in tool.parameter_names:
            if pattern.search(pname):
                score += 1.0
        for pdesc in tool.parameter_descriptions.values():
            if pattern.search(pdesc):
                score += 1.0

        return score
