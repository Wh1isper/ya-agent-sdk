"""BM25 search strategy for tool discovery.

Uses rank-bm25 to rank tool metadata with a lightweight lexical retrieval model.
"""

from __future__ import annotations

import re
from typing import Protocol, cast

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class _ScoreVector(Protocol):
    def __getitem__(self, index: int) -> float: ...


class _BM25Index(Protocol):
    def __init__(self, corpus: list[list[str]], *, epsilon: float = 0.25) -> None: ...

    def get_scores(self, query: list[str]) -> _ScoreVector: ...


class BM25SearchStrategy:
    """Lightweight BM25 ranking for tool search.

    BM25 scores query terms against ``ToolMetadata.searchable_text``. The
    strategy keeps the original metadata objects in the index so search can
    filter to the provided candidate list while preserving ranked order.

    Requires the ``rank-bm25`` package::

        pip install rank-bm25
    """

    def __init__(self, *, epsilon: float = 0.25) -> None:
        """Initialize the strategy.

        Args:
            epsilon: BM25Okapi epsilon floor for terms with negative IDF.
        """
        self._epsilon = epsilon
        self._index: _BM25Index | None = None
        self._indexed_tools: list[ToolMetadata] = []
        self._tokenized_corpus: list[list[str]] = []

    def get_search_hint(self) -> str:
        """Hint for BM25 search usage."""
        return (
            "Search uses BM25 ranking. Use natural language with concrete capability words, "
            'tool names, action verbs, resource names, or parameter names (e.g., "send email", '
            '"read file", "convert currency", "stock ticker").'
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25.

        Splits snake_case and other punctuation into searchable lowercase terms.
        """
        normalized = text.replace("_", " ").replace("-", " ")
        return [token.lower() for token in _TOKEN_RE.findall(normalized)]

    @classmethod
    def _normalized_tokens(cls, text: str) -> str:
        """Return a stable token-normalized string for exact name matching."""
        return " ".join(cls._tokenize(text))

    @staticmethod
    def _import_bm25() -> type[_BM25Index]:
        """Import rank-bm25 with an actionable error message."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            msg = "rank-bm25 is required for BM25SearchStrategy. Install it with: pip install ya-agent-sdk[tool-search]"
            raise ImportError(msg) from None
        return cast(type[_BM25Index], BM25Okapi)

    async def build_index(self, tools: list[ToolMetadata]) -> None:
        """Build a BM25 index from tool metadata."""
        self._indexed_tools = list(tools)
        self._tokenized_corpus = [self._tokenize(tool.searchable_text) for tool in tools]

        if not self._tokenized_corpus:
            self._index = None
            return

        BM25Okapi = self._import_bm25()
        self._index = BM25Okapi(self._tokenized_corpus, epsilon=self._epsilon)
        logger.debug("BM25 index built: %d entries", len(self._indexed_tools))

    async def search(
        self,
        query: str,
        candidates: list[ToolMetadata],
        max_results: int = 5,
    ) -> list[ToolMetadata]:
        """Search tools using BM25 ranking."""
        if not query or not candidates:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if self._index is None or not self._indexed_tools:
            logger.warning("BM25 index not built, returning empty results")
            return []

        candidate_ids = {id(tool) for tool in candidates}
        scores = self._index.get_scores(query_tokens)

        query_token_set = set(query_tokens)
        normalized_query = self._normalized_tokens(query)
        scored: list[tuple[float, int, ToolMetadata]] = []
        for idx, tool in enumerate(self._indexed_tools):
            if id(tool) in candidate_ids:
                score = float(scores[idx])
                overlap = len(query_token_set.intersection(self._tokenized_corpus[idx]))
                if score > 0 or overlap > 0:
                    # rank-bm25 can produce zero/negative scores for very small
                    # corpora because IDF is corpus-relative. Keep lexical
                    # matches discoverable while preserving BM25 rank whenever
                    # positive scores exist.
                    adjusted_score = max(score, 0.0) + (overlap * 1e-6)
                    if normalized_query == self._normalized_tokens(tool.name):
                        adjusted_score += 100.0
                    scored.append((adjusted_score, overlap, tool))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2].name))
        results = [tool for _, _, tool in scored[:max_results]]

        if results:
            logger.debug("BM25 search for %r: top result=%s, %d total", query, results[0].name, len(results))

        return results
