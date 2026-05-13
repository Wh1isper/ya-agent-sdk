"""Deterministic retrieval evals for tool search strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata
from ya_agent_sdk.toolsets.tool_search.strategies import SearchStrategy
from ya_agent_sdk.toolsets.tool_search.strategies.bm25 import BM25SearchStrategy
from ya_agent_sdk.toolsets.tool_search.strategies.keyword import KeywordSearchStrategy

from .tool_search_eval_cases import EVAL_CASES, ToolSearchEvalCase, build_eval_catalog


@dataclass(frozen=True)
class CaseResult:
    """Search results for a single eval case."""

    case: ToolSearchEvalCase
    result_names: tuple[str, ...]

    @property
    def hit_rank(self) -> int | None:
        """One-based rank of the first expected result."""
        if not self.case.expected:
            return None
        for index, name in enumerate(self.result_names, start=1):
            if name in self.case.expected:
                return index
        return None

    @property
    def acceptable_rank(self) -> int | None:
        """One-based rank of the first acceptable result."""
        acceptable = self.case.expected | self.case.acceptable
        if not acceptable:
            return None
        for index, name in enumerate(self.result_names, start=1):
            if name in acceptable:
                return index
        return None

    @property
    def has_forbidden_hit(self) -> bool:
        """Whether top-k contains a forbidden result."""
        return any(name in self.case.forbidden for name in self.result_names)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate retrieval metrics for a strategy."""

    strategy: str
    total_positive: int
    top1: int
    hit_at_3: int
    acceptable_hit_at_3: int
    mrr: float
    forbidden_hits: int
    results: tuple[CaseResult, ...]

    def format_failures(self, other: EvalSummary | None = None) -> str:
        """Format case-level details for assertion messages."""
        lines = [
            f"strategy={self.strategy} top1={self.top1}/{self.total_positive} "
            f"hit@3={self.hit_at_3}/{self.total_positive} acceptable@3={self.acceptable_hit_at_3}/{self.total_positive} "
            f"mrr={self.mrr:.3f} forbidden={self.forbidden_hits}"
        ]
        other_by_id = {result.case.id: result for result in other.results} if other else {}
        lines.append("case_id | query | expected | results | other_results")
        for result in self.results:
            other_result = other_by_id.get(result.case.id)
            other_names = ",".join(other_result.result_names) if other_result else ""
            lines.append(
                f"{result.case.id} | {result.case.query!r} | {sorted(result.case.expected)} | "
                f"{','.join(result.result_names)} | {other_names}"
            )
        return "\n".join(lines)


async def run_eval(strategy: SearchStrategy, catalog: list[ToolMetadata], *, strategy_name: str) -> EvalSummary:
    """Run all eval cases against a strategy."""
    await strategy.build_index(catalog)
    results: list[CaseResult] = []
    for case in EVAL_CASES:
        matches = await strategy.search(case.query, catalog, max_results=case.max_results)
        results.append(CaseResult(case=case, result_names=tuple(match.name for match in matches)))

    positives = [result for result in results if result.case.expected]
    top1 = sum(1 for result in positives if result.hit_rank == 1)
    hit_at_3 = sum(1 for result in positives if result.hit_rank is not None and result.hit_rank <= 3)
    acceptable_hit_at_3 = sum(
        1 for result in positives if result.acceptable_rank is not None and result.acceptable_rank <= 3
    )
    reciprocal_ranks = [1 / result.hit_rank if result.hit_rank else 0.0 for result in positives]
    forbidden_hits = sum(1 for result in results if result.has_forbidden_hit)
    return EvalSummary(
        strategy=strategy_name,
        total_positive=len(positives),
        top1=top1,
        hit_at_3=hit_at_3,
        acceptable_hit_at_3=acceptable_hit_at_3,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        forbidden_hits=forbidden_hits,
        results=tuple(results),
    )


@pytest.mark.anyio
async def test_keyword_search_eval_baseline() -> None:
    """Keyword strategy should keep a stable baseline on core retrieval cases."""
    catalog = build_eval_catalog()
    summary = await run_eval(KeywordSearchStrategy(), catalog, strategy_name="keyword")

    assert summary.top1 >= 2, summary.format_failures()
    assert summary.hit_at_3 >= 3, summary.format_failures()
    assert summary.acceptable_hit_at_3 >= 3, summary.format_failures()
    assert summary.forbidden_hits <= 4, summary.format_failures()


@pytest.fixture
def bm25_available() -> None:
    """Skip BM25 evals when optional dependency is absent."""
    pytest.importorskip("rank_bm25", reason="rank-bm25 not installed")


@pytest.mark.anyio
async def test_bm25_search_eval_quality(bm25_available: None) -> None:
    """BM25 should pass deterministic retrieval quality gates."""
    catalog = build_eval_catalog()
    summary = await run_eval(BM25SearchStrategy(), catalog, strategy_name="bm25")

    assert summary.top1 >= 14, summary.format_failures()
    assert summary.hit_at_3 >= 17, summary.format_failures()
    assert summary.acceptable_hit_at_3 >= 18, summary.format_failures()
    assert summary.mrr >= 0.86, summary.format_failures()
    assert summary.forbidden_hits <= 2, summary.format_failures()


@pytest.mark.anyio
async def test_bm25_search_eval_compares_with_keyword(bm25_available: None) -> None:
    """BM25 should match or improve aggregate quality against keyword baseline."""
    catalog = build_eval_catalog()
    keyword = await run_eval(KeywordSearchStrategy(), catalog, strategy_name="keyword")
    bm25 = await run_eval(BM25SearchStrategy(), catalog, strategy_name="bm25")

    assert bm25.hit_at_3 >= keyword.hit_at_3, bm25.format_failures(keyword)
    assert bm25.acceptable_hit_at_3 >= keyword.acceptable_hit_at_3, bm25.format_failures(keyword)
    assert bm25.mrr >= keyword.mrr, bm25.format_failures(keyword)
    assert bm25.forbidden_hits <= 2, bm25.format_failures(keyword)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case_id", "expected_top1"),
    [
        ("file_regex_search", "grep"),
        ("snake_case_stock_price", "get_stock_price"),
        ("snake_case_old_new_string", "edit"),
        ("download_file_url", "download"),
        ("email_subject_body", "send_email"),
        ("search_chat_history", "search_messages"),
        ("web_search_docs", "search"),
        ("local_file_search", "grep"),
    ],
)
async def test_bm25_search_eval_key_cases_top1(
    bm25_available: None,
    case_id: str,
    expected_top1: str,
) -> None:
    """Key BM25 cases should keep stable top-1 results."""
    catalog = build_eval_catalog()
    strategy = BM25SearchStrategy()
    await strategy.build_index(catalog)
    case = next(item for item in EVAL_CASES if item.id == case_id)

    results = await strategy.search(case.query, catalog, max_results=case.max_results)
    result_names = [result.name for result in results]

    assert result_names and result_names[0] == expected_top1, (
        f"case_id={case_id} query={case.query!r} expected_top1={expected_top1} results={result_names}"
    )


@pytest.mark.anyio
async def test_keyword_search_eval_negative_cases() -> None:
    """Keyword strategy should return no results for empty and unrelated queries."""
    catalog = build_eval_catalog()
    keyword = KeywordSearchStrategy()
    await keyword.build_index(catalog)

    for case_id in ["empty_query", "unknown_capability"]:
        case = next(item for item in EVAL_CASES if item.id == case_id)
        assert await keyword.search(case.query, catalog, max_results=case.max_results) == []


@pytest.mark.anyio
async def test_bm25_search_eval_negative_cases(bm25_available: None) -> None:
    """BM25 strategy should return no results for empty and unrelated queries."""
    catalog = build_eval_catalog()
    bm25 = BM25SearchStrategy()
    await bm25.build_index(catalog)

    for case_id in ["empty_query", "unknown_capability"]:
        case = next(item for item in EVAL_CASES if item.id == case_id)
        assert await bm25.search(case.query, catalog, max_results=case.max_results) == []
