"""Tests for Tool Proxy search metadata and ranking strategies."""

from __future__ import annotations

import pytest
from ya_agent_sdk.toolsets.search.metadata import ToolMetadata, extract_metadata_from_schema
from ya_agent_sdk.toolsets.search.strategies.bm25 import BM25SearchStrategy
from ya_agent_sdk.toolsets.search.strategies.keyword import KeywordSearchStrategy

# ToolMetadata tests
# ---------------------------------------------------------------------------


def test_tool_metadata_searchable_text():
    meta = ToolMetadata(
        name="get_weather",
        description="Get current weather",
        parameter_names=["location", "unit"],
        parameter_descriptions={"location": "City name", "unit": "Temperature unit"},
    )
    text = meta.searchable_text
    assert "get_weather" in text
    assert "Get current weather" in text
    assert "location" in text
    assert "City name" in text
    assert "unit" in text


def test_tool_metadata_searchable_text_with_namespace():
    meta = ToolMetadata(
        name="get_weather",
        description="Get current weather",
        namespace="weather",
    )
    assert "Namespace: weather" in meta.searchable_text


def test_tool_metadata_namespace_entry_searchable_text():
    meta = ToolMetadata(
        name="weather",
        description="Weather related tools",
        is_namespace_entry=True,
        namespace="weather",
        namespace_tool_names=["get_weather", "get_forecast"],
    )
    text = meta.searchable_text
    assert "Namespace: weather" in text
    assert "Weather related tools" in text
    assert "get_weather" in text
    assert "get_forecast" in text


def test_tool_metadata_brief():
    meta = ToolMetadata(
        name="get_weather",
        description="Get current weather",
        parameter_names=["location", "unit"],
    )
    brief = meta.brief
    assert "get_weather" in brief
    assert "location, unit" in brief


def test_tool_metadata_brief_no_params():
    meta = ToolMetadata(name="ping", description="Ping the server")
    assert "none" in meta.brief


def test_tool_metadata_namespace_entry_brief():
    meta = ToolMetadata(
        name="weather",
        description="Weather tools",
        is_namespace_entry=True,
        namespace_tool_names=["get_weather", "get_forecast"],
    )
    brief = meta.brief
    assert "[weather]" in brief
    assert "get_weather" in brief
    assert "get_forecast" in brief


def test_extract_metadata_from_schema():
    schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city name"},
            "unit": {"type": "string", "description": "Temperature unit"},
        },
        "required": ["location"],
    }
    meta = extract_metadata_from_schema(
        name="get_weather",
        description="Get current weather",
        parameters_json_schema=schema,
        namespace="weather",
    )
    assert meta.name == "get_weather"
    assert meta.description == "Get current weather"
    assert "location" in meta.parameter_names
    assert "unit" in meta.parameter_names
    assert meta.parameter_descriptions["location"] == "The city name"
    assert meta.namespace == "weather"


def test_extract_metadata_empty_schema():
    meta = extract_metadata_from_schema(
        name="ping",
        description=None,
        parameters_json_schema={"type": "object", "properties": {}},
    )
    assert meta.name == "ping"
    assert meta.description == ""
    assert meta.parameter_names == []
    assert meta.namespace is None


# ---------------------------------------------------------------------------
# KeywordSearchStrategy tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_keyword_search_basic():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(name="get_weather", description="Get current weather for a location"),
        ToolMetadata(name="get_stock_price", description="Get stock price for a ticker"),
        ToolMetadata(name="convert_currency", description="Convert between currencies"),
    ]
    results = await strategy.search("weather", candidates)
    assert len(results) >= 1
    assert results[0].name == "get_weather"


@pytest.mark.anyio
async def test_keyword_search_regex():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(name="get_weather", description="Get current weather"),
        ToolMetadata(name="get_forecast", description="Get weather forecast"),
        ToolMetadata(name="get_stock_price", description="Get stock price"),
    ]
    results = await strategy.search("get_.*cast", candidates)
    assert len(results) == 1
    assert results[0].name == "get_forecast"


@pytest.mark.anyio
async def test_keyword_search_invalid_regex_fallback():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(name="get_weather", description="Get current weather"),
        ToolMetadata(name="test[tool", description="A tool with brackets in name"),
    ]
    results = await strategy.search("test[tool", candidates)
    assert len(results) == 1
    assert results[0].name == "test[tool"


@pytest.mark.anyio
async def test_keyword_search_scoring_name_beats_description():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(name="weather_tool", description="Some tool"),
        ToolMetadata(name="other_tool", description="Does weather stuff"),
    ]
    results = await strategy.search("weather", candidates)
    assert len(results) == 2
    assert results[0].name == "weather_tool"


@pytest.mark.anyio
async def test_keyword_search_max_results():
    strategy = KeywordSearchStrategy()
    candidates = [ToolMetadata(name=f"tool_{i}", description="Generic tool") for i in range(10)]
    results = await strategy.search("tool", candidates, max_results=3)
    assert len(results) == 3


@pytest.mark.anyio
async def test_keyword_search_no_match():
    strategy = KeywordSearchStrategy()
    candidates = [ToolMetadata(name="get_weather", description="Get weather")]
    results = await strategy.search("database", candidates)
    assert len(results) == 0


@pytest.mark.anyio
async def test_keyword_search_empty_query():
    strategy = KeywordSearchStrategy()
    candidates = [ToolMetadata(name="tool", description="A tool")]
    assert await strategy.search("", candidates) == []


@pytest.mark.anyio
async def test_keyword_search_empty_candidates():
    strategy = KeywordSearchStrategy()
    assert await strategy.search("weather", []) == []


@pytest.mark.anyio
async def test_keyword_search_parameter_match():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(
            name="send_email",
            description="Send an email message",
            parameter_names=["recipient", "subject", "body"],
            parameter_descriptions={"recipient": "Email address of the recipient"},
        ),
        ToolMetadata(name="get_weather", description="Get weather"),
    ]
    results = await strategy.search("recipient", candidates)
    assert len(results) == 1
    assert results[0].name == "send_email"


@pytest.mark.anyio
async def test_keyword_search_namespace_match():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(name="list_orders", description="List orders", namespace="crm"),
        ToolMetadata(name="get_weather", description="Get weather", namespace="weather"),
    ]
    results = await strategy.search("crm", candidates)
    assert len(results) == 1
    assert results[0].name == "list_orders"


@pytest.mark.anyio
async def test_keyword_search_namespace_entry():
    strategy = KeywordSearchStrategy()
    candidates = [
        ToolMetadata(
            name="weather",
            description="Weather related tools",
            is_namespace_entry=True,
            namespace="weather",
            namespace_tool_names=["get_weather", "get_forecast"],
        ),
        ToolMetadata(name="get_stock_price", description="Get stock price"),
    ]
    results = await strategy.search("weather", candidates)
    assert len(results) >= 1
    # Namespace entry should match (name match + description match)
    ns_results = [r for r in results if r.is_namespace_entry]
    assert len(ns_results) == 1
    assert ns_results[0].name == "weather"


# ---------------------------------------------------------------------------
# BM25SearchStrategy tests (requires rank-bm25)
# ---------------------------------------------------------------------------


@pytest.fixture
def bm25_available():
    """Skip BM25 tests when the optional dependency is absent."""
    pytest.importorskip("rank_bm25", reason="rank-bm25 not installed")


@pytest.fixture
def bm25_candidates():
    """Create test metadata for BM25 search."""
    return [
        ToolMetadata(
            name="get_weather",
            description="Get the current weather in a given location",
            parameter_names=["location", "unit"],
            parameter_descriptions={"location": "The city and state", "unit": "Temperature unit"},
        ),
        ToolMetadata(
            name="get_forecast",
            description="Get the weather forecast for multiple days ahead",
            parameter_names=["location", "days"],
            parameter_descriptions={"location": "The city name", "days": "Number of days to forecast"},
        ),
        ToolMetadata(
            name="get_stock_price",
            description="Get the current stock price for a ticker symbol",
            parameter_names=["ticker"],
            parameter_descriptions={"ticker": "Stock ticker symbol like AAPL"},
        ),
        ToolMetadata(
            name="convert_currency",
            description="Convert an amount from one currency to another using exchange rates",
            parameter_names=["amount", "from_currency", "to_currency"],
            parameter_descriptions={
                "amount": "Amount to convert",
                "from_currency": "Source currency code",
                "to_currency": "Target currency code",
            },
        ),
        ToolMetadata(
            name="send_email",
            description="Send an email message to a recipient",
            parameter_names=["recipient", "subject", "body"],
            parameter_descriptions={
                "recipient": "Email address",
                "subject": "Email subject line",
                "body": "Email body content",
            },
        ),
    ]


@pytest.fixture
async def indexed_bm25_strategy(bm25_available, bm25_candidates):
    """Build a BM25 index with test tools."""
    strategy = BM25SearchStrategy()
    await strategy.build_index(bm25_candidates)
    return strategy


@pytest.mark.anyio
async def test_bm25_search_weather(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """BM25 search for weather should rank weather tools first."""
    results = await indexed_bm25_strategy.search("current weather location", bm25_candidates, max_results=3)
    assert len(results) >= 1
    assert results[0].name == "get_weather"


@pytest.mark.anyio
async def test_bm25_search_finance(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """BM25 search for stock ticker should find stock tools."""
    results = await indexed_bm25_strategy.search("stock ticker price", bm25_candidates, max_results=3)
    assert len(results) >= 1
    assert results[0].name == "get_stock_price"


@pytest.mark.anyio
async def test_bm25_search_currency(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """BM25 search for currency conversion."""
    results = await indexed_bm25_strategy.search("convert currency exchange rates", bm25_candidates, max_results=3)
    assert len(results) >= 1
    assert results[0].name == "convert_currency"


@pytest.mark.anyio
async def test_bm25_search_email(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """BM25 search for email should find send_email."""
    results = await indexed_bm25_strategy.search("email recipient subject body", bm25_candidates, max_results=3)
    assert len(results) >= 1
    assert results[0].name == "send_email"


@pytest.mark.anyio
async def test_bm25_search_max_results(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """Should respect max_results parameter."""
    results = await indexed_bm25_strategy.search("get current", bm25_candidates, max_results=2)
    assert len(results) <= 2


@pytest.mark.anyio
async def test_bm25_search_empty_query(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """Empty query should return empty results."""
    assert await indexed_bm25_strategy.search("", bm25_candidates) == []


@pytest.mark.anyio
async def test_bm25_search_empty_candidates(bm25_available, indexed_bm25_strategy):
    """Empty candidates should return empty results."""
    assert await indexed_bm25_strategy.search("weather", []) == []


@pytest.mark.anyio
async def test_bm25_search_candidate_filtering(bm25_available, indexed_bm25_strategy, bm25_candidates):
    """Should only return tools from the candidates list."""
    weather_only = [t for t in bm25_candidates if "weather" in t.name or "forecast" in t.name]
    results = await indexed_bm25_strategy.search("stock price", weather_only, max_results=5)
    result_names = {r.name for r in results}
    assert "get_stock_price" not in result_names


@pytest.mark.anyio
async def test_bm25_build_index_empty(bm25_available):
    """Building index with empty list should succeed."""
    strategy = BM25SearchStrategy()
    await strategy.build_index([])
    results = await strategy.search("anything", [])
    assert results == []


@pytest.mark.anyio
async def test_bm25_build_index_rebuild(bm25_available, indexed_bm25_strategy):
    """Rebuilding index should replace previous index."""
    new_tools = [
        ToolMetadata(name="ping", description="Ping a server to check if it is alive"),
    ]
    await indexed_bm25_strategy.build_index(new_tools)
    results = await indexed_bm25_strategy.search("ping server", new_tools, max_results=3)
    assert len(results) == 1
    assert results[0].name == "ping"


@pytest.mark.anyio
async def test_bm25_tokenizes_snake_case(bm25_available):
    """Snake_case tool names should be searchable by separated words."""
    strategy = BM25SearchStrategy()
    candidates = [ToolMetadata(name="get_stock_price", description="Fetch market data")]
    await strategy.build_index(candidates)
    results = await strategy.search("stock price", candidates, max_results=3)
    assert len(results) == 1
    assert results[0].name == "get_stock_price"
