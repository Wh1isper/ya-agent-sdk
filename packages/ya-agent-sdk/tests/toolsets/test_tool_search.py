"""Tests for ToolSearchToolSet, ToolMetadata, and search strategies (namespace-aware)."""

from __future__ import annotations

from typing import Annotated
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata, extract_metadata_from_schema
from ya_agent_sdk.toolsets.tool_search.strategies.bm25 import BM25SearchStrategy
from ya_agent_sdk.toolsets.tool_search.strategies.keyword import KeywordSearchStrategy
from ya_agent_sdk.toolsets.tool_search.toolset import ToolSearchToolSet

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather in a given location"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        location: Annotated[str, Field(description="The city and state")],
        unit: Annotated[str, Field(description="Temperature unit", default="celsius")] = "celsius",
    ) -> str:
        return f"Weather in {location}: sunny, 25{unit[0]}"


class GetForecastTool(BaseTool):
    name = "get_forecast"
    description = "Get the weather forecast for multiple days ahead"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        location: Annotated[str, Field(description="The city name")],
        days: Annotated[int, Field(description="Number of days to forecast")] = 5,
    ) -> str:
        return f"Forecast for {location}: {days} days"


class GetStockPriceTool(BaseTool):
    name = "get_stock_price"
    description = "Get the current stock price for a ticker symbol"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        ticker: Annotated[str, Field(description="Stock ticker symbol like AAPL")],
    ) -> str:
        return f"Stock {ticker}: $150.00"


class ConvertCurrencyTool(BaseTool):
    name = "convert_currency"
    description = "Convert an amount from one currency to another"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        amount: Annotated[float, Field(description="Amount to convert")],
        from_currency: Annotated[str, Field(description="Source currency code")],
        to_currency: Annotated[str, Field(description="Target currency code")],
    ) -> str:
        return f"{amount} {from_currency} = {amount * 0.85} {to_currency}"


class ViewFileTool(BaseTool):
    name = "view"
    description = "Read contents of a file from the filesystem"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the file to read")],
    ) -> str:
        return f"Contents of {file_path}"


class EditFileTool(BaseTool):
    name = "edit"
    description = "Edit a file by replacing text"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the file")],
        old_string: Annotated[str, Field(description="Text to replace")],
        new_string: Annotated[str, Field(description="Replacement text")],
    ) -> str:
        return f"Edited {file_path}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx() -> AgentContext:
    return AgentContext(run_id="test-tool-search")


@pytest.fixture
def mock_run_context(mock_ctx: AgentContext) -> RunContext[AgentContext]:
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_ctx
    return ctx


@pytest.fixture
def weather_toolset() -> Toolset:
    """Weather toolset with namespace id."""
    return Toolset(tools=[GetWeatherTool, GetForecastTool], toolset_id="weather")


@pytest.fixture
def finance_toolset() -> Toolset:
    """Finance toolset with namespace id."""
    return Toolset(tools=[GetStockPriceTool, ConvertCurrencyTool], toolset_id="finance")


@pytest.fixture
def loose_toolset() -> Toolset:
    """Loose toolset without id (individual tool loading)."""
    return Toolset(tools=[ViewFileTool, EditFileTool])


@pytest.fixture
def weather_toolset_no_id() -> Toolset:
    """Weather toolset without namespace id (loose)."""
    return Toolset(tools=[GetWeatherTool, GetForecastTool])


# ---------------------------------------------------------------------------
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
# ToolSearchToolSet tests - namespace loading
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_toolset_all_deferred_initially(weather_toolset, mock_run_context):
    """With no prior state, only tool_search should be visible."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    assert "tool_search" in tools
    assert "get_weather" not in tools
    assert "get_forecast" not in tools


@pytest.mark.anyio
async def test_tool_search_always_first_in_tool_list(weather_toolset, loose_toolset, mock_run_context):
    """tool_search must always be the first tool for stable positioning."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, loose_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )

    # Initially: tool_search is first (and only)
    tools = await ts.get_tools(mock_run_context)
    assert next(iter(tools.keys())) == "tool_search"

    # After loading namespace: tool_search still first
    await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert next(iter(tools.keys())) == "tool_search"
    assert "get_weather" in tools
    assert "get_forecast" in tools

    # After loading loose tools: tool_search still first
    await ts.call_tool("tool_search", {"query": "file"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert next(iter(tools.keys())) == "tool_search"


@pytest.mark.anyio
async def test_toolset_namespace_search_loads_all_tools(weather_toolset, mock_run_context):
    """Searching by namespace name should load all tools in that namespace."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Weather related tools"},
    )
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
    assert "[weather]" in result
    assert "Found" in result

    # Both tools should now be visible
    tools = await ts.get_tools(mock_run_context)
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "tool_search" in tools


@pytest.mark.anyio
async def test_toolset_tool_search_loads_entire_namespace(weather_toolset, mock_run_context):
    """Searching for a specific tool in a namespace loads the entire namespace."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    # Search for forecast specifically
    result = await ts.call_tool("tool_search", {"query": "forecast"}, mock_run_context, tools["tool_search"])
    assert "get_forecast" in result

    # Both tools should be loaded (entire namespace)
    tools = await ts.get_tools(mock_run_context)
    assert "get_weather" in tools
    assert "get_forecast" in tools


@pytest.mark.anyio
async def test_toolset_loose_tools_load_individually(loose_toolset, mock_run_context):
    """Loose tools (no namespace) load individually."""
    ts = ToolSearchToolSet(toolsets=[loose_toolset])
    tools = await ts.get_tools(mock_run_context)

    # Search for view - keyword "file" matches both view and edit descriptions
    # Use "read" which only matches view's description "Read contents of a file"
    result = await ts.call_tool("tool_search", {"query": "read"}, mock_run_context, tools["tool_search"])
    assert "view" in result

    # Only matched tool(s) should be loaded
    tools = await ts.get_tools(mock_run_context)
    assert "view" in tools


@pytest.mark.anyio
async def test_toolset_mixed_namespace_and_loose(weather_toolset, loose_toolset, mock_run_context):
    """Mix of namespace and loose toolsets should work correctly."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, loose_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)
    assert "tool_search" in tools
    assert len(tools) == 1  # Only tool_search

    # Search for weather -> loads namespace
    await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "view" not in tools
    assert "edit" not in tools

    # Search for file -> loads individual loose tool
    await ts.call_tool("tool_search", {"query": "file"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert "view" in tools or "edit" in tools


@pytest.mark.anyio
async def test_toolset_multiple_namespaces(weather_toolset, finance_toolset, mock_run_context):
    """Multiple namespaces should load independently."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={
            "weather": "Weather tools",
            "finance": "Finance tools",
        },
    )
    tools = await ts.get_tools(mock_run_context)

    # Load weather
    await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "get_stock_price" not in tools

    # Load finance
    await ts.call_tool("tool_search", {"query": "finance"}, mock_run_context, tools["tool_search"])
    tools = await ts.get_tools(mock_run_context)
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "get_stock_price" in tools
    assert "convert_currency" in tools


@pytest.mark.anyio
async def test_toolset_search_no_results(weather_toolset, mock_run_context):
    """Search with no matches should return helpful message."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("tool_search", {"query": "database_migration"}, mock_run_context, tools["tool_search"])
    assert "No tools found" in result


@pytest.mark.anyio
async def test_toolset_search_all_loaded(weather_toolset, mock_run_context):
    """Search when all tools are loaded should say so."""
    # Pre-load the namespace
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")

    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("tool_search", {"query": "anything"}, mock_run_context, tools["tool_search"])
    assert "already loaded" in result


@pytest.mark.anyio
async def test_toolset_call_tool_delegation(weather_toolset, mock_run_context):
    """Calling a discovered tool should delegate to the owning toolset."""
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")

    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("get_weather", {"location": "Tokyo"}, mock_run_context, tools["get_weather"])
    assert "Tokyo" in result
    assert "sunny" in result


@pytest.mark.anyio
async def test_toolset_call_unknown_tool(weather_toolset, mock_run_context):
    """Calling an unknown tool should return an error message."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "nonexistent",
        {},
        mock_run_context,
        tools["tool_search"],  # dummy tool
    )
    assert "not found" in result


@pytest.mark.anyio
async def test_toolset_search_empty_query(weather_toolset, mock_run_context):
    """Empty query should return error."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("tool_search", {"query": ""}, mock_run_context, tools["tool_search"])
    assert "Error" in result or "required" in result


@pytest.mark.anyio
async def test_toolset_context_manager(weather_toolset, finance_toolset):
    """Context manager should enter/exit wrapped toolsets."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset, finance_toolset])
    async with ts as entered:
        assert entered is ts


@pytest.mark.anyio
async def test_toolset_search_result_includes_params(weather_toolset, mock_run_context):
    """Search results for loose tools should include parameter information."""
    # Use loose toolset for individual tool results
    loose_ts = Toolset(tools=[GetWeatherTool])
    ts = ToolSearchToolSet(toolsets=[loose_ts])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
    assert "location" in result
    assert "Parameters" in result


# ---------------------------------------------------------------------------
# State persistence tests (session restore via AgentContext)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_state_persists_in_context(weather_toolset, mock_run_context):
    """Loaded state should be stored in AgentContext fields."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    assert mock_run_context.deps.tool_search_loaded_namespaces == []
    assert mock_run_context.deps.tool_search_loaded_tools == []

    await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])

    assert "weather" in mock_run_context.deps.tool_search_loaded_namespaces


@pytest.mark.anyio
async def test_state_restore_from_context(weather_toolset, finance_toolset, mock_run_context):
    """Pre-populated context should restore loaded tools."""
    mock_run_context.deps.tool_search_loaded_namespaces = ["weather"]

    ts = ToolSearchToolSet(toolsets=[weather_toolset, finance_toolset])
    tools = await ts.get_tools(mock_run_context)

    # Weather should be visible, finance should not
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "get_stock_price" not in tools
    assert "convert_currency" not in tools


@pytest.mark.anyio
async def test_state_restore_loose_tools(loose_toolset, mock_run_context):
    """Pre-populated loose tools should be restored."""
    mock_run_context.deps.tool_search_loaded_tools = ["view"]

    ts = ToolSearchToolSet(toolsets=[loose_toolset])
    tools = await ts.get_tools(mock_run_context)

    assert "view" in tools
    assert "edit" not in tools


@pytest.mark.anyio
async def test_export_and_restore_state(weather_toolset, finance_toolset):
    """Full round-trip: export state -> restore -> tools visible."""
    ctx1 = AgentContext(run_id="session-1")
    run_ctx1 = MagicMock(spec=RunContext)
    run_ctx1.deps = ctx1

    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={"weather": "Weather tools", "finance": "Finance tools"},
    )

    # Load weather namespace
    tools = await ts.get_tools(run_ctx1)
    await ts.call_tool("tool_search", {"query": "weather"}, run_ctx1, tools["tool_search"])

    # Export state
    state = ctx1.export_state()
    assert "weather" in state.tool_search_loaded_namespaces

    # Restore in new context
    ctx2 = AgentContext(run_id="session-2").with_state(state)
    run_ctx2 = MagicMock(spec=RunContext)
    run_ctx2.deps = ctx2

    tools = await ts.get_tools(run_ctx2)
    assert "get_weather" in tools
    assert "get_forecast" in tools
    assert "get_stock_price" not in tools


# ---------------------------------------------------------------------------
# Namespace description resolution tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_namespace_description_from_explicit(weather_toolset, mock_run_context):
    """namespace_descriptions takes priority over toolset.description."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Custom weather description"},
    )
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "Custom weather description" in instruction


@pytest.mark.anyio
async def test_namespace_description_from_toolset_description(mock_run_context):
    """Toolset.description should be used when namespace_descriptions is not provided."""
    ts_with_desc = Toolset(
        tools=[GetWeatherTool],
        toolset_id="weather",
        description="Weather toolset from description property",
    )
    ts = ToolSearchToolSet(toolsets=[ts_with_desc])
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "Weather toolset from description property" in instruction


@pytest.mark.anyio
async def test_namespace_description_auto_fallback(mock_run_context):
    """Without any description source, auto-generate from toolset id."""
    ts_no_desc = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    ts = ToolSearchToolSet(toolsets=[ts_no_desc])
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "weather" in instruction


@pytest.mark.anyio
async def test_namespace_description_from_instructions_property(mock_run_context):
    """Toolsets with an 'instructions' property (e.g., MCPServer) use first line as description."""
    ts_with_instructions = Toolset(tools=[GetWeatherTool], toolset_id="mcp_server")
    # Simulate MCPServer.instructions property (available after __aenter__)
    ts_with_instructions.instructions = "MCP server for weather data\nMore details here"  # type: ignore[attr-defined]

    ts = ToolSearchToolSet(toolsets=[ts_with_instructions])
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "MCP server for weather data" in instruction
    # Should not include the second line
    assert "More details here" not in instruction


@pytest.mark.anyio
async def test_namespace_description_instructions_lower_priority_than_explicit(mock_run_context):
    """namespace_descriptions should take priority over instructions property."""
    ts_with_instructions = Toolset(tools=[GetWeatherTool], toolset_id="mcp_server")
    ts_with_instructions.instructions = "From MCP instructions"  # type: ignore[attr-defined]

    ts = ToolSearchToolSet(
        toolsets=[ts_with_instructions],
        namespace_descriptions={"mcp_server": "User-provided description"},
    )
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "User-provided description" in instruction
    assert "From MCP instructions" not in instruction


# ---------------------------------------------------------------------------
# Instructions tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_instructions_include_namespace_info(weather_toolset, finance_toolset, mock_run_context):
    """Instructions should list available namespaces."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={
            "weather": "Weather tools",
            "finance": "Finance tools",
        },
    )
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    assert instruction is not None
    assert "weather" in instruction
    assert "finance" in instruction
    assert "namespaces" in instruction.lower()


@pytest.mark.anyio
async def test_instructions_only_for_loaded_toolsets(weather_toolset, mock_run_context):
    """Wrapped toolset instructions should only be included when loaded."""
    ts = ToolSearchToolSet(toolsets=[weather_toolset])

    # No tools loaded yet - should not include weather toolset instructions
    await ts.get_tools(mock_run_context)
    instruction = await ts.get_instructions(mock_run_context)
    # Should only have tool_search instruction, not weather toolset instructions
    assert instruction is not None
    assert "tool_search" in instruction


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


@pytest.mark.anyio
async def test_toolset_with_bm25_strategy(bm25_available, weather_toolset, finance_toolset, mock_run_context):
    """ToolSearchToolSet should work with BM25SearchStrategy."""
    ts = ToolSearchToolSet(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={"weather": "Weather tools", "finance": "Finance tools"},
        search_strategy=BM25SearchStrategy(),
    )
    tools = await ts.get_tools(mock_run_context)
    assert "tool_search" in tools
    assert "get_weather" not in tools

    result = await ts.call_tool("tool_search", {"query": "stock ticker price"}, mock_run_context, tools["tool_search"])
    assert "get_stock_price" in result or "finance" in result

    tools = await ts.get_tools(mock_run_context)
    assert "get_stock_price" in tools
    assert "convert_currency" in tools


# ---------------------------------------------------------------------------
# optional_namespaces tests
# ---------------------------------------------------------------------------


class FailingToolset(Toolset):
    """A toolset that raises on __aenter__ to simulate connection failure."""

    async def __aenter__(self):
        raise ConnectionError("Failed to connect to MCP server")


@pytest.mark.anyio
async def test_optional_namespace_skipped_on_failure(mock_run_context):
    """Optional toolset that fails to init should be skipped."""
    good_toolset = Toolset(tools=[GetWeatherTool, GetForecastTool], toolset_id="weather")
    bad_toolset = FailingToolset(tools=[GetStockPriceTool], toolset_id="failing_mcp")

    ts = ToolSearchToolSet(
        toolsets=[good_toolset, bad_toolset],
        optional_namespaces={"failing_mcp"},
    )

    # Should not raise, the failing toolset is optional
    async with ts:
        tools = await ts.get_tools(mock_run_context)
        assert "tool_search" in tools

        # Load weather namespace - should work
        await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
        tools = await ts.get_tools(mock_run_context)
        assert "get_weather" in tools
        assert "get_forecast" in tools


@pytest.mark.anyio
async def test_required_namespace_raises_on_failure(mock_run_context):
    """Required toolset (not in optional_namespaces) should raise on init failure."""
    good_toolset = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    bad_toolset = FailingToolset(tools=[GetStockPriceTool], toolset_id="critical_mcp")

    ts = ToolSearchToolSet(
        toolsets=[good_toolset, bad_toolset],
        optional_namespaces=set(),  # No optional namespaces
    )

    with pytest.raises(ConnectionError, match="Failed to connect"):
        async with ts:
            pass


@pytest.mark.anyio
async def test_required_namespace_default_raises_on_failure(mock_run_context):
    """Without optional_namespaces parameter, all toolsets are required (backward compat)."""
    bad_toolset = FailingToolset(tools=[GetStockPriceTool], toolset_id="some_mcp")

    ts = ToolSearchToolSet(toolsets=[bad_toolset])

    with pytest.raises(ConnectionError, match="Failed to connect"):
        async with ts:
            pass


@pytest.mark.anyio
async def test_optional_namespace_no_id_still_raises():
    """Toolset without id that fails should always raise (can't match optional_namespaces)."""
    bad_toolset = FailingToolset(tools=[GetStockPriceTool])  # No toolset_id

    ts = ToolSearchToolSet(
        toolsets=[bad_toolset],
        optional_namespaces=set(),
    )

    with pytest.raises(ConnectionError, match="Failed to connect"):
        async with ts:
            pass


@pytest.mark.anyio
async def test_optional_namespace_rollback_on_required_failure(mock_run_context):
    """When a required toolset fails, already-entered optional toolsets are rolled back."""
    good_toolset = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    bad_toolset = FailingToolset(tools=[GetStockPriceTool], toolset_id="critical_mcp")

    ts = ToolSearchToolSet(
        toolsets=[good_toolset, bad_toolset],
        optional_namespaces=set(),  # Both required
    )

    with pytest.raises(ConnectionError):
        async with ts:
            pass


@pytest.mark.anyio
async def test_multiple_optional_namespaces_some_fail(mock_run_context):
    """Multiple optional toolsets: some fail, some succeed."""
    good_weather = Toolset(tools=[GetWeatherTool, GetForecastTool], toolset_id="weather")
    bad_mcp1 = FailingToolset(tools=[GetStockPriceTool], toolset_id="mcp1")
    bad_mcp2 = FailingToolset(tools=[ConvertCurrencyTool], toolset_id="mcp2")

    ts = ToolSearchToolSet(
        toolsets=[good_weather, bad_mcp1, bad_mcp2],
        optional_namespaces={"mcp1", "mcp2"},
    )

    async with ts:
        tools = await ts.get_tools(mock_run_context)
        assert "tool_search" in tools

        # Weather should still work
        await ts.call_tool("tool_search", {"query": "weather"}, mock_run_context, tools["tool_search"])
        tools = await ts.get_tools(mock_run_context)
        assert "get_weather" in tools


# ---------------------------------------------------------------------------
# init_report and ToolSearchInitEvent tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_init_report_all_connected(mock_run_context):
    """init_report shows connected status for all successfully initialized namespaces."""
    from ya_agent_sdk.events import NamespaceStatus

    weather = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    finance = Toolset(tools=[GetStockPriceTool], toolset_id="finance")

    ts = ToolSearchToolSet(toolsets=[weather, finance])

    async with ts:
        report = ts.init_report
        assert report == {
            "weather": NamespaceStatus.connected,
            "finance": NamespaceStatus.connected,
        }


@pytest.mark.anyio
async def test_init_report_mixed_status(mock_run_context):
    """init_report shows skipped status for failed optional namespaces."""
    from ya_agent_sdk.events import NamespaceStatus

    good = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    bad = FailingToolset(tools=[GetStockPriceTool], toolset_id="bad_mcp")

    ts = ToolSearchToolSet(
        toolsets=[good, bad],
        optional_namespaces={"bad_mcp"},
    )

    async with ts:
        report = ts.init_report
        assert report == {
            "weather": NamespaceStatus.connected,
            "bad_mcp": NamespaceStatus.skipped,
        }


@pytest.mark.anyio
async def test_init_report_empty_without_namespaces():
    """Toolsets without id are not tracked in init_report."""
    loose = Toolset(tools=[GetWeatherTool])  # No toolset_id

    ts = ToolSearchToolSet(toolsets=[loose])
    async with ts:
        assert ts.init_report == {}


@pytest.mark.anyio
async def test_init_report_is_copy():
    """init_report returns a copy, not a reference to internal state."""
    from ya_agent_sdk.events import NamespaceStatus

    weather = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    ts = ToolSearchToolSet(toolsets=[weather])

    async with ts:
        report1 = ts.init_report
        report1["weather"] = NamespaceStatus.skipped  # Mutate the copy
        assert ts.init_report["weather"] == NamespaceStatus.connected  # Original unchanged


@pytest.mark.anyio
async def test_init_event_emitted_on_first_get_tools(mock_run_context):
    """ToolSearchInitEvent is emitted on get_tools call."""
    weather = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    ts = ToolSearchToolSet(toolsets=[weather])

    async with ts:
        # Enable streaming to capture the event
        mock_run_context.deps._stream_queue_enabled = True
        await ts.get_tools(mock_run_context)

        # Check event was placed on the queue
        queue = mock_run_context.deps.agent_stream_queues[mock_run_context.deps._agent_id]
        assert not queue.empty()
        event = queue.get_nowait()

        from ya_agent_sdk.events import NamespaceStatus, ToolSearchInitEvent

        assert isinstance(event, ToolSearchInitEvent)
        assert event.namespace_status == {"weather": NamespaceStatus.connected}


@pytest.mark.anyio
async def test_init_event_emitted_every_call(mock_run_context):
    """ToolSearchInitEvent is emitted on every get_tools call (status can change)."""
    weather = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    ts = ToolSearchToolSet(toolsets=[weather])

    async with ts:
        mock_run_context.deps._stream_queue_enabled = True
        await ts.get_tools(mock_run_context)
        await ts.get_tools(mock_run_context)

        # Two events should be on the queue
        queue = mock_run_context.deps.agent_stream_queues[mock_run_context.deps._agent_id]
        assert queue.qsize() == 2


@pytest.mark.anyio
async def test_init_event_not_emitted_without_namespaces(mock_run_context):
    """No event emitted if there are no namespace toolsets."""
    loose = Toolset(tools=[GetWeatherTool])  # No toolset_id
    ts = ToolSearchToolSet(toolsets=[loose])

    async with ts:
        mock_run_context.deps._stream_queue_enabled = True
        await ts.get_tools(mock_run_context)

        # No event because init_report is empty
        queue = mock_run_context.deps.agent_stream_queues[mock_run_context.deps._agent_id]
        assert queue.empty()


class RuntimeFailingToolset(Toolset):
    """A toolset that initializes OK but fails on get_tools (simulates runtime disconnect)."""

    _fail_get_tools: bool = False

    async def get_tools(self, ctx):
        if self._fail_get_tools:
            raise ConnectionError("MCP server disconnected")
        return await super().get_tools(ctx)


@pytest.mark.anyio
async def test_optional_namespace_runtime_error(mock_run_context):
    """Optional namespace that fails during get_tools should be skipped with error status."""
    from ya_agent_sdk.events import NamespaceStatus

    good = Toolset(tools=[GetWeatherTool], toolset_id="weather")
    flaky = RuntimeFailingToolset(tools=[GetStockPriceTool], toolset_id="flaky_mcp")

    ts = ToolSearchToolSet(
        toolsets=[good, flaky],
        optional_namespaces={"flaky_mcp"},
    )

    async with ts:
        # First call: both work
        tools = await ts.get_tools(mock_run_context)
        assert ts.init_report["flaky_mcp"] == NamespaceStatus.connected

        # Simulate runtime disconnect
        flaky._fail_get_tools = True
        tools = await ts.get_tools(mock_run_context)

        # flaky_mcp should now be error, weather still connected
        assert ts.init_report == {
            "weather": NamespaceStatus.connected,
            "flaky_mcp": NamespaceStatus.error,
        }
        # tool_search and weather tools should still work
        assert "tool_search" in tools


@pytest.mark.anyio
async def test_required_namespace_runtime_error_raises(mock_run_context):
    """Required namespace that fails during get_tools should raise."""
    flaky = RuntimeFailingToolset(tools=[GetStockPriceTool], toolset_id="critical_mcp")
    flaky._fail_get_tools = True

    ts = ToolSearchToolSet(toolsets=[flaky])

    async with ts:
        with pytest.raises(ConnectionError, match="MCP server disconnected"):
            await ts.get_tools(mock_run_context)
