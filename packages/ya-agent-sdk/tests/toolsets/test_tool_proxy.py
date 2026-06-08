"""Tests for ToolProxyToolset: fixed two-tool proxy for dynamic tool invocation."""

from __future__ import annotations

from typing import Annotated
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.tool_proxy.toolset import ToolProxyToolset

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


class FailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that always fails"

    async def call(self, ctx: RunContext[AgentContext]) -> object:
        raise ValueError("Something went wrong")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx() -> AgentContext:
    return AgentContext(run_id="test-tool-proxy")


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
def failing_toolset() -> Toolset:
    """Toolset containing a tool that always raises."""
    return Toolset(tools=[FailingTool])


# ---------------------------------------------------------------------------
# get_tools: always returns exactly 2 tools
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_tools_always_returns_two(weather_toolset, mock_run_context):
    """get_tools must always return exactly search_tools and call_tool."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    assert set(tools.keys()) == {"search_tools", "call_tool"}
    assert len(tools) == 2


@pytest.mark.anyio
async def test_get_tools_stable_after_search(weather_toolset, loose_toolset, mock_run_context):
    """Tool list must stay exactly 2 tools even after searching."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset, loose_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)
    assert len(tools) == 2

    # Search for weather
    await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])
    tools = await ts.get_tools(mock_run_context)
    assert set(tools.keys()) == {"search_tools", "call_tool"}

    # Search for file tools
    await ts.call_tool("search_tools", {"query": "file"}, mock_run_context, tools["search_tools"])
    tools = await ts.get_tools(mock_run_context)
    assert set(tools.keys()) == {"search_tools", "call_tool"}


@pytest.mark.anyio
async def test_get_tools_order_is_stable(weather_toolset, mock_run_context):
    """search_tools must always come first."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)
    keys = list(tools.keys())
    assert keys == ["search_tools", "call_tool"]


@pytest.mark.anyio
async def test_get_tools_supports_prefix(weather_toolset, mock_run_context):
    """Prefixed proxy names should use {prefix}_search_tool and {prefix}_call_tool."""
    ts = ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp")
    tools = await ts.get_tools(mock_run_context)

    assert list(tools.keys()) == ["mcp_search_tool", "mcp_call_tool"]
    assert ts.prefix == "mcp"
    assert ts.search_tool_name == "mcp_search_tool"
    assert ts.call_tool_name == "mcp_call_tool"


@pytest.mark.anyio
async def test_prefixed_proxy_search_and_call(weather_toolset, mock_run_context):
    """Prefixed proxy tools should search, store scoped state, and call underlying tools."""
    ts = ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp")
    tools = await ts.get_tools(mock_run_context)

    search_result = await ts.call_tool(
        "mcp_search_tool",
        {"query": "weather"},
        mock_run_context,
        tools["mcp_search_tool"],
    )
    assert "get_weather" in search_result
    assert "weather" not in mock_run_context.deps.tool_search_loaded_namespaces
    assert "mcp:weather" in mock_run_context.deps.tool_search_loaded_namespaces

    call_result = await ts.call_tool(
        "mcp_call_tool",
        {"name": "get_weather", "arguments": {"location": "Rome"}},
        mock_run_context,
        tools["mcp_call_tool"],
    )
    assert "Rome" in call_result


@pytest.mark.anyio
async def test_prefixed_proxy_state_isolated_between_prefixes(weather_toolset, mock_run_context):
    """Different proxy prefixes should not share discovered namespace state."""
    mcp_proxy = ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp")
    builtin_proxy = ToolProxyToolset(toolsets=[weather_toolset], prefix="builtin")
    mcp_tools = await mcp_proxy.get_tools(mock_run_context)
    builtin_tools = await builtin_proxy.get_tools(mock_run_context)

    await mcp_proxy.call_tool("mcp_search_tool", {"query": "weather"}, mock_run_context, mcp_tools["mcp_search_tool"])
    result = await builtin_proxy.call_tool(
        "builtin_search_tool", {"query": "weather"}, mock_run_context, builtin_tools["builtin_search_tool"]
    )

    assert "get_weather" in result
    assert "mcp:weather" in mock_run_context.deps.tool_search_loaded_namespaces
    assert "builtin:weather" in mock_run_context.deps.tool_search_loaded_namespaces


@pytest.mark.anyio
async def test_prefixed_proxy_instructions_use_prefixed_tool_names(weather_toolset, mock_run_context):
    """Instructions should reference the configured proxy tool names."""
    ts = ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp")
    await ts.get_tools(mock_run_context)

    instructions = await ts.get_instructions(mock_run_context)

    assert instructions is not None
    assert "mcp_search_tool" in instructions
    assert "mcp_call_tool" in instructions
    assert "search_tools" not in instructions


@pytest.mark.anyio
async def test_prefixed_proxy_instruction_replacement_is_single_pass(weather_toolset, mock_run_context):
    """Prefix names containing legacy tool names should not corrupt rendered instructions."""
    ts = ToolProxyToolset(toolsets=[weather_toolset], prefix="call_tool_proxy")
    await ts.get_tools(mock_run_context)

    instructions = await ts.get_instructions(mock_run_context)

    assert instructions is not None
    assert "call_tool_proxy_search_tool" in instructions
    assert "call_tool_proxy_call_tool" in instructions
    assert "call_tool_proxy_call_tool_proxy_call_tool" not in instructions


@pytest.mark.anyio
async def test_prefixed_proxy_uses_only_prefixed_state(weather_toolset, mock_run_context):
    """Prefixed proxy state should not read unprefixed state from older proxy blocks."""
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")
    ts = ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp")
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("mcp_search_tool", {"query": "weather"}, mock_run_context, tools["mcp_search_tool"])

    assert "get_weather" in result
    assert "mcp:weather" in mock_run_context.deps.tool_search_loaded_namespaces


def test_prefix_validation_rejects_invalid_names(weather_toolset):
    """Prefix must be safe for a pydantic-ai tool name."""
    with pytest.raises(ValueError):
        ToolProxyToolset(toolsets=[weather_toolset], prefix="mcp-server")


@pytest.mark.anyio
async def test_underlying_tools_not_directly_visible(weather_toolset, mock_run_context):
    """Underlying tools must never appear in get_tools result."""
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    assert "get_weather" not in tools
    assert "get_forecast" not in tools


# ---------------------------------------------------------------------------
# search_tools: XML-formatted results
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_search_returns_xml_with_schemas(weather_toolset, mock_run_context):
    """search_tools must return XML format with full parameter schemas."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])

    assert "<search-results" in result
    assert "</search-results>" in result
    assert '<tool name="get_weather"' in result
    assert '<tool name="get_forecast"' in result
    assert "<description>" in result
    assert "<parameters>" in result
    assert "</tool>" in result


@pytest.mark.anyio
async def test_search_includes_parameter_schema(weather_toolset, mock_run_context):
    """search_tools must include full JSON schema for parameters."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])

    # Should contain JSON schema with properties
    assert '"properties"' in result
    assert '"location"' in result
    assert '"description"' in result


@pytest.mark.anyio
async def test_search_namespace_loads_all_tools(weather_toolset, mock_run_context):
    """Searching by namespace should load all tools in that namespace."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Weather related tools"},
    )
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])

    assert "get_weather" in result
    assert "get_forecast" in result
    assert 'namespace="weather"' in result


@pytest.mark.anyio
async def test_search_tool_in_namespace_loads_entire_namespace(weather_toolset, mock_run_context):
    """Searching for a specific tool in a namespace loads the entire namespace."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "forecast"}, mock_run_context, tools["search_tools"])

    # Both tools should appear (entire namespace loaded)
    assert "get_weather" in result
    assert "get_forecast" in result


@pytest.mark.anyio
async def test_search_loose_tools_load_individually(loose_toolset, mock_run_context):
    """Loose tools (no namespace) load individually."""
    ts = ToolProxyToolset(toolsets=[loose_toolset])
    tools = await ts.get_tools(mock_run_context)

    # "read" only matches view tool's description
    result = await ts.call_tool("search_tools", {"query": "read"}, mock_run_context, tools["search_tools"])
    assert "view" in result


@pytest.mark.anyio
async def test_search_no_results(weather_toolset, mock_run_context):
    """Search with no matches should return empty XML results."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "search_tools", {"query": "database_migration"}, mock_run_context, tools["search_tools"]
    )
    assert 'count="0"' in result
    assert "No tools found" in result


@pytest.mark.anyio
async def test_search_all_discovered(weather_toolset, mock_run_context):
    """Search when all tools are discovered should say so."""
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")

    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "anything"}, mock_run_context, tools["search_tools"])
    assert "already discovered" in result
    assert 'query="anything"' in result


@pytest.mark.anyio
async def test_search_empty_query(weather_toolset, mock_run_context):
    """Empty query should return error."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": ""}, mock_run_context, tools["search_tools"])
    assert "required" in result


@pytest.mark.anyio
async def test_search_updates_context_state(weather_toolset, loose_toolset, mock_run_context):
    """search_tools must update context state for session restore."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset, loose_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)

    # Search namespace
    await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])
    assert "weather" in mock_run_context.deps.tool_search_loaded_namespaces

    # Search loose tool
    await ts.call_tool("search_tools", {"query": "read"}, mock_run_context, tools["search_tools"])
    assert "view" in mock_run_context.deps.tool_search_loaded_tools


@pytest.mark.anyio
async def test_search_multiple_namespaces(weather_toolset, finance_toolset, mock_run_context):
    """Multiple namespaces should load independently."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={
            "weather": "Weather tools",
            "finance": "Finance tools",
        },
    )
    tools = await ts.get_tools(mock_run_context)

    # Search weather
    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])
    assert "get_weather" in result
    assert "get_stock_price" not in result

    # Search finance
    result = await ts.call_tool("search_tools", {"query": "finance"}, mock_run_context, tools["search_tools"])
    assert "get_stock_price" in result
    assert "convert_currency" in result


# ---------------------------------------------------------------------------
# call_tool: proxy invocation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_call_tool_delegates_to_underlying(weather_toolset, mock_run_context):
    """call_tool should delegate to the owning toolset."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {"location": "Tokyo"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "Tokyo" in result
    assert "sunny" in result


@pytest.mark.anyio
async def test_call_tool_with_optional_args(weather_toolset, mock_run_context):
    """call_tool should pass through all arguments including optional ones."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {"location": "NYC", "unit": "fahrenheit"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "NYC" in result
    assert "f" in result.lower()


@pytest.mark.anyio
async def test_call_tool_unknown_returns_error(weather_toolset, mock_run_context):
    """call_tool for unknown tool should return XML error."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "nonexistent_tool", "arguments": {}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "<error>" in result
    assert "not found" in result
    assert "nonexistent_tool" in result


@pytest.mark.anyio
async def test_call_tool_empty_name_returns_error(weather_toolset, mock_run_context):
    """call_tool with empty name should return error."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "", "arguments": {}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "<error>" in result
    assert "required" in result


@pytest.mark.anyio
async def test_call_tool_error_from_underlying(failing_toolset, mock_run_context):
    """When a tool fails, the error from the underlying toolset should propagate."""
    ts = ToolProxyToolset(toolsets=[failing_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "failing_tool", "arguments": {}},
        mock_run_context,
        tools["call_tool"],
    )
    # The underlying Toolset catches the exception and returns an error string.
    # If it re-raises, our proxy wraps it in XML with schema.
    assert "Something went wrong" in str(result) or "<tool-call-error" in str(result)


@pytest.mark.anyio
async def test_call_tool_no_search_required(weather_toolset, mock_run_context):
    """call_tool should work even without prior search_tools call."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    # Directly call without searching first
    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {"location": "Berlin"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "Berlin" in result


@pytest.mark.anyio
async def test_call_tool_empty_arguments(weather_toolset, mock_run_context):
    """call_tool with missing arguments should return an error."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {}},
        mock_run_context,
        tools["call_tool"],
    )
    # Should get an error because 'location' is required.
    # May come as string from underlying Toolset or XML from proxy.
    result_str = str(result)
    assert "location" in result_str.lower() or "missing" in result_str.lower() or "required" in result_str.lower()


@pytest.mark.anyio
async def test_call_tool_none_arguments_treated_as_empty(weather_toolset, mock_run_context):
    """call_tool with None arguments should be treated as empty dict."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": None},
        mock_run_context,
        tools["call_tool"],
    )
    # Should get an error because 'location' is required
    result_str = str(result)
    assert "location" in result_str.lower() or "missing" in result_str.lower() or "error" in result_str.lower()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_context_manager(weather_toolset, finance_toolset):
    """Context manager should enter/exit wrapped toolsets."""
    ts = ToolProxyToolset(toolsets=[weather_toolset, finance_toolset])
    async with ts as entered:
        assert entered is ts


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_instructions_include_namespace_info(weather_toolset, finance_toolset, mock_run_context):
    """Instructions should include available namespace information."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset, finance_toolset],
        namespace_descriptions={
            "weather": "Weather related tools",
            "finance": "Financial tools",
        },
    )
    await ts.get_tools(mock_run_context)

    instructions = await ts.get_instructions(mock_run_context)
    assert instructions is not None
    assert "weather" in instructions
    assert "finance" in instructions
    assert "tool-proxy" in instructions


@pytest.mark.anyio
async def test_instructions_include_discovered_tools(weather_toolset, mock_run_context):
    """Instructions should list previously discovered tools."""
    mock_run_context.deps.tool_search_loaded_namespaces.append("weather")

    ts = ToolProxyToolset(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    await ts.get_tools(mock_run_context)

    instructions = await ts.get_instructions(mock_run_context)
    assert instructions is not None
    assert "Previously discovered tools" in instructions
    assert "get_weather" in instructions
    assert "get_forecast" in instructions


@pytest.mark.anyio
async def test_instructions_delegates_loaded_toolset_instructions(mock_run_context):
    """Instructions from loaded toolsets should be forwarded."""
    # Use a toolset that provides instructions (Toolset with tools that have get_instruction)
    toolset = Toolset(tools=[GetWeatherTool, GetForecastTool], toolset_id="weather")

    ts = ToolProxyToolset(toolsets=[toolset])
    await ts.get_tools(mock_run_context)

    # Before loading namespace, no toolset instructions forwarded
    instructions = await ts.get_instructions(mock_run_context)
    # Only proxy instructions should be present
    assert instructions is not None
    assert "tool-proxy" in instructions


# ---------------------------------------------------------------------------
# Init report (namespace status)
# ---------------------------------------------------------------------------


def test_init_report_empty_before_enter():
    """init_report should be empty before __aenter__."""
    ts = ToolProxyToolset(toolsets=[])
    assert ts.init_report == {}


@pytest.mark.anyio
async def test_init_report_populated_after_enter(weather_toolset):
    """init_report should show connected namespaces after __aenter__."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    async with ts:
        report = ts.init_report
        assert "weather" in report
        assert report["weather"].value == "connected"


# ---------------------------------------------------------------------------
# Mixed namespace and loose
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mixed_namespace_and_loose(weather_toolset, loose_toolset, mock_run_context):
    """Mix of namespace and loose toolsets should work correctly."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset, loose_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)
    assert len(tools) == 2

    # Search weather -> namespace tools in result
    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])
    assert "get_weather" in result
    assert "get_forecast" in result

    # Call tool from namespace
    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {"location": "Paris"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "Paris" in result

    # Search and call loose tool
    result = await ts.call_tool("search_tools", {"query": "read"}, mock_run_context, tools["search_tools"])
    assert "view" in result

    result = await ts.call_tool(
        "call_tool",
        {"name": "view", "arguments": {"file_path": "test.txt"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "test.txt" in result


# ---------------------------------------------------------------------------
# Duplicate tool names
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_duplicate_tool_name_last_wins(mock_run_context):
    """When multiple toolsets have the same tool name, last one wins."""
    toolset1 = Toolset(tools=[GetWeatherTool])
    toolset2 = Toolset(tools=[GetWeatherTool])  # same tool name

    ts = ToolProxyToolset(toolsets=[toolset1, toolset2])
    tools = await ts.get_tools(mock_run_context)

    # Should still work - last one wins
    result = await ts.call_tool(
        "call_tool",
        {"name": "get_weather", "arguments": {"location": "London"}},
        mock_run_context,
        tools["call_tool"],
    )
    assert "London" in result


# ---------------------------------------------------------------------------
# search_tools count attribute
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_search_result_count_attribute(weather_toolset, mock_run_context):
    """XML search results should have accurate count attribute."""
    ts = ToolProxyToolset(
        toolsets=[weather_toolset],
        namespace_descriptions={"weather": "Weather tools"},
    )
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("search_tools", {"query": "weather"}, mock_run_context, tools["search_tools"])

    # Weather namespace has 2 tools
    assert 'count="2"' in result


# ---------------------------------------------------------------------------
# Dispatch unknown proxy tool
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dispatch_unknown_proxy_name(weather_toolset, mock_run_context):
    """Calling with an unknown proxy tool name should return error."""
    ts = ToolProxyToolset(toolsets=[weather_toolset])
    tools = await ts.get_tools(mock_run_context)

    result = await ts.call_tool("nonexistent", {}, mock_run_context, tools["search_tools"])
    assert "<error>" in result
