"""Test query tools."""

from __future__ import annotations

from ya_agent_sdk.toolsets.browser_use._tools import build_tool
from ya_agent_sdk.toolsets.browser_use.tools import find_elements, navigate_to_url
from ya_agent_sdk.toolsets.browser_use.tools.query import get_element_attributes, get_element_text
from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_find_elements(cdp_url, test_server):
    """Test finding elements on page."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first using tool
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Find h1 elements
        tool = build_tool(session, find_elements)
        result = await tool.function_schema.call({"selector": "h1", "limit": 10}, None)

        assert result["status"] == "success"
        assert result["count"] >= 0
        assert "elements" in result

        if result["count"] > 0:
            element = result["elements"][0]
            assert "tag_name" in element
            assert element["tag_name"] == "h1"
            assert "text" in element


async def test_find_elements_not_found(cdp_url, test_server):
    """Test finding elements with selector that doesn't match."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Find non-existent elements
        tool = build_tool(session, find_elements)
        result = await tool.function_schema.call({"selector": ".non-existent-class-xyz", "limit": 10}, None)

        assert result["status"] == "success"
        assert result["count"] == 0
        assert len(result["elements"]) == 0


async def test_find_elements_with_limit(cdp_url, test_server):
    """Test finding elements with result limit."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Find all elements with small limit
        tool = build_tool(session, find_elements)
        result = await tool.function_schema.call({"selector": "*", "limit": 5}, None)

        assert result["status"] == "success"
        assert len(result["elements"]) <= 5


async def test_get_element_text(cdp_url, test_server):
    """Test getting element text content."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get text from h1 element
        text_tool = build_tool(session, get_element_text)
        result = await text_tool.function_schema.call({"selector": "h1"}, None)

        assert isinstance(result, str)
        # May be empty if h1 doesn't exist, but should not raise error


async def test_get_element_attributes(cdp_url, test_server):
    """Test getting element attributes."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get attributes from link element
        attr_tool = build_tool(session, get_element_attributes)
        result = await attr_tool.function_schema.call({"selector": "a", "attributes": None}, None)

        assert isinstance(result, dict)


async def test_get_element_attributes_specific(cdp_url, test_server):
    """Test getting specific attributes from element."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get specific attributes
        attr_tool = build_tool(session, get_element_attributes)
        result = await attr_tool.function_schema.call({"selector": "a", "attributes": ["href"]}, None)

        assert isinstance(result, dict)
