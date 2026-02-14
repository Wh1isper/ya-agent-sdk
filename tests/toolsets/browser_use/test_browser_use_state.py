"""Test state inspection tools."""

from __future__ import annotations

import io

from PIL import Image
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ToolReturn

from ya_agent_sdk.toolsets.browser_use._tools import build_tool
from ya_agent_sdk.toolsets.browser_use.tools import (
    get_page_content,
    get_page_info,
    navigate_to_url,
    take_screenshot,
)
from ya_agent_sdk.toolsets.browser_use.tools.state import get_viewport_info, take_element_screenshot
from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_get_page_info(cdp_url, test_server):
    """Test getting page information."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first using tool
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get page info
        tool = build_tool(session, get_page_info)
        result = await tool.function_schema.call({}, None)

        # Verify structure
        assert "url" in result
        assert "title" in result
        assert "ready_state" in result
        assert "viewport" in result
        assert "basic.html" in result["url"]


async def test_get_page_content(cdp_url, test_server):
    """Test getting page content."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first using tool
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get text content
        tool = build_tool(session, get_page_content)
        result = await tool.function_schema.call({"content_format": "text"}, None)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Example" in result or "example" in result.lower()


async def test_get_page_content_html(cdp_url, test_server):
    """Test getting page HTML content."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Get HTML content
        tool = build_tool(session, get_page_content)
        result = await tool.function_schema.call({"content_format": "html"}, None)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "<html" in result.lower() or "<!doctype" in result.lower()


async def test_take_screenshot(cdp_url, test_server):
    """Test screenshot capture with ToolReturn structure."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first using tool
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Take screenshot
        tool = build_tool(session, take_screenshot)
        result = await tool.function_schema.call({"full_page": False}, None)

        # Verify it's a ToolReturn
        assert isinstance(result, ToolReturn)

        # Verify return_value structure
        assert "status" in result.return_value
        assert result.return_value["status"] == "success"
        assert "url" in result.return_value
        assert "segments_count" in result.return_value
        assert result.return_value["segments_count"] > 0

        # Verify content structure (multi-modal)
        assert isinstance(result.content, list)
        assert len(result.content) > 0
        assert len(result.content) <= 20

        # All content items should be BinaryContent
        for item in result.content:
            assert isinstance(item, BinaryContent)
            assert item.media_type == "image/png"

        # Verify image data is valid
        first_image = result.content[0]
        img = Image.open(io.BytesIO(first_image.data))
        assert img.size[0] > 0
        assert img.size[1] > 0


async def test_take_screenshot_full_page(cdp_url, test_server):
    """Test taking full page screenshot."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Take full page screenshot
        tool = build_tool(session, take_screenshot)
        result = await tool.function_schema.call({"full_page": True, "img_format": "image/png"}, None)

        assert isinstance(result, ToolReturn)
        assert result.return_value["status"] == "success"
        assert result.return_value["full_page"] is True
        assert len(result.content) > 0


async def test_take_screenshot_jpeg(cdp_url, test_server):
    """Test taking screenshot in JPEG format."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Take JPEG screenshot
        tool = build_tool(session, take_screenshot)
        result = await tool.function_schema.call({"full_page": False, "img_format": "image/jpeg"}, None)

        assert isinstance(result, ToolReturn)
        assert result.return_value["status"] == "success"
        assert result.return_value["format"] == "jpeg"


async def test_take_element_screenshot(cdp_url, test_server):
    """Test taking screenshot of specific element."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Take element screenshot
        tool = build_tool(session, take_element_screenshot)
        result = await tool.function_schema.call({"selector": "h1", "img_format": "image/png"}, None)

        assert isinstance(result, ToolReturn)
        # Status could be success or not_found
        assert result.return_value["status"] in ["success", "not_found"]


async def test_take_element_screenshot_not_found(cdp_url, test_server):
    """Test taking screenshot of non-existent element."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try non-existent element
        tool = build_tool(session, take_element_screenshot)
        result = await tool.function_schema.call({"selector": ".non-existent-xyz", "img_format": "image/png"}, None)

        assert isinstance(result, ToolReturn)
        assert result.return_value["status"] == "not_found"
        assert len(result.content) == 0


async def test_get_viewport_info(cdp_url, test_server):
    """Test getting viewport information."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Get viewport info
        tool = build_tool(session, get_viewport_info)
        result = await tool.function_schema.call({}, None)

        assert isinstance(result, dict)
        assert "width" in result
        assert "height" in result
        assert result["width"] > 0
        assert result["height"] > 0
