"""Test validation tools."""

from __future__ import annotations

from ya_agent_sdk.toolsets.browser_use._tools import build_tool
from ya_agent_sdk.toolsets.browser_use.tools import execute_javascript, navigate_to_url
from ya_agent_sdk.toolsets.browser_use.tools.validation import is_checked, is_enabled, is_visible
from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_is_visible_true(cdp_url, test_server):
    """Test checking if a visible element is visible."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Check if h1 is visible
        visible_tool = build_tool(session, is_visible)
        result = await visible_tool.function_schema.call({"selector": "h1"}, None)

        assert result["status"] == "success"
        assert result["result"] is True


async def test_is_visible_false(cdp_url, test_server):
    """Test checking if a hidden element is visible."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create a hidden element
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const div = document.createElement('div');
            div.id = 'hidden-element';
            div.style.display = 'none';
            document.body.appendChild(div);
        """
            },
            None,
        )

        # Check if hidden element is visible
        visible_tool = build_tool(session, is_visible)
        result = await visible_tool.function_schema.call({"selector": "#hidden-element"}, None)

        assert result["status"] == "success"
        assert result["result"] is False


async def test_is_enabled_true(cdp_url, test_server):
    """Test checking if an enabled element is enabled."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create an enabled input
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const input = document.createElement('input');
            input.id = 'enabled-input';
            document.body.appendChild(input);
        """
            },
            None,
        )

        # Check if input is enabled
        enabled_tool = build_tool(session, is_enabled)
        result = await enabled_tool.function_schema.call({"selector": "#enabled-input"}, None)

        assert result["status"] == "success"
        assert result["result"] is True


async def test_is_enabled_false(cdp_url, test_server):
    """Test checking if a disabled element is enabled."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create a disabled input
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const input = document.createElement('input');
            input.id = 'disabled-input';
            input.disabled = true;
            document.body.appendChild(input);
        """
            },
            None,
        )

        # Check if input is enabled
        enabled_tool = build_tool(session, is_enabled)
        result = await enabled_tool.function_schema.call({"selector": "#disabled-input"}, None)

        assert result["status"] == "success"
        assert result["result"] is False


async def test_is_checked_true(cdp_url, test_server):
    """Test checking if a checked checkbox is checked."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create a checked checkbox
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = 'checked-checkbox';
            checkbox.checked = true;
            document.body.appendChild(checkbox);
        """
            },
            None,
        )

        # Check if checkbox is checked
        checked_tool = build_tool(session, is_checked)
        result = await checked_tool.function_schema.call({"selector": "#checked-checkbox"}, None)

        assert result["status"] == "success"
        assert result["result"] is True


async def test_is_checked_false(cdp_url, test_server):
    """Test checking if an unchecked checkbox is checked."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create an unchecked checkbox
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = 'unchecked-checkbox';
            document.body.appendChild(checkbox);
        """
            },
            None,
        )

        # Check if checkbox is checked
        checked_tool = build_tool(session, is_checked)
        result = await checked_tool.function_schema.call({"selector": "#unchecked-checkbox"}, None)

        assert result["status"] == "success"
        assert result["result"] is False


async def test_is_visible_not_found(cdp_url, test_server):
    """Test checking visibility of non-existent element."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Check non-existent element
        visible_tool = build_tool(session, is_visible)
        result = await visible_tool.function_schema.call({"selector": "#non-existent"}, None)

        assert result["status"] == "not_found"


async def test_is_checked_wrong_element_type(cdp_url, test_server):
    """Test checking if a non-checkbox element is checked."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create a text input (not checkbox)
        js_tool = build_tool(session, execute_javascript)
        await js_tool.function_schema.call(
            {
                "script": """
            const input = document.createElement('input');
            input.type = 'text';
            input.id = 'text-input';
            document.body.appendChild(input);
        """
            },
            None,
        )

        # Try to check if text input is checked
        checked_tool = build_tool(session, is_checked)
        result = await checked_tool.function_schema.call({"selector": "#text-input"}, None)

        assert result["status"] == "error"
        assert "error_message" in result
