"""Test dialog handling tools."""

from __future__ import annotations

from ya_agent_sdk.toolsets.browser_use._tools import build_tool
from ya_agent_sdk.toolsets.browser_use.tools import navigate_to_url
from ya_agent_sdk.toolsets.browser_use.tools.dialog import accept_dialog, dismiss_dialog, handle_dialog
from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_handle_dialog_no_dialog(cdp_url, test_server):
    """Test handling dialog when no dialog is present."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to handle dialog with short timeout
        dialog_tool = build_tool(session, handle_dialog)
        result = await dialog_tool.function_schema.call({"accept": True, "timeout": 500}, None)

        # Should report no dialog found
        assert result["status"] in ["no_dialog", "success"]


async def test_accept_dialog_convenience(cdp_url, test_server):
    """Test accept_dialog convenience function."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to accept dialog
        dialog_tool = build_tool(session, accept_dialog)
        result = await dialog_tool.function_schema.call({"timeout": 500}, None)

        assert result["status"] in ["no_dialog", "success"]


async def test_dismiss_dialog_convenience(cdp_url, test_server):
    """Test dismiss_dialog convenience function."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to dismiss dialog
        dialog_tool = build_tool(session, dismiss_dialog)
        result = await dialog_tool.function_schema.call({"timeout": 500}, None)

        assert result["status"] in ["no_dialog", "success"]


async def test_handle_dialog_with_prompt_text(cdp_url, test_server):
    """Test handling dialog with prompt text."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to handle dialog with prompt text
        dialog_tool = build_tool(session, handle_dialog)
        result = await dialog_tool.function_schema.call(
            {"accept": True, "prompt_text": "test input", "timeout": 500}, None
        )

        assert result["status"] in ["no_dialog", "success"]
        if result["status"] == "success":
            assert result.get("prompt_text") == "test input"
