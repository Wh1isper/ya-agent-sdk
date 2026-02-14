"""Test navigation tools."""

from __future__ import annotations

from ya_agent_sdk.toolsets.browser_use._tools import build_tool
from ya_agent_sdk.toolsets.browser_use.tools import go_back, go_forward, navigate_to_url, reload_page
from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_navigate_to_url(cdp_url, test_server):
    """Test navigation to a URL."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Build and call tool
        tool = build_tool(session, navigate_to_url)
        result = await tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Verify result structure
        assert result["status"] == "success"
        assert "basic.html" in result["url"]
        assert result["title"] != ""

        # Verify session state updated
        assert "basic.html" in session.current_url
        assert session.current_title != ""
        assert len(session.navigation_history) > 0


async def test_go_back(cdp_url, test_server):
    """Test going back in navigation history."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to first page
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Navigate to second page
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/navigation/page1.html"}, None)

        # Go back
        back_tool = build_tool(session, go_back)
        result = await back_tool.function_schema.call({}, None)

        assert result["status"] == "success"
        assert "basic.html" in result["url"]


async def test_go_forward(cdp_url, test_server):
    """Test going forward in navigation history."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to pages and go back
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/navigation/page1.html"}, None)

        back_tool = build_tool(session, go_back)
        await back_tool.function_schema.call({}, None)

        # Go forward
        forward_tool = build_tool(session, go_forward)
        result = await forward_tool.function_schema.call({}, None)

        assert result["status"] == "success"
        assert "page1.html" in result["url"]


async def test_reload_page(cdp_url, test_server):
    """Test reloading the current page."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Reload page
        reload_tool = build_tool(session, reload_page)
        result = await reload_tool.function_schema.call({"ignore_cache": False}, None)

        assert result["status"] == "success"
        assert "basic.html" in result["url"]


async def test_reload_page_ignore_cache(cdp_url, test_server):
    """Test reloading page with cache ignored."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        reload_tool = build_tool(session, reload_page)
        result = await reload_tool.function_schema.call({"ignore_cache": True}, None)

        assert result["status"] == "success"


async def test_go_back_no_history(cdp_url, test_server):
    """Test going back when already at the first page."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to only one page
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to go back - may succeed (about:blank) or fail depending on browser state
        back_tool = build_tool(session, go_back)
        result = await back_tool.function_schema.call({}, None)

        # Just verify the call completed (success or error both valid)
        assert result["status"] in ["success", "error"]


async def test_go_forward_no_history(cdp_url, test_server):
    """Test going forward when already at the last page."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to a page (no forward history)
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Try to go forward (should fail)
        forward_tool = build_tool(session, go_forward)
        result = await forward_tool.function_schema.call({}, None)

        # Should return error for no next page
        assert result["status"] == "error"
        assert "history" in result.get("error_message", "").lower()


async def test_navigate_with_timeout(cdp_url, test_server):
    """Test navigation with very short timeout to trigger timeout handling."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Use extremely short timeout (1ms) to trigger timeout path
        nav_tool = build_tool(session, navigate_to_url)
        result = await nav_tool.function_schema.call(
            {"url": f"{test_server}/test_fixtures/basic.html", "timeout": 1}, None
        )

        # Should still succeed but with timeout warning logged
        # The function continues after timeout to get page info
        assert result["status"] == "success"
        assert "basic.html" in result["url"]


async def test_go_back_with_timeout(cdp_url, test_server):
    """Test go_back with timeout during history navigation."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to multiple pages to build history
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/navigation/page1.html"}, None)

        # Monkey patch _wait_for_page_ready to trigger timeout for go_back
        from ya_agent_sdk.toolsets.browser_use.tools import navigation

        original_wait = navigation._wait_for_page_ready

        async def mock_wait_timeout(*args, **kwargs):
            raise TimeoutError("Simulated timeout")

        navigation._wait_for_page_ready = mock_wait_timeout

        try:
            back_tool = build_tool(session, go_back)
            result = await back_tool.function_schema.call({}, None)

            # Should still succeed despite timeout
            assert result["status"] == "success"
            assert "basic.html" in result["url"]
        finally:
            # Restore original function
            navigation._wait_for_page_ready = original_wait


async def test_go_forward_with_timeout(cdp_url, test_server):
    """Test go_forward with timeout during history navigation."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate and go back to build forward history
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/navigation/page1.html"}, None)

        back_tool = build_tool(session, go_back)
        await back_tool.function_schema.call({}, None)

        # Monkey patch _wait_for_page_ready to trigger timeout for go_forward
        from ya_agent_sdk.toolsets.browser_use.tools import navigation

        original_wait = navigation._wait_for_page_ready

        async def mock_wait_timeout(*args, **kwargs):
            raise TimeoutError("Simulated timeout")

        navigation._wait_for_page_ready = mock_wait_timeout

        try:
            forward_tool = build_tool(session, go_forward)
            result = await forward_tool.function_schema.call({}, None)

            # Should still succeed despite timeout
            assert result["status"] == "success"
            assert "page1.html" in result["url"]
        finally:
            # Restore original function
            navigation._wait_for_page_ready = original_wait


async def test_reload_page_with_timeout(cdp_url, test_server):
    """Test reload_page with timeout during reload wait."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Monkey patch _wait_for_page_ready to trigger timeout for reload
        from ya_agent_sdk.toolsets.browser_use.tools import navigation

        original_wait = navigation._wait_for_page_ready

        async def mock_wait_timeout(*args, **kwargs):
            raise TimeoutError("Simulated timeout")

        navigation._wait_for_page_ready = mock_wait_timeout

        try:
            reload_tool = build_tool(session, reload_page)
            result = await reload_tool.function_schema.call({"ignore_cache": False}, None)

            # Should still succeed despite timeout
            assert result["status"] == "success"
            assert "basic.html" in result["url"]
        finally:
            # Restore original function
            navigation._wait_for_page_ready = original_wait
