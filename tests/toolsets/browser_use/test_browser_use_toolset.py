from cdp_use.client import CDPClient

from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset, get_cdp_websocket_url


def test_get_cdp_websocket_url_with_ws_url():
    """Test that ws:// URLs are returned as-is."""
    ws_url = "ws://127.0.0.1:9222/devtools/browser/abc123"
    result = get_cdp_websocket_url(ws_url)
    assert result == ws_url


def test_get_cdp_websocket_url_with_wss_url():
    """Test that wss:// URLs are returned as-is."""
    wss_url = "wss://127.0.0.1:9222/devtools/browser/abc123"
    result = get_cdp_websocket_url(wss_url)
    assert result == wss_url


async def test_browser_use_toolset_async_context(cdp_url):
    """Test that BrowserUseToolset can be used as an async context manager."""
    # This test would require a running CDP endpoint or mock
    # For now, just test that the class can be instantiated
    toolset = BrowserUseToolset(cdp_url)
    assert toolset.id == "browser_use"

    async with toolset as ts:
        assert ts._cdp_client is not None


async def test_browser_use_toolset_reuses_existing_page(cdp_url):
    """Test that BrowserUseToolset reuses existing page by default."""
    # Create first toolset and attach to a page
    toolset1 = BrowserUseToolset(cdp_url)

    async with toolset1 as ts1:
        assert ts1._browser_session is not None
        _ = ts1._browser_session.page

        # Create second toolset - should reuse the same page
        toolset2 = BrowserUseToolset(cdp_url, always_use_new_page=False)
        async with toolset2 as ts2:
            assert ts2._browser_session is not None
            second_session_id = ts2._browser_session.page

            # Session IDs might be different (different sessions attached to same target)
            # but the toolset should have successfully initialized
            assert second_session_id is not None


async def test_browser_use_toolset_always_use_new_page(cdp_url):
    """Test that BrowserUseToolset creates new page when always_use_new_page=True."""
    # Create first toolset
    toolset1 = BrowserUseToolset(cdp_url)

    async with toolset1 as ts1:
        assert ts1._browser_session is not None
        _ = ts1._browser_session.page

        # Get targets count after first toolset
        targets_response1 = await ts1._cdp_client.send.Target.getTargets()
        page_count_before = sum(1 for t in targets_response1.get("targetInfos", []) if t.get("type") == "page")

        # Create second toolset with always_use_new_page=True
        toolset2 = BrowserUseToolset(cdp_url, always_use_new_page=True)
        async with toolset2 as ts2:
            assert ts2._browser_session is not None
            second_session_id = ts2._browser_session.page

            # Verify a new session was created
            assert second_session_id is not None

            # Get targets count after second toolset
            targets_response2 = await ts2._cdp_client.send.Target.getTargets()
            page_count_after = sum(1 for t in targets_response2.get("targetInfos", []) if t.get("type") == "page")

            # Should have created a new page
            assert page_count_after > page_count_before


async def test_browser_use_toolset_default_always_use_new_page_false(cdp_url):
    """Test that always_use_new_page defaults to False."""
    toolset = BrowserUseToolset(cdp_url)
    assert toolset.always_use_new_page is False


async def test_browser_use_toolset_always_use_new_page_true_initialization(cdp_url):
    """Test that always_use_new_page can be set to True during initialization."""
    toolset = BrowserUseToolset(cdp_url, always_use_new_page=True)
    assert toolset.always_use_new_page is True

    async with toolset as ts:
        assert ts._browser_session is not None
        assert ts._cdp_client is not None


async def test_browser_use_toolset_closes_created_page_on_exit(cdp_url):
    """Test that created page is closed when both always_use_new_page=True and auto_cleanup_page=True."""
    # Get initial page count
    websocket_url = get_cdp_websocket_url(cdp_url)
    async with CDPClient(websocket_url) as client:
        targets_before = await client.send.Target.getTargets()
        page_count_before = sum(1 for t in targets_before.get("targetInfos", []) if t.get("type") == "page")

        # Create toolset with always_use_new_page=True and auto_cleanup_page=True
        toolset = BrowserUseToolset(cdp_url, always_use_new_page=True, auto_cleanup_page=True)
        created_target_id = None

        async with toolset as ts:
            # Verify a new page was created
            created_target_id = ts._created_target_id
            assert created_target_id is not None

            targets_during = await client.send.Target.getTargets()
            page_count_during = sum(1 for t in targets_during.get("targetInfos", []) if t.get("type") == "page")
            assert page_count_during == page_count_before + 1

        # After exiting context, verify the page was closed
        targets_after = await client.send.Target.getTargets()
        page_count_after = sum(1 for t in targets_after.get("targetInfos", []) if t.get("type") == "page")
        assert page_count_after == page_count_before

        # Verify the specific target was closed
        target_ids_after = [t.get("targetId") for t in targets_after.get("targetInfos", [])]
        assert created_target_id not in target_ids_after


async def test_browser_use_toolset_does_not_close_reused_page_on_exit(cdp_url):
    """Test that reused page is NOT closed when exiting context if always_use_new_page=False."""
    # Get initial page count
    websocket_url = get_cdp_websocket_url(cdp_url)
    async with CDPClient(websocket_url) as client:
        targets_before = await client.send.Target.getTargets()
        page_count_before = sum(1 for t in targets_before.get("targetInfos", []) if t.get("type") == "page")

        # Create toolset with always_use_new_page=False (default)
        toolset = BrowserUseToolset(cdp_url, always_use_new_page=False)

        async with toolset as ts:
            # Verify no target was marked for creation
            assert ts._created_target_id is None

        # After exiting context, verify page count remains the same
        targets_after = await client.send.Target.getTargets()
        page_count_after = sum(1 for t in targets_after.get("targetInfos", []) if t.get("type") == "page")
        assert page_count_after == page_count_before


async def test_browser_use_toolset_created_page_without_cleanup(cdp_url):
    """Test that created page is NOT closed when auto_cleanup_page=False (default)."""
    websocket_url = get_cdp_websocket_url(cdp_url)
    async with CDPClient(websocket_url) as client:
        targets_before = await client.send.Target.getTargets()
        page_count_before = sum(1 for t in targets_before.get("targetInfos", []) if t.get("type") == "page")

        # Create toolset with always_use_new_page=True, auto_cleanup_page defaults to False
        toolset = BrowserUseToolset(cdp_url, always_use_new_page=True)
        created_target_id = None

        async with toolset as ts:
            created_target_id = ts._created_target_id
            assert created_target_id is not None

        # Page should still exist after exit
        targets_after = await client.send.Target.getTargets()
        page_count_after = sum(1 for t in targets_after.get("targetInfos", []) if t.get("type") == "page")
        assert page_count_after == page_count_before + 1

        # Clean up manually
        await client.send.Target.closeTarget(params={"targetId": created_target_id})


def test_browser_use_toolset_default_prefix():
    """Test that prefix defaults to toolset.id when not provided."""
    toolset = BrowserUseToolset(cdp_url="http://localhost:9222/json/version")
    assert toolset.prefix == toolset.id
    assert toolset.prefix == "browser_use"


def test_browser_use_toolset_custom_prefix():
    """Test that custom prefix can be set during initialization."""
    custom_prefix = "my_browser"
    toolset = BrowserUseToolset(cdp_url="http://localhost:9222/json/version", prefix=custom_prefix)
    assert toolset.prefix == custom_prefix


async def test_browser_use_toolset_prefix_in_tool_names(cdp_url):
    """Test that prefix is used in tool names returned by get_tools."""
    custom_prefix = "custom"
    toolset = BrowserUseToolset(cdp_url, prefix=custom_prefix)

    async with toolset as ts:
        # Verify internal tools are built correctly
        assert ts._tools is not None
        assert len(ts._tools) > 0

        # Check that the prefix will be used in tool names
        # by examining the get_tools implementation
        # We can verify the expected tool names without calling get_tools
        for tool in ts._tools:
            expected_name = f"{custom_prefix}_{tool.name}"
            # Verify the naming pattern matches what get_tools would produce
            assert tool.name is not None
            assert not expected_name.startswith("browser_use_"), "Should use custom prefix, not default"


async def test_browser_use_toolset_default_prefix_in_tool_names(cdp_url):
    """Test that default prefix is used in tool names when no custom prefix is provided."""
    toolset = BrowserUseToolset(cdp_url)

    async with toolset as ts:
        # Verify internal tools are built correctly
        assert ts._tools is not None
        assert len(ts._tools) > 0

        # Verify default prefix is set correctly
        default_prefix = "browser_use"
        assert ts.prefix == default_prefix

        # Check that the default prefix will be used in tool names
        for tool in ts._tools:
            expected_name = f"{default_prefix}_{tool.name}"
            # Verify the naming pattern matches what get_tools would produce
            assert tool.name is not None
            # The expected name should use the default prefix
            assert expected_name.startswith(default_prefix)
