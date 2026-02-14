"""Basic integration tests for browser automation tools.

This file contains basic integration tests that verify the core functionality
works together. More detailed tests are organized in separate test files:

- test_image_utils.py: Image processing utilities
- test_navigation.py: Navigation tools
- test_interaction.py: Interaction tools
- test_query.py: Query and element inspection tools
- test_state.py: State inspection and screenshot tools
- test_toolset.py: Toolset lifecycle and configuration
"""

from __future__ import annotations

from ya_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_toolset_context_manager(cdp_url):
    """Test BrowserUseToolset as context manager."""
    async with BrowserUseToolset(cdp_url) as toolset:
        assert toolset._cdp_client is not None
        assert toolset._browser_session is not None
        assert toolset._browser_session.page is not None
        assert len(toolset._tools) > 0

    # After exit, should be cleaned up
    assert toolset._cdp_client is None
