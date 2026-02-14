"""Tests for Docker-based browser sandbox."""

from uuid import uuid4

import httpx

from ya_agent_sdk.sandbox.browser.docker_ import DockerBrowserSandbox, get_port


async def test_browser_sandbox_context_manager():
    """Test browser sandbox with async context manager."""
    async with DockerBrowserSandbox() as cdp_url:
        # Verify CDP URL format
        assert cdp_url.startswith("http://localhost:")
        assert "/json/version" in cdp_url

        # Verify Chrome is accessible
        async with httpx.AsyncClient() as client:
            response = await client.get(cdp_url, timeout=10)
            assert response.status_code == 200

            # Verify response contains Chrome version info
            data = response.json()
            assert "Browser" in data
            assert "Protocol-Version" in data


async def test_browser_sandbox_manual_start_stop():
    """Test browser sandbox with manual start/stop."""
    sandbox = DockerBrowserSandbox()

    try:
        # Start browser
        cdp_url = await sandbox.start_browser()

        # Verify browser is running
        async with httpx.AsyncClient() as client:
            response = await client.get(cdp_url, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "Browser" in data

    finally:
        # Stop browser
        await sandbox.stop_browser()


async def test_browser_sandbox_custom_port():
    """Test browser sandbox with custom port."""
    custom_port = get_port()

    async with DockerBrowserSandbox(port=custom_port) as cdp_url:
        # Verify custom port is used
        assert f"localhost:{custom_port}" in cdp_url

        # Verify accessibility
        async with httpx.AsyncClient() as client:
            response = await client.get(cdp_url, timeout=10)
            assert response.status_code == 200


async def test_browser_sandbox_multiple_instances():
    """Test running multiple browser sandbox instances concurrently."""
    async with DockerBrowserSandbox(container_name=str(uuid4())) as cdp_url1:
        async with DockerBrowserSandbox(container_name=str(uuid4())) as cdp_url2:
            # Verify both instances are running on different ports
            assert cdp_url1 != cdp_url2

            # Verify both are accessible
            async with httpx.AsyncClient() as client:
                response1 = await client.get(cdp_url1, timeout=10)
                response2 = await client.get(cdp_url2, timeout=10)

                assert response1.status_code == 200
                assert response2.status_code == 200


async def test_browser_sandbox_exception_cleanup():
    """Test that browser is properly cleaned up on exception."""
    sandbox = DockerBrowserSandbox()
    cdp_url = None

    try:
        async with sandbox as url:
            cdp_url = url
            # Verify browser is running
            async with httpx.AsyncClient() as client:
                response = await client.get(cdp_url, timeout=10)
                assert response.status_code == 200

            # Simulate exception
            raise ValueError("Test exception")

    except ValueError:
        pass  # Expected exception
