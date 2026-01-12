"""Common test fixtures for pai-agent-sdk tests."""

import contextlib
import socket
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture(autouse=True)
def clear_http_client_cache():
    """Clear the cached HTTP client before and after each test.

    The cached HTTP client may be bound to a closed event loop from a previous test,
    causing 'Event loop is closed' errors when reused in a new test.
    """
    from pai_agent_sdk.toolsets.core.web._http_client import _get_http_client

    _get_http_client.cache_clear()
    yield
    _get_http_client.cache_clear()


def get_port() -> int:
    """Get an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def agent_context(tmp_path: Path) -> AsyncIterator[AgentContext]:
    """Create an AgentContext for testing with an entered environment."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            yield ctx


@pytest.fixture
def mock_run_ctx(agent_context: AgentContext) -> MagicMock:
    """Create a mock RunContext for testing is_available and other methods.

    This fixture provides a MagicMock spec'd to RunContext with deps set to agent_context.
    Use this for testing tool methods that require a RunContext parameter.
    """
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = agent_context
    return mock_ctx


@pytest.fixture(scope="session")
def docker_client():
    """Get Docker client, skip if not available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return client
    except Exception:
        pytest.skip("Docker is not available")


@pytest.fixture(scope="session")
def docker_network(docker_client):
    """Create a Docker network for test containers."""
    network_name = f"test-network-{uuid4().hex}"

    with contextlib.suppress(Exception):
        old_network = docker_client.networks.get(network_name)
        old_network.remove()

    network = docker_client.networks.create(network_name, driver="bridge")

    yield network_name

    with contextlib.suppress(Exception):
        network.remove()


@pytest.fixture(scope="session")
def chrome_cdp_url(docker_client, docker_network):
    """Start headless Chrome container and return CDP URL.

    This is a general-purpose Chrome fixture that can be used by any test
    that needs a headless Chrome browser.
    """
    chrome_port = get_port()
    image = "zenika/alpine-chrome:latest"
    container_name = f"chrome-{uuid4().hex}"
    container = None

    try:
        container = docker_client.containers.run(
            image,
            command=[
                "chromium-browser",
                "--headless",
                "--remote-debugging-port=9222",
                "--remote-debugging-address=0.0.0.0",
                "--no-sandbox",
            ],
            detach=True,
            name=container_name,
            network=docker_network,
            ports={"9222": chrome_port},
            remove=True,
        )

        # Wait for Chrome to start
        cdp_endpoint = f"http://localhost:{chrome_port}/json/version"
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = httpx.get(cdp_endpoint, timeout=5)
                if response.status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            if container:
                container.stop()
            raise RuntimeError("Chrome container failed to start within timeout")

        yield cdp_endpoint

    finally:
        if container:
            with contextlib.suppress(Exception):
                container.stop()
