"""Common test fixtures for ya-agent-sdk tests."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture(autouse=True)
def clear_http_client_cache():
    """Clear the cached HTTP client before and after each test.

    The cached HTTP client may be bound to a closed event loop from a previous test,
    causing 'Event loop is closed' errors when reused in a new test.
    """
    from ya_agent_sdk.toolsets.core.web._http_client import _get_http_client

    _get_http_client.cache_clear()
    yield
    _get_http_client.cache_clear()


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
