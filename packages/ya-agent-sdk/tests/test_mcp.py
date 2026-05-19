"""Tests for ya_agent_sdk.mcp module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from ya_agent_sdk.mcp import (
    MCPConfig,
    MCPServerConfig,
    MCPServerSpec,
    build_mcp_server,
    build_mcp_servers,
    create_mcp_approval_hook,
    extract_mcp_descriptions,
    extract_optional_mcps,
    filter_mcp_config,
    load_mcp_config_file,
)


def test_mcp_server_spec_defaults() -> None:
    spec = MCPServerSpec()
    assert spec.transport == "stdio"
    assert spec.command is None
    assert spec.args == []
    assert spec.env == {}
    assert spec.url is None
    assert spec.headers == {}


def test_mcp_server_config_defaults() -> None:
    config = MCPServerConfig(command="uvx")
    assert config.description == ""
    assert config.required is True


def test_mcp_server_spec_stdio() -> None:
    spec = MCPServerSpec(
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem"],
        env={"HOME": "/home/user"},
    )
    assert spec.transport == "stdio"
    assert spec.command == "uvx"
    assert spec.args == ["mcp-server-filesystem"]
    assert spec.env == {"HOME": "/home/user"}


def test_mcp_server_spec_streamable_http() -> None:
    spec = MCPServerSpec(
        transport="streamable_http",
        url="http://localhost:8000/mcp",
        headers={"Authorization": "Bearer token"},
    )
    assert spec.transport == "streamable_http"
    assert spec.url == "http://localhost:8000/mcp"
    assert spec.headers == {"Authorization": "Bearer token"}


def test_load_mcp_config_file(tmp_path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({
            "servers": {
                "context7": {
                    "transport": "streamable_http",
                    "url": "https://mcp.context7.com/mcp",
                    "description": "Library docs",
                    "required": False,
                }
            }
        }),
        encoding="utf-8",
    )

    config = load_mcp_config_file(config_path)

    assert isinstance(config, MCPConfig)
    assert config.servers["context7"].url == "https://mcp.context7.com/mcp"
    assert config.servers["context7"].required is False


def test_filter_mcp_config_enabled_and_disabled() -> None:
    config = MCPConfig(
        servers={
            "github": MCPServerConfig(transport="stdio", command="npx"),
            "context7": MCPServerConfig(transport="streamable_http", url="https://mcp.context7.com/mcp"),
            "filesystem": MCPServerConfig(transport="stdio", command="uvx"),
        }
    )

    filtered = filter_mcp_config(config, enabled_mcps=["github", "filesystem"], disabled_mcps=["filesystem"])

    assert list(filtered.servers) == ["github"]


def test_build_mcp_server_stdio() -> None:
    config = MCPServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "test-token"},
    )

    server = build_mcp_server("github", config)

    assert server is not None
    assert server.tool_prefix == "github"


def test_build_mcp_server_stdio_no_command() -> None:
    config = MCPServerConfig(
        transport="stdio",
        command=None,
    )

    server = build_mcp_server("test", config)

    assert server is None


def test_build_mcp_server_streamable_http() -> None:
    config = MCPServerConfig(
        transport="streamable_http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer test"},
    )

    server = build_mcp_server("api", config)

    assert server is not None
    assert server.tool_prefix == "api"


def test_build_mcp_server_streamable_http_no_url() -> None:
    config = MCPServerConfig(
        transport="streamable_http",
        url=None,
    )

    server = build_mcp_server("test", config)

    assert server is None


def test_build_mcp_servers_empty() -> None:
    mcp_config = MCPConfig(servers={})
    servers = build_mcp_servers(mcp_config)
    assert servers == []


def test_build_mcp_servers_multiple() -> None:
    mcp_config = MCPConfig(
        servers={
            "github": MCPServerConfig(
                transport="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
            ),
            "api": MCPServerConfig(
                transport="streamable_http",
                url="http://localhost:8080/mcp",
            ),
        }
    )

    servers = build_mcp_servers(mcp_config)

    assert len(servers) == 2


def test_extract_mcp_descriptions() -> None:
    mcp_config = MCPConfig(
        servers={
            "github": MCPServerConfig(transport="stdio", command="npx", description="GitHub operations"),
            "context7": MCPServerConfig(
                transport="streamable_http",
                url="https://mcp.context7.com/mcp",
                description="Docs search",
            ),
            "empty": MCPServerConfig(transport="stdio", command="uvx"),
        }
    )

    assert extract_mcp_descriptions(mcp_config) == {
        "github": "GitHub operations",
        "context7": "Docs search",
    }


def test_extract_optional_mcps_empty() -> None:
    mcp_config = MCPConfig(servers={})
    assert extract_optional_mcps(mcp_config) == set()


def test_extract_optional_mcps_mixed() -> None:
    mcp_config = MCPConfig(
        servers={
            "github": MCPServerConfig(transport="stdio", command="npx", required=True),
            "context7": MCPServerConfig(
                transport="streamable_http",
                url="https://mcp.context7.com/mcp",
                required=False,
            ),
            "docs": MCPServerConfig(
                transport="streamable_http",
                url="http://localhost:3000/mcp",
                required=False,
            ),
        }
    )
    result = extract_optional_mcps(mcp_config)
    assert result == {"context7", "docs"}


def test_build_mcp_servers_skips_invalid() -> None:
    mcp_config = MCPConfig(
        servers={
            "valid": MCPServerConfig(
                transport="stdio",
                command="npx",
            ),
            "invalid": MCPServerConfig(
                transport="stdio",
                command=None,
            ),
        }
    )

    servers = build_mcp_servers(mcp_config)

    assert len(servers) == 1


@pytest.fixture
def mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.need_user_approve_mcps = []
    ctx.tool_call_approved = False
    return ctx


@pytest.fixture
def mock_call_tool() -> AsyncMock:
    return AsyncMock(return_value="tool result")


@pytest.mark.asyncio
async def test_hook_no_approval_needed(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = []

    result = await hook(mock_context, mock_call_tool, "read_file", {"path": "/home/user/test.txt"})

    assert result == "tool result"
    mock_call_tool.assert_called_once_with("read_file", {"path": "/home/user/test.txt"}, metadata=None)


@pytest.mark.asyncio
async def test_hook_approval_required_raises(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    from pydantic_ai import ApprovalRequired

    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = False

    with pytest.raises(ApprovalRequired) as exc_info:
        await hook(mock_context, mock_call_tool, "write_file", {"path": "/home/user/test.txt"})

    assert exc_info.value.metadata["mcp_server"] == "filesystem"
    assert exc_info.value.metadata["mcp_tool"] == "write_file"
    assert exc_info.value.metadata["full_name"] == "filesystem_write_file"
    mock_call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_hook_already_approved(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = True

    result = await hook(mock_context, mock_call_tool, "write_file", {"path": "/home/user/test.txt"})

    assert result == "tool result"
    mock_call_tool.assert_called_once_with("write_file", {"path": "/home/user/test.txt"}, metadata=None)


@pytest.mark.asyncio
async def test_hook_different_server_not_affected(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    hook = create_mcp_approval_hook("github")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = False

    result = await hook(mock_context, mock_call_tool, "create_issue", {"title": "Test"})

    assert result == "tool result"
    mock_call_tool.assert_called_once()
