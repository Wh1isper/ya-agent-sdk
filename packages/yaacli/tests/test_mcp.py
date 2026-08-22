"""Tests for yaacli.mcp module."""

from __future__ import annotations

from ya_agent_sdk.mcp import MCPConfig, MCPServerConfig
from yaacli.mcp import build_mcp_server, build_mcp_servers, extract_optional_mcps

# =============================================================================
# build_mcp_server Tests
# =============================================================================


def test_build_mcp_server_stdio() -> None:
    """Test building stdio MCP server."""
    config = MCPServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "test-token"},
    )

    server = build_mcp_server("github", config)

    assert server is not None
    # MCPServerStdio attributes
    assert server.tool_prefix == "github"


def test_build_mcp_server_stdio_no_command() -> None:
    """Test building stdio server without command returns None."""
    config = MCPServerConfig(
        transport="stdio",
        command=None,  # Missing command
    )

    server = build_mcp_server("test", config)

    assert server is None


def test_build_mcp_server_streamable_http() -> None:
    """Test building streamable_http MCP server."""
    config = MCPServerConfig(
        transport="streamable_http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer test"},
    )

    server = build_mcp_server("api", config)

    assert server is not None
    assert server.tool_prefix == "api"


def test_build_mcp_server_streamable_http_no_url() -> None:
    """Test building streamable_http server without url returns None."""
    config = MCPServerConfig(
        transport="streamable_http",
        url=None,  # Missing url
    )

    server = build_mcp_server("test", config)

    assert server is None


# =============================================================================
# build_mcp_servers Tests
# =============================================================================


def test_build_mcp_servers_empty() -> None:
    """Test building from empty config."""
    mcp_config = MCPConfig(servers={})

    servers = build_mcp_servers(mcp_config)

    assert servers == []


def test_build_mcp_servers_multiple() -> None:
    """Test building multiple servers."""
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


# =============================================================================
# extract_optional_mcps Tests
# =============================================================================


def test_extract_optional_mcps_empty() -> None:
    """Test with empty config returns empty set."""
    mcp_config = MCPConfig(servers={})
    assert extract_optional_mcps(mcp_config) == set()


def test_extract_optional_mcps_all_required() -> None:
    """Test with all required servers returns empty set."""
    mcp_config = MCPConfig(
        servers={
            "github": MCPServerConfig(transport="stdio", command="npx"),
            "api": MCPServerConfig(transport="streamable_http", url="http://localhost:8080/mcp"),
        }
    )
    assert extract_optional_mcps(mcp_config) == set()


def test_extract_optional_mcps_mixed() -> None:
    """Test with mix of required and optional servers."""
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


def test_extract_optional_mcps_all_optional() -> None:
    """Test with all optional servers."""
    mcp_config = MCPConfig(
        servers={
            "a": MCPServerConfig(transport="stdio", command="cmd", required=False),
            "b": MCPServerConfig(transport="stdio", command="cmd", required=False),
        }
    )
    assert extract_optional_mcps(mcp_config) == {"a", "b"}


def test_build_mcp_servers_skips_invalid() -> None:
    """Test that invalid configs are skipped."""
    mcp_config = MCPConfig(
        servers={
            "valid": MCPServerConfig(
                transport="stdio",
                command="npx",
            ),
            "invalid": MCPServerConfig(
                transport="stdio",
                command=None,  # Invalid - no command
            ),
        }
    )

    servers = build_mcp_servers(mcp_config)

    assert len(servers) == 1
