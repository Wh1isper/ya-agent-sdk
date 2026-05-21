"""Compatibility wrapper for MCP helpers.

YAACLI now relies on the shared MCP utilities in ``ya_agent_sdk.mcp``.
This module keeps the old import path stable for internal and external users.
"""

from ya_agent_sdk.mcp import (
    MCPConfig,
    MCPServerConfig,
    build_mcp_server,
    build_mcp_servers,
    create_mcp_approval_hook,
    extract_mcp_descriptions,
    extract_optional_mcps,
    filter_mcp_config,
    load_mcp_config_file,
)

__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "build_mcp_server",
    "build_mcp_servers",
    "create_mcp_approval_hook",
    "extract_mcp_descriptions",
    "extract_optional_mcps",
    "filter_mcp_config",
    "load_mcp_config_file",
]
