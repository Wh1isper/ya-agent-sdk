"""MCP server configuration, loading, and construction utilities.

This module provides:
- MCPServerSpec: transport-agnostic MCP server spec
- MCPServerConfig: server spec with runtime metadata
- MCPConfig: collection of named MCP servers
- load_mcp_config_file(): load JSON config from disk
- filter_mcp_config(): apply namespace filters
- build_mcp_server()/build_mcp_servers(): construct MCP toolsets
- extract_mcp_descriptions()/extract_optional_mcps(): metadata helpers
- create_mcp_approval_hook(): approval hook factory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel, Field
from pydantic_ai import ApprovalRequired, RunContext
from pydantic_ai.mcp import MCPToolset, ProcessToolCallback
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk._logger import get_logger

if TYPE_CHECKING:
    from pydantic_ai.mcp import CallToolFunc, ToolResult

    from ya_agent_sdk.context import AgentContext

logger = get_logger(__name__)


class MCPServerSpec(BaseModel):
    """Transport-agnostic MCP server specification."""

    transport: Literal["stdio", "streamable_http"] = "stdio"
    """Transport type: stdio or streamable_http."""

    command: str | None = None
    """Command for stdio transport."""

    args: list[str] = Field(default_factory=list)
    """Command arguments for stdio transport."""

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables for the server."""

    url: str | None = None
    """URL for streamable_http transport."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Headers for streamable_http transport."""


class MCPServerConfig(MCPServerSpec):
    """MCP server configuration with runtime metadata."""

    description: str = ""
    """Human-readable namespace description."""

    required: bool = True
    """Whether startup/toolset initialization treats this server as required."""


class MCPConfig(BaseModel):
    """Collection of MCP server configurations keyed by namespace."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class NamedMCPToolset(MCPToolset):
    """MCPToolset with a stable ``tool_prefix`` compatibility attribute."""

    def __init__(self, *args: Any, tool_prefix: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_prefix = tool_prefix


def load_mcp_config_file(file_path: Path) -> MCPConfig:
    """Load MCP JSON configuration from disk."""

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return MCPConfig.model_validate(payload)


def filter_mcp_config(
    mcp_config: MCPConfig,
    *,
    enabled_mcps: list[str] | set[str] | None = None,
    disabled_mcps: list[str] | set[str] | None = None,
) -> MCPConfig:
    """Return a filtered MCP config preserving original namespace order."""

    enabled_names = _normalize_namespace_names(enabled_mcps)
    disabled_names = _normalize_namespace_names(disabled_mcps)

    filtered_servers: dict[str, MCPServerConfig] = {}
    for name, config in mcp_config.servers.items():
        if enabled_names and name not in enabled_names:
            continue
        if name in disabled_names:
            continue
        filtered_servers[name] = config.model_copy(deep=True)
    return MCPConfig(servers=filtered_servers)


def create_mcp_approval_hook(server_name: str) -> ProcessToolCallback:
    """Create a process_tool_call hook for MCP tool approval."""

    async def hook(
        ctx: RunContext[AgentContext],
        call_tool: CallToolFunc,
        name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        if server_name in ctx.deps.need_user_approve_mcps and not ctx.tool_call_approved:
            full_name = f"{server_name}_{name}"
            logger.debug("MCP tool %r requires approval", full_name)
            raise ApprovalRequired(metadata={"mcp_server": server_name, "mcp_tool": name, "full_name": full_name})

        return await call_tool(name, tool_args, metadata=None)

    return hook


def build_mcp_server(
    name: str,
    config: MCPServerConfig,
    need_approval: bool = False,
) -> AbstractToolset[Any] | None:
    """Build a single MCP toolset instance from configuration."""

    process_tool_call = create_mcp_approval_hook(name) if need_approval else None

    match config.transport:
        case "stdio":
            if not config.command:
                logger.warning("MCP server %r has stdio transport but no command, skipping", name)
                return None
            null_path = "NUL" if sys.platform == "win32" else "/dev/null"
            return NamedMCPToolset(
                StdioTransport(
                    command=config.command, args=config.args, env=config.env or None, log_file=Path(null_path)
                ),
                tool_prefix=name,
                id=name,
                process_tool_call=process_tool_call,
            )
        case "streamable_http":
            if not config.url:
                logger.warning("MCP server %r has streamable_http transport but no url, skipping", name)
                return None
            return NamedMCPToolset(
                config.url,
                headers=config.headers or None,
                tool_prefix=name,
                id=name,
                process_tool_call=process_tool_call,
            )
        case _:
            logger.warning("MCP server %r has unknown transport type %r, skipping", name, config.transport)
            return None


def build_mcp_servers(
    mcp_config: MCPConfig,
    need_approval_mcps: list[str] | None = None,
) -> list[AbstractToolset[Any]]:
    """Build MCP toolset instances from MCPConfig."""

    servers: list[AbstractToolset[Any]] = []
    approval_names = _normalize_namespace_names(need_approval_mcps)

    for name, config in mcp_config.servers.items():
        server = build_mcp_server(name, config, need_approval=name in approval_names)
        if server is not None:
            servers.append(server)
            logger.info("Added MCP toolset: %s (%s, approval=%s)", name, config.transport, name in approval_names)

    logger.debug("Built %d MCP toolsets from config", len(servers))
    return servers


def _normalize_namespace_names(names: list[str] | set[str] | None) -> set[str]:
    return {name.strip() for name in names or [] if isinstance(name, str) and name.strip() != ""}


def extract_mcp_descriptions(mcp_config: MCPConfig) -> dict[str, str]:
    """Extract non-empty namespace descriptions from config."""

    descriptions: dict[str, str] = {}
    for name, config in mcp_config.servers.items():
        if config.description:
            descriptions[name] = config.description
    return descriptions


def extract_optional_mcps(mcp_config: MCPConfig) -> set[str]:
    """Extract server names marked as optional."""

    return {name for name, config in mcp_config.servers.items() if not config.required}
