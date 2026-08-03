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

import base64
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, assert_never

import httpx
import pydantic_core
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from mcp import types as mcp_types
from mcp.shared import exceptions as mcp_exceptions
from pydantic import BaseModel, Field
from pydantic_ai import ApprovalRequired, RunContext, ToolFailed, ToolReturn
from pydantic_ai.mcp import MCPToolset, ProcessToolCallback
from pydantic_ai.messages import BinaryContent, BinaryImage, is_multi_modal_content
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from ya_agent_sdk._logger import get_logger

if TYPE_CHECKING:
    from pydantic_ai.mcp import CallToolFunc, ToolResult

    from ya_agent_sdk.context import AgentContext

logger = get_logger(__name__)

_TRANSPORT_RUNTIME_ERRORS = frozenset({
    "Server session was closed unexpectedly",
    "Failed to initialize server session",
    "Session task completed without exception but connection failed",
    "Session task completed unexpectedly",
    "Client is not connected. Use the 'async with client:' context manager first.",
    ("Cannot access task results outside client context. Task futures must be used within 'async with client:' block."),
})
_TRANSPORT_RUNTIME_PREFIXES = ("Client failed to connect:",)
_PROTOCOL_RUNTIME_PREFIXES = (
    "Invalid structured content returned by tool ",
    "Invalid schema for tool ",
)


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

    prefix: str | None = None
    """Optional host-facing tool prefix; ``None`` leaves the host default unchanged."""


class MCPConfig(BaseModel):
    """Collection of MCP server configurations keyed by namespace."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class NamedMCPToolset(MCPToolset):
    """Host-managed MCP tools with stable namespace and CodeAct metadata.

    Provider-native MCP integrations do not pass through this class. Tools
    constructed here remain subject to MCP approval hooks and all ordinary
    ToolManager policy when invoked from CodeAct. The host-managed client
    transport is expected to release local request ownership on cancellation;
    cancellation cannot roll back work already accepted by a remote server.
    """

    def __init__(self, *args: Any, tool_prefix: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_prefix = tool_prefix

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await super().get_tools(ctx)
        return {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def,
                    metadata={**(tool.tool_def.metadata or {}), "codeact": True},
                ),
            )
            for name, tool in tools.items()
        }

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        use_task: bool = False,
    ) -> Any:
        """Call an MCP tool while preserving structured output alongside media."""

        try:
            async with self:
                result = await self._invoke_raw_tool(name, args, metadata=metadata, use_task=use_task)
        except BaseExceptionGroup as group:
            _raise_mcp_exception_group(group)
        except Exception as exc:
            if _is_mcp_transport_failure(exc):
                raise ToolFailed(str(exc)) from exc
            raise

        return _map_mcp_call_tool_result(result)

    async def _invoke_raw_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None,
        use_task: bool,
    ) -> CallToolResult:
        if use_task:
            tool_task = await self.client.call_tool(
                name=name,
                arguments=args,
                task=True,
                meta=metadata,
                raise_on_error=False,
            )
            return await tool_task.result()
        return await self.client.call_tool(
            name=name,
            arguments=args,
            meta=metadata,
            raise_on_error=False,
        )


def _map_mcp_call_tool_result(result: CallToolResult) -> Any:
    """Keep MCP structured output as the Python value and media as model-facing content."""

    has_structured_result = result.structured_content is not None
    structured = _structured_result(result.structured_content)
    mapped_content = [_map_mcp_content(part) for part in result.content]
    if has_structured_result:
        media = [item for item in mapped_content if is_multi_modal_content(item)]
        if media:
            return ToolReturn(return_value=structured, content=media)
        return structured
    return mapped_content[0] if len(mapped_content) == 1 else mapped_content


def _structured_result(value: dict[str, Any] | None) -> Any:
    if isinstance(value, dict) and len(value) == 1 and "result" in value:
        return value["result"]
    return value


def _map_mcp_content(part: mcp_types.ContentBlock) -> Any:
    if isinstance(part, mcp_types.TextContent):
        text = part.text
        if text.startswith(("[", "{")):
            try:
                return pydantic_core.from_json(text)
            except ValueError:
                pass
        return text
    if isinstance(part, mcp_types.ImageContent):
        return BinaryImage(data=base64.b64decode(part.data), media_type=part.mimeType)
    if isinstance(part, mcp_types.AudioContent):
        return BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
    if isinstance(part, mcp_types.EmbeddedResource):
        resource = part.resource
        if isinstance(resource, mcp_types.TextResourceContents):
            return resource.text
        if isinstance(resource, mcp_types.BlobResourceContents):
            return BinaryContent.narrow_type(
                BinaryContent(
                    data=base64.b64decode(resource.blob),
                    media_type=resource.mimeType or "application/octet-stream",
                )
            )
        assert_never(resource)
    if isinstance(part, mcp_types.ResourceLink):
        return str(part.uri)
    assert_never(part)


def _is_mcp_transport_failure(exc: BaseException) -> bool:
    if isinstance(exc, (ToolError, mcp_exceptions.McpError, httpx.TransportError, httpx.HTTPStatusError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc)
    if message in _TRANSPORT_RUNTIME_ERRORS or message.startswith((
        *_TRANSPORT_RUNTIME_PREFIXES,
        *_PROTOCOL_RUNTIME_PREFIXES,
    )):
        return True
    return message.startswith("Tool ") and message.endswith(
        " has an output schema but did not return structured content"
    )


def _raise_mcp_exception_group(group: BaseExceptionGroup) -> NoReturn:
    matched, rest = group.split(_is_mcp_transport_failure)
    if matched is None or rest is not None:
        raise group
    error: BaseException = matched
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    raise ToolFailed(str(error)) from group


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


def create_mcp_approval_hook(server_name: str, *, tool_prefix: str | None = None) -> ProcessToolCallback:
    """Create a process_tool_call hook for MCP tool approval."""

    effective_prefix = server_name if tool_prefix is None else tool_prefix

    async def hook(
        ctx: RunContext[AgentContext],
        call_tool: CallToolFunc,
        name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        if server_name in ctx.deps.need_user_approve_mcps and not ctx.tool_call_approved:
            full_name = f"{effective_prefix}_{name}" if effective_prefix else name
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

    tool_prefix = name if config.prefix is None else config.prefix
    process_tool_call = create_mcp_approval_hook(name, tool_prefix=tool_prefix) if need_approval else None

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
                tool_prefix=tool_prefix,
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
                tool_prefix=tool_prefix,
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
