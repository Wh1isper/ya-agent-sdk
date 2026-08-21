"""Tests for ya_agent_sdk.mcp module."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastmcp.client.client import CallToolResult
from mcp import types as mcp_types
from pydantic_ai import (
    ApprovalRequired,
    BinaryContent,
    FunctionToolset,
    RunContext,
    Tool,
    ToolFailed,
    ToolReturn,
    UserError,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.codeact import CodeActConfig
from ya_agent_sdk.codeact.toolset import CodeActToolset
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.mcp import (
    MCPConfig,
    MCPServerConfig,
    MCPServerSpec,
    _ManagedMCPToolset,
    _map_mcp_call_tool_result,
    build_mcp_server,
    build_mcp_servers,
    create_mcp_approval_hook,
    extract_mcp_descriptions,
    extract_optional_mcps,
    filter_mcp_config,
    load_mcp_config_file,
)


class _StubManagedMCPToolset(_ManagedMCPToolset):
    def __init__(
        self,
        client: Any,
        *,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        self.client = client
        self.enter_error = enter_error
        self.exit_error = exit_error

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.exit_error is not None:
            raise self.exit_error


def _mcp_result(
    *,
    structured: dict[str, Any] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    return CallToolResult(
        content=[mcp_types.TextContent(type="text", text="result")],
        structured_content=structured,
        meta=None,
        is_error=is_error,
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
    assert config.prefix is None


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
                    "prefix": "",
                }
            }
        }),
        encoding="utf-8",
    )

    config = load_mcp_config_file(config_path)

    assert isinstance(config, MCPConfig)
    assert config.servers["context7"].url == "https://mcp.context7.com/mcp"
    assert config.servers["context7"].required is False
    assert config.servers["context7"].prefix == ""


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


def test_build_mcp_server_preserves_custom_and_empty_prefixes() -> None:
    custom = build_mcp_server("github", MCPServerConfig(command="npx", prefix="gh"))
    unprefixed = build_mcp_server("local", MCPServerConfig(command="local-server", prefix=""))

    assert custom is not None
    assert custom.tool_prefix == "gh"
    assert unprefixed is not None
    assert unprefixed.tool_prefix == ""
    assert custom.id == "github"
    assert unprefixed.id == "local"


def test_mcp_mixed_structured_and_image_result_preserves_both_channels() -> None:
    observation = {
        "catalog_version": {"major": 1, "minor": 1},
        "observation": {"observation_id": "obs-123", "frame_generation": 4},
    }
    result = CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="computer_observe succeeded"),
            mcp_types.ImageContent(
                type="image",
                data=base64.b64encode(b"png-data").decode("ascii"),
                mimeType="image/png",
            ),
        ],
        structured_content=observation,
        meta=None,
    )

    mapped = _map_mcp_call_tool_result(result)

    assert isinstance(mapped, ToolReturn)
    assert mapped.return_value == observation
    assert isinstance(mapped.content, list)
    assert len(mapped.content) == 1
    assert isinstance(mapped.content[0], BinaryContent)
    assert mapped.content[0].data == b"png-data"
    assert mapped.content[0].media_type == "image/png"


def test_mcp_structured_result_remains_plain_python_value() -> None:
    result = CallToolResult(
        content=[mcp_types.TextContent(type="text", text='{"result": 7}')],
        structured_content={"result": 7},
        meta=None,
    )

    assert _map_mcp_call_tool_result(result) == 7

    null_result = CallToolResult(
        content=[mcp_types.TextContent(type="text", text="null")],
        structured_content={"result": None},
        meta=None,
    )
    assert _map_mcp_call_tool_result(null_result) is None


@pytest.mark.asyncio
async def test_mcp_completed_error_is_returned_without_model_retry() -> None:
    error = {
        "success": False,
        "status": "error",
        "error": {
            "code": "stale_observation",
            "message": "observation is unknown, evicted, or stale",
            "retry": "after_fresh_observation",
        },
    }
    result = CallToolResult(
        content=[mcp_types.TextContent(type="text", text="stale_observation")],
        structured_content=error,
        meta=None,
        is_error=True,
    )

    client = MagicMock()
    client.call_tool = AsyncMock(return_value=result)
    mapped = await _StubManagedMCPToolset(client).direct_call_tool("computer_click", {})

    assert mapped == error
    client.call_tool.assert_awaited_once_with(
        name="computer_click",
        arguments={},
        meta=None,
        raise_on_error=False,
    )


@pytest.mark.asyncio
async def test_mcp_task_result_preserves_completed_error_without_retry() -> None:
    result = _mcp_result(structured={"success": False, "error": {"code": "stale_observation"}}, is_error=True)
    task = MagicMock()
    task.result = AsyncMock(return_value=result)
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=task)

    mapped = await _StubManagedMCPToolset(client).direct_call_tool(
        "computer_click",
        {},
        metadata={"request": "metadata"},
        use_task=True,
    )

    assert mapped == result.structured_content
    client.call_tool.assert_awaited_once_with(
        name="computer_click",
        arguments={},
        task=True,
        meta={"request": "metadata"},
        raise_on_error=False,
    )
    task.result.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection failed"),
        httpx.HTTPStatusError(
            "service unavailable",
            request=httpx.Request("POST", "https://example.test/mcp"),
            response=httpx.Response(503),
        ),
        RuntimeError("Server session was closed unexpectedly"),
        RuntimeError("Failed to initialize server session"),
        RuntimeError("Session task completed unexpectedly"),
        RuntimeError("Client failed to connect: connection refused"),
        RuntimeError("Client is not connected. Use the 'async with client:' context manager first."),
        RuntimeError("Tool computer_observe has an output schema but did not return structured content"),
        RuntimeError("Invalid structured content returned by tool computer_observe: schema mismatch"),
        RuntimeError("Invalid schema for tool computer_observe: invalid reference"),
    ],
)
async def test_mcp_transport_failure_becomes_terminal_tool_failure(failure: Exception) -> None:
    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=failure)

    with pytest.raises(ToolFailed, match=str(failure)):
        await _StubManagedMCPToolset(client).direct_call_tool("computer_observe", {})


@pytest.mark.asyncio
async def test_mcp_task_session_reset_becomes_terminal_tool_failure() -> None:
    task = MagicMock()
    task.result = AsyncMock(
        side_effect=RuntimeError(
            "Cannot access task results outside client context. "
            "Task futures must be used within 'async with client:' block."
        )
    )
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=task)

    with pytest.raises(ToolFailed, match="Cannot access task results outside client context"):
        await _StubManagedMCPToolset(client).direct_call_tool(
            "computer_click",
            {},
            use_task=True,
        )


@pytest.mark.asyncio
async def test_mcp_lifecycle_transport_failure_becomes_terminal_tool_failure() -> None:
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=_mcp_result(structured={"success": True}))

    with pytest.raises(ToolFailed, match="service unavailable"):
        await _StubManagedMCPToolset(
            client,
            enter_error=httpx.HTTPStatusError(
                "service unavailable",
                request=httpx.Request("POST", "https://example.test/mcp"),
                response=httpx.Response(503),
            ),
        ).direct_call_tool("computer_observe", {})

    with pytest.raises(ToolFailed, match="Server session was closed unexpectedly"):
        await _StubManagedMCPToolset(
            client,
            exit_error=RuntimeError("Server session was closed unexpectedly"),
        ).direct_call_tool("computer_observe", {})


@pytest.mark.asyncio
async def test_mcp_unknown_or_mixed_failures_propagate_without_hiding_cancellation() -> None:
    client = MagicMock()
    unknown = RuntimeError("application callback failed")
    client.call_tool = AsyncMock(side_effect=unknown)
    with pytest.raises(RuntimeError, match="application callback failed"):
        await _StubManagedMCPToolset(client).direct_call_tool("computer_observe", {})

    transport_group = ExceptionGroup(
        "transport failures",
        [httpx.ConnectError("one"), httpx.ReadError("two")],
    )
    client.call_tool = AsyncMock(side_effect=transport_group)
    with pytest.raises(ToolFailed, match="one"):
        await _StubManagedMCPToolset(client).direct_call_tool("computer_observe", {})

    mixed_group = BaseExceptionGroup(
        "mixed failure",
        [httpx.ConnectError("connection failed"), asyncio.CancelledError()],
    )
    client.call_tool = AsyncMock(side_effect=mixed_group)
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await _StubManagedMCPToolset(client).direct_call_tool("computer_observe", {})
    assert exc_info.value is mixed_group


@pytest.mark.asyncio
async def test_named_mcp_tools_are_codeact_eligible_by_default(monkeypatch) -> None:
    async def sample() -> str:
        return "ok"

    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage())
    base_tools = await FunctionToolset([Tool(sample)]).get_tools(ctx)

    async def get_tools(_self, _ctx):
        return base_tools

    monkeypatch.setattr(MCPToolset, "get_tools", get_tools)
    server = build_mcp_server(
        "sample",
        MCPServerConfig(transport="streamable_http", url="https://example.test/mcp"),
    )
    assert server is not None

    tools = await server.get_tools(ctx)

    assert tools["sample"].tool_def.metadata == {"codeact": True}


@pytest.mark.asyncio
async def test_named_mcp_tool_with_invalid_python_parameter_fails_codeact_closed(monkeypatch) -> None:
    async def sample(value: str) -> str:
        return value

    ctx = RunContext(deps=AgentContext(), model=TestModel(), usage=RunUsage())
    base_tools = await FunctionToolset([Tool(sample)]).get_tools(ctx)
    base_tool = base_tools["sample"]
    invalid_definition = replace(
        base_tool.tool_def,
        parameters_json_schema={
            "type": "object",
            "properties": {"invalid-name": {"type": "string"}},
            "required": ["invalid-name"],
        },
    )
    invalid_tools = {"sample": replace(base_tool, tool_def=invalid_definition)}

    async def get_tools(_self, _ctx):
        return invalid_tools

    monkeypatch.setattr(MCPToolset, "get_tools", get_tools)
    server = build_mcp_server(
        "sample",
        MCPServerConfig(transport="streamable_http", url="https://example.test/mcp"),
    )
    assert server is not None
    codeact = CodeActToolset(wrapped=server, config=CodeActConfig())

    with pytest.raises(UserError, match="not valid Python identifiers"):
        await codeact.get_tools(ctx)


@pytest.mark.asyncio
async def test_named_mcp_nested_invalid_python_parameter_fails_codeact_closed(monkeypatch) -> None:
    async def sample(payload: dict[str, str]) -> str:
        return str(payload)

    ctx = RunContext(deps=AgentContext(), model=TestModel(), usage=RunUsage())
    base_tools = await FunctionToolset([Tool(sample)]).get_tools(ctx)
    base_tool = base_tools["sample"]
    invalid_definition = replace(
        base_tool.tool_def,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"invalid-name": {"type": "string"}},
                    "required": ["invalid-name"],
                }
            },
            "required": ["payload"],
        },
    )
    invalid_tools = {"sample": replace(base_tool, tool_def=invalid_definition)}

    async def get_tools(_self, _ctx):
        return invalid_tools

    monkeypatch.setattr(MCPToolset, "get_tools", get_tools)
    server = build_mcp_server(
        "sample",
        MCPServerConfig(transport="streamable_http", url="https://example.test/mcp"),
    )
    assert server is not None
    codeact = CodeActToolset(wrapped=server, config=CodeActConfig())

    with pytest.raises(UserError, match="cannot be rendered as valid Python"):
        await codeact.get_tools(ctx)


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
async def test_hook_approval_uses_custom_or_empty_tool_prefix(
    mock_context: MagicMock, mock_call_tool: AsyncMock
) -> None:
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = False

    for prefix, expected_name in [("fs", "fs_write_file"), ("", "write_file")]:
        hook = create_mcp_approval_hook("filesystem", tool_prefix=prefix)
        with pytest.raises(ApprovalRequired) as exc_info:
            await hook(mock_context, mock_call_tool, "write_file", {"path": "/home/user/test.txt"})
        assert exc_info.value.metadata["mcp_server"] == "filesystem"
        assert exc_info.value.metadata["full_name"] == expected_name

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
