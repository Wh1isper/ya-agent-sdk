"""Tests for ya_agent_sdk.toolsets.core.web.fetch module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import BinaryContent, RunContext, ToolReturn
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.web.fetch import FetchTool


def test_fetch_tool_attributes() -> None:
    """Should have correct name and description."""
    assert FetchTool.name == "fetch"
    assert "web" in FetchTool.description.lower()


async def test_fetch_tool_head_only(tmp_path: Path, httpx_mock) -> None:
    """Should return metadata with head_only=True."""
    httpx_mock.add_response(
        url="https://example.com/file.json",
        method="HEAD",
        headers={
            "Content-Type": "application/json",
            "Content-Length": "1234",
        },
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="https://example.com/file.json", head_only=True)
        assert result == snapshot({
            "exists": True,
            "accessible": True,
            "status_code": 200,
            "content_type": "application/json",
            "content_length": "1234",
            "last_modified": None,
            "url": "https://example.com/file.json",
        })


async def test_fetch_tool_get_text(tmp_path: Path, httpx_mock) -> None:
    """Should return text content."""
    httpx_mock.add_response(
        url="https://example.com/data.json",
        text='{"key": "value"}',
        headers={"Content-Type": "application/json"},
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="https://example.com/data.json")
        assert result == snapshot('{"key": "value"}')


async def test_fetch_tool_get_image(tmp_path: Path, httpx_mock) -> None:
    """Should return BinaryContent for images."""
    image_data = b"\x89PNG\r\n\x1a\n"  # PNG header
    httpx_mock.add_response(
        url="https://example.com/image.png",
        content=image_data,
        headers={"Content-Type": "image/png"},
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="https://example.com/image.png")
        assert isinstance(result, ToolReturn)
        assert result.return_value == "The image is attached in the user message."
        assert result.content is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], BinaryContent)
        assert result.content[0].media_type == "image/png"


async def test_fetch_tool_reject_large_image_by_content_length(tmp_path: Path, httpx_mock) -> None:
    """Should reject large binary responses before buffering them."""
    httpx_mock.add_response(
        url="https://example.com/huge-image.png",
        content=b"",
        headers={
            "Content-Type": "image/png",
            "Content-Length": str(35 * 1024 * 1024),
        },
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="https://example.com/huge-image.png")
        assert result == snapshot({
            "success": False,
            "error": "Resource too large to inline (36700160 bytes). Maximum supported size is 31457280 bytes.",
        })


async def test_fetch_tool_uses_tool_config_binary_limit(tmp_path: Path, httpx_mock) -> None:
    """Should honor custom binary size limit from ToolConfig."""
    from ya_agent_sdk.context import ToolConfig

    httpx_mock.add_response(
        url="https://example.com/custom-limit.png",
        content=b"",
        headers={
            "Content-Type": "image/png",
            "Content-Length": "2048",
        },
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                tool_config=ToolConfig(fetch_max_inline_binary_bytes=1024),
            )
        )
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="https://example.com/custom-limit.png")
        assert result == snapshot({
            "success": False,
            "error": "Resource too large to inline (2048 bytes). Maximum supported size is 1024 bytes.",
        })


async def test_fetch_tool_forbidden_url(tmp_path: Path) -> None:
    """Should return error for forbidden URLs when verification enabled."""
    from ya_agent_sdk.context import ToolConfig

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                tool_config=ToolConfig(skip_url_verification=False),
            )
        )
        tool = FetchTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="http://192.168.1.1/secret")
        assert result == snapshot({
            "success": False,
            "error": "URL access forbidden - Access to private IP range is forbidden: 192.168.1.1",
        })
