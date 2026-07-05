"""Tests for the read_media tool."""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import httpx2
import pytest
from PIL import Image
from pydantic_ai import BinaryContent, RunContext, ToolReturn, VideoUrl
from ya_agent_sdk.context import AgentContext, ModelCapability, ModelConfig, ToolConfig
from ya_agent_sdk.toolsets.core.content import LoadMediaUrlTool, ReadMediaTool, tools

PNG_1X1 = bytes([
    0x89,
    0x50,
    0x4E,
    0x47,
    0x0D,
    0x0A,
    0x1A,
    0x0A,
    0x00,
    0x00,
    0x00,
    0x0D,
    0x49,
    0x48,
    0x44,
    0x52,
    0x00,
    0x00,
    0x00,
    0x01,
    0x00,
    0x00,
    0x00,
    0x01,
    0x08,
    0x06,
    0x00,
    0x00,
    0x00,
    0x1F,
    0x15,
    0xC4,
    0x89,
    0x00,
    0x00,
    0x00,
    0x0A,
    0x49,
    0x44,
    0x41,
    0x54,
    0x78,
    0x9C,
    0x63,
    0x00,
    0x01,
    0x00,
    0x00,
    0x05,
    0x00,
    0x01,
    0x0D,
    0x0A,
    0x2D,
    0xB4,
    0x00,
    0x00,
    0x00,
    0x00,
    0x49,
    0x45,
    0x4E,
    0x44,
    0xAE,
    0x42,
    0x60,
    0x82,
])


@dataclass
class _RegisteredResponse:
    url: str
    response: httpx2.Response
    method: str | None = None
    used: bool = False


class _HTTPX2Mock:
    def __init__(self) -> None:
        self._responses: list[_RegisteredResponse] = []

    def add_response(
        self,
        *,
        url: str,
        method: str | None = None,
        status_code: int = 200,
        content: bytes | None = None,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._responses.append(
            _RegisteredResponse(
                url=url,
                method=method.upper() if method else None,
                response=httpx2.Response(status_code, content=content, text=text, headers=headers),
            )
        )

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        request_url = str(request.url)
        request_method = request.method.upper()
        for registered in self._responses:
            if registered.used:
                continue
            if registered.method is not None and registered.method != request_method:
                continue
            if registered.url != request_url:
                continue
            registered.used = True
            return httpx2.Response(
                registered.response.status_code,
                headers=registered.response.headers,
                content=registered.response.content,
                request=request,
            )
        raise AssertionError(f"No mocked HTTPX2 response for {request_method} {request_url}")


@pytest.fixture
def httpx2_mock(monkeypatch: pytest.MonkeyPatch) -> _HTTPX2Mock:
    from ya_agent_sdk.toolsets.core.web import _http_client

    original_get_http_client = _http_client._get_http_client
    mock = _HTTPX2Mock()
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(mock.handler),
        follow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ya-agent-sdk/1.0)"},
    )

    original_get_http_client.cache_clear()
    monkeypatch.setattr(_http_client, "_get_http_client", lambda: client)
    monkeypatch.setattr(_http_client, "get_http_client", lambda: client)
    yield mock
    original_get_http_client.cache_clear()


def _run_context(
    *capabilities: ModelCapability,
    model_cfg: ModelConfig | None = None,
    tool_config: ToolConfig | None = None,
) -> RunContext[AgentContext]:
    ctx = MagicMock(spec=RunContext)
    ctx.deps = AgentContext(
        model_cfg=model_cfg or ModelConfig(capabilities=set(capabilities)),
        tool_config=tool_config or ToolConfig(),
    )
    return ctx


def _make_random_png(width: int = 1200, height: int = 1200) -> bytes:
    raw = os.urandom(width * height * 3)
    image = Image.frombytes("RGB", (width, height), raw)
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def test_content_toolset_registers_read_media_by_default() -> None:
    """read_media is the default URL media tool; legacy URL tool remains explicit only."""
    assert tools == [ReadMediaTool]
    assert LoadMediaUrlTool not in tools


async def test_read_media_instruction_describes_download_view_fallback() -> None:
    instruction = await ReadMediaTool().get_instruction(_run_context(ModelCapability.vision))

    assert instruction is not None
    assert "read_media" in instruction
    assert "YouTube" in instruction
    assert "`download`" in instruction
    assert "`view`" in instruction
    assert "`instructions`" in instruction


async def test_read_media_returns_youtube_video_url_for_youtube_url_model() -> None:
    url = "https://www.youtube.com/watch?v=9hE5-98ZeCg"

    result = await ReadMediaTool().call(
        _run_context(ModelCapability.video_understanding, ModelCapability.youtube_url),
        url=url,
        instructions="Summarize the video.",
    )

    assert isinstance(result, ToolReturn)
    assert "The video is attached" in result.return_value
    assert "Summarize the video." in result.return_value
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], VideoUrl)
    assert result.content[0].url == url
    assert result.content[0].is_youtube is True
    assert result.content[0].media_type == "video/mp4"


async def test_read_media_returns_fallback_for_youtube_without_youtube_url_capability() -> None:
    result = await ReadMediaTool().call(
        _run_context(ModelCapability.video_understanding),
        url="https://youtu.be/9hE5-98ZeCg",
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "does not support direct YouTube URLs" in result["error"]
    assert "`download`" in result["fallback"]
    assert "`view`" in result["fallback"]


async def test_read_media_returns_image_binary_with_instructions(httpx2_mock: Any) -> None:
    httpx2_mock.add_response(
        url="https://example.com/image.png",
        content=PNG_1X1,
        headers={"Content-Type": "image/png", "Content-Length": str(len(PNG_1X1))},
    )

    result = await ReadMediaTool().call(
        _run_context(ModelCapability.vision),
        url="https://example.com/image.png",
        instructions="Extract all visible text.",
    )

    assert isinstance(result, ToolReturn)
    assert "The image is attached" in result.return_value
    assert "Analysis instructions" in result.return_value
    assert "Extract all visible text." in result.return_value
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], BinaryContent)
    assert result.content[0].data == PNG_1X1
    assert result.content[0].media_type == "image/png"


async def test_read_media_returns_video_binary_from_extension_when_content_type_is_generic(httpx2_mock: Any) -> None:
    video_data = b"video bytes"
    httpx2_mock.add_response(
        url="https://example.com/movie.mp4",
        content=video_data,
        headers={"Content-Type": "application/octet-stream"},
    )

    result = await ReadMediaTool().call(
        _run_context(ModelCapability.video_understanding),
        url="https://example.com/movie.mp4",
    )

    assert isinstance(result, ToolReturn)
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], BinaryContent)
    assert result.content[0].data == video_data
    assert result.content[0].media_type == "video/mp4"


async def test_read_media_rejects_declared_oversized_media_before_reading(httpx2_mock: Any) -> None:
    httpx2_mock.add_response(
        url="https://example.com/huge.png",
        content=b"",
        headers={"Content-Type": "image/png", "Content-Length": "2048"},
    )

    result = await ReadMediaTool().call(
        _run_context(
            ModelCapability.vision,
            tool_config=ToolConfig(view_max_inline_image_bytes=1024),
        ),
        url="https://example.com/huge.png",
    )

    assert result == {
        "success": False,
        "error": "The image URL is too large to read into memory safely (2048 bytes). Maximum supported size is 1024 bytes.",
        "fallback": (
            "Use `download` to save the URL to a local file, compress or transcode it if needed, "
            "then call `view` on the local path with focused `instructions`."
        ),
    }


class _ChunkedResponse:
    async def aiter_bytes(self, chunk_size: int):
        assert chunk_size == 2
        yield b"12"
        yield b"345"


async def test_read_media_stops_when_stream_exceeds_limit() -> None:
    result = await ReadMediaTool()._read_limited_body(
        _run_context(
            tool_config=ToolConfig(fetch_stream_chunk_size=2),
        ),
        _ChunkedResponse(),
        kind="audio",
        max_bytes=4,
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error"] == (
        "The audio URL exceeded the safe in-memory limit while downloading (5 bytes). "
        "Maximum supported size is 4 bytes."
    )
    assert "`download`" in result["fallback"]
    assert "`view`" in result["fallback"]


async def test_read_media_compresses_images_to_model_limit(httpx2_mock: Any) -> None:
    from ya_agent_sdk.utils import raw_bytes_limit_for_base64

    image_data = _make_random_png()
    max_image_bytes = 5 * 1024 * 1024
    raw_budget = raw_bytes_limit_for_base64(max_image_bytes)
    assert len(image_data) > raw_budget
    httpx2_mock.add_response(
        url="https://example.com/large-image.png",
        content=image_data,
        headers={"Content-Type": "image/png", "Content-Length": str(len(image_data))},
    )

    result = await ReadMediaTool().call(
        _run_context(
            model_cfg=ModelConfig(max_image_bytes=max_image_bytes, capabilities={ModelCapability.vision}),
            tool_config=ToolConfig(view_max_inline_image_bytes=len(image_data) + 1),
        ),
        url="https://example.com/large-image.png",
    )

    assert isinstance(result, ToolReturn)
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], BinaryContent)
    assert result.content[0].media_type == "image/jpeg"
    assert len(result.content[0].data) <= raw_budget


async def test_read_media_returns_fallback_for_missing_model_capability(httpx2_mock: Any) -> None:
    httpx2_mock.add_response(
        url="https://example.com/image.png",
        content=PNG_1X1,
        headers={"Content-Type": "image/png"},
    )

    result = await ReadMediaTool().call(_run_context(), url="https://example.com/image.png")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "does not support vision" in result["error"]
    assert "`download`" in result["fallback"]
    assert "`view`" in result["fallback"]


async def test_read_media_returns_fallback_for_unknown_content(httpx2_mock: Any) -> None:
    httpx2_mock.add_response(
        url="https://example.com/page.html",
        text="<html></html>",
        headers={"Content-Type": "text/html"},
    )

    result = await ReadMediaTool().call(_run_context(ModelCapability.vision), url="https://example.com/page.html")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "does not look like a supported image, video, or audio resource" in result["error"]
    assert "`download`" in result["fallback"]
    assert "`view`" in result["fallback"]


async def test_read_media_rejects_non_http_urls() -> None:
    result = await ReadMediaTool().call(_run_context(ModelCapability.vision), url="file:///tmp/image.png")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "Only HTTP and HTTPS URLs are supported" in result["error"]
