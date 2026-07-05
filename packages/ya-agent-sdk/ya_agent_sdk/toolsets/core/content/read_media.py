"""Read remote media URLs as inline binary content."""

from __future__ import annotations

import contextlib
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx2 as httpx
from pydantic import Field
from pydantic_ai import BinaryContent, RunContext, ToolReturn, VideoUrl

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext, ModelCapability
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.content._url_helper import (
    ContentCategory,
    get_category_from_extension,
    get_category_from_mime_type,
    is_valid_http_url,
)
from ya_agent_sdk.toolsets.core.web._http_client import ForbiddenUrlError, safe_stream_request
from ya_agent_sdk.utils import compress_image_to_model_limit, detect_image_media_type, raw_bytes_limit_for_base64

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

MediaKind = Literal["image", "video", "audio"]
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

IMAGE_MEDIA_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
VIDEO_MEDIA_TYPE_MAP = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".ogv": "video/ogg",
}
AUDIO_MEDIA_TYPE_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
}


@cache
def _load_instruction() -> str:
    """Load read_media instruction text."""
    return (_PROMPTS_DIR / "read_media.md").read_text()


def _main_content_type(value: str | None) -> str:
    """Return a normalized MIME type without parameters."""
    if not value:
        return ""
    return value.split(";")[0].strip().lower()


def _url_extension(url: str) -> str:
    """Extract a lower-case extension from a URL path."""
    path = urlparse(url).path.lower()
    dot = path.rfind(".")
    if dot == -1:
        return ""
    return path[dot:]


def _is_youtube_url(url: str) -> bool:
    """Return whether a URL points at a YouTube video host."""
    hostname = urlparse(url).hostname
    return hostname in {"youtu.be", "youtube.com", "www.youtube.com"}


def _extension_media_type(url: str, kind: MediaKind) -> str | None:
    ext = _url_extension(url)
    if kind == "image":
        return IMAGE_MEDIA_TYPE_MAP.get(ext)
    if kind == "video":
        return VIDEO_MEDIA_TYPE_MAP.get(ext)
    return AUDIO_MEDIA_TYPE_MAP.get(ext)


def _kind_for_category(category: ContentCategory) -> MediaKind | None:
    if category == ContentCategory.image:
        return "image"
    if category == ContentCategory.video:
        return "video"
    if category == ContentCategory.audio:
        return "audio"
    return None


def _fallback_guidance() -> str:
    return (
        "Use `download` to save the URL to a local file, compress or transcode it if needed, "
        "then call `view` on the local path with focused `instructions`."
    )


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message, "fallback": _fallback_guidance()}


class ReadMediaTool(BaseTool):
    """Read image, video, or audio URLs into model-consumable media content."""

    name = "read_media"
    description = (
        "Read an HTTP/HTTPS image, video, or audio URL as model-consumable media content. "
        "Public YouTube URLs are passed directly to models that support direct YouTube URLs. "
        "Use `instructions` for focused image, video, or audio analysis."
    )

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    def _category_and_media_type(self, response: httpx.Response, url: str) -> tuple[ContentCategory, str | None]:
        content_type = _main_content_type(response.headers.get("Content-Type"))
        category = get_category_from_mime_type(content_type) if content_type else ContentCategory.unknown

        if category == ContentCategory.unknown:
            extension_category = get_category_from_extension(url)
            category = extension_category

        kind = _kind_for_category(category)
        if kind is None:
            return category, None

        if content_type.startswith(f"{kind}/"):
            return category, content_type
        return category, _extension_media_type(url, kind)

    def _model_supports(self, ctx: RunContext[AgentContext], kind: MediaKind) -> bool:
        model_cfg = ctx.deps.model_cfg
        if kind == "image":
            return model_cfg.has_capability(ModelCapability.vision)
        if kind == "video":
            return model_cfg.has_capability(ModelCapability.video_understanding)
        return model_cfg.has_capability(ModelCapability.audio_understanding)

    def _unsupported_capability_error(self, url: str, kind: MediaKind) -> dict[str, Any]:
        if kind == "image":
            capability = "vision"
        elif kind == "video":
            capability = "video understanding"
        else:
            capability = "audio understanding"
        return _error(
            f"The URL '{url}' appears to be {kind} media, but the current model does not support {capability}."
        )

    def _max_inline_bytes(self, ctx: RunContext[AgentContext], kind: MediaKind) -> int:
        tool_config = ctx.deps.tool_config
        if kind == "image":
            return tool_config.view_max_inline_image_bytes
        if kind == "video":
            return tool_config.view_max_inline_video_bytes
        return tool_config.view_max_inline_audio_bytes

    def _declared_size(self, response: httpx.Response) -> int | None:
        content_length = response.headers.get("Content-Length")
        if not content_length:
            return None
        with contextlib.suppress(ValueError, OverflowError):
            return int(content_length)
        return None

    def _size_error(
        self,
        *,
        kind: MediaKind,
        max_bytes: int,
        size: int,
        downloaded: bool = False,
    ) -> dict[str, Any]:
        if downloaded:
            return _error(
                f"The {kind} URL exceeded the safe in-memory limit while downloading ({size} bytes). "
                f"Maximum supported size is {max_bytes} bytes."
            )
        return _error(
            f"The {kind} URL is too large to read into memory safely ({size} bytes). "
            f"Maximum supported size is {max_bytes} bytes."
        )

    async def _read_limited_body(
        self,
        ctx: RunContext[AgentContext],
        response: httpx.Response,
        *,
        kind: MediaKind,
        max_bytes: int,
    ) -> bytes | dict[str, Any]:
        data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=ctx.deps.tool_config.fetch_stream_chunk_size):
            data.extend(chunk)
            if len(data) > max_bytes:
                return self._size_error(kind=kind, max_bytes=max_bytes, size=len(data), downloaded=True)
        return bytes(data)

    async def _prepare_image(
        self,
        ctx: RunContext[AgentContext],
        data: bytes,
        media_type: str | None,
        *,
        url: str,
    ) -> tuple[bytes, str] | dict[str, Any]:
        detected_type = detect_image_media_type(data)
        if detected_type is not None:
            media_type = detected_type
        media_type = _main_content_type(media_type)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
            return _error(
                f"Unsupported image format '{media_type or 'unknown'}' for URL '{url}'. Supported formats: {supported}."
            )

        max_encoded_bytes = ctx.deps.model_cfg.max_image_bytes
        if max_encoded_bytes <= 0:
            return data, media_type

        max_raw_bytes = raw_bytes_limit_for_base64(max_encoded_bytes)
        if len(data) <= max_raw_bytes:
            return data, media_type

        try:
            compressed_data, compressed_media_type = await compress_image_to_model_limit(
                data,
                max_encoded_bytes=max_encoded_bytes,
                media_type=media_type,
            )
        except Exception:
            logger.exception("Failed to compress remote image from %s before inlining", url)
            return _error(f"Image from URL '{url}' could not be compressed for inline model input.")

        if len(compressed_data) > max_raw_bytes:
            return _error(
                f"Image from URL '{url}' could not be compressed below the {max_encoded_bytes} byte API limit "
                "after accounting for base64 encoding."
            )

        logger.info(
            "Compressed remote image from %s from %d bytes to %d bytes before inlining",
            url,
            len(data),
            len(compressed_data),
        )
        return compressed_data, compressed_media_type

    def _build_media_return(
        self,
        *,
        kind: MediaKind,
        content: BinaryContent | VideoUrl,
        instructions: str | None,
    ) -> ToolReturn:
        return_value = f"The {kind} is attached in the user message."
        if instructions and instructions.strip():
            return_value = f"{return_value}\n\nAnalysis instructions:\n{instructions.strip()}"
        return ToolReturn(return_value=return_value, content=[content])

    def _read_youtube_url(
        self,
        ctx: RunContext[AgentContext],
        *,
        url: str,
        instructions: str | None,
    ) -> dict[str, Any] | ToolReturn:
        if not self._model_supports(ctx, "video"):
            return self._unsupported_capability_error(url, "video")
        if not ctx.deps.model_cfg.has_youtube_url:
            return _error(
                f"The URL '{url}' appears to be a YouTube video, but the current model does not support direct YouTube URLs."
            )

        return self._build_media_return(
            kind="video",
            content=VideoUrl(url=url),
            instructions=instructions,
        )

    async def _read_response(
        self,
        ctx: RunContext[AgentContext],
        response: httpx.Response,
        *,
        url: str,
        instructions: str | None,
    ) -> dict[str, Any] | ToolReturn:
        category, media_type = self._category_and_media_type(response, url)
        kind = _kind_for_category(category)
        if kind is None:
            return _error(f"The URL '{url}' does not look like a supported image, video, or audio resource.")

        if not self._model_supports(ctx, kind):
            return self._unsupported_capability_error(url, kind)

        max_bytes = self._max_inline_bytes(ctx, kind)
        declared_size = self._declared_size(response)
        if declared_size is not None and declared_size > max_bytes:
            return self._size_error(kind=kind, max_bytes=max_bytes, size=declared_size)

        body = await self._read_limited_body(ctx, response, kind=kind, max_bytes=max_bytes)
        if isinstance(body, dict):
            return body

        if kind == "image":
            prepared_image = await self._prepare_image(ctx, body, media_type, url=url)
            if isinstance(prepared_image, dict):
                return prepared_image
            body, media_type = prepared_image
        elif media_type is None:
            return _error(f"Could not determine a media type for URL '{url}'.")

        return self._build_media_return(
            kind=kind,
            content=BinaryContent(data=body, media_type=media_type),
            instructions=instructions,
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[str, Field(description="HTTP or HTTPS URL of the image, video, or audio resource to read.")],
        instructions: Annotated[
            str | None,
            Field(
                description=(
                    "Optional focused analysis instructions for the attached media, such as OCR, UI review, "
                    "transcription, timestamped summary, or speaker identification."
                ),
                default=None,
            ),
        ] = None,
    ) -> dict[str, Any] | ToolReturn:
        """Download a media URL into bounded in-memory binary content."""
        if not is_valid_http_url(url):
            return _error(f"Only HTTP and HTTPS URLs are supported. The provided URL '{url}' is not supported.")

        if _is_youtube_url(url):
            return self._read_youtube_url(ctx, url=url, instructions=instructions)

        try:
            async with safe_stream_request(
                url,
                method="GET",
                timeout=60.0,
                skip_verification=ctx.deps.tool_config.skip_url_verification,
            ) as response:
                response.raise_for_status()
                return await self._read_response(ctx, response, url=url, instructions=instructions)

        except ForbiddenUrlError as e:
            return _error(f"URL forbidden: {e}")
        except Exception:
            logger.exception("Failed to read media URL %s", url)
            return _error(f"Failed to read media URL '{url}'.")


__all__ = ["ReadMediaTool"]
