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
from ya_agent_sdk.utils import (
    compress_image_to_model_limit,
    detect_image_media_type,
    image_exceeds_limits,
    raw_bytes_limit_for_base64,
)

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
    ".avif": "image/avif",
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
    return "Use `download` to save the URL to a local file, then call `view` on it with focused `instructions`."


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message, "fallback": _fallback_guidance()}


def _fallback_analysis_error(
    *,
    source: str,
    kind: MediaKind,
    exc: Exception,
) -> dict[str, Any]:
    return _error(f"{_analysis_failure_reason(kind=kind, exc=exc)} URL: '{source}'.")


def _analysis_failure_reason(*, kind: MediaKind, exc: Exception) -> str:
    status_code, reason_phrase = _http_status_from_exception(exc)
    if status_code is not None:
        if status_code == 429:
            return (
                f"{kind.capitalize()} analysis is rate limited by the configured {kind} understanding model "
                f"(429 Too Many Requests)."
            )
        if reason_phrase:
            return f"{kind.capitalize()} analysis request failed with HTTP {status_code} {reason_phrase}."
        return f"{kind.capitalize()} analysis request failed with HTTP {status_code}."

    message = str(exc).splitlines()[0].strip()
    if " for url " in message:
        message = message.split(" for url ", maxsplit=1)[0].rstrip()
    if not message:
        return f"{kind.capitalize()} analysis failed."
    return f"{kind.capitalize()} analysis failed: {message}."


def _http_status_from_exception(exc: BaseException) -> tuple[int | None, str | None]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            reason_phrase = getattr(response, "reason_phrase", None)
            return status_code, str(reason_phrase) if reason_phrase else None

        cause = getattr(current, "cause", None)
        if isinstance(cause, BaseException):
            current = cause
            continue
        current = current.__cause__ or current.__context__
    return None, None


def _url_read_error(*, url: str, exc: Exception) -> dict[str, Any]:
    status_code, reason_phrase = _http_status_from_exception(exc)
    if status_code is not None:
        if reason_phrase:
            return _error(f"Failed to read media URL: HTTP {status_code} {reason_phrase}. URL: '{url}'.")
        return _error(f"Failed to read media URL: HTTP {status_code}. URL: '{url}'.")

    message = str(exc).splitlines()[0].strip()
    if " for url " in message:
        message = message.split(" for url ", maxsplit=1)[0].rstrip()
    if message:
        return _error(f"Failed to read media URL: {message}. URL: '{url}'.")
    return _error(f"Failed to read media URL '{url}'.")


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
        max_dimension = ctx.deps.model_cfg.max_image_dimension
        max_raw_bytes = raw_bytes_limit_for_base64(max_encoded_bytes) if max_encoded_bytes > 0 else None
        if not image_exceeds_limits(
            data,
            max_bytes=max_raw_bytes,
            max_dimension=max_dimension,
        ):
            return data, media_type

        try:
            compressed_data, compressed_media_type = await compress_image_to_model_limit(
                data,
                max_encoded_bytes=max_encoded_bytes,
                media_type=media_type,
                max_dimension=max_dimension,
            )
        except Exception:
            logger.exception("Failed to compress remote image from %s before inlining", url)
            return _error(f"Image from URL '{url}' could not be compressed for inline model input.")

        if image_exceeds_limits(
            compressed_data,
            max_bytes=max_raw_bytes,
            max_dimension=max_dimension,
        ):
            return _error(
                f"Image from URL '{url}' could not be compressed within the configured model image limits "
                f"({max_encoded_bytes} encoded bytes, {max_dimension} pixels per dimension)."
            )

        logger.info(
            "Compressed remote image from %s from %d bytes to %d bytes before inlining",
            url,
            len(data),
            len(compressed_data),
        )
        return compressed_data, compressed_media_type

    def _fallback_image_media_type(self, data: bytes, media_type: str | None, *, url: str) -> str | None:
        detected_type = detect_image_media_type(data)
        if detected_type is not None:
            return detected_type

        media_type = _main_content_type(media_type)
        if media_type:
            return media_type

        return _extension_media_type(url, "image")

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

    def _record_understanding_usage(
        self,
        ctx: RunContext[AgentContext],
        *,
        agent_name: str,
        model_id: str,
        usage: Any,
    ) -> None:
        tool_call_id = getattr(ctx, "tool_call_id", None)
        if not tool_call_id:
            return

        ctx.deps.update_usage_snapshot_entry(
            agent_id=agent_name,
            agent_name=agent_name,
            model_id=model_id,
            usage=usage,
            source=agent_name,
            usage_id=tool_call_id,
            ledger_key=tool_call_id,
        )

    async def _describe_image(
        self,
        ctx: RunContext[AgentContext],
        *,
        source: str,
        media_type: str,
        instructions: str | None,
        image_data: bytes,
    ) -> str | dict[str, Any]:
        """Describe image via the fallback image-understanding agent."""
        try:
            from ya_agent_sdk.agents.image_understanding import get_image_description

            model = None
            model_settings = None
            if ctx.deps.tool_config:
                tool_config = ctx.deps.tool_config
                model = tool_config.image_understanding_model
                model_settings = tool_config.image_understanding_model_settings

            description, model_id, usage = await get_image_description(
                image_url=None,
                image_data=image_data,
                media_type=media_type,
                instruction=instructions,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            self._record_understanding_usage(
                ctx,
                agent_name="image_understanding",
                model_id=model_id,
                usage=usage,
            )
            return f"Image description (via image analysis):\n{description}"
        except Exception as e:
            logger.warning("Failed to analyze image URL %s with image understanding: %s", source, e)
            return _fallback_analysis_error(
                source=source,
                kind="image",
                exc=e,
            )

    async def _describe_video(
        self,
        ctx: RunContext[AgentContext],
        *,
        source: str,
        media_type: str,
        instructions: str | None,
        video_url: str | None = None,
        video_data: bytes | None = None,
    ) -> str | dict[str, Any]:
        """Describe video via the fallback video-understanding agent."""
        try:
            from ya_agent_sdk.agents.video_understanding import get_video_description

            model = None
            model_settings = None
            if ctx.deps.tool_config:
                tool_config = ctx.deps.tool_config
                model = tool_config.video_understanding_model
                model_settings = tool_config.video_understanding_model_settings

            description, model_id, usage = await get_video_description(
                video_url=video_url,
                video_data=video_data,
                media_type=media_type,
                instruction=instructions,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            self._record_understanding_usage(
                ctx,
                agent_name="video_understanding",
                model_id=model_id,
                usage=usage,
            )
            return f"Video description (via video understanding agent):\n{description}"
        except Exception as e:
            logger.warning("Failed to analyze video URL %s with video understanding: %s", source, e)
            return _fallback_analysis_error(
                source=source,
                kind="video",
                exc=e,
            )

    async def _describe_audio(
        self,
        ctx: RunContext[AgentContext],
        *,
        source: str,
        media_type: str,
        instructions: str | None,
        audio_data: bytes,
    ) -> str | dict[str, Any]:
        """Describe audio via the fallback audio-understanding agent."""
        try:
            from ya_agent_sdk.agents.audio_understanding import get_audio_description

            model = None
            model_settings = None
            if ctx.deps.tool_config:
                tool_config = ctx.deps.tool_config
                model = tool_config.audio_understanding_model
                model_settings = tool_config.audio_understanding_model_settings

            description, model_id, usage = await get_audio_description(
                audio_url=None,
                audio_data=audio_data,
                media_type=media_type,
                instruction=instructions,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            self._record_understanding_usage(
                ctx,
                agent_name="audio_understanding",
                model_id=model_id,
                usage=usage,
            )
            return f"Audio description (via audio understanding agent):\n{description}"
        except Exception as e:
            logger.warning("Failed to analyze audio URL %s with audio understanding: %s", source, e)
            return _fallback_analysis_error(
                source=source,
                kind="audio",
                exc=e,
            )

    async def _read_youtube_url(
        self,
        ctx: RunContext[AgentContext],
        *,
        url: str,
        instructions: str | None,
    ) -> str | dict[str, Any] | ToolReturn:
        if self._model_supports(ctx, "video") and ctx.deps.model_cfg.has_youtube_url:
            return self._build_media_return(
                kind="video",
                content=VideoUrl(url=url),
                instructions=instructions,
            )

        return await self._describe_video(
            ctx,
            source=url,
            video_url=url,
            media_type="video/mp4",
            instructions=instructions,
        )

    async def _describe_media_fallback(
        self,
        ctx: RunContext[AgentContext],
        *,
        kind: MediaKind,
        source: str,
        data: bytes,
        media_type: str,
        instructions: str | None,
    ) -> str | dict[str, Any]:
        if kind == "image":
            return await self._describe_image(
                ctx,
                source=source,
                image_data=data,
                media_type=media_type,
                instructions=instructions,
            )
        if kind == "video":
            return await self._describe_video(
                ctx,
                source=source,
                video_data=data,
                media_type=media_type,
                instructions=instructions,
            )
        return await self._describe_audio(
            ctx,
            source=source,
            audio_data=data,
            media_type=media_type,
            instructions=instructions,
        )

    async def _prepare_image_or_describe(
        self,
        ctx: RunContext[AgentContext],
        *,
        data: bytes,
        media_type: str | None,
        url: str,
        instructions: str | None,
        model_supports_image: bool,
    ) -> tuple[bytes, str] | str | dict[str, Any]:
        prepared_image = await self._prepare_image(ctx, data, media_type, url=url)
        if not isinstance(prepared_image, dict):
            return prepared_image

        model_cfg = ctx.deps.model_cfg
        max_encoded_bytes = model_cfg.max_image_bytes
        max_raw_bytes = raw_bytes_limit_for_base64(max_encoded_bytes) if max_encoded_bytes > 0 else None
        if image_exceeds_limits(
            data,
            max_bytes=max_raw_bytes,
            max_dimension=model_cfg.max_image_dimension,
        ):
            # Compression failures and unverifiable dimensions must fail closed;
            # otherwise the original image could exceed the fallback model limits.
            return prepared_image

        fallback_media_type = self._fallback_image_media_type(data, media_type, url=url)
        if fallback_media_type is None:
            return prepared_image
        if not model_supports_image:
            return data, fallback_media_type

        return await self._describe_image(
            ctx,
            source=url,
            image_data=data,
            media_type=fallback_media_type,
            instructions=instructions,
        )

    async def _read_response(
        self,
        ctx: RunContext[AgentContext],
        response: httpx.Response,
        *,
        url: str,
        instructions: str | None,
    ) -> str | dict[str, Any] | ToolReturn:
        category, media_type = self._category_and_media_type(response, url)
        kind = _kind_for_category(category)
        if kind is None:
            return _error(f"The URL '{url}' does not look like a supported image, video, or audio resource.")

        max_bytes = self._max_inline_bytes(ctx, kind)
        declared_size = self._declared_size(response)
        if declared_size is not None and declared_size > max_bytes:
            return self._size_error(kind=kind, max_bytes=max_bytes, size=declared_size)

        body = await self._read_limited_body(ctx, response, kind=kind, max_bytes=max_bytes)
        if isinstance(body, dict):
            return body

        model_supports_kind = self._model_supports(ctx, kind)
        if kind == "image":
            prepared_image = await self._prepare_image_or_describe(
                ctx,
                data=body,
                media_type=media_type,
                url=url,
                instructions=instructions,
                model_supports_image=model_supports_kind,
            )
            if isinstance(prepared_image, str | dict):
                return prepared_image
            body, media_type = prepared_image
        elif media_type is None:
            return _error(f"Could not determine a media type for URL '{url}'.")

        if not model_supports_kind:
            return await self._describe_media_fallback(
                ctx,
                kind=kind,
                source=url,
                data=body,
                media_type=media_type,
                instructions=instructions,
            )

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
    ) -> str | dict[str, Any] | ToolReturn:
        """Download a media URL into bounded in-memory binary content."""
        if not is_valid_http_url(url):
            return _error(f"Only HTTP and HTTPS URLs are supported. The provided URL '{url}' is not supported.")

        if _is_youtube_url(url):
            return await self._read_youtube_url(ctx, url=url, instructions=instructions)

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
        except Exception as e:
            logger.exception("Failed to read media URL %s", url)
            return _url_read_error(url=url, exc=e)


__all__ = ["ReadMediaTool"]
