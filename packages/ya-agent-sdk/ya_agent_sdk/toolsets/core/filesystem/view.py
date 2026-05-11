"""View tool for reading files.

Supports text files, images, videos, and audio files.
All file operations use the FileOperator abstraction for remote filesystem support.
"""

import inspect
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import Field
from pydantic_ai import BinaryContent, ImageUrl, RunContext, ToolReturn, VideoUrl
from pydantic_ai.messages import AudioUrl
from y_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import (
    ViewMetadata,
    ViewReadingParams,
    ViewSegment,
    ViewTruncationInfo,
)
from ya_agent_sdk.toolsets.core.filesystem._utils import is_binary_file
from ya_agent_sdk.utils import detect_image_media_type, run_in_threadpool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Image file extensions that can be displayed as BinaryContent
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp"})

# Image media types supported for display in the LLM context
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

# Video file extensions
VIDEO_EXTENSIONS = frozenset({
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".mkv",
    ".m4v",
    ".ogv",
})

# Audio file extensions
AUDIO_EXTENSIONS = frozenset({
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".aiff",
    ".aif",
})

# Media type mapping for common extensions
MEDIA_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
}

# Video media type mapping
VIDEO_MEDIA_TYPE_MAP = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".ogv": "video/ogg",
}

# Audio media type mapping
AUDIO_MEDIA_TYPE_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".opus": "audio/opus",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}


@cache
def _load_instruction() -> str:
    """Load view instruction from prompts/view.md."""
    prompt_file = _PROMPTS_DIR / "view.md"
    return prompt_file.read_text()


class ViewTool(BaseTool):
    """Tool for reading files from the filesystem.

    Supports text files, images, videos, and audio files.
    All operations use FileOperator abstraction for remote filesystem support.
    """

    name = "view"
    description = (
        "Read files from local filesystem. Supports text, images (PNG/JPEG/WebP), videos (MP4/WebM/MOV), "
        "and audio (MP3/WAV/OGG). For PDF files, use `pdf_convert` tool instead."
    )

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("ViewTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/view.md."""
        return _load_instruction()

    # --- Path and type utilities (string-based, no Path dependency) ---

    def _get_extension(self, file_path: str) -> str:
        """Extract file extension from path string."""
        idx = file_path.rfind(".")
        if idx == -1:
            return ""
        # Handle cases like "/path/to/.hidden" where dot is part of filename
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        if idx < last_sep:
            return ""
        return file_path[idx:].lower()

    def _is_image_file(self, file_path: str) -> bool:
        """Check if a file is a displayable image based on extension."""
        return self._get_extension(file_path) in IMAGE_EXTENSIONS

    def _is_video_file(self, file_path: str) -> bool:
        """Check if a file is a video based on extension."""
        return self._get_extension(file_path) in VIDEO_EXTENSIONS

    def _is_audio_file(self, file_path: str) -> bool:
        """Check if a file is an audio file based on extension."""
        return self._get_extension(file_path) in AUDIO_EXTENSIONS

    def _get_media_type(self, file_path: str) -> str:
        """Get media type from file extension."""
        ext = self._get_extension(file_path)
        return MEDIA_TYPE_MAP.get(ext, "application/octet-stream")

    def _get_video_media_type(self, file_path: str) -> str:
        """Get video media type from file extension."""
        ext = self._get_extension(file_path)
        return VIDEO_MEDIA_TYPE_MAP.get(ext, "video/mp4")

    def _get_audio_media_type(self, file_path: str) -> str:
        """Get audio media type from file extension."""
        ext = self._get_extension(file_path)
        return AUDIO_MEDIA_TYPE_MAP.get(ext, "audio/mpeg")

    # --- File reading methods (async, using FileOperator) ---

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size into a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    async def _check_inline_size(
        self,
        file_operator: FileOperator,
        file_path: str,
        *,
        max_bytes: int,
        kind: str,
    ) -> str | None:
        """Validate that a media file is small enough to inline."""
        stat = await file_operator.stat(file_path)
        if stat["size"] <= max_bytes:
            return None

        return (
            f"Error: {kind} file is too large to inline ({self._format_size(stat['size'])}). "
            f"Maximum supported inline size is {self._format_size(max_bytes)}."
        )

    async def _maybe_convert_media_to_url(
        self,
        hook: Any,
        ctx: RunContext[AgentContext],
        data: bytes,
        media_type: str,
        *,
        log_name: str,
    ) -> str | None:
        """Run an optional media-to-URL hook and normalize empty results."""
        try:
            if inspect.iscoroutinefunction(hook):
                result = await hook(ctx, data, media_type)
            else:
                result = await run_in_threadpool(hook, ctx, data, media_type)
            result = cast("str | None", result)
            return result if result and result.strip() else None
        except Exception:
            logger.warning("%s failed, falling back to data", log_name, exc_info=True)
            return None

    async def _describe_image(
        self,
        ctx: RunContext[AgentContext],
        file_path: str,
        image_data: bytes,
        media_type: str,
        image_url: str | None,
    ) -> str:
        """Describe an image via the fallback image-understanding agent."""
        try:
            from ya_agent_sdk.agents.image_understanding import get_image_description

            model = None
            model_settings = None
            if ctx.deps.tool_config:
                tool_config = ctx.deps.tool_config
                model = tool_config.image_understanding_model
                model_settings = tool_config.image_understanding_model_settings

            description, model_id, usage = await get_image_description(
                image_url=image_url,
                image_data=None if image_url else image_data,
                media_type=media_type,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            if ctx.tool_call_id:
                await ctx.deps.emit_usage_snapshot(
                    agent_id="image_understanding",
                    agent_name="image_understanding",
                    model_id=model_id,
                    usage=usage,
                    source="image_understanding",
                    usage_id=ctx.tool_call_id,
                    ledger_key=ctx.tool_call_id,
                )

            return f"Image description (via image analysis):\n{description}"
        except Exception as e:
            logger.warning(f"Failed to analyze image with image understanding: {e}")
            return f"Image file: {file_path}. Model does not support vision and fallback analysis failed."

    async def _describe_video(
        self,
        ctx: RunContext[AgentContext],
        file_path: str,
        video_data: bytes,
        media_type: str,
        video_url: str | None,
    ) -> str:
        """Describe a video via the fallback video-understanding agent."""
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
                video_data=None if video_url else video_data,
                media_type=media_type,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            if ctx.tool_call_id:
                await ctx.deps.emit_usage_snapshot(
                    agent_id="video_understanding",
                    agent_name="video_understanding",
                    model_id=model_id,
                    usage=usage,
                    source="video_understanding",
                    usage_id=ctx.tool_call_id,
                    ledger_key=ctx.tool_call_id,
                )

            return f"Video description (via video understanding agent):\n{description}"
        except Exception as e:
            logger.warning(f"Failed to analyze video with video understanding: {e}")
            return f"Video file: {file_path}. Model does not support video understanding and fallback analysis failed."

    async def _describe_audio(
        self,
        ctx: RunContext[AgentContext],
        file_path: str,
        audio_data: bytes,
        media_type: str,
        audio_url: str | None,
    ) -> str:
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
                audio_url=audio_url,
                audio_data=None if audio_url else audio_data,
                media_type=media_type,
                model=model,
                model_settings=model_settings,
                model_wrapper=ctx.deps.model_wrapper,
                wrapper_metadata=ctx.deps.get_wrapper_metadata(),
            )

            if ctx.tool_call_id:
                await ctx.deps.emit_usage_snapshot(
                    agent_id="audio_understanding",
                    agent_name="audio_understanding",
                    model_id=model_id,
                    usage=usage,
                    source="audio_understanding",
                    usage_id=ctx.tool_call_id,
                    ledger_key=ctx.tool_call_id,
                )

            return f"Audio description (via audio understanding agent):\n{description}"
        except Exception as e:
            logger.warning(f"Failed to analyze audio with audio understanding: {e}")
            return f"Audio file: {file_path}. Model does not support audio understanding and fallback analysis failed."

    def _build_inline_media_return(
        self,
        *,
        kind: Literal["image", "video", "audio"],
        media_url: str | None,
        data: bytes,
        media_type: str,
    ) -> ToolReturn:
        """Build the ToolReturn payload for inline media."""
        if kind == "image":
            content = [ImageUrl(url=media_url)] if media_url else [BinaryContent(data=data, media_type=media_type)]
            return ToolReturn(return_value="The image is attached in the user message.", content=content)

        if kind == "audio":
            content = [AudioUrl(url=media_url)] if media_url else [BinaryContent(data=data, media_type=media_type)]
            return ToolReturn(return_value="The audio is attached in the user message.", content=content)

        content = [VideoUrl(url=media_url)] if media_url else [BinaryContent(data=data, media_type=media_type)]
        return ToolReturn(return_value="The video is attached in the user message.", content=content)

    async def _read_image_with_fallback(
        self,
        file_operator: FileOperator,
        file_path: str,
        ctx: RunContext[AgentContext],
    ) -> str | ToolReturn:
        """Read image file, falling back to description if vision not supported."""
        if error := await self._check_inline_size(
            file_operator,
            file_path,
            max_bytes=ctx.deps.tool_config.view_max_inline_image_bytes,
            kind="Image",
        ):
            return error

        image_data = await file_operator.read_bytes(file_path)
        # Detect media type from content first to handle extension/content mismatches
        # (e.g. a JPEG file named .png, or an image with no extension)
        media_type = detect_image_media_type(image_data) or self._get_media_type(file_path)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
            return f"Error: unsupported image format '{media_type}' for {file_path}. Supported formats: {supported}."

        image_url: str | None = None
        if ctx.deps.tool_config and ctx.deps.tool_config.image_to_url_hook:
            image_url = await self._maybe_convert_media_to_url(
                ctx.deps.tool_config.image_to_url_hook,
                ctx,
                image_data,
                media_type,
                log_name="image_to_url_hook",
            )

        if ctx.deps.model_cfg.has_vision:
            return self._build_inline_media_return(
                kind="image",
                media_url=image_url,
                data=image_data,
                media_type=media_type,
            )

        return await self._describe_image(ctx, file_path, image_data, media_type, image_url)

    async def _read_video_with_fallback(
        self,
        file_operator: FileOperator,
        file_path: str,
        ctx: RunContext[AgentContext],
    ) -> str | ToolReturn:
        """Read video file, falling back to video understanding agent if not supported."""
        if error := await self._check_inline_size(
            file_operator,
            file_path,
            max_bytes=ctx.deps.tool_config.view_max_inline_video_bytes,
            kind="Video",
        ):
            return error

        video_data = await file_operator.read_bytes(file_path)
        media_type = self._get_video_media_type(file_path)

        video_url: str | None = None
        if ctx.deps.tool_config and ctx.deps.tool_config.video_to_url_hook:
            video_url = await self._maybe_convert_media_to_url(
                ctx.deps.tool_config.video_to_url_hook,
                ctx,
                video_data,
                media_type,
                log_name="video_to_url_hook",
            )

        if ctx.deps.model_cfg.has_video_understanding:
            return self._build_inline_media_return(
                kind="video",
                media_url=video_url,
                data=video_data,
                media_type=media_type,
            )

        return await self._describe_video(ctx, file_path, video_data, media_type, video_url)

    async def _read_audio_with_fallback(
        self,
        file_operator: FileOperator,
        file_path: str,
        ctx: RunContext[AgentContext],
    ) -> str | ToolReturn:
        """Read audio file, falling back to audio understanding agent if not supported."""
        if error := await self._check_inline_size(
            file_operator,
            file_path,
            max_bytes=ctx.deps.tool_config.view_max_inline_audio_bytes,
            kind="Audio",
        ):
            return error

        audio_data = await file_operator.read_bytes(file_path)
        media_type = self._get_audio_media_type(file_path)

        audio_url: str | None = None
        if ctx.deps.tool_config and ctx.deps.tool_config.audio_to_url_hook:
            audio_url = await self._maybe_convert_media_to_url(
                ctx.deps.tool_config.audio_to_url_hook,
                ctx,
                audio_data,
                media_type,
                log_name="audio_to_url_hook",
            )

        if ctx.deps.model_cfg.has_audio_understanding:
            return self._build_inline_media_return(
                kind="audio",
                media_url=audio_url,
                data=audio_data,
                media_type=media_type,
            )

        return await self._describe_audio(ctx, file_path, audio_data, media_type, audio_url)

    def _build_text_metadata_response(
        self,
        *,
        file_path: str,
        content: str,
        total_lines: int,
        total_chars: int,
        file_size: int,
        line_offset: int,
        line_limit: int,
        has_offset: bool,
        lines_truncated: bool,
        content_truncated: bool,
        max_line_length: int,
        actual_lines_read: int,
    ) -> dict[str, Any]:
        """Build metadata-rich text response for paged/truncated reads."""
        start_line = line_offset + 1
        end_line = start_line + actual_lines_read - 1 if actual_lines_read > 0 else start_line

        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        filename = file_path[last_sep + 1 :] if last_sep != -1 else file_path

        return {
            "content": content,
            "metadata": ViewMetadata(
                file_path=filename,
                total_lines=total_lines,
                total_characters=total_chars,
                file_size_bytes=file_size,
                current_segment=ViewSegment(
                    start_line=start_line,
                    end_line=end_line,
                    lines_to_show=actual_lines_read,
                    has_more_content=end_line < total_lines,
                ),
                reading_parameters=ViewReadingParams(
                    line_offset=line_offset if has_offset else None,
                    line_limit=line_limit,
                ),
                truncation_info=ViewTruncationInfo(
                    lines_truncated=lines_truncated,
                    content_truncated=content_truncated,
                    max_line_length=max_line_length,
                ),
            ),
            "system": "Increase the `line_limit` and `max_line_length` if you need more context.",
        }

    async def _read_text_file(
        self,
        ctx: RunContext[AgentContext],
        file_operator: FileOperator,
        file_path: str,
        line_offset: int | None,
        line_limit: int,
        max_line_length: int,
    ) -> str | dict[str, Any]:
        """Read text file with pagination and truncation support."""
        stat = await file_operator.stat(file_path)
        file_size = stat["size"]
        max_text_file_size = ctx.deps.tool_config.view_max_text_file_size

        if file_size > max_text_file_size:
            return {
                "error": (
                    f"File is too large to inspect safely ({self._format_size(file_size)}). "
                    f"Maximum supported text view size is {self._format_size(max_text_file_size)}. "
                    f"Use shell tools (e.g. `head`, `tail`, `sed -n`) to read portions of this file."
                ),
                "success": False,
            }

        # Safe to read in one shot: stat check above guarantees bounded size
        if await is_binary_file(file_operator, file_path):
            return (
                f"Error: {file_path} appears to be a binary file. "
                f"Use appropriate tools (e.g. `pdf_convert` for PDFs, `xxd` for hex dumps) instead."
            )

        full_content = await file_operator.read_file(file_path)
        all_lines = full_content.splitlines(keepends=True)
        total_lines = len(all_lines)
        total_chars = len(full_content)
        start_index = line_offset if line_offset is not None and line_offset > 0 else 0
        has_offset = start_index > 0
        selected_lines = all_lines[start_index : start_index + line_limit]
        has_line_limit = len(all_lines[start_index:]) > line_limit
        lines_truncated = False
        content_truncated = False
        processed_lines: list[str] = []

        for line in selected_lines:
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... (line truncated)\n"
                lines_truncated = True
            processed_lines.append(line)

        content = "".join(processed_lines)
        if len(content) > 60000:
            content = content[:60000] + "\n... (content truncated)"
            content_truncated = True

        line_offset = start_index
        needs_metadata = has_offset or has_line_limit or lines_truncated or content_truncated
        if not needs_metadata:
            return content

        return self._build_text_metadata_response(
            file_path=file_path,
            content=content,
            total_lines=total_lines,
            total_chars=total_chars,
            file_size=file_size,
            line_offset=line_offset,
            line_limit=line_limit,
            has_offset=has_offset,
            lines_truncated=lines_truncated,
            content_truncated=content_truncated,
            max_line_length=max_line_length,
            actual_lines_read=len(processed_lines),
        )

    # --- Main entry point ---

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[
            str,
            Field(description="Relative path to the file to read"),
        ],
        line_offset: Annotated[
            int | None,
            Field(
                description="Line number to start reading from (0-indexed)",
                default=None,
            ),
        ] = None,
        line_limit: Annotated[
            int,
            Field(
                description="Maximum number of lines to read (default: 300)",
                default=300,
            ),
        ] = 300,
        max_line_length: Annotated[
            int,
            Field(
                description="Maximum length of each line before truncation",
                default=2000,
            ),
        ] = 2000,
    ) -> str | dict[str, Any] | ToolReturn:
        """Read a file from the filesystem."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not await file_operator.exists(file_path):
            return f"Error: File not found: {file_path}"

        if await file_operator.is_dir(file_path):
            return f"Error: Path is a directory, not a file: {file_path}"

        if self._is_image_file(file_path):
            return await self._read_image_with_fallback(file_operator, file_path, ctx)

        if self._is_video_file(file_path):
            return await self._read_video_with_fallback(file_operator, file_path, ctx)

        if self._is_audio_file(file_path):
            return await self._read_audio_with_fallback(file_operator, file_path, ctx)

        return await self._read_text_file(ctx, file_operator, file_path, line_offset, line_limit, max_line_length)


__all__ = ["ViewTool"]
