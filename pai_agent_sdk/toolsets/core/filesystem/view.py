"""View tool for reading files.

Supports text files, images, videos, and audio files.
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import BinaryContent, RunContext, ToolReturn

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._types import (
    ViewMetadata,
    ViewReadingParams,
    ViewSegment,
    ViewTruncationInfo,
)

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Image file extensions that can be displayed as BinaryContent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp"}

# Image media types supported for display in the LLM context
SUPPORTED_IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}

# Video file extensions
VIDEO_EXTENSIONS = {
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
}

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


@cache
def _load_instruction() -> str:
    """Load view instruction from prompts/view.md."""
    prompt_file = _PROMPTS_DIR / "view.md"
    return prompt_file.read_text()


class ViewTool(BaseTool):
    """Tool for reading files from the filesystem.

    Supports text files, images, and videos.
    Default line_limit is 300.
    """

    name = "view"
    description = (
        "Read files from local filesystem. Supports text, images (PNG/JPEG/WebP), and videos (MP4/WebM/MOV). "
        "For PDF files, use `pdf_convert` tool instead."
    )

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/view.md."""
        return _load_instruction()

    def _is_image_file(self, path: Path) -> bool:
        """Check if a file is a displayable image based on extension."""
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _is_video_file(self, path: Path) -> bool:
        """Check if a file is a video based on extension."""
        return path.suffix.lower() in VIDEO_EXTENSIONS

    def _get_media_type(self, path: Path) -> str:
        """Get media type from file extension."""
        ext = path.suffix.lower()
        return MEDIA_TYPE_MAP.get(ext, "application/octet-stream")

    def _read_as_image(self, path: Path) -> BinaryContent:
        """Read a file as an image and return BinaryContent."""
        media_type = self._get_media_type(path)

        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            media_type = "image/gif" if media_type == "image/gif" else "image/png"

        with path.open("rb") as f:
            content = f.read()

        return BinaryContent(data=content, media_type=media_type)

    async def _read_video_with_fallback(
        self, path: Path, file_path: str, ctx: RunContext[AgentContext]
    ) -> str | ToolReturn:
        """Read video file, falling back to image description if video not supported."""
        # Check if current model supports video understanding via model_config
        model_config = getattr(ctx.deps, "model_config", None)
        is_video_model = model_config.is_video_understanding_model if model_config else False

        if is_video_model:
            # Return video content directly
            media_type = "video/mp4"  # Default
            ext = path.suffix.lower()
            if ext == ".webm":
                media_type = "video/webm"
            elif ext == ".mov":
                media_type = "video/quicktime"

            with path.open("rb") as f:
                content = f.read()

            return ToolReturn(
                return_value="The video is attached in the user message.",
                content=[BinaryContent(data=content, media_type=media_type)],
            )
        else:
            # Fall back to image understanding agent for video frames
            try:
                from pai_agent_sdk.agents.image_understanding import get_image_description

                # Read video file
                with path.open("rb") as f:
                    video_data = f.read()

                # Get model and settings from tool_config if available
                model = None
                model_settings = None
                if ctx.deps.tool_config:
                    tool_config = ctx.deps.tool_config
                    model = tool_config.image_understanding_model
                    model_settings = tool_config.image_understanding_model_settings

                # Use image understanding to describe video
                description, usage = await get_image_description(
                    image_data=video_data,
                    media_type="image/png",
                    instruction="This is a video file. Please describe what you can see in this video content.",
                    model=model,
                    model_settings=model_settings,
                )

                # Store usage in extra_usages with tool_call_id
                if ctx.tool_call_id:
                    ctx.deps.add_extra_usage(agent="image_understanding", usage=usage, uuid=ctx.tool_call_id)

                return f"Video description (via image analysis):\n{description}"
            except Exception as e:
                logger.warning(f"Failed to analyze video with image understanding: {e}")
                return (
                    f"Video file: {file_path}. Model does not support video understanding and fallback analysis failed."
                )

    def _return_text_file(
        self, path: Path, line_offset: int | None, line_limit: int, max_line_length: int
    ) -> str | dict[str, Any]:
        """Read text file and return either simple content string or detailed dict with metadata."""
        lines_truncated = False
        content_truncated = False

        with path.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        total_chars = sum(len(line) for line in all_lines)
        file_size = path.stat().st_size

        if line_offset is not None and line_offset > 0:
            all_lines = all_lines[line_offset:]
            has_offset = True
        else:
            has_offset = False
            line_offset = 0

        if len(all_lines) > line_limit:
            all_lines = all_lines[:line_limit]
            has_line_limit = True
        else:
            has_line_limit = False

        processed_lines = []
        for line in all_lines:
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... (line truncated)\n"
                lines_truncated = True
            processed_lines.append(line)

        content = "".join(processed_lines)

        if len(content) > 60000:
            content = content[:60000] + "\n... (content truncated)"
            content_truncated = True

        needs_metadata = has_offset or has_line_limit or lines_truncated or content_truncated

        if not needs_metadata:
            return content
        else:
            start_line = line_offset + 1
            actual_lines_read = len(processed_lines)
            end_line = start_line + actual_lines_read - 1 if actual_lines_read > 0 else start_line

            return {
                "content": content,
                "metadata": ViewMetadata(
                    file_path=path.name,
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
        """Read a file from the local filesystem."""
        file_operator = ctx.deps.file_operator

        if not await file_operator.exists(file_path):
            return f"Error: File not found: {file_path}"

        if await file_operator.is_dir(file_path):
            return f"Error: Path is a directory, not a file: {file_path}"

        path = ctx.deps.file_operator._default_path / file_path
        if not path.exists():
            path = Path(file_path)
            if not path.is_absolute():
                path = ctx.deps.file_operator._default_path / file_path

        if self._is_image_file(path):
            return ToolReturn(
                return_value="The image is attached in the user message.",
                content=[self._read_as_image(path)],
            )

        if self._is_video_file(path):
            return await self._read_video_with_fallback(path, file_path, ctx)

        return self._return_text_file(path, line_offset, line_limit, max_line_length)


__all__ = ["ViewTool"]
