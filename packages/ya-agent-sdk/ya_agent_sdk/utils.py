from __future__ import annotations

import functools
import io
import socket
import typing
from typing import TYPE_CHECKING, Literal, cast

import anyio.to_thread
from PIL import Image
from pydantic_ai import AbstractToolset, Agent, ModelMessage, ModelResponse, RequestUsage, ToolCallPart
from pydantic_ai.messages import BinaryContent
from pydantic_ai.output import OutputDataT
from typing_extensions import TypeVar
from ya_agent_environment import Environment

from ya_agent_sdk._logger import get_logger

if TYPE_CHECKING:
    from ya_agent_sdk.context import AgentContext

logger = get_logger(__name__)

P = typing.ParamSpec("P")
T = typing.TypeVar("T")
AgentDepsT = TypeVar("AgentDepsT", bound="AgentContext")
EnvT = TypeVar("EnvT", bound=Environment, default=Environment)

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

# PIL format name -> MIME type mapping for image content detection
_PIL_FORMAT_TO_MEDIA_TYPE: dict[str, ImageMediaType] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(_PIL_FORMAT_TO_MEDIA_TYPE.values())


def normalize_image_media_type(data: bytes, media_type: str | None = None) -> ImageMediaType:
    """Return a supported image media type using content detection first."""
    detected = detect_image_media_type(data)
    if detected:
        return detected
    if media_type in _SUPPORTED_IMAGE_MEDIA_TYPES:
        return cast(ImageMediaType, media_type)
    return "image/jpeg"


def detect_image_media_type(data: bytes) -> ImageMediaType | None:
    """Detect actual image media type from raw bytes using PIL.

    Inspects the image content (magic bytes / file header) rather than relying
    on the file extension, which may not match the actual content.  This prevents
    Anthropic API rejections caused by a declared ``media_type`` that disagrees
    with the real payload.

    Args:
        data: Raw image bytes.

    Returns:
        The detected media type (one of the ``ImageMediaType`` literals), or
        ``None`` when the format cannot be determined (e.g. corrupted data or
        an unsupported image format).
    """
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format  # e.g. "JPEG", "PNG", "GIF", "WEBP"
        if fmt is None:
            return None
        return _PIL_FORMAT_TO_MEDIA_TYPE.get(fmt.upper())
    except Exception:
        return None


def get_available_port() -> int:
    """Get an available port on localhost.

    Note: There is a small race condition window between getting the port
    and actually binding to it. For most use cases this is acceptable.

    Returns:
        int: Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_in_threadpool(func: typing.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    # copied from fastapi.concurrency import run_in_threadpool
    func = functools.partial(func, *args, **kwargs)
    return await anyio.to_thread.run_sync(func)


def get_latest_request_usage(message_history: list[ModelMessage]) -> RequestUsage | None:
    """
    Retrieve the latest RequestUsage from the message history.

    Args:
        message_history: List of model messages from conversation

    Returns:
        The latest RequestUsage if available, otherwise None
    """
    for message in reversed(message_history):
        if isinstance(message, ModelResponse) and message.usage:
            return message.usage
    return None


def add_toolset_instructions(
    agent: Agent[AgentDepsT, OutputDataT], toolsets: list[AbstractToolset]
) -> Agent[AgentDepsT, OutputDataT]:
    """Return *agent* unchanged; Pydantic AI owns toolset instruction collection.

    ``ya-agent-sdk`` depends on Pydantic AI versions where
    ``AbstractToolset.get_instructions()`` is native. Keeping this small helper
    preserves the existing factory call sites without re-injecting or flattening
    toolset instructions through ``@agent.instructions``. This lets toolsets
    return ``InstructionPart`` objects directly so static/dynamic cache metadata
    reaches providers such as Anthropic and Bedrock.
    """
    _ = toolsets
    return agent


def get_tool_name_from_id(tool_id: str, message_history: list[ModelMessage]) -> str | None:
    """
    Retrieve the tool name corresponding to a given tool ID from message history.

    Args:
        tool_id: The tool call ID to look for
        message_history: List of model messages from conversation

    Returns:
        The tool name if found, otherwise None
    """
    if not message_history:
        return None
    for message in message_history:
        if isinstance(message, ModelResponse) and any(
            isinstance(p, ToolCallPart) and p.tool_call_id == tool_id for p in message.parts
        ):
            for p in message.parts:
                if isinstance(p, ToolCallPart) and p.tool_call_id == tool_id:
                    return p.tool_name
    return None


async def compress_image_data(
    image_bytes: bytes,
    max_bytes: int = 5 * 1024 * 1024,
    media_type: ImageMediaType = "image/jpeg",
) -> tuple[bytes, ImageMediaType]:
    """Compress an image to fit within a maximum byte size.

    Uses a multi-step strategy:
    1. If already under the limit, return as-is.
    2. Convert to JPEG and reduce quality progressively (95 -> 20).
    3. If still too large, resize the image (halve dimensions) and repeat.

    Args:
        image_bytes: The raw image data as bytes.
        max_bytes: Maximum allowed size in bytes. Defaults to 5 MB.
        media_type: The original MIME type. Used to detect format when
            the image is already small enough.

    Returns:
        A tuple of (compressed_bytes, media_type). The media_type will be
        ``"image/jpeg"`` if compression was applied, or the detected/original
        type if the image was already small enough.
    """
    return await run_in_threadpool(_compress_image_data_sync, image_bytes, max_bytes, media_type)


async def compress_image_to_model_limit(
    image_bytes: bytes,
    max_encoded_bytes: int,
    media_type: str | None = "image/jpeg",
) -> tuple[bytes, ImageMediaType]:
    """Compress an image so its base64-encoded payload fits a model API limit."""
    normalized_media_type = normalize_image_media_type(image_bytes, media_type)
    if max_encoded_bytes <= 0:
        return image_bytes, normalized_media_type

    return await compress_image_data(
        image_bytes=image_bytes,
        max_bytes=raw_bytes_limit_for_base64(max_encoded_bytes),
        media_type=normalized_media_type,
    )


def raw_bytes_limit_for_base64(max_encoded_bytes: int) -> int:
    """Return the raw-byte budget for a base64-encoded byte limit."""
    return (max_encoded_bytes // 4) * 3


def _compress_image_data_sync(
    image_bytes: bytes,
    max_bytes: int = 5 * 1024 * 1024,
    media_type: ImageMediaType = "image/jpeg",
) -> tuple[bytes, ImageMediaType]:
    """Synchronous implementation of compress_image_data."""
    if len(image_bytes) <= max_bytes:
        detected = detect_image_media_type(image_bytes)
        return image_bytes, detected or media_type

    img = Image.open(io.BytesIO(image_bytes))

    # Skip animated images (multi-frame GIF/WebP) -- JPEG cannot preserve animation.
    # Return original data so the downstream filter can drop it with a clear hint.
    n_frames = getattr(img, "n_frames", 1) or 1
    if n_frames > 1:
        return image_bytes, detect_image_media_type(image_bytes) or media_type

    # Convert to RGB for JPEG output, compositing alpha onto white background
    # to keep transparent content legible (avoids black-background artifacts).
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Strategy: reduce JPEG quality, then resize if needed
    for _resize_pass in range(5):  # At most 5 resize passes (1/32 of original)
        for quality in (95, 85, 75, 60, 45, 30, 20):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            result = buf.getvalue()
            if len(result) <= max_bytes:
                return result, "image/jpeg"

        # Still too large -- halve dimensions and try again
        w, h = img.size
        img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)

    # Final fallback: return the smallest we could produce
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20, optimize=True)
    result = buf.getvalue()
    if len(result) > max_bytes:
        logger.warning(
            "Image compression could not reach target size: %d bytes > %d bytes limit",
            len(result),
            max_bytes,
        )
    return result, "image/jpeg"


async def split_image_data(
    image_bytes: bytes,
    max_height: int = 4096,
    overlap: int = 50,
    media_type: ImageMediaType = "image/png",
) -> list[BinaryContent]:
    """Split a large image into smaller vertical segments.

    This function takes an image and splits it into multiple segments if the height
    exceeds max_height. Each segment overlaps with the next by the specified amount.

    Args:
        image_bytes: The raw image data as bytes.
        max_height: Maximum height for each segment. Defaults to 4096.
        overlap: Number of pixels to overlap between segments. Defaults to 50.
        media_type: The MIME type for output images. Defaults to "image/png".

    Returns:
        A list of BinaryContent objects, each containing a segment of the image.
    """
    return await run_in_threadpool(_split_image_data_sync, image_bytes, max_height, overlap, media_type)


def _split_image_data_sync(
    image_bytes: bytes,
    max_height: int = 4096,
    overlap: int = 50,
    media_type: ImageMediaType = "image/png",
) -> list[BinaryContent]:
    """Synchronous implementation of split_image_data."""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    if height <= max_height:
        # Detect actual media type from content to avoid mismatch with declared type
        detected = detect_image_media_type(image_bytes)
        actual_type = detected or media_type
        return [BinaryContent(data=image_bytes, media_type=actual_type)]

    segments: list[BinaryContent] = []
    y = 0

    format_map = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }
    pil_format = format_map.get(media_type, "PNG")

    while y < height:
        segment_height = min(max_height, height - y)
        segment = image.crop((0, y, width, y + segment_height))

        buffer = io.BytesIO()
        segment.save(buffer, format=pil_format)
        segment_bytes = buffer.getvalue()

        segments.append(BinaryContent(data=segment_bytes, media_type=media_type))

        y += max_height - overlap
        if y + overlap >= height:
            break

    return segments
