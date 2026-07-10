"""Image and video content filter for message history.

This module provides history processors that limit the number of images and videos
in message history and validates image content using Pillow.

Example::

    from contextlib import AsyncExitStack
    from pydantic_ai import Agent

    from ya_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
    from ya_agent_sdk.environment.local import LocalEnvironment
    from ya_agent_sdk.filters.image import drop_extra_images, drop_extra_videos

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment())
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                model_cfg=ModelConfig(
                    max_images=20,  # Limit to 20 images (default)
                    max_videos=1,   # Limit to 1 video (default)
                    capabilities={ModelCapability.vision},
                ),
            )
        )
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_images, drop_extra_videos],
        )
        result = await agent.run('Describe these images', deps=ctx)
"""

import io
from collections.abc import Mapping, Sequence
from typing import Any, cast

from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.tools import RunContext

from ya_agent_sdk._logger import logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.utils import (
    ImageMediaType,
    compress_image_to_model_limit,
    image_exceeds_limits,
    raw_bytes_limit_for_base64,
    split_image_data,
)


def _raw_bytes_limit_for_base64(max_encoded_bytes: int) -> int:
    """Compute the maximum raw byte size that stays within a base64-encoded size limit.

    Base64 encoding expands data in 4-character blocks (each 3 raw bytes become
    4 encoded characters). Given an API limit on the *encoded* size, the raw
    payload must fit in the complete base64 blocks available under that limit.
    """
    return raw_bytes_limit_for_base64(max_encoded_bytes)


def _is_image_content(item: UserContent) -> bool:
    """Check if content item is an image."""
    if isinstance(item, ImageUrl):
        return True
    return isinstance(item, BinaryContent) and item.media_type.startswith("image/")


def _is_video_content(item: UserContent) -> bool:
    """Check if content item is a video."""
    if isinstance(item, VideoUrl):
        return True
    return isinstance(item, BinaryContent) and item.media_type.startswith("video/")


def _validate_image(content: BinaryContent) -> bool:
    """Validate image content using Pillow.

    Args:
        content: Binary content to validate.

    Returns:
        True if the image is valid, False otherwise.
    """
    try:
        image = Image.open(io.BytesIO(content.data))
        image.verify()
        return True
    except Exception:
        return False


async def _split_image_content_list(
    content_list: list[UserContent],
    *,
    max_height: int,
    overlap: int,
) -> tuple[list[UserContent], bool]:
    """Split oversized binary images in a content list.

    Returns:
        A tuple of (processed_content, modified).
    """
    new_content: list[UserContent] = []
    modified = False

    for item in content_list:
        if not isinstance(item, BinaryContent) or not item.media_type.startswith("image/"):
            new_content.append(item)
            continue

        media_type = cast(ImageMediaType, item.media_type)

        try:
            segments = await split_image_data(
                image_bytes=item.data,
                max_height=max_height,
                overlap=overlap,
                media_type=media_type,
            )
        except Exception:
            logger.exception("Failed to split image; keeping original binary content")
            new_content.append(item)
            continue

        if len(segments) <= 1:
            new_content.append(item)
            continue

        logger.info(
            "Split large image into %d segments (max_height=%d, overlap=%d)",
            len(segments),
            max_height,
            overlap,
        )
        new_content.extend(segments)
        modified = True

    return new_content, modified


async def split_large_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Split oversized binary image content in message history.

    This is a pydantic-ai history_processor that:
    1. Splits BinaryContent images whose height exceeds configured threshold
    2. Preserves image order by replacing one image with multiple segments
    3. Leaves non-image content unchanged

    Behavior is controlled by ModelConfig:
    - split_large_images: master enable/disable switch
    - image_split_max_height: max height per segment
    - image_split_overlap: vertical overlap between segments

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with oversized images split into segments.
    """
    model_cfg = ctx.deps.model_cfg

    if model_cfg and not model_cfg.split_large_images:
        return message_history

    max_height = model_cfg.image_split_max_height if model_cfg else 4096
    overlap = model_cfg.image_split_overlap if model_cfg else 50

    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue

        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue

            content_list: list[UserContent] = (
                list(part.content) if isinstance(part.content, Sequence) else [part.content]
            )
            new_content, modified = await _split_image_content_list(
                content_list,
                max_height=max_height,
                overlap=overlap,
            )
            if modified:
                part.content = new_content

    return message_history


def drop_extra_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop extra image content from message history and validate images.

    This is a pydantic-ai history_processor that:
    1. Limits the number of images to max_images (configured in ModelConfig)
    2. Validates images using Pillow and replaces broken images with text messages
    3. Keeps the most recent images (processes from newest to oldest)

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with extra images dropped or replaced.

    Example:
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_images],
        )
    """
    model_cfg = ctx.deps.model_cfg
    max_images = model_cfg.max_images if model_cfg else 20
    current_image_count = 0

    # Reverse iterate message history to keep the latest images
    for i in range(len(message_history) - 1, -1, -1):
        message = message_history[i]
        if not isinstance(message, ModelRequest):
            continue

        # Reverse iterate parts to keep the latest images
        for j in range(len(message.parts) - 1, -1, -1):
            part = message.parts[j]
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            # Reverse iterate content to keep the latest images
            for k in range(len(content_list) - 1, -1, -1):
                item = content_list[k]
                if not _is_image_content(item):
                    continue

                # Validate image using Pillow
                if isinstance(item, BinaryContent) and not _validate_image(item):
                    logger.info(f"Removing broken image at position {k}")
                    content_list[k] = (
                        "<system-reminder>This image content has been removed "
                        "because the image is broken or corrupted.</system-reminder>"
                    )
                    continue

                current_image_count += 1
                if current_image_count <= max_images:
                    continue

                # Drop the extra image
                logger.info(f"Dropping extra image content: {current_image_count} > {max_images}")
                content_list[k] = (
                    f"<system-reminder>This image content has been dropped "
                    f"as it exceeds the maximum allowed images (max_images={max_images}).</system-reminder>"
                )

            # Update the content
            part.content = content_list

    return message_history


def drop_gif_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop GIF images from message history when model doesn't support them.

    This is a pydantic-ai history_processor that removes GIF images when
    support_gif is False in ModelConfig. GIF images are replaced with a
    system reminder text.

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with GIF images removed if not supported.

    Example:
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_gif_images],
        )
    """
    model_cfg = ctx.deps.model_cfg
    support_gif = model_cfg.support_gif if model_cfg else True

    if support_gif:
        return message_history

    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue

        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            new_content: list[UserContent] = []
            for item in content_list:
                if isinstance(item, BinaryContent) and item.media_type == "image/gif":
                    logger.info("Dropping GIF image as model does not support GIF")
                    new_content.append(
                        "<system-reminder>This GIF image has been removed "
                        "because the model does not support GIF images.</system-reminder>"
                    )
                else:
                    new_content.append(item)

            part.content = new_content

    return message_history


async def _compress_binary_image_content(
    item: BinaryContent,
    *,
    max_image_bytes: int,
    max_raw_bytes: int | None,
    max_image_dimension: int,
) -> tuple[object, bool]:
    """Compress BinaryContent that exceeds model byte or dimension limits."""
    if not image_exceeds_limits(
        item.data,
        max_bytes=max_raw_bytes,
        max_dimension=max_image_dimension,
    ):
        return item, False

    original_size = len(item.data)
    original_media_type = cast(ImageMediaType, item.media_type)
    try:
        compressed_data, compressed_type = await compress_image_to_model_limit(
            image_bytes=item.data,
            max_encoded_bytes=max_image_bytes,
            media_type=original_media_type,
            max_dimension=max_image_dimension,
        )
    except Exception:
        logger.exception("Failed to compress image; dropping it")
        return (
            "<system-reminder>An image was removed because compression failed. "
            "If the image is needed, try compressing it to a smaller size before viewing.</system-reminder>",
            True,
        )

    if image_exceeds_limits(
        compressed_data,
        max_bytes=max_raw_bytes,
        max_dimension=max_image_dimension,
    ):
        logger.warning(
            "Image compression could not satisfy model limits "
            "(raw bytes=%d, raw byte limit=%s, max dimension=%d); dropping image",
            len(compressed_data),
            max_raw_bytes,
            max_image_dimension,
        )
        return (
            f"<system-reminder>An image ({original_size} bytes) was removed because it could not be "
            "compressed within the configured model image byte and dimension limits. "
            "If you need this image, try resizing or converting it to a smaller format first, "
            "then use the view tool again.</system-reminder>",
            True,
        )

    logger.info(
        "Prepared image for model limits: %d bytes -> %d bytes (max dimension=%d)",
        original_size,
        len(compressed_data),
        max_image_dimension,
    )
    return BinaryContent(data=compressed_data, media_type=compressed_type), True


async def _compress_sequence_content(
    items: list[object] | tuple[object, ...],
    *,
    max_image_bytes: int,
    max_raw_bytes: int | None,
    max_image_dimension: int,
) -> tuple[list[object] | tuple[object, ...], bool]:
    """Compress image content nested inside a sequence."""
    modified = False
    new_items: list[object] = []
    for child in items:
        new_child, child_modified = await _compress_content_item(
            child,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
        new_items.append(new_child)
        modified = modified or child_modified
    return (tuple(new_items) if isinstance(items, tuple) else new_items), modified


async def _compress_mapping_content(
    item: Mapping[object, object],
    *,
    max_image_bytes: int,
    max_raw_bytes: int | None,
    max_image_dimension: int,
) -> tuple[dict[object, object], bool]:
    """Compress image content nested inside a mapping."""
    modified = False
    new_dict: dict[object, object] = {}
    for key, value in item.items():
        new_value, value_modified = await _compress_content_item(
            value,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
        new_dict[key] = new_value
        modified = modified or value_modified
    return new_dict, modified


async def _compress_content_item(
    item: object,
    *,
    max_image_bytes: int,
    max_raw_bytes: int | None,
    max_image_dimension: int,
) -> tuple[object, bool]:
    """Compress BinaryContent images in an arbitrary content item."""
    if isinstance(item, BinaryContent) and item.media_type.startswith("image/"):
        return await _compress_binary_image_content(
            item,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
    if isinstance(item, (list, tuple)):
        return await _compress_sequence_content(
            item,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
    if isinstance(item, Mapping):
        return await _compress_mapping_content(
            item,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
    return item, False


async def _compress_content_list(
    content_list: list[UserContent | object],
    *,
    max_image_bytes: int,
    max_image_dimension: int,
) -> bool:
    """Compress images that exceed model byte or dimension limits in-place."""
    max_raw_bytes = _raw_bytes_limit_for_base64(max_image_bytes) if max_image_bytes > 0 else None
    modified = False

    for idx, item in enumerate(content_list):
        new_item, item_modified = await _compress_content_item(
            item,
            max_image_bytes=max_image_bytes,
            max_raw_bytes=max_raw_bytes,
            max_image_dimension=max_image_dimension,
        )
        if item_modified:
            content_list[idx] = new_item
            modified = True

    return modified


async def compress_large_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Compress oversized binary image content in message history.

    This is a pydantic-ai history_processor that compresses BinaryContent images
    whose encoded size exceeds ``max_image_bytes`` or whose width/height exceeds
    ``max_image_dimension`` (configured in ModelConfig). Images are converted to
    JPEG with progressively reduced quality and, if necessary, resized until they
    fit both limits. If compression cannot meet the target, the image is dropped
    and replaced with a system reminder.

    Set both limits to 0 in ModelConfig to disable this filter.

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with oversized images compressed.
    """
    model_cfg = ctx.deps.model_cfg
    max_image_bytes = model_cfg.max_image_bytes if model_cfg else 0
    max_image_dimension = model_cfg.max_image_dimension if model_cfg else 0

    if max_image_bytes <= 0 and max_image_dimension <= 0:
        return message_history

    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue

        for part in message.parts:
            if isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    continue

                content_list: list[UserContent | object] = (
                    list(part.content) if isinstance(part.content, Sequence) else [part.content]
                )
                modified = await _compress_content_list(
                    content_list,
                    max_image_bytes=max_image_bytes,
                    max_image_dimension=max_image_dimension,
                )

                if modified:
                    part.content = cast(list[UserContent], content_list)
                continue

            if isinstance(part, ToolReturnPart):
                tool_content_list: list[object] = list(part.content_items(mode="raw"))
                modified = await _compress_content_list(
                    tool_content_list,
                    max_image_bytes=max_image_bytes,
                    max_image_dimension=max_image_dimension,
                )

                if modified:
                    tool_return = cast(Any, part)
                    tool_return.content = tool_content_list[0] if len(tool_content_list) == 1 else tool_content_list

    return message_history


def drop_extra_videos(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop extra video content from message history.

    This is a pydantic-ai history_processor that limits the number of videos
    to max_videos (configured in ModelConfig). Older videos (appearing earlier
    in message history) are dropped first.

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with excess videos replaced by system reminders.

    Example:
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_videos],
        )
    """
    model_cfg = ctx.deps.model_cfg
    max_videos = model_cfg.max_videos if model_cfg else 1
    current_video_count = 0

    # Reverse iterate message history to keep the latest videos
    for i in range(len(message_history) - 1, -1, -1):
        message = message_history[i]
        if not isinstance(message, ModelRequest):
            continue

        # Reverse iterate parts to keep the latest videos
        for j in range(len(message.parts) - 1, -1, -1):
            part = message.parts[j]
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            # Reverse iterate content to keep the latest videos
            for k in range(len(content_list) - 1, -1, -1):
                item = content_list[k]
                if not _is_video_content(item):
                    continue

                current_video_count += 1
                if current_video_count <= max_videos:
                    continue

                # Drop the extra video
                logger.info(f"Dropping extra video content: {current_video_count} > {max_videos}")
                content_list[k] = (
                    f"<system-reminder>This video content has been dropped "
                    f"as it exceeds the maximum allowed videos (max_videos={max_videos}).</system-reminder>"
                )

            # Update the content
            part.content = content_list

    return message_history
