"""Capability-based content filter for message history.

This module provides a history processor that filters message content
based on model capabilities. For example, if a model doesn't support
vision, image content will be filtered out and replaced with explanatory text.

Example::

    from contextlib import AsyncExitStack
    from pydantic_ai import Agent

    from ya_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
    from ya_agent_sdk.environment.local import LocalEnvironment
    from ya_agent_sdk.filters.capability import filter_by_capability

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment())
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                model_cfg=ModelConfig(
                    capabilities={ModelCapability.vision},  # Model supports vision
                ),
            )
        )
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[filter_by_capability],
        )
        result = await agent.run('Describe this image', deps=ctx)
"""

from collections.abc import Sequence

from pydantic_ai import DocumentUrl
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.tools import RunContext

from ya_agent_sdk._logger import logger
from ya_agent_sdk.context import AgentContext, ModelCapability


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


def _is_document_content(item: UserContent) -> bool:
    """Check if content item is a document (PDF)."""
    if isinstance(item, DocumentUrl):
        return True
    return isinstance(item, BinaryContent) and item.media_type == "application/pdf"


def _add_filtered_messages(
    filtered: list[UserContent],
    removed_images: bool,
    removed_videos: bool,
    removed_documents: bool,
) -> None:
    """Add explanatory messages for filtered content types."""
    if removed_images:
        logger.info("Filtering out image content - model does not have vision capability")
        filtered.append(
            "<filtered-content type='image'>"
            "Image content has been filtered out as the current model does not support vision."
            "</filtered-content>"
        )

    if removed_videos:
        logger.info("Filtering out video content - model does not have video_understanding capability")
        filtered.append(
            "<filtered-content type='video'>"
            "Video content has been filtered out as the current model does not support video understanding."
            "</filtered-content>"
        )

    if removed_documents:
        logger.info("Filtering out document content - model does not have document_understanding capability")
        filtered.append(
            "<filtered-content type='document'>"
            "Document content has been filtered out as the current model does not support document understanding."
            "</filtered-content>"
        )


def _filter_content(
    content: str | Sequence[UserContent],
    has_vision: bool,
    has_video: bool,
    has_document: bool,
) -> str | list[UserContent]:
    """Filter content based on model capabilities.

    Args:
        content: The content to filter (string or sequence of content items).
        has_vision: Whether the model supports vision.
        has_video: Whether the model supports video understanding.
        has_document: Whether the model supports document understanding.

    Returns:
        Filtered content with explanatory text for removed items.
    """
    # If content is a string, no filtering needed
    if isinstance(content, str):
        return content

    # Convert to list for processing
    items: list[UserContent] = list(content)
    filtered: list[UserContent] = []
    removed_images = False
    removed_videos = False
    removed_documents = False

    for item in items:
        if _is_image_content(item):
            if has_vision:
                filtered.append(item)
            else:
                removed_images = True
        elif _is_video_content(item):
            if has_video:
                filtered.append(item)
            else:
                removed_videos = True
        elif _is_document_content(item):
            if has_document:
                filtered.append(item)
            else:
                removed_documents = True
        else:
            filtered.append(item)

    _add_filtered_messages(filtered, removed_images, removed_videos, removed_documents)
    return filtered


def filter_by_capability(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Filter message content based on model capabilities.

    This is a pydantic-ai history_processor that filters out unsupported
    content types based on the model's declared capabilities. For example,
    if the model doesn't have the 'vision' capability, image content will
    be replaced with explanatory text.

    Supported capability checks:
    - vision: Filters ImageUrl and image BinaryContent
    - video_understanding: Filters VideoUrl and video BinaryContent
    - document_understanding: Filters DocumentUrl and PDF BinaryContent

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The message history with filtered content.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[filter_by_capability],
        )
    """
    model_cfg = ctx.deps.model_cfg

    # If no model config or no capability restrictions, return as-is
    if model_cfg is None:
        return message_history

    has_vision = model_cfg.has_capability(ModelCapability.vision)
    has_video = model_cfg.has_capability(ModelCapability.video_understanding)
    has_document = model_cfg.has_capability(ModelCapability.document_understanding)

    # If model has all capabilities, no filtering needed
    if has_vision and has_video and has_document:
        return message_history

    # Process each message
    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue

        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue

            # Filter the content based on capabilities
            filtered_content = _filter_content(part.content, has_vision, has_video, has_document)

            # Update content - handle type conversion
            if isinstance(filtered_content, str):
                part.content = filtered_content
            else:
                part.content = filtered_content

    return message_history
