"""Tests for the load_media_url tool."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.toolsets.core.content._url_helper import ContentCategory
from ya_agent_sdk.toolsets.core.content.load_media_url import LoadMediaUrlTool
from ya_agent_sdk.toolsets.core.multimodal import ReadAudioTool, ReadImageTool, ReadVideoTool, tools


def _run_context(model_cfg: ModelConfig | None = None) -> RunContext[AgentContext]:
    agent_ctx = AgentContext(model_cfg=model_cfg)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = agent_ctx
    return ctx


def test_multimodal_toolset_no_longer_registers_deprecated_read_tools() -> None:
    """Deprecated read_* tools remain importable but are not registered by default."""
    assert tools == []
    assert ReadImageTool.name == "read_image"
    assert ReadVideoTool.name == "read_video"
    assert ReadAudioTool.name == "read_audio"


@pytest.mark.parametrize(
    ("category", "url", "expected"),
    [
        (
            ContentCategory.image,
            "https://example.com/image.png",
            (
                "The URL 'https://example.com/image.png' points to an image, but the current model does not "
                "support vision capability. Use the `view` tool instead to analyze this image."
            ),
        ),
        (
            ContentCategory.video,
            "https://example.com/video.mp4",
            (
                "The URL 'https://example.com/video.mp4' points to a video, but the current model does not "
                "support video understanding. Use the `view` tool instead to analyze this video."
            ),
        ),
        (
            ContentCategory.audio,
            "https://example.com/audio.mp3",
            (
                "The URL 'https://example.com/audio.mp3' points to audio, but the current model does not "
                "support audio understanding. Use the `view` tool instead to analyze this audio."
            ),
        ),
    ],
)
def test_load_media_url_fallback_messages_point_to_view(
    category: ContentCategory,
    url: str,
    expected: str,
) -> None:
    """Unsupported media fallback messages should point to view, not deprecated read_* tools."""
    result = LoadMediaUrlTool()._resolve_content(
        url,
        category,
        has_vision=False,
        has_video=False,
        has_audio=False,
        has_document=False,
        enable_load_document=False,
    )

    assert result == expected
    assert "read_image" not in result
    assert "read_video" not in result
    assert "read_audio" not in result


async def test_load_media_url_instruction_points_to_view_for_unsupported_media() -> None:
    """Dynamic instructions should direct unsupported media users to view."""
    instruction = await LoadMediaUrlTool().get_instruction(_run_context(ModelConfig(capabilities=set())))

    assert "<note>Image loading not supported. Use `view` tool instead.</note>" in instruction
    assert "<note>Video/YouTube loading not supported. Use `view` tool instead.</note>" in instruction
    assert "<note>Audio loading not supported. Use `view` tool instead.</note>" in instruction
    assert "read_image" not in instruction
    assert "read_video" not in instruction
    assert "read_audio" not in instruction
