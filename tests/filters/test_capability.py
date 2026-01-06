"""Tests for pai_agent_sdk.filters.capability module."""

from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)

from pai_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.capability import filter_by_capability


async def test_filter_by_capability_no_model_config(tmp_path: Path) -> None:
    """Should return unchanged history when no model config is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = filter_by_capability(mock_ctx, history)

            assert result == history
            assert request.parts[0].content == "Hello"  # type: ignore[union-attr]


async def test_filter_by_capability_with_all_capabilities(tmp_path: Path) -> None:
    """Should return unchanged history when model has all capabilities."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                capabilities={ModelCapability.vision, ModelCapability.video_understanding},
            ),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            image_url = ImageUrl(url="https://example.com/image.png")
            request = ModelRequest(parts=[UserPromptPart(content=["Describe this", image_url])])
            history = [request]

            result = filter_by_capability(mock_ctx, history)

            assert result == history
            # Content should be unchanged
            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            assert content[1] == image_url


async def test_filter_by_capability_filters_images_without_vision(tmp_path: Path) -> None:
    """Should filter out images when model doesn't have vision capability."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                capabilities=set(),  # No capabilities
            ),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            image_url = ImageUrl(url="https://example.com/image.png")
            request = ModelRequest(parts=[UserPromptPart(content=["Describe this", image_url])])
            history = [request]

            filter_by_capability(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            assert content[0] == "Describe this"
            assert content[1] == snapshot(
                "<filtered-content type='image'>Image content has been filtered out as the current model does not support vision.</filtered-content>"
            )


async def test_filter_by_capability_filters_videos_without_video_understanding(tmp_path: Path) -> None:
    """Should filter out videos when model doesn't have video_understanding capability."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                capabilities={ModelCapability.vision},  # Only vision, no video
            ),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            video_url = VideoUrl(url="https://example.com/video.mp4")
            request = ModelRequest(parts=[UserPromptPart(content=["Watch this", video_url])])
            history = [request]

            filter_by_capability(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            assert content[0] == "Watch this"
            assert content[1] == snapshot(
                "<filtered-content type='video'>Video content has been filtered out as the current model does not support video understanding.</filtered-content>"
            )


async def test_filter_by_capability_filters_binary_image_content(tmp_path: Path) -> None:
    """Should filter out binary image content when model doesn't have vision."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(capabilities=set()),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            binary_image = BinaryContent(data=b"fake image data", media_type="image/png")
            request = ModelRequest(parts=[UserPromptPart(content=["Look at this", binary_image])])
            history = [request]

            filter_by_capability(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            assert content[0] == "Look at this"
            assert content[1] == snapshot(
                "<filtered-content type='image'>Image content has been filtered out as the current model does not support vision.</filtered-content>"
            )


async def test_filter_by_capability_preserves_string_content(tmp_path: Path) -> None:
    """Should not modify string-only content."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(capabilities=set()),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Just text")])
            history = [request]

            filter_by_capability(mock_ctx, history)

            assert request.parts[0].content == "Just text"  # type: ignore[union-attr]


async def test_filter_by_capability_skips_model_response(tmp_path: Path) -> None:
    """Should not process ModelResponse messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(capabilities=set()),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            response = ModelResponse(parts=[TextPart(content="Response text")])
            history = [response]

            result = filter_by_capability(mock_ctx, history)

            assert result == history
            assert response.parts[0].content == "Response text"
