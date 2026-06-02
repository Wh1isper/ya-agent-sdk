"""Tests for image compression utilities and filter."""

import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart, UserPromptPart
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.filters.image import _raw_bytes_limit_for_base64, compress_large_images
from ya_agent_sdk.utils import compress_image_data


def _make_png(width: int, height: int) -> bytes:
    """Create a simple PNG image of given size."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgba_png(width: int, height: int) -> bytes:
    """Create a RGBA PNG image (with alpha channel)."""
    img = Image.new("RGBA", (width, height), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_png(min_bytes: int = 6 * 1024 * 1024) -> bytes:
    """Create a PNG with random pixel data that exceeds min_bytes.

    Random data resists PNG compression, producing large files.
    """
    # 2000x2000 RGB random pixels -> ~12MB raw, PNG ~11-12MB
    width, height = 2000, 2000
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)  # Low compression for speed
    data = buf.getvalue()
    assert len(data) > min_bytes, f"Generated PNG is only {len(data)} bytes, need > {min_bytes}"
    return data


# ---------------------------------------------------------------------------
# compress_image_data tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compress_small_image_passthrough():
    """Images already under the limit should pass through unchanged."""
    small_png = _make_png(100, 100)
    assert len(small_png) < 1024 * 1024  # well under 1MB

    result_data, result_type = await compress_image_data(small_png, max_bytes=5 * 1024 * 1024)
    # Should return the same bytes (no re-encoding)
    assert result_data == small_png
    assert result_type == "image/png"  # detected from content


@pytest.mark.anyio
async def test_compress_reduces_size():
    """A large image should be compressed below the target."""
    # Create a large-ish PNG (random-ish data compresses poorly in PNG)
    large_png = _make_png(4000, 4000)
    max_bytes = len(large_png) // 2  # target is half the original

    result_data, result_type = await compress_image_data(large_png, max_bytes=max_bytes)
    assert len(result_data) <= max_bytes
    assert result_type == "image/jpeg"


@pytest.mark.anyio
async def test_compress_rgba_image():
    """RGBA images should be converted to RGB for JPEG compression."""
    rgba_png = _make_rgba_png(2000, 2000)
    max_bytes = len(rgba_png) // 2

    result_data, result_type = await compress_image_data(rgba_png, max_bytes=max_bytes)
    assert len(result_data) <= max_bytes
    assert result_type == "image/jpeg"

    # Verify the result is a valid JPEG
    img = Image.open(io.BytesIO(result_data))
    assert img.format == "JPEG"
    assert img.mode == "RGB"


@pytest.mark.anyio
async def test_compress_returns_jpeg():
    """Compressed output should always be JPEG."""
    large_png = _make_png(3000, 3000)
    max_bytes = 1024  # very small target to force compression

    result_data, result_type = await compress_image_data(large_png, max_bytes=max_bytes, media_type="image/png")
    assert result_type == "image/jpeg"

    # Verify it's a valid JPEG
    img = Image.open(io.BytesIO(result_data))
    assert img.format == "JPEG"


@pytest.mark.anyio
async def test_compress_large_image_to_5mb():
    """Compress a generated 6+ MB PNG to under 5 MB."""
    image_bytes = _make_large_png()
    original_size = len(image_bytes)
    assert original_size > 5 * 1024 * 1024, f"Test image should be > 5MB, got {original_size}"

    max_bytes = 5 * 1024 * 1024
    result_data, result_type = await compress_image_data(image_bytes, max_bytes=max_bytes)

    assert len(result_data) <= max_bytes, f"Compressed size {len(result_data)} exceeds {max_bytes} limit"
    assert result_type == "image/jpeg"

    # Verify it's a valid image
    img = Image.open(io.BytesIO(result_data))
    assert img.format == "JPEG"


# ---------------------------------------------------------------------------
# compress_large_images filter tests
# ---------------------------------------------------------------------------


def _make_ctx(max_image_bytes: int = 5 * 1024 * 1024) -> MagicMock:
    """Create a mock RunContext with the given max_image_bytes."""
    ctx = MagicMock()
    ctx.deps = MagicMock(spec=AgentContext)
    ctx.deps.model_cfg = ModelConfig(max_image_bytes=max_image_bytes)
    return ctx


@pytest.mark.anyio
async def test_filter_disabled_when_zero():
    """Filter should be no-op when max_image_bytes is 0."""
    ctx = _make_ctx(max_image_bytes=0)
    large_png = _make_png(4000, 4000)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=large_png, media_type="image/png")]),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    # Content should be unchanged
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], BinaryContent)
    assert content[0].data == large_png  # not compressed


@pytest.mark.anyio
async def test_filter_compresses_oversized_image():
    """Filter should compress images exceeding the limit."""
    large_png = _make_png(4000, 4000)
    max_bytes = len(large_png) // 2
    ctx = _make_ctx(max_image_bytes=max_bytes)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=large_png, media_type="image/png")]),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], BinaryContent)
    assert len(content[0].data) <= max_bytes
    assert content[0].media_type == "image/jpeg"


@pytest.mark.anyio
async def test_filter_preserves_small_images():
    """Filter should not touch images already under the limit."""
    small_png = _make_png(100, 100)
    ctx = _make_ctx(max_image_bytes=5 * 1024 * 1024)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=small_png, media_type="image/png")]),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], BinaryContent)
    assert content[0].data == small_png  # untouched


@pytest.mark.anyio
async def test_filter_handles_mixed_content():
    """Filter should only compress oversized images, leaving text and small images alone."""
    small_png = _make_png(100, 100)
    large_png = _make_png(4000, 4000)
    max_bytes = len(large_png) // 2
    ctx = _make_ctx(max_image_bytes=max_bytes)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Hello world",
                        BinaryContent(data=small_png, media_type="image/png"),
                        BinaryContent(data=large_png, media_type="image/png"),
                    ]
                ),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)

    assert content[0] == "Hello world"
    assert isinstance(content[1], BinaryContent)
    assert content[1].data == small_png  # small image untouched
    assert isinstance(content[2], BinaryContent)
    assert len(content[2].data) <= max_bytes  # large image compressed
    assert content[2].media_type == "image/jpeg"


@pytest.mark.anyio
async def test_filter_with_large_image():
    """End-to-end filter test with a generated oversized image."""
    image_bytes = _make_large_png()
    max_bytes = 5 * 1024 * 1024
    ctx = _make_ctx(max_image_bytes=max_bytes)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=image_bytes, media_type="image/png"),
                    ]
                ),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], BinaryContent)
    assert len(content[0].data) <= max_bytes
    assert content[0].media_type == "image/jpeg"


@pytest.mark.anyio
async def test_filter_drops_image_when_compression_fails():
    """Filter should drop the image and insert a hint when compress_image_data raises."""
    large_png = _make_png(4000, 4000)
    max_bytes = len(large_png) // 2
    ctx = _make_ctx(max_image_bytes=max_bytes)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=large_png, media_type="image/png")]),
            ]
        ),
    ]

    with patch(
        "ya_agent_sdk.filters.image.compress_image_data",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await compress_large_images(ctx, messages)

    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], str)
    assert "removed" in content[0]
    assert "compress" in content[0].lower()


@pytest.mark.anyio
async def test_filter_drops_image_when_still_over_limit():
    """Filter should drop the image when compression cannot reach the target size."""
    large_png = _make_png(4000, 4000)
    # Set an impossibly small limit
    max_bytes = 10
    ctx = _make_ctx(max_image_bytes=max_bytes)
    # Raw budget = 10 * 3 // 4 = 7 bytes
    max_raw = _raw_bytes_limit_for_base64(max_bytes)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=large_png, media_type="image/png")]),
            ]
        ),
    ]

    # Mock compress_image_data to return data that's still over the raw budget
    with patch(
        "ya_agent_sdk.filters.image.compress_image_data",
        new_callable=AsyncMock,
        return_value=(b"x" * (max_raw + 1), "image/jpeg"),
    ):
        result = await compress_large_images(ctx, messages)

    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], str)
    assert "removed" in content[0]
    assert "view" in content[0].lower()


@pytest.mark.anyio
async def test_compress_animated_gif_skipped():
    """Animated GIF should not be compressed (returned as-is for downstream drop)."""
    # Create a simple 2-frame GIF
    frames = [Image.new("RGB", (200, 200), color=c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0)
    gif_bytes = buf.getvalue()

    # Set max_bytes smaller than the GIF to trigger compression attempt
    result_data, result_type = await compress_image_data(gif_bytes, max_bytes=1, media_type="image/gif")
    # Should return original bytes unchanged (animation preserved)
    assert result_data == gif_bytes
    assert result_type == "image/gif"


@pytest.mark.anyio
async def test_compress_rgba_white_background():
    """RGBA images should be composited onto white, not black."""
    # Create an RGBA image with transparency
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 0))  # Fully transparent red
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result_data, result_type = await compress_image_data(
        png_bytes, max_bytes=len(png_bytes) // 2, media_type="image/png"
    )
    assert result_type == "image/jpeg"

    # Verify background is white (not black)
    result_img = Image.open(io.BytesIO(result_data))
    w, h = result_img.size
    # Sample center pixel -- should be close to white (255,255,255)
    pixel = result_img.getpixel((w // 2, h // 2))
    # JPEG compression may shift values slightly, but should be near white
    assert all(v > 200 for v in pixel[:3]), f"Expected near-white pixel, got {pixel}"


# ---------------------------------------------------------------------------
# base64 budget tests
# ---------------------------------------------------------------------------


def test_raw_bytes_limit_for_base64():
    """Verify the base64 budget calculation."""
    # 5 MB API limit -> raw budget is 3/4 of that
    assert _raw_bytes_limit_for_base64(5 * 1024 * 1024) == 5 * 1024 * 1024 * 3 // 4
    # 0 -> 0
    assert _raw_bytes_limit_for_base64(0) == 0
    # Small values
    assert _raw_bytes_limit_for_base64(4) == 3
    assert _raw_bytes_limit_for_base64(100) == 75


@pytest.mark.anyio
async def test_filter_compresses_image_near_base64_boundary():
    """An image between 3.75MB and 5MB raw should be compressed.

    Even though the raw bytes are under 5MB, the base64-encoded payload
    would exceed the 5MB API limit, so compression must kick in.
    """
    # Create an image whose raw size is ~4MB (under 5MB but over 3.75MB base64 budget)
    api_limit = 5 * 1024 * 1024  # 5MB
    raw_budget = _raw_bytes_limit_for_base64(api_limit)  # ~3.75MB

    # Build a PNG that's larger than the raw budget but smaller than 5MB
    width = 1400
    height = 1400
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    png_bytes = buf.getvalue()

    # If the generated image happens to be too small, skip the test
    if len(png_bytes) <= raw_budget:
        pytest.skip(f"Generated PNG ({len(png_bytes)} bytes) is not large enough for this test")

    ctx = _make_ctx(max_image_bytes=api_limit)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=png_bytes, media_type="image/png")]),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = list(part.content)
    assert isinstance(content[0], BinaryContent)
    # Compressed result must be under the raw budget (not just under 5MB)
    assert len(content[0].data) <= raw_budget, (
        f"Compressed image ({len(content[0].data)} bytes) exceeds "
        f"raw budget ({raw_budget} bytes) for {api_limit} byte API limit"
    )


@pytest.mark.anyio
async def test_filter_compresses_tool_return_image_content():
    """ToolReturnPart binary images should be compressed before being sent back to the model."""
    large_png = _make_png(4000, 4000)
    max_bytes = len(large_png) // 2
    ctx = _make_ctx(max_image_bytes=max_bytes)

    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="fetch",
                    tool_call_id="call_1",
                    content=BinaryContent(data=large_png, media_type="image/png"),
                ),
            ]
        ),
    ]

    result = await compress_large_images(ctx, messages)
    part = result[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert isinstance(part.content, BinaryContent)
    assert len(part.content.data) <= max_bytes
    assert part.content.media_type == "image/jpeg"
