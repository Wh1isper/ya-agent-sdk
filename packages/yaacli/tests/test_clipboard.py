"""Tests for clipboard image strategy helpers."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from yaacli.clipboard import (
    ClipboardImage,
    ClipboardImageReadResult,
    _clipboard_image_from_pillow_payload,
    _merge_errors,
    _read_macos_pasteboard_file_image_sync,
    _read_windows_clipboard_image,
    read_clipboard_image,
)

_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pQe0AAAAASUVORK5CYII="
_PNG_BYTES = base64.b64decode(_PNG_BASE64)


def test_clipboard_image_from_pillow_payload_reads_file_list(tmp_path: Path) -> None:
    """Pillow file-list payloads should load the first supported image file."""
    image_path = tmp_path / "clipboard.png"
    image_path.write_bytes(_PNG_BYTES)

    image = _clipboard_image_from_pillow_payload([str(image_path)])

    assert image == ClipboardImage(data=_PNG_BYTES, media_type="image/png")


def test_merge_errors_deduplicates_and_preserves_order() -> None:
    """Clipboard error merging should keep unique messages in order."""
    assert _merge_errors("one", None, "two", "one") == "one two"


def test_read_macos_pasteboard_file_image_sync_reads_public_file_url(tmp_path: Path) -> None:
    """macOS Cocoa fallback should load Finder-copied image files."""
    image_path = tmp_path / "finder-copy.png"
    image_path.write_bytes(_PNG_BYTES)

    class FakePasteboardItem:
        def __init__(self, file_url: str) -> None:
            self._file_url = file_url

        def stringForType_(self, media_type: str) -> str | None:
            if media_type == "public.file-url":
                return self._file_url
            return None

    class FakePasteboard:
        def __init__(self, file_url: str) -> None:
            self._items = [FakePasteboardItem(file_url)]

        def pasteboardItems(self) -> list[FakePasteboardItem]:
            return self._items

    fake_pasteboard = FakePasteboard(image_path.as_uri())
    fake_appkit = SimpleNamespace(NSPasteboard=SimpleNamespace(generalPasteboard=lambda: fake_pasteboard))
    fake_foundation = SimpleNamespace(
        NSURL=SimpleNamespace(URLWithString_=lambda value: SimpleNamespace(path=lambda: str(image_path)))
    )

    with patch.dict(sys.modules, {"AppKit": fake_appkit, "Foundation": fake_foundation}):
        result = _read_macos_pasteboard_file_image_sync()

    assert result == ClipboardImageReadResult(image=ClipboardImage(data=_PNG_BYTES, media_type="image/png"))


@pytest.mark.asyncio
async def test_read_clipboard_image_prefers_pillow_result() -> None:
    """Cross-platform Pillow read should short-circuit platform fallbacks."""
    with (
        patch("yaacli.clipboard._read_pillow_clipboard_image", new=AsyncMock()) as mock_pillow,
        patch("yaacli.clipboard._read_macos_pasteboard_file_image", new=AsyncMock()) as mock_macos,
        patch.object(sys, "platform", "darwin"),
    ):
        mock_pillow.return_value = ClipboardImageReadResult(image=ClipboardImage(data=b"img", media_type="image/png"))

        result = await read_clipboard_image()

    assert result.image == ClipboardImage(data=b"img", media_type="image/png")
    mock_macos.assert_not_called()


@pytest.mark.asyncio
async def test_read_clipboard_image_uses_macos_file_fallback() -> None:
    """macOS should fall back to Cocoa file URLs when Pillow returns no image."""
    with (
        patch("yaacli.clipboard._read_pillow_clipboard_image", new=AsyncMock()) as mock_pillow,
        patch("yaacli.clipboard._read_macos_pasteboard_file_image", new=AsyncMock()) as mock_macos,
        patch.object(sys, "platform", "darwin"),
    ):
        mock_pillow.return_value = ClipboardImageReadResult(image=None)
        mock_macos.return_value = ClipboardImageReadResult(image=ClipboardImage(data=b"finder", media_type="image/png"))

        result = await read_clipboard_image()

    assert result.image == ClipboardImage(data=b"finder", media_type="image/png")


@pytest.mark.asyncio
async def test_read_windows_clipboard_image_retries_with_pwsh() -> None:
    """Windows fallback should prefer PowerShell and retry with pwsh."""
    encoded = base64.b64encode(_PNG_BYTES).decode("ascii")

    with (
        patch(
            "yaacli.clipboard.shutil.which",
            side_effect=lambda name: {"powershell": "powershell", "pwsh": "pwsh"}.get(name),
        ),
        patch("yaacli.clipboard._run_command", new=AsyncMock()) as mock_run,
    ):
        mock_run.side_effect = [
            (1, b"", b"powershell failed"),
            (0, encoded.encode("utf-8"), b""),
        ]

        result = await _read_windows_clipboard_image()

    assert result == ClipboardImageReadResult(image=ClipboardImage(data=_PNG_BYTES, media_type="image/png"))
    assert mock_run.await_args_list[0].args[0][:4] == ["powershell", "-NoProfile", "-STA", "-Command"]
    assert mock_run.await_args_list[1].args[0][:4] == ["pwsh", "-NoProfile", "-STA", "-Command"]


@pytest.mark.asyncio
async def test_read_windows_clipboard_image_rejects_invalid_base64() -> None:
    """Windows fallback should surface invalid clipboard payloads."""
    with (
        patch(
            "yaacli.clipboard.shutil.which",
            side_effect=lambda name: {"powershell": "powershell", "pwsh": None}.get(name),
        ),
        patch("yaacli.clipboard._run_command", new=AsyncMock(return_value=(0, b"not-base64", b""))),
    ):
        result = await _read_windows_clipboard_image()

    assert result == ClipboardImageReadResult(image=None, error="Invalid clipboard image payload from PowerShell.")


@pytest.mark.asyncio
async def test_read_clipboard_image_linux_merges_wayland_and_x11_errors() -> None:
    """Linux should report the active clipboard backend requirements."""
    with (
        patch("yaacli.clipboard._read_pillow_clipboard_image", new=AsyncMock()) as mock_pillow,
        patch("yaacli.clipboard._read_wayland_clipboard_image", new=AsyncMock()) as mock_wayland,
        patch("yaacli.clipboard._read_x11_clipboard_image", new=AsyncMock()) as mock_x11,
        patch.object(sys, "platform", "linux"),
        patch.dict("yaacli.clipboard.os.environ", {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, clear=True),
    ):
        mock_pillow.return_value = ClipboardImageReadResult(image=None)
        mock_wayland.return_value = ClipboardImageReadResult(image=None, error="wayland missing")
        mock_x11.return_value = ClipboardImageReadResult(image=None, error="x11 missing")

        result = await read_clipboard_image()

    assert result == ClipboardImageReadResult(image=None, error="wayland missing x11 missing")
