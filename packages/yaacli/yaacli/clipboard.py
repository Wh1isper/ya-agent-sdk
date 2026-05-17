"""System clipboard helpers for YAACLI TUI paste handling."""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import io
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from ya_agent_sdk.utils import detect_image_media_type

_SUPPORTED_MEDIA_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
)


class _PillowImageLike(Protocol):
    """Minimal Pillow image surface used for PNG encoding."""

    def save(self, fp: io.BytesIO, format: str) -> None: ...  # noqa: A002


@dataclass(frozen=True)
class ClipboardImage:
    """Image content read from the system clipboard."""

    data: bytes
    media_type: str


@dataclass(frozen=True)
class ClipboardImageReadResult:
    """Clipboard image read outcome."""

    image: ClipboardImage | None
    error: str | None = None


async def _run_command(args: list[str], timeout_seconds: float = 2.0) -> tuple[int, bytes, bytes]:
    """Run a subprocess and capture stdout/stderr."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return 124, stdout, b"clipboard command timed out"

    returncode = process.returncode if process.returncode is not None else 1
    return returncode, stdout, stderr


def _encode_image_as_png_bytes(image: _PillowImageLike) -> bytes:
    """Encode a Pillow image object as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _read_image_file(path: str | os.PathLike[str]) -> ClipboardImage | None:
    """Load an image file path into clipboard image bytes."""
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        return None

    try:
        file_bytes = image_path.read_bytes()
    except OSError:
        return None

    media_type = detect_image_media_type(file_bytes)
    if media_type in _SUPPORTED_MEDIA_TYPES:
        return ClipboardImage(data=file_bytes, media_type=media_type)

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(image_path) as image:
            image_bytes = _encode_image_as_png_bytes(image)
    except Exception:
        return None

    detected_type = detect_image_media_type(image_bytes) or "image/png"
    return ClipboardImage(data=image_bytes, media_type=detected_type)


def _clipboard_image_from_pillow_payload(payload: object) -> ClipboardImage | None:
    """Normalize Pillow ImageGrab payload into clipboard image bytes."""
    try:
        from PIL import Image
    except ImportError:
        return None

    if isinstance(payload, Image.Image):
        image_bytes = _encode_image_as_png_bytes(payload)
        media_type = detect_image_media_type(image_bytes) or "image/png"
        return ClipboardImage(data=image_bytes, media_type=media_type)

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, str):
                continue
            image = _read_image_file(item)
            if image is not None:
                return image

    return None


def _grab_clipboard_with_pillow_sync() -> ClipboardImageReadResult:
    """Read clipboard image data through Pillow's cross-platform API."""
    try:
        from PIL import ImageGrab
    except ImportError:
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image paste requires Pillow.",
        )

    try:
        payload = ImageGrab.grabclipboard()
    except NotImplementedError:
        return ClipboardImageReadResult(image=None)
    except Exception:
        return ClipboardImageReadResult(image=None)

    return ClipboardImageReadResult(image=_clipboard_image_from_pillow_payload(payload))


async def _read_pillow_clipboard_image() -> ClipboardImageReadResult:
    """Read clipboard image through Pillow in a worker thread."""
    return await asyncio.to_thread(_grab_clipboard_with_pillow_sync)


def _parse_file_url_path(file_url: str) -> str | None:
    """Parse a file:// URL into a local path."""
    parsed = urlparse(file_url)
    if parsed.scheme and parsed.scheme != "file":
        return None

    path = unquote(parsed.path or file_url)
    if not path:
        return None

    if sys.platform.startswith("win") and path.startswith("/") and len(path) > 2 and path[2] == ":":
        return path[1:]

    return path


def _read_macos_pasteboard_file_image_sync() -> ClipboardImageReadResult:
    """Read image file references from the macOS pasteboard via Cocoa."""
    try:
        appkit = importlib.import_module("AppKit")
        foundation = importlib.import_module("Foundation")
        NSPasteboard = appkit.NSPasteboard
        NSURL = foundation.NSURL
    except ImportError:
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image file paste on macOS requires pyobjc-framework-Cocoa.",
        )

    try:
        pasteboard = NSPasteboard.generalPasteboard()
        items = pasteboard.pasteboardItems() or []
    except Exception:
        return ClipboardImageReadResult(image=None)

    for item in items:
        file_url = None
        try:
            file_url = item.stringForType_("public.file-url")
        except Exception:
            file_url = None

        path: str | None = None
        if file_url:
            try:
                url = NSURL.URLWithString_(file_url)
                if url is not None:
                    path = str(url.path())
            except Exception:
                path = None

            if not path:
                path = _parse_file_url_path(str(file_url))

        if not path:
            continue

        image = _read_image_file(path)
        if image is not None:
            return ClipboardImageReadResult(image=image)

    return ClipboardImageReadResult(image=None)


async def _read_macos_pasteboard_file_image() -> ClipboardImageReadResult:
    """Read image file references from the macOS pasteboard in a worker thread."""
    return await asyncio.to_thread(_read_macos_pasteboard_file_image_sync)


async def _read_wayland_clipboard_image() -> ClipboardImageReadResult:
    if shutil.which("wl-paste") is None:
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image paste requires wl-paste on Wayland.",
        )

    returncode, stdout, _stderr = await _run_command(["wl-paste", "--list-types"])
    if returncode != 0:
        return ClipboardImageReadResult(image=None)

    available_types = {line.strip() for line in stdout.decode("utf-8", errors="ignore").splitlines() if line.strip()}
    for media_type in _SUPPORTED_MEDIA_TYPES:
        if media_type not in available_types:
            continue
        read_returncode, image_bytes, _read_stderr = await _run_command([
            "wl-paste",
            "--no-newline",
            "--type",
            media_type,
        ])
        if read_returncode == 0 and image_bytes:
            detected_type = detect_image_media_type(image_bytes) or media_type
            return ClipboardImageReadResult(image=ClipboardImage(data=image_bytes, media_type=detected_type))

    return ClipboardImageReadResult(image=None)


async def _read_x11_clipboard_image() -> ClipboardImageReadResult:
    if shutil.which("xclip") is None:
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image paste requires xclip on X11.",
        )

    returncode, stdout, _stderr = await _run_command(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"])
    if returncode != 0:
        return ClipboardImageReadResult(image=None)

    available_types = {line.strip() for line in stdout.decode("utf-8", errors="ignore").splitlines() if line.strip()}
    for media_type in _SUPPORTED_MEDIA_TYPES:
        if media_type not in available_types:
            continue
        read_returncode, image_bytes, _read_stderr = await _run_command([
            "xclip",
            "-selection",
            "clipboard",
            "-t",
            media_type,
            "-o",
        ])
        if read_returncode == 0 and image_bytes:
            detected_type = detect_image_media_type(image_bytes) or media_type
            return ClipboardImageReadResult(image=ClipboardImage(data=image_bytes, media_type=detected_type))

    return ClipboardImageReadResult(image=None)


async def _read_windows_clipboard_image() -> ClipboardImageReadResult:
    candidates = [path for path in (shutil.which("powershell"), shutil.which("pwsh")) if path]
    if not candidates:
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image paste requires PowerShell on Windows.",
        )

    command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "if (-not [System.Windows.Forms.Clipboard]::ContainsImage()) { exit 0 }; "
        "$image = [System.Windows.Forms.Clipboard]::GetImage(); "
        "$stream = New-Object System.IO.MemoryStream; "
        "$image.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png); "
        "[Convert]::ToBase64String($stream.ToArray())"
    )

    last_error: str | None = None
    for powershell in candidates:
        returncode, stdout, stderr = await _run_command([powershell, "-NoProfile", "-STA", "-Command", command])
        if returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="ignore").strip()
            if stderr_text:
                last_error = f"PowerShell clipboard image read failed: {stderr_text}"
            continue

        payload = stdout.decode("utf-8", errors="ignore").strip()
        if not payload:
            return ClipboardImageReadResult(image=None)

        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return ClipboardImageReadResult(
                image=None,
                error="Invalid clipboard image payload from PowerShell.",
            )

        media_type = detect_image_media_type(image_bytes)
        if media_type is None:
            return ClipboardImageReadResult(
                image=None,
                error="Clipboard payload from PowerShell was not an image.",
            )

        return ClipboardImageReadResult(image=ClipboardImage(data=image_bytes, media_type=media_type))

    return ClipboardImageReadResult(
        image=None,
        error=last_error or "Clipboard image paste requires PowerShell on Windows.",
    )


def _merge_errors(*errors: str | None) -> str | None:
    """Join unique clipboard errors while preserving order."""
    unique_errors = list(dict.fromkeys(error for error in errors if error))
    if not unique_errors:
        return None
    return " ".join(unique_errors)


async def read_clipboard_image() -> ClipboardImageReadResult:
    """Read an image from the system clipboard when available."""
    pillow_result = await _read_pillow_clipboard_image()
    if pillow_result.image is not None:
        return pillow_result

    if sys.platform == "darwin":
        macos_result = await _read_macos_pasteboard_file_image()
        if macos_result.image is not None:
            return macos_result
        return ClipboardImageReadResult(image=None, error=_merge_errors(pillow_result.error, macos_result.error))

    if sys.platform.startswith("win"):
        windows_result = await _read_windows_clipboard_image()
        if windows_result.image is not None:
            return windows_result
        return ClipboardImageReadResult(image=None, error=_merge_errors(pillow_result.error, windows_result.error))

    if sys.platform.startswith("linux"):
        errors: list[str] = []
        if pillow_result.error:
            errors.append(pillow_result.error)

        if os.environ.get("WAYLAND_DISPLAY"):
            wayland_result = await _read_wayland_clipboard_image()
            if wayland_result.image is not None:
                return wayland_result
            if wayland_result.error:
                errors.append(wayland_result.error)

        if os.environ.get("DISPLAY"):
            x11_result = await _read_x11_clipboard_image()
            if x11_result.image is not None:
                return x11_result
            if x11_result.error:
                errors.append(x11_result.error)

        if errors:
            return ClipboardImageReadResult(image=None, error=_merge_errors(*errors))
        return ClipboardImageReadResult(
            image=None,
            error="Clipboard image paste requires wl-paste on Wayland or xclip on X11.",
        )

    return ClipboardImageReadResult(
        image=None,
        error=f"Clipboard image paste is not supported on platform: {sys.platform}.",
    )
