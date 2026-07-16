"""Terminal theme detection and resolved YAACLI color themes."""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

ThemePreference: TypeAlias = Literal["auto", "dark", "light"]
ThemeVariant: TypeAlias = Literal["dark", "light"]
ThemeSource: TypeAlias = Literal["config", "osc11", "colorfgbg", "fallback"]

_DEFAULT_QUERY_TIMEOUT_SECONDS = 0.05
_OSC_11_RESPONSE_RE = re.compile(rb"(?:\x1b\]|\x9d)11;([^\x07\x1b\x9c]{1,64})(?:\x07|\x1b\\|\x9c)")
_XTERM_COLOR_LEVELS = (0, 95, 135, 175, 215, 255)
_ANSI_COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
)


@dataclass(frozen=True, slots=True)
class RGBColor:
    """An RGB terminal color with 8-bit channels."""

    red: int
    green: int
    blue: int

    @property
    def is_light(self) -> bool:
        """Classify the color by its perceived brightness."""
        brightness = (299 * self.red + 587 * self.green + 114 * self.blue) / 1000
        return brightness >= 128


@dataclass(frozen=True, slots=True)
class ResolvedTheme:
    """A concrete UI and syntax-highlighting theme."""

    variant: ThemeVariant
    syntax_theme: str
    source: ThemeSource


BackgroundQuery: TypeAlias = Callable[[float], RGBColor | None]


def resolve_theme(
    preference: ThemePreference,
    *,
    environ: Mapping[str, str] | None = None,
    query_background: BackgroundQuery | None = None,
    query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> ResolvedTheme:
    """Resolve a configured preference to a concrete terminal theme.

    Automatic resolution first asks the terminal for its default background
    through OSC 11, then falls back to COLORFGBG and finally to the dark theme.
    """
    if preference == "dark" or preference == "light":
        return _resolved_theme(preference, "config")

    environment = os.environ if environ is None else environ
    query = _query_terminal_background if query_background is None else query_background
    should_query = query_background is not None or _supports_safe_osc_11_query(environment)
    background = query(query_timeout_seconds) if should_query else None
    if background is not None:
        return _resolved_theme("light" if background.is_light else "dark", "osc11")

    colorfgbg_background = _background_from_colorfgbg(environment.get("COLORFGBG"))
    if colorfgbg_background is not None:
        return _resolved_theme("light" if colorfgbg_background.is_light else "dark", "colorfgbg")

    return _resolved_theme("dark", "fallback")


def fallback_theme(preference: object) -> ResolvedTheme:
    """Resolve without terminal I/O, for construction and non-interactive use."""
    if preference == "light":
        return _resolved_theme("light", "config")
    if preference == "dark":
        return _resolved_theme("dark", "config")
    return _resolved_theme("dark", "fallback")


def prompt_toolkit_style_rules(theme: ResolvedTheme) -> dict[str, str]:
    """Return prompt_toolkit style rules for a resolved theme."""
    if theme.variant == "light":
        return {
            "status-bar": "bg:#dbeafe fg:#1e3a5f",
            "status-bar.mode-act": "bg:#86efac fg:#14532d bold",
            "status-bar.mode-plan": "bg:#93c5fd fg:#1e3a8a bold",
            "status-bar.warning": "fg:#9a3412 bold",
            "task-pane": "bg:#f8fafc fg:#334155",
            "task-pane.summary": "bg:#e2e8f0 fg:#1e293b bold",
            "task-pane.status-active": "fg:#0369a1 bold",
            "task-pane.status-pending": "fg:#854d0e",
            "task-pane.status-completed": "fg:#15803d",
            "task-pane.status-blocked": "fg:#b91c1c",
            "model-selector": "bg:ansiwhite fg:ansiblack",
            "input-area": "",
        }
    return {
        "status-bar": "bg:#1e3a5f fg:#e2e8f0",
        "status-bar.mode-act": "bg:#166534 fg:#dcfce7 bold",
        "status-bar.mode-plan": "bg:#1d4ed8 fg:#dbeafe bold",
        "status-bar.warning": "fg:#fbbf24 bold",
        "task-pane": "bg:#111827 fg:#cbd5e1",
        "task-pane.summary": "bg:#1f2937 fg:#f1f5f9 bold",
        "task-pane.status-active": "fg:#38bdf8 bold",
        "task-pane.status-pending": "fg:#facc15",
        "task-pane.status-completed": "fg:#4ade80",
        "task-pane.status-blocked": "fg:#f87171",
        "model-selector": "bg:ansibrightblack fg:ansiwhite",
        "input-area": "",
    }


def _resolved_theme(variant: ThemeVariant, source: ThemeSource) -> ResolvedTheme:
    syntax_theme = "ansi_light" if variant == "light" else "ansi_dark"
    return ResolvedTheme(variant=variant, syntax_theme=syntax_theme, source=source)


def _query_terminal_background(timeout_seconds: float) -> RGBColor | None:
    """Query a POSIX terminal's default background with OSC 11."""
    if os.name != "posix" or timeout_seconds <= 0:
        return None

    try:
        import select
        import termios
    except ImportError:
        return None

    flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        terminal_fd = os.open("/dev/tty", flags)
    except OSError:
        return None

    previous_attributes: list[Any] | None = None
    try:
        if not os.isatty(terminal_fd):
            return None

        readable, _, _ = select.select([terminal_fd], [], [], 0)
        if readable:
            # Do not consume input that was queued before theme detection.
            return None

        deadline = time.monotonic() + timeout_seconds
        previous_attributes = termios.tcgetattr(terminal_fd)
        query_attributes = list(previous_attributes)
        query_attributes[6] = list(previous_attributes[6])
        query_attributes[3] = int(query_attributes[3]) & ~(termios.ICANON | termios.ECHO)
        query_attributes[6][termios.VMIN] = 0
        query_attributes[6][termios.VTIME] = 0
        termios.tcsetattr(terminal_fd, termios.TCSANOW, query_attributes)

        readable, _, _ = select.select([terminal_fd], [], [], 0)
        if readable:
            # Canonical mode can hide an unfinished input line from the first
            # readiness check. Preserve it by skipping the query without reading.
            return None

        query = b"\x1b]11;?\x07"
        written = 0
        while written < len(query):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            _, writable, _ = select.select([], [terminal_fd], [], remaining)
            if not writable:
                return None
            try:
                written += os.write(terminal_fd, query[written:])
            except BlockingIOError:
                continue

        response = bytearray()
        while len(response) < 1024:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            readable, _, _ = select.select([terminal_fd], [], [], remaining)
            if not readable:
                break
            try:
                chunk = os.read(terminal_fd, 256)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            response.extend(chunk)
            background = _parse_osc_11_response(bytes(response))
            if background is not None:
                return background
        return None
    except (OSError, ValueError):
        return None
    finally:
        if previous_attributes is not None:
            with contextlib.suppress(OSError, ValueError):
                termios.tcsetattr(terminal_fd, termios.TCSANOW, previous_attributes)
        with contextlib.suppress(OSError):
            os.close(terminal_fd)


def _supports_safe_osc_11_query(environ: Mapping[str, str]) -> bool:
    """Limit active queries to recognized local terminal emulators."""
    if environ.get("SSH_CONNECTION") or environ.get("SSH_TTY"):
        return False
    term_program = environ.get("TERM_PROGRAM", "").casefold()
    return term_program in {
        "apple_terminal",
        "ghostty",
        "hyper",
        "iterm.app",
        "vscode",
        "warpterminal",
        "wezterm",
    } or any(environ.get(name) for name in ("KITTY_WINDOW_ID", "WT_SESSION"))


def _parse_osc_11_response(response: bytes) -> RGBColor | None:
    """Parse an OSC 11 response in rgb: or #RGB form."""
    match = _OSC_11_RESPONSE_RE.search(response)
    if match is None:
        return None

    color_spec = match.group(1).decode("ascii", errors="ignore")
    if color_spec.startswith("rgb:"):
        channels = color_spec[4:].split("/")
        if len(channels) != 3:
            return None
        parsed_channels = [_normalize_hex_channel(channel) for channel in channels]
    elif color_spec.startswith("#"):
        hex_value = color_spec[1:]
        if len(hex_value) not in {3, 6, 9, 12}:
            return None
        channel_width = len(hex_value) // 3
        parsed_channels = [
            _normalize_hex_channel(hex_value[offset : offset + channel_width])
            for offset in range(0, len(hex_value), channel_width)
        ]
    else:
        return None

    if any(channel is None for channel in parsed_channels):
        return None
    red, green, blue = parsed_channels
    if not isinstance(red, int) or not isinstance(green, int) or not isinstance(blue, int):
        return None
    return RGBColor(red=red, green=green, blue=blue)


def _normalize_hex_channel(value: str) -> int | None:
    if not 1 <= len(value) <= 4:
        return None
    try:
        numeric_value = int(value, 16)
    except ValueError:
        return None
    maximum = (16 ** len(value)) - 1
    return round(numeric_value * 255 / maximum)


def _background_from_colorfgbg(value: str | None) -> RGBColor | None:
    if not value:
        return None
    try:
        background_index = int(value.rsplit(";", maxsplit=1)[-1].strip())
    except ValueError:
        return None
    return _xterm_index_to_rgb(background_index)


def _xterm_index_to_rgb(index: int) -> RGBColor | None:
    if 0 <= index < len(_ANSI_COLORS):
        return RGBColor(*_ANSI_COLORS[index])
    if 16 <= index <= 231:
        cube_index = index - 16
        red = _XTERM_COLOR_LEVELS[cube_index // 36]
        green = _XTERM_COLOR_LEVELS[(cube_index % 36) // 6]
        blue = _XTERM_COLOR_LEVELS[cube_index % 6]
        return RGBColor(red=red, green=green, blue=blue)
    if 232 <= index <= 255:
        gray = 8 + (index - 232) * 10
        return RGBColor(red=gray, green=gray, blue=gray)
    return None
