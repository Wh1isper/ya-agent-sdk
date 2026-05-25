"""Adapter for the optional ya_ripgrep_core native extension."""

from __future__ import annotations

from functools import cache
from typing import Any


@cache
def _native() -> Any | None:
    """Return the native ripgrep extension when installed."""
    try:
        import ya_ripgrep_core
    except ImportError:
        return None
    return ya_ripgrep_core


def is_available() -> bool:
    """Return True when ya_ripgrep_core is importable."""
    return _native() is not None


def match_glob(path: str, pattern: str) -> bool | None:
    """Match with ripgrep globset when the native extension is available."""
    native = _native()
    if native is None:
        return None
    return bool(native.match_glob(path, pattern))


class NativeRegex:
    """Small wrapper around ya_ripgrep_core.RustRegex."""

    def __init__(self, pattern: str) -> None:
        native = _native()
        if native is None:
            raise ImportError("ya_ripgrep_core is not installed")
        self._regex = native.RustRegex(pattern)

    def search(self, text: str) -> bool:
        """Return True when the compiled native regex matches text."""
        return bool(self._regex.is_match(text))


__all__ = ["NativeRegex", "is_available", "match_glob"]
