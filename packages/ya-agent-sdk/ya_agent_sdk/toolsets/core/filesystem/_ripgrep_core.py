"""Adapter for the optional ya_ripgrep_core native extension."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cache
from typing import Any, Protocol, runtime_checkable

_DISABLE_ENV = "YA_RIPGREP_CORE_DISABLE"


@runtime_checkable
class _NativeGlobModule(Protocol):
    """Native module surface for batch glob matching."""

    def match_globs(self, paths: Sequence[str], pattern: str) -> Sequence[bool]: ...


@runtime_checkable
class _NativeRegexBytes(Protocol):
    """Native regex surface for whole-file byte search."""

    def search_bytes(
        self,
        data: bytes,
        context_lines: int,
        max_matches: int,
    ) -> Sequence[tuple[int, str, str, int]]: ...


def is_disabled() -> bool:
    """Return True when the native ripgrep core is disabled by environment."""
    return os.getenv(_DISABLE_ENV, "").lower() in {"1", "true", "yes", "on"}


@cache
def _native() -> Any | None:
    """Return the native ripgrep extension when installed and enabled."""
    if is_disabled():
        return None
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
    try:
        return bool(native.match_glob(path, pattern))
    except Exception:
        return None


def match_globs(paths: list[str], pattern: str) -> list[bool] | None:
    """Batch-match paths with ripgrep globset when the native extension is available."""
    native = _native()
    if not isinstance(native, _NativeGlobModule):
        return None
    try:
        return [bool(value) for value in native.match_globs(paths, pattern)]
    except Exception:
        return None


class NativeRegex:
    """Small wrapper around ya_ripgrep_core.RustRegex."""

    def __init__(self, pattern: str) -> None:
        native = _native()
        if native is None:
            raise ImportError("ya_ripgrep_core is not installed")
        self._regex = native.RustRegex(pattern)

    @property
    def supports_search_bytes(self) -> bool:
        """Return True when this native regex supports whole-file byte search."""
        return isinstance(self._regex, _NativeRegexBytes)

    def search(self, text: str) -> bool:
        """Return True when the compiled native regex matches text."""
        return bool(self._regex.is_match(text))

    def search_bytes(
        self,
        data: bytes,
        *,
        context_lines: int,
        max_matches: int,
    ) -> list[tuple[int, str, str, int]] | None:
        """Search a whole file in native code and return line match tuples."""
        if not isinstance(self._regex, _NativeRegexBytes):
            return None
        return [
            (int(line_number), str(matching_line), str(context), int(context_start_line))
            for line_number, matching_line, context, context_start_line in self._regex.search_bytes(
                data,
                max(0, context_lines),
                max_matches,
            )
        ]


__all__ = ["NativeRegex", "is_available", "is_disabled", "match_glob", "match_globs"]
