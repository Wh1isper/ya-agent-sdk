"""Defensive exception formatting helpers."""

from __future__ import annotations


def safe_exception_str(exc: BaseException) -> str:
    """Return useful exception text without trusting ``__str__`` or ``__repr__``."""
    try:
        message = str(exc)
    except BaseException:
        message = ""
    if message and message != "None":
        return message

    try:
        representation = repr(exc)
    except BaseException:
        representation = ""
    if representation:
        return representation

    try:
        exception_type = type(exc).__name__
    except BaseException:  # pragma: no cover - defensive against hostile metaclasses
        exception_type = "BaseException"
    return f"<{exception_type}: exception text unavailable>"
