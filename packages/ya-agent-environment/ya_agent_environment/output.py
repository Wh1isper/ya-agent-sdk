"""Shared bounded-output helpers for environment and agent tool streams."""

from __future__ import annotations


def split_head_tail_budget(limit: int, *, marker_length: int = 0) -> tuple[int, int]:
    """Split a hard limit between head and tail, favoring the head by one unit."""
    content_limit = max(0, limit - marker_length)
    return (content_limit + 1) // 2, content_limit // 2


def truncate_text_head_tail(text: str, max_chars: int, *, marker: str) -> str:
    """Fit text within a character budget while retaining both its head and tail."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""

    retained_marker = marker if len(marker) < max_chars else ""
    head_chars, tail_chars = split_head_tail_budget(max_chars, marker_length=len(retained_marker))
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + retained_marker + tail


def truncate_utf8_head_tail(text: str, max_bytes: int) -> tuple[str, bool]:
    """Fit UTF-8 text within a byte budget without splitting terminal code points."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    if max_bytes <= 0:
        return "", True

    head_bytes, tail_bytes = split_head_tail_budget(max_bytes)
    first_char_bytes = len(text[0].encode("utf-8", errors="replace"))
    last_char_bytes = len(text[-1].encode("utf-8", errors="replace"))
    if first_char_bytes + last_char_bytes <= max_bytes:
        if head_bytes < first_char_bytes:
            tail_bytes -= first_char_bytes - head_bytes
            head_bytes = first_char_bytes
        if tail_bytes < last_char_bytes:
            head_bytes -= last_char_bytes - tail_bytes
            tail_bytes = last_char_bytes
    elif first_char_bytes <= max_bytes:
        head_bytes = max_bytes
        tail_bytes = 0
    elif last_char_bytes <= max_bytes:
        head_bytes = 0
        tail_bytes = max_bytes

    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    return head + tail, True


class BoundedTextAccumulator:
    """Incrementally retain text, switching to bounded head+tail storage on overflow."""

    def __init__(self, max_length: int, *, marker: str) -> None:
        self.max_length = max_length
        self.marker = marker if len(marker) < max_length else ""
        self._head_length, self._tail_length = split_head_tail_budget(
            max_length,
            marker_length=len(self.marker),
        )
        self._value = ""
        self._head = ""
        self._tail = ""
        self._truncated = False

    @property
    def empty(self) -> bool:
        """Whether no text has been appended since the last reset."""
        return not self._value and not self._truncated

    def append(self, value: str) -> None:
        """Append text without allowing retained state to grow beyond its budget."""
        if not value:
            return
        if self._truncated:
            if self._tail_length:
                self._tail = (self._tail + value)[-self._tail_length :]
            return

        combined = self._value + value
        if len(combined) <= self.max_length:
            self._value = combined
            return

        self._truncated = True
        self._head = combined[: self._head_length]
        self._tail = combined[-self._tail_length :] if self._tail_length else ""
        self._value = ""

    def finish(self) -> str:
        """Return the retained text and reset the accumulator."""
        value = self._head + self.marker + self._tail if self._truncated else self._value
        self.__init__(self.max_length, marker=self.marker)
        return value
