"""Bounded transcript storage for the TUI output viewport."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import NewType

BlockId = NewType("BlockId", int)

_TRUNCATION_MARKER = "[... output truncated ...]"


@dataclass(frozen=True, slots=True)
class TranscriptLimits:
    """Retention limits applied to rendered transcript blocks."""

    max_lines: int = 1000
    max_blocks: int = 1000
    max_bytes: int = 4 * 1024 * 1024

    def normalized(self) -> TranscriptLimits:
        return TranscriptLimits(
            max_lines=max(1, self.max_lines),
            max_blocks=max(1, self.max_blocks),
            max_bytes=max(1, self.max_bytes),
        )


class BoundedTextAccumulator:
    """Accumulate a UTF-8 text tail with bounded bytes, lines, and fragments."""

    _TARGET_FRAGMENT_CHARS = 4096

    def __init__(self, *, max_bytes: int, max_lines: int) -> None:
        self._max_bytes = max(1, max_bytes)
        self._max_lines = max(1, max_lines)
        self._fragments: deque[str] = deque()
        self._total_bytes = 0
        self._total_newlines = 0
        self._truncated = False

    @property
    def retained_bytes(self) -> int:
        return self._total_bytes

    @property
    def fragment_count(self) -> int:
        return len(self._fragments)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def append(self, value: str) -> None:
        if not value:
            return
        value = _bound_text_tail(value, max_bytes=self._max_bytes, max_lines=self._max_lines)
        if self._fragments and len(self._fragments[-1]) < self._TARGET_FRAGMENT_CHARS:
            previous = self._fragments.pop()
            self._total_bytes -= len(previous.encode("utf-8"))
            self._total_newlines -= previous.count("\n")
            value = previous + value
        self._fragments.append(value)
        self._total_bytes += len(value.encode("utf-8"))
        self._total_newlines += value.count("\n")
        self._trim()

    def text(self) -> str:
        content = "".join(self._fragments)
        if not self._truncated:
            return content
        if self._max_lines == 1:
            return _bound_text_tail(_TRUNCATION_MARKER, max_bytes=self._max_bytes, max_lines=1)
        marker = _TRUNCATION_MARKER + "\n"
        marker_bytes = marker.encode("utf-8")
        if self._max_bytes <= len(marker_bytes):
            return marker_bytes[: self._max_bytes].decode("utf-8", errors="ignore")
        tail = _bound_text_tail(
            content,
            max_bytes=self._max_bytes - len(marker_bytes),
            max_lines=self._max_lines - 1,
        )
        return marker + tail

    def clear(self) -> None:
        self._fragments.clear()
        self._total_bytes = 0
        self._total_newlines = 0
        self._truncated = False

    def _trim(self) -> None:
        while len(self._fragments) > 1 and (
            self._total_bytes > self._max_bytes or self._total_newlines + 1 > self._max_lines
        ):
            removed = self._fragments.popleft()
            self._total_bytes -= len(removed.encode("utf-8"))
            self._total_newlines -= removed.count("\n")
            self._truncated = True

        if self._fragments and (self._total_bytes > self._max_bytes or self._total_newlines + 1 > self._max_lines):
            previous = self._fragments.pop()
            bounded = _bound_text_tail(previous, max_bytes=self._max_bytes, max_lines=self._max_lines)
            self._fragments.append(bounded)
            self._total_bytes = len(bounded.encode("utf-8"))
            self._total_newlines = bounded.count("\n")
            self._truncated = True


class TranscriptStore:
    """Store rendered blocks under line, block, and UTF-8 byte budgets.

    Blocks receive stable IDs so live entries can still be replaced after older
    blocks are evicted. Compatibility list views are intentionally exposed for
    the existing TUI and its tests; callers must not mutate them directly.
    """

    def __init__(self, limits: TranscriptLimits | None = None) -> None:
        self._limits = (limits or TranscriptLimits()).normalized()
        self._blocks: list[str] = []
        self._line_counts: list[int] = []
        self._byte_counts: list[int] = []
        self._block_ids: list[BlockId] = []
        self._id_to_index: dict[BlockId, int] = {}
        self._cumulative_line_ends: list[int] = []
        self._next_id = 1
        self._total_lines = 0
        self._total_bytes = 0

    @property
    def blocks(self) -> list[str]:
        return self._blocks

    @property
    def line_counts(self) -> list[int]:
        return self._line_counts

    @property
    def total_lines(self) -> int:
        return self._total_lines

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._blocks)

    def configure(self, limits: TranscriptLimits) -> None:
        """Apply limits immediately, evicting old content when necessary."""
        self._limits = limits.normalized()
        self._enforce_limits()

    def append(self, content: str) -> BlockId:
        """Append a block and return its stable identifier."""
        bounded = self._bound_content(content)
        block_id = BlockId(self._next_id)
        self._next_id += 1
        line_count = _line_count(bounded)
        byte_count = len(bounded.encode("utf-8"))

        self._blocks.append(bounded)
        self._line_counts.append(line_count)
        self._byte_counts.append(byte_count)
        self._block_ids.append(block_id)
        self._id_to_index[block_id] = len(self._blocks) - 1
        self._total_lines += line_count
        self._total_bytes += byte_count
        self._cumulative_line_ends.append(self._total_lines)
        self._enforce_limits()
        return block_id

    def replace(self, block_id: BlockId, content: str) -> bool:
        """Replace a retained block. Return False if it has been evicted."""
        index = self._id_to_index.get(block_id)
        if index is None:
            return False

        bounded = self._bound_content(content)
        line_count = _line_count(bounded)
        byte_count = len(bounded.encode("utf-8"))
        self._total_lines += line_count - self._line_counts[index]
        self._total_bytes += byte_count - self._byte_counts[index]
        self._blocks[index] = bounded
        self._line_counts[index] = line_count
        self._byte_counts[index] = byte_count
        self._rebuild_line_ends(start=index)
        self._enforce_limits()
        return block_id in self._id_to_index

    def replace_at(self, index: int, content: str) -> bool:
        """Compatibility replacement by current list index."""
        if index < 0 or index >= len(self._block_ids):
            return False
        return self.replace(self._block_ids[index], content)

    def id_at(self, index: int) -> BlockId | None:
        if index < 0 or index >= len(self._block_ids):
            return None
        return self._block_ids[index]

    def index_of(self, block_id: BlockId | None) -> int | None:
        if block_id is None:
            return None
        return self._id_to_index.get(block_id)

    def contains(self, block_id: BlockId | None) -> bool:
        return block_id is not None and block_id in self._id_to_index

    def clear(self) -> None:
        self._blocks.clear()
        self._line_counts.clear()
        self._byte_counts.clear()
        self._block_ids.clear()
        self._id_to_index.clear()
        self._cumulative_line_ends.clear()
        self._total_lines = 0
        self._total_bytes = 0

    def visible_text(self, start_line: int, end_line: int) -> str:
        """Return a line range, locating its first block with binary search."""
        if not self._blocks or end_line <= start_line:
            return ""
        start_line = max(0, start_line)
        end_line = min(max(start_line, end_line), self._total_lines)
        if end_line <= start_line:
            return ""

        index = bisect_right(self._cumulative_line_ends, start_line)
        previous_end = self._cumulative_line_ends[index - 1] if index else 0
        parts: list[str] = []

        while index < len(self._blocks) and previous_end < end_line:
            block_end = self._cumulative_line_ends[index]
            block_count = self._line_counts[index]
            local_start = max(0, start_line - previous_end)
            local_end = min(block_count, end_line - previous_end)
            block = self._blocks[index]
            if local_start == 0 and local_end == block_count:
                parts.append(block)
            else:
                parts.append("\n".join(block.split("\n")[local_start:local_end]))
            previous_end = block_end
            index += 1

        return "\n".join(parts)

    def _bound_content(self, content: str) -> str:
        limits = self._limits
        if _line_count(content) > limits.max_lines:
            keep_lines = max(0, limits.max_lines - 1)
            tail = content.rsplit("\n", keep_lines)[-keep_lines:] if keep_lines else []
            content = "\n".join([_TRUNCATION_MARKER, *tail])

        encoded = content.encode("utf-8")
        if len(encoded) <= limits.max_bytes:
            return content

        marker = (_TRUNCATION_MARKER + "\n").encode()
        tail_budget = max(0, limits.max_bytes - len(marker))
        tail = encoded[-tail_budget:] if tail_budget else b""
        # Dropping an incomplete leading code point is intentional.
        bounded_tail = tail.decode("utf-8", errors="ignore")
        result = marker.decode() + bounded_tail
        while len(result.encode("utf-8")) > limits.max_bytes:
            result = result[:-1]
        return result

    def _enforce_limits(self) -> None:
        limits = self._limits
        remove_count = 0
        remaining_lines = self._total_lines
        remaining_bytes = self._total_bytes
        remaining_blocks = len(self._blocks)
        while remove_count < len(self._blocks) and (
            remaining_lines > limits.max_lines
            or remaining_bytes > limits.max_bytes
            or remaining_blocks > limits.max_blocks
        ):
            remaining_lines -= self._line_counts[remove_count]
            remaining_bytes -= self._byte_counts[remove_count]
            remaining_blocks -= 1
            remove_count += 1

        if remove_count:
            del self._blocks[:remove_count]
            del self._line_counts[:remove_count]
            del self._byte_counts[:remove_count]
            del self._block_ids[:remove_count]
            self._total_lines = remaining_lines
            self._total_bytes = remaining_bytes
            self._reindex()
            self._rebuild_line_ends()

    def _reindex(self) -> None:
        self._id_to_index.clear()
        self._id_to_index.update((block_id, index) for index, block_id in enumerate(self._block_ids))

    def _rebuild_line_ends(self, start: int = 0) -> None:
        if start <= 0:
            self._cumulative_line_ends.clear()
            running = 0
            start = 0
        else:
            del self._cumulative_line_ends[start:]
            running = self._cumulative_line_ends[start - 1]
        for count in self._line_counts[start:]:
            running += count
            self._cumulative_line_ends.append(running)


def _bound_text_tail(content: str, *, max_bytes: int, max_lines: int) -> str:
    if _line_count(content) > max_lines:
        content = "\n".join(content.rsplit("\n", max_lines)[-max_lines:])
    encoded = content.encode("utf-8")
    if len(encoded) > max_bytes:
        content = encoded[-max_bytes:].decode("utf-8", errors="ignore")
    return content


def _line_count(content: str) -> int:
    return content.count("\n") + 1
