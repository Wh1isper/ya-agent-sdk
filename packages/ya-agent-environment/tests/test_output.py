"""Tests for shared bounded-output helpers."""

from ya_agent_environment.output import (
    BoundedTextAccumulator,
    split_head_tail_budget,
    truncate_text_head_tail,
    truncate_utf8_head_tail,
)


def test_split_head_tail_budget_favors_head() -> None:
    """Odd retained budgets should allocate the extra unit to the head."""
    assert split_head_tail_budget(10, marker_length=3) == (4, 3)
    assert split_head_tail_budget(0) == (0, 0)


def test_truncate_text_head_tail_retains_boundaries_and_marker() -> None:
    """Character truncation should retain both ends within the exact budget."""
    result = truncate_text_head_tail("HEAD-" + "x" * 100 + "-TAIL", 30, marker="...[truncated]...")
    assert len(result) == 30
    assert result.startswith("HEAD-")
    assert result.endswith("-TAIL")
    assert "truncated" in result


def test_truncate_text_head_tail_omits_marker_when_it_cannot_fit() -> None:
    """Tiny budgets should prioritize source boundaries over a partial marker."""
    assert truncate_text_head_tail("hello world", 5, marker="...[truncated]...") == "helld"
    assert truncate_text_head_tail("hello", 0, marker="...") == ""


def test_truncate_utf8_head_tail_ascii() -> None:
    """ASCII text should retain both boundaries within the exact byte count."""
    text = "HEAD-" + "a" * 90 + "-TAIL"
    result, truncated = truncate_utf8_head_tail(text, 50)
    assert len(result) == 50
    assert result.startswith("HEAD-")
    assert result.endswith("-TAIL")
    assert truncated is True


def test_truncate_utf8_head_tail_returns_unchanged_text_within_limit() -> None:
    """Text within the byte limit should be returned unchanged."""
    result, truncated = truncate_utf8_head_tail("hello", 100)
    assert result == "hello"
    assert truncated is False


def test_truncate_utf8_head_tail_multibyte() -> None:
    """Multibyte text should fit by bytes without partial terminal characters."""
    text = "\u4e2d" * 100
    result, truncated = truncate_utf8_head_tail(text, 150)
    assert truncated is True
    assert len(result) == 50
    assert len(result.encode("utf-8")) == 150


def test_truncate_utf8_head_tail_handles_uneven_boundaries() -> None:
    """Byte truncation should preserve valid boundary characters whenever possible."""
    result, truncated = truncate_utf8_head_tail("a\u4e2d" * 50, 5)
    assert result == "a\u4e2d"
    assert len(result.encode("utf-8")) == 4
    assert truncated is True

    result, truncated = truncate_utf8_head_tail("\u4e2da", 1)
    assert result == "a"
    assert truncated is True


def test_bounded_text_accumulator_retains_streamed_head_and_tail() -> None:
    """Incremental accumulation should use the same bounded head-tail policy."""
    accumulator = BoundedTextAccumulator(30, marker="...[truncated]...")
    accumulator.append("HEAD-")
    accumulator.append("x" * 100)
    accumulator.append("-TAIL")

    result = accumulator.finish()
    assert len(result) == 30
    assert result.startswith("HEAD-")
    assert result.endswith("-TAIL")
    assert "truncated" in result
    assert accumulator.empty is True
