from __future__ import annotations

from yaacli.rendering.transcript import BoundedTextAccumulator


def test_stream_accumulator_bounds_multibyte_bytes_lines_and_fragments() -> None:
    accumulator = BoundedTextAccumulator(max_bytes=1024, max_lines=20)

    for _ in range(100_000):
        accumulator.append("四\n")

    rendered = accumulator.text()
    assert accumulator.retained_bytes <= 1024
    assert rendered.count("\n") + 1 <= 20
    assert accumulator.fragment_count <= 2
    assert accumulator.truncated is True
    assert "output truncated" in rendered


def test_stream_accumulator_keeps_small_content_complete() -> None:
    accumulator = BoundedTextAccumulator(max_bytes=1024, max_lines=20)
    accumulator.append("hello")
    accumulator.append(" world")

    assert accumulator.text() == "hello world"
    assert accumulator.truncated is False
