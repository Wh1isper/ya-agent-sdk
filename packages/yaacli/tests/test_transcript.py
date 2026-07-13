from __future__ import annotations

from yaacli.rendering.transcript import TranscriptLimits, TranscriptStore


def test_transcript_enforces_real_line_budget_for_one_large_block() -> None:
    store = TranscriptStore(TranscriptLimits(max_lines=10, max_blocks=100, max_bytes=10_000))

    store.append("\n".join(f"line-{index}" for index in range(100)))

    assert store.total_lines == 10
    assert len(store.blocks) == 1
    assert "output truncated" in store.blocks[0]
    assert "line-99" in store.blocks[0]
    assert "line-0\n" not in store.blocks[0]


def test_transcript_enforces_block_and_byte_budgets() -> None:
    store = TranscriptStore(TranscriptLimits(max_lines=100, max_blocks=3, max_bytes=128))

    for index in range(10):
        store.append(f"block-{index}-" + "x" * 30)

    assert len(store.blocks) <= 3
    assert store.total_bytes <= 128
    assert "block-9" in store.blocks[-1]


def test_transcript_stable_id_updates_after_older_blocks_are_evicted() -> None:
    store = TranscriptStore(TranscriptLimits(max_lines=3, max_blocks=3, max_bytes=10_000))
    store.append("old-1")
    live_id = store.append("live")
    store.append("new-1")
    store.append("new-2")

    assert store.index_of(live_id) == 0
    assert store.replace(live_id, "live-updated") is True
    assert store.blocks[0] == "live-updated"


def test_transcript_visible_text_slices_from_indexed_blocks() -> None:
    store = TranscriptStore(TranscriptLimits(max_lines=100, max_blocks=100, max_bytes=10_000))
    store.append("a\nb\nc")
    store.append("d")
    store.append("e\nf")

    assert store.visible_text(2, 5) == "c\nd\ne"
