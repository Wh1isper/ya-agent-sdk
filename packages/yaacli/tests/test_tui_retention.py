from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from yaacli.app import TUIApp
from yaacli.clipboard import ClipboardImage, ClipboardImageReadResult
from yaacli.config import CommandDefinition, YaacliConfig


@dataclass
class MockConfig:
    general: object = field(default_factory=lambda: MagicMock(max_requests=10, mode="act"))
    display: object = field(default_factory=lambda: MagicMock(max_lines=500, mouse=True))
    commands: dict[str, CommandDefinition] = field(default_factory=dict)

    def get_commands(self) -> dict[str, CommandDefinition]:
        return self.commands


@dataclass
class MockConfigManager:
    def get_sessions_dir(self) -> object:
        return MagicMock(exists=lambda: False)


def make_app() -> TUIApp:
    return TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]


def test_all_append_paths_share_real_line_and_byte_limits() -> None:
    app = make_app()
    app._max_output_lines = 20
    app._max_output_blocks = 10
    app._max_output_bytes = 512

    for index in range(100):
        app._append_block(f"event-{index}\nsecond-line")
    app._append_block("\n".join(f"huge-{index}" for index in range(1000)))

    assert app._total_line_count <= 20
    assert len(app._output_lines) <= 10
    assert app._transcript.total_bytes <= 512
    assert app._total_line_count == sum(app._block_line_counts)


def test_streaming_markdown_is_rendered_before_finalization_and_throttled() -> None:
    app = make_app()
    app._renderer.render_markdown = MagicMock(return_value="MARKDOWN_PREVIEW\n")  # type: ignore[method-assign]
    app._renderer.render_text = MagicMock(return_value="PLAIN_PREVIEW\n")  # type: ignore[method-assign]

    app._start_streaming_text("")
    with patch("yaacli.app.tui.time.monotonic", side_effect=[1.0, 1.01, 1.02]):
        app._update_streaming_text("**bold**")
        app._update_streaming_text(" text")

    assert app._renderer.render_markdown.call_count == 1
    assert app._renderer.render_markdown.call_args.args[0] == "**bold**"
    app._renderer.render_text.assert_not_called()
    assert app._output_lines == ["MARKDOWN_PREVIEW"]

    app._finalize_streaming_text()
    assert app._renderer.render_markdown.call_count == 2
    assert app._renderer.render_markdown.call_args.args[0] == "**bold** text"


def test_streaming_ui_retains_only_bounded_raw_tail() -> None:
    app = make_app()
    app._max_stream_render_bytes = 4096
    app._max_output_bytes = 8192
    app._max_output_lines = 20
    app._stream_render_interval = 3600

    app._start_streaming_text("")
    for _ in range(50_000):
        app._update_streaming_text("0123456789\n")

    assert app._streaming_text_buffer is not None
    assert app._streaming_text_buffer.retained_bytes <= 4096
    assert app._streaming_text_buffer.fragment_count <= 2
    app._finalize_streaming_text()
    assert app._transcript.total_bytes <= 8192
    assert app._total_line_count <= 20
    assert any("output truncated" in block for block in app._output_lines)


def test_prompt_history_is_bounded_and_clear_removes_it() -> None:
    app = make_app()
    app._max_prompt_history = 3

    for index in range(10):
        app._add_prompt_history(f"prompt-{index}")

    assert app._prompt_history == ["prompt-7", "prompt-8", "prompt-9"]
    app._clear_session()
    assert app._prompt_history == []


@pytest.mark.asyncio
async def test_pending_attachment_count_and_byte_budgets() -> None:
    config = YaacliConfig()
    config.media.max_pending_attachments = 1
    config.media.max_pending_attachment_bytes = 4
    app = TUIApp(config=config, config_manager=MockConfigManager())  # type: ignore[arg-type]

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock()) as mock_read:
        mock_read.return_value = ClipboardImageReadResult(image=ClipboardImage(data=b"12345", media_type="image/png"))
        await app._paste_clipboard_image()
        assert app._pending_attachments == []
        assert any("byte limit exceeded" in block for block in app._output_lines)

        mock_read.return_value = ClipboardImageReadResult(image=ClipboardImage(data=b"1234", media_type="image/png"))
        await app._paste_clipboard_image()
        await app._paste_clipboard_image()

    assert len(app._pending_attachments) == 1
    assert any("Attachment limit reached" in block for block in app._output_lines)


@pytest.mark.asyncio
async def test_async_session_save_does_not_block_event_loop() -> None:
    app = make_app()
    started = threading.Event()
    release = threading.Event()

    def blocking_save(**_: object) -> object:
        started.set()
        assert release.wait(timeout=2)
        return object()

    runtime = MagicMock()
    runtime.ctx.export_state.return_value.model_dump_json.return_value = "{}"
    app._runtime = runtime
    app._display_replay.append({"type": "CUSTOM", "name": "test", "value": "payload"})
    with patch("yaacli.app.tui.save_session_turn", side_effect=blocking_save):
        save_task = asyncio.create_task(
            app._save_session_snapshot_async(include_usage_ledger=False, save_reason="test")
        )
        assert await asyncio.to_thread(started.wait, 1)

        # This event-loop turn would be blocked if persistence ran synchronously.
        await asyncio.sleep(0)
        assert save_task.done() is False
        release.set()
        assert await save_task is True
