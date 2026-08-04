from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from yaacli.app import TUIApp, TUIState
from yaacli.clipboard import ClipboardImage, ClipboardImageReadResult
from yaacli.config import CommandDefinition, YaacliConfig


@dataclass
class MockConfig:
    general: object = field(default_factory=lambda: MagicMock(max_requests=10))
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


def test_scroll_up_pauses_auto_follow_until_scrolled_back_to_bottom(monkeypatch: pytest.MonkeyPatch) -> None:
    app = make_app()
    monkeypatch.setattr(app, "_get_viewport_height", lambda: 10)
    app._state = TUIState.RUNNING

    app._append_output("\n".join(f"initial-{index}" for index in range(30)))
    assert app._follow_latest is True
    assert app._scroll_offset == app._get_max_scroll() + 4

    app._scroll_output(-10)
    paused_offset = app._scroll_offset
    assert app._follow_latest is False

    app._append_output("\n".join(f"new-{index}" for index in range(10)))
    assert app._scroll_offset == paused_offset

    app._scroll_output(10_000)
    assert app._follow_latest is True
    assert app._scroll_offset == app._get_max_scroll() + 4

    app._append_output("one more line")
    assert app._scroll_offset == app._get_max_scroll() + 4


def test_scroll_to_bottom_explicitly_reenables_auto_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    app = make_app()
    monkeypatch.setattr(app, "_get_viewport_height", lambda: 10)
    app._append_block("\n".join(f"line-{index}" for index in range(30)))

    app._scroll_output(-10)
    assert app._follow_latest is False

    app._scroll_to_bottom()

    assert app._follow_latest is True
    assert app._scroll_offset == app._get_max_scroll() + 4


def test_viewport_cache_distinguishes_terminal_width(monkeypatch: pytest.MonkeyPatch) -> None:
    app = make_app()
    app._append_block("visible output")
    monkeypatch.setattr(app, "_get_viewport_height", lambda: 10)
    monkeypatch.setattr(app, "_get_terminal_height", lambda: 24)
    width = 80
    monkeypatch.setattr(app, "_get_terminal_width", lambda: width)

    app._get_output_text()
    first_key = app._viewport_cache_key
    width = 120
    app._get_output_text()

    assert first_key == (0, 10, 80, app._output_generation)
    assert app._viewport_cache_key == (0, 10, 120, app._output_generation)


@pytest.mark.asyncio
async def test_terminal_resize_burst_schedules_one_settled_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    app = make_app()
    app._app = MagicMock()
    monkeypatch.setattr("yaacli.app.tui._RESIZE_SETTLE_SECONDS", 0.01)

    app._observe_terminal_size(80, 24)
    app._observe_terminal_size(81, 24)
    first_handle = app._pending_resize_settle_handle
    app._observe_terminal_size(82, 24)

    assert first_handle is not None
    assert first_handle.cancelled()
    assert app._resize_active is True
    await asyncio.sleep(0.02)

    assert app._resize_active is False
    assert app._pending_resize_settle_handle is None
    app._app.invalidate.assert_called_once_with()


def test_stream_render_interval_adapts_to_content_size_and_resize() -> None:
    app = make_app()
    app._stream_render_interval = 1 / 15
    app._streaming_text_buffer = app._new_stream_accumulator()

    app._streaming_text_buffer.append("x" * (32 * 1024))
    assert app._effective_stream_render_interval() == 0.1

    app._streaming_text_buffer.append("x" * (128 * 1024))
    assert app._effective_stream_render_interval() == 0.2

    app._streaming_text_buffer.clear()
    app._resize_active = True
    assert app._effective_stream_render_interval() == 1 / 8


@pytest.mark.asyncio
async def test_pending_stream_frame_moves_when_resize_begins() -> None:
    app = make_app()
    render = MagicMock()
    app._observe_terminal_size(80, 24)
    app._last_stream_render_time = time.monotonic()

    app._request_stream_render(render)
    first_handle = app._pending_stream_render_handle
    first_deadline = app._pending_stream_render_deadline
    app._observe_terminal_size(81, 24)

    assert first_handle is not None
    assert first_handle.cancelled()
    assert first_deadline is not None
    assert app._pending_stream_render_deadline is not None
    assert app._pending_stream_render_deadline > first_deadline
    assert render.call_count == 0
    app._cancel_pending_stream_render()
    assert app._pending_resize_settle_handle is not None
    app._pending_resize_settle_handle.cancel()


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


@pytest.mark.asyncio
async def test_streaming_markdown_commits_coalesced_trailing_frame() -> None:
    app = make_app()
    app._stream_render_interval = 0.01
    app._renderer.render_markdown = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda text, **_: f"rendered: {text}\n"
    )

    app._start_streaming_text("")
    app._update_streaming_text("first")
    app._update_streaming_text(" second")

    assert app._output_lines == ["rendered: first"]
    assert app._pending_stream_render_handle is not None

    await asyncio.sleep(0.02)

    assert app._output_lines == ["rendered: first second"]
    assert app._pending_stream_render_handle is None
    app._finalize_streaming_text()


@pytest.mark.asyncio
async def test_text_to_thinking_switch_flushes_text_and_commits_thinking_tail() -> None:
    app = make_app()
    app._stream_render_interval = 0.01
    app._renderer.render_markdown = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda text, **_: f"text: {text}\n"
    )
    app._event_renderer.render_thinking = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda text, **_: f"thinking: {text}\n"
    )

    app._handle_display_events([
        {"type": "TEXT_MESSAGE_START", "messageId": "text-1"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": "first"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": " tail"},
        {"type": "REASONING_MESSAGE_START", "messageId": "thinking-1"},
        {"type": "REASONING_MESSAGE_CHUNK", "messageId": "thinking-1", "delta": "next"},
        {"type": "REASONING_MESSAGE_CHUNK", "messageId": "thinking-1", "delta": " thought"},
    ])

    assert app._output_lines == ["text: first tail", "thinking: next"]
    assert app._pending_stream_render_handle is not None

    await asyncio.sleep(0.02)

    assert app._output_lines == ["text: first tail", "thinking: next thought"]
    assert app._pending_stream_render_handle is None
    app._finalize_streaming_thinking()


@pytest.mark.asyncio
async def test_display_reset_cancels_stale_stream_render() -> None:
    app = make_app()
    app._stream_render_interval = 0.01

    app._start_streaming_text("")
    app._update_streaming_text("first")
    app._update_streaming_text(" stale")
    assert app._pending_stream_render_handle is not None

    app._restore_output_from_display_events([])
    await asyncio.sleep(0.02)

    assert app._output_lines == []
    assert app._pending_stream_render_handle is None
    assert app._streaming_text_buffer is None


@pytest.mark.asyncio
async def test_tool_boundary_flushes_text_and_cancels_trailing_frame() -> None:
    app = make_app()
    app._stream_render_interval = 60.0

    app._handle_display_events([
        {"type": "TEXT_MESSAGE_START", "messageId": "text-1"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": "complete"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": " sentence"},
    ])

    assert app._pending_stream_render_handle is not None
    assert "sentence" not in app._output_lines[0]

    app._handle_display_events([
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "tool-1",
            "toolCallName": "view",
        }
    ])

    assert "complete sentence" in app._output_lines[0]
    assert "Calling:" in app._output_lines[1]
    assert "view" in app._output_lines[1]
    assert app._pending_stream_render_handle is None

    await asyncio.sleep(0)
    assert len(app._output_lines) == 2


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


@pytest.mark.asyncio
async def test_prompt_history_is_bounded_and_clear_removes_it() -> None:
    app = make_app()
    app._max_prompt_history = 3

    for index in range(10):
        app._add_prompt_history(f"prompt-{index}")

    assert app._prompt_history == ["prompt-7", "prompt-8", "prompt-9"]
    await app._clear_session()
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
