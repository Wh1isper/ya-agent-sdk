from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from yaacli.app import TUIApp
from yaacli.config import CommandDefinition


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


def test_tui_display_tool_call_chunks_render_calling_once() -> None:
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {"type": "TOOL_CALL_CHUNK", "toolCallId": "tool-1", "toolCallName": "shell", "delta": '{"command":'},
        {"type": "TOOL_CALL_CHUNK", "toolCallId": "tool-1", "toolCallName": "shell", "delta": '"pytest"}'},
    ])

    calling_blocks = [line for line in app._output_lines if "Calling:" in line and "shell" in line]
    assert len(calling_blocks) == 1
    assert app._append_block.call_count == 1


def test_tui_display_tool_result_renders_once() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {"type": "TOOL_CALL_RESULT", "toolCallId": "tool-1", "toolCallName": "shell", "content": "done"},
        {"type": "TOOL_CALL_RESULT", "toolCallId": "tool-1", "toolCallName": "shell", "content": "done"},
    ])

    assert app._append_block.call_count == 1


def test_tui_display_skips_subagent_tool_events() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": "{}",
            "yaacliAgentId": "subagent-1",
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "content": "done",
            "yaacliAgentId": "subagent-1",
        },
    ])

    assert app._append_block.call_count == 0


def test_tui_display_user_input_attachment_fallback() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._handle_display_events([
        {
            "type": "CUSTOM",
            "name": "yaacli.user_input",
            "value": {"text": "", "attachments": [{"media_type": "image/png", "size_bytes": 1}]},
        }
    ])

    assert any("[Attached 1 image]" in line for line in app._output_lines)
