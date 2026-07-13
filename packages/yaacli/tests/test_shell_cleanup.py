from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


@pytest.mark.asyncio
async def test_direct_shell_drain_exception_terminates_process() -> None:
    app = TUIApp(
        config=MockConfig(),  # type: ignore[arg-type]
        config_manager=MockConfigManager(),  # type: ignore[arg-type]
        working_dir=Path.cwd(),
    )
    process = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.wait = AsyncMock(return_value=1)
    process.returncode = None
    app._terminate_direct_shell_process = AsyncMock()  # type: ignore[method-assign]

    with (
        patch("yaacli.app.tui.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)),
        patch(
            "yaacli.app.tui._drain_direct_shell_stream",
            new=AsyncMock(side_effect=RuntimeError("reader failed")),
        ),
    ):
        await app._execute_shell_command("echo test")

    app._terminate_direct_shell_process.assert_awaited_once_with(process)
    assert any("reader failed" in block for block in app._output_lines)
