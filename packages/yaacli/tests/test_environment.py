"""Tests for TUIEnvironment."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pytest
from yaacli.environment import TUIEnvironment


class TestTUIEnvironment:
    """Tests for TUIEnvironment."""

    @pytest.mark.asyncio
    async def test_enter_exit(self, tmp_path: Path) -> None:
        """TUIEnvironment should enter and exit cleanly."""
        async with TUIEnvironment(default_path=tmp_path) as env:
            assert env.file_operator is not None
            assert env.shell is not None
            assert env.resources is not None

    @pytest.mark.asyncio
    async def test_background_shell_via_shell_abc(self, tmp_path: Path) -> None:
        """Shell ABC should support background process management."""
        async with TUIEnvironment(default_path=tmp_path) as env:
            # Start a background process via Shell ABC
            process_id = await env.shell.start("echo hello")

            # Should be tracked
            assert process_id in env.shell.active_background_processes

            # Wait and drain output
            stdout, stderr, is_running, exit_code = await env.shell.wait_process(process_id, timeout=5.0)
            assert exit_code == 0
            assert "hello" in stdout

    @pytest.mark.asyncio
    async def test_background_processes_killed_on_exit(self, tmp_path: Path) -> None:
        """Background shell processes should be killed when exiting context."""
        shell_ref = None

        async with TUIEnvironment(default_path=tmp_path) as env:
            await env.shell.start("sleep 10")
            shell_ref = env.shell

            # Process should be running
            assert env.shell.has_active_background_processes is True

        # After exit, shell.close() should have killed all processes
        assert shell_ref is not None
        assert shell_ref.has_active_background_processes is False

    @pytest.mark.asyncio
    async def test_inherits_local_environment_features(self, tmp_path: Path) -> None:
        """Should inherit file_operator and shell from LocalEnvironment."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        async with TUIEnvironment(default_path=tmp_path) as env:
            # File operator should work
            content = await env.file_operator.read_file("test.txt")
            assert content == "hello"

            # Shell should work
            exit_code, stdout, _ = await env.shell.execute("echo test")
            assert exit_code == 0
            assert "test" in stdout

    @pytest.mark.asyncio
    async def test_tmp_dir_disabled(self, tmp_path: Path) -> None:
        """tmp_dir should be None when disabled."""
        async with TUIEnvironment(default_path=tmp_path, enable_tmp_dir=False) as env:
            assert env.tmp_dir is None


@pytest.mark.asyncio
async def test_tui_tmp_dir_created_in_system_temp(tmp_path: Path) -> None:
    """Managed TUI tmp storage defaults to the system temporary directory."""
    async with TUIEnvironment(default_path=tmp_path, enable_tmp_dir=True) as env:
        assert isinstance(env.tmp_dir, Path)
        tmp_dir = env.tmp_dir
        assert tmp_dir.exists()
        assert tmp_dir.parent == Path(tempfile.gettempdir()).resolve()
        assert tmp_dir.parent != tmp_path.resolve()
        assert tmp_dir.name.startswith("ya-agent-")
        assert (tmp_dir / ".gitignore").read_bytes() == b"*\n"

    assert not tmp_dir.exists()
    assert not (tmp_path / ".tmp").exists()


def test_tui_tmp_base_dir_override_is_not_exposed() -> None:
    """TUIEnvironment intentionally owns the LocalEnvironment tmp policy."""
    assert "tmp_base_dir" not in inspect.signature(TUIEnvironment).parameters
