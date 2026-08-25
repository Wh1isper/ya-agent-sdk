"""Regression tests for lightweight CLI discovery paths."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_import_does_not_load_runtime_stack() -> None:
    _run_isolated(
        """
import sys
import yaacli.cli

blocked = {
    "pydantic_ai",
    "ya_agent_sdk.capabilities",
    "yaacli.app.tui",
    "yaacli.config",
    "yaacli.durable.sqlite",
}
loaded = sorted(blocked.intersection(sys.modules))
assert not loaded, loaded
"""
    )


def test_fast_tui_shell_import_does_not_load_runtime_stack() -> None:
    _run_isolated(
        """
import sys
import yaacli.tui_startup

blocked = {
    "pydantic_ai",
    "ya_agent_sdk.capabilities",
    "yaacli.app.tui",
    "yaacli.config",
    "yaacli.durable.sqlite",
}
loaded = sorted(blocked.intersection(sys.modules))
assert not loaded, loaded
"""
    )


def test_lazy_app_exports_support_module_introspection() -> None:
    _run_isolated(
        """
import yaacli.app

assert "TUIApp" in dir(yaacli.app)
from yaacli.app import TUIApp
assert vars(yaacli.app)["TUIApp"] is TUIApp
"""
    )


def test_help_and_version_do_not_load_runtime_stack() -> None:
    _run_isolated(
        """
import sys
from click.testing import CliRunner
from yaacli.cli import cli

runner = CliRunner()
help_result = runner.invoke(cli, ["--help"])
version_result = runner.invoke(cli, ["--version"])
assert help_result.exit_code == 0, help_result.output
assert version_result.exit_code == 0, version_result.output
assert "Usage:" in help_result.output
assert "yaacli, version" in version_result.output
blocked = {
    "pydantic_ai",
    "ya_agent_sdk.capabilities",
    "yaacli.app.tui",
    "yaacli.config",
    "yaacli.durable.sqlite",
}
loaded = sorted(blocked.intersection(sys.modules))
assert not loaded, loaded
"""
    )


def test_config_import_defers_sdk_runtime_modules() -> None:
    _run_isolated(
        """
import sys
import yaacli.config

blocked = {
    "pydantic_ai",
    "ya_agent_sdk.capabilities",
    "ya_agent_sdk.mcp",
}
loaded = sorted(blocked.intersection(sys.modules))
assert not loaded, loaded
"""
    )


def test_config_lazily_preserves_mcp_type_aliases() -> None:
    _run_isolated(
        """
import sys
from typing import get_type_hints
import yaacli.config

assert "ya_agent_sdk.mcp" not in sys.modules
assert "return" in get_type_hints(yaacli.config.ConfigManager.load_mcp_config)
assert "ya_agent_sdk.mcp" not in sys.modules
from yaacli.config import MCPConfig, MCPServerConfig
from ya_agent_sdk.mcp import MCPConfig as SDKMCPConfig, MCPServerConfig as SDKMCPServerConfig

assert MCPConfig is SDKMCPConfig
assert MCPServerConfig is SDKMCPServerConfig
"""
    )
