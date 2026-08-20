from __future__ import annotations

from pathlib import Path

import click
from yaacli.cli import ensure_builtin_assets, run_setup_wizard
from yaacli.config import ConfigManager


def test_first_run_setup_writes_loadable_section_scoped_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "explorer.md").write_text(
        "---\nname: explorer\ndescription: Existing explorer\n---\n\nExisting instructions.\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    answers = iter([
        "anthropic:claude-sonnet-4-20250514",
        'setup-smoke-"quoted"-\\value',
        "https://example.invalid/v1",
    ])
    monkeypatch.setattr(click, "prompt", lambda *_args, **_kwargs: next(answers))
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    assert run_setup_wizard(manager) is True

    config = manager.reload()
    content = (config_dir / "config.toml").read_text(encoding="utf-8")
    assert config.general.model == "anthropic:claude-sonnet-4-20250514"
    assert config.general.model_settings == "anthropic_default"
    assert config.general.model_cfg == "claude_1m"
    assert config.env == {
        "ANTHROPIC_API_KEY": 'setup-smoke-"quoted"-\\value',
        "ANTHROPIC_BASE_URL": "https://example.invalid/v1",
    }
    assert content.count('model_settings = "anthropic_default"') == 1
    assert content.count('model_cfg = "claude_1m"') == 1
    assert (config_dir / "mcp.json").is_file()
    assert any((config_dir / "subagents").glob("*.yaml"))
    assert (config_dir / "subagents" / "explorer.md").is_file()
    assert not (config_dir / "subagents" / "explorer.yaml").exists()
    assert any((config_dir / "skills").glob("*/SKILL.md"))


def test_builtin_assets_preserve_same_basename_markdown_subagent(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    markdown_path = subagents_dir / "explorer.md"
    markdown_path.write_text(
        "---\nname: explorer\ndescription: Custom explorer\n---\n\nCustom instructions.\n",
        encoding="utf-8",
    )

    ensure_builtin_assets(manager)

    assert markdown_path.is_file()
    assert not (subagents_dir / "explorer.yaml").exists()
    assert (subagents_dir / "executor.yaml").is_file()
    assert (subagents_dir / "code-reviewer.yaml").is_file()
