"""Tests for YAACLI terminal theme detection."""

from __future__ import annotations

from unittest.mock import patch

from yaacli.rendering.renderer import RichRenderer
from yaacli.theme import RGBColor, _background_from_colorfgbg, prompt_toolkit_style_rules, resolve_theme


def test_rgb_color_classifies_perceived_brightness() -> None:
    assert RGBColor(18, 18, 18).is_light is False
    assert RGBColor(245, 245, 245).is_light is True


def test_resolve_theme_honors_explicit_preference() -> None:
    dark = resolve_theme("dark")
    light = resolve_theme("light")

    assert (dark.variant, dark.syntax_theme, dark.source) == ("dark", "ansi_dark", "config")
    assert (light.variant, light.syntax_theme, light.source) == ("light", "ansi_light", "config")


def test_resolve_auto_theme_does_not_issue_active_terminal_queries() -> None:
    with patch("yaacli.theme.os.open") as open_terminal:
        theme = resolve_theme(
            "auto",
            environ={"TERM_PROGRAM": "vscode", "TERM": "xterm-256color"},
        )

    open_terminal.assert_not_called()
    assert (theme.variant, theme.source) == ("dark", "fallback")


def test_resolve_auto_theme_uses_colorfgbg() -> None:
    dark = resolve_theme("auto", environ={"COLORFGBG": "15;0"})
    light = resolve_theme("auto", environ={"COLORFGBG": "0;15"})

    assert (dark.variant, dark.source) == ("dark", "colorfgbg")
    assert (light.variant, light.source) == ("light", "colorfgbg")


def test_resolve_auto_theme_defaults_to_dark() -> None:
    theme = resolve_theme("auto", environ={})

    assert (theme.variant, theme.syntax_theme, theme.source) == ("dark", "ansi_dark", "fallback")


def test_colorfgbg_supports_xterm_256_color_indexes() -> None:
    assert _background_from_colorfgbg("15;16") == RGBColor(0, 0, 0)
    assert _background_from_colorfgbg("0;231") == RGBColor(255, 255, 255)
    assert _background_from_colorfgbg("invalid") is None


def test_resolved_syntax_themes_render_with_rich() -> None:
    renderer = RichRenderer(width=80)

    for preference in ("dark", "light"):
        theme = resolve_theme(preference)
        output = renderer.render_markdown("```python\nanswer = 42\n```", code_theme=theme.syntax_theme)
        assert "answer" in output


def test_prompt_toolkit_rules_adapt_status_and_task_surfaces() -> None:
    dark = resolve_theme("dark")
    light = resolve_theme("light")

    dark_rules = prompt_toolkit_style_rules(dark)
    light_rules = prompt_toolkit_style_rules(light)

    assert dark_rules["status-bar"] != light_rules["status-bar"]
    assert dark_rules["task-pane"] != light_rules["task-pane"]
    assert dark_rules["task-pane.summary"] != light_rules["task-pane.summary"]
    assert dark_rules["model-selector"] == "bg:ansibrightblack fg:ansiwhite"
    assert light_rules["model-selector"] == "bg:ansiwhite fg:ansiblack"
    assert "steering-pane" not in dark_rules
    assert "steering-pane" not in light_rules
