"""Tests for YAACLI terminal theme detection."""

from __future__ import annotations

from yaacli.rendering.renderer import RichRenderer
from yaacli.theme import (
    RGBColor,
    _background_from_colorfgbg,
    _parse_osc_11_response,
    _supports_safe_osc_11_query,
    prompt_toolkit_style_rules,
    resolve_theme,
)


def test_rgb_color_classifies_perceived_brightness() -> None:
    assert RGBColor(18, 18, 18).is_light is False
    assert RGBColor(245, 245, 245).is_light is True


def test_parse_osc_11_rgb_response() -> None:
    color = _parse_osc_11_response(b"\x1b]11;rgb:ffff/8000/0000\x07")

    assert color == RGBColor(255, 128, 0)


def test_parse_osc_11_hash_response_with_string_terminator() -> None:
    color = _parse_osc_11_response(b"prefix\x1b]11;#fefefe\x1b\\suffix")

    assert color == RGBColor(254, 254, 254)


def test_parse_osc_11_accepts_c1_control_characters() -> None:
    color = _parse_osc_11_response(b"\x9d11;rgb:0000/ffff/0000\x9c")

    assert color == RGBColor(0, 255, 0)


def test_parse_osc_11_rejects_incomplete_or_unrelated_responses() -> None:
    assert _parse_osc_11_response(b"user 11;rgb:ffff/ffff/ffff\x07") is None
    assert _parse_osc_11_response(b"\x1b]10;rgb:ffff/ffff/ffff\x07") is None
    assert _parse_osc_11_response(b"\x1b]11;rgb:ffff/ffff\x07") is None
    assert _parse_osc_11_response(b"\x1b]11;rgb:ffff/ffff/ffff") is None


def test_resolve_theme_honors_explicit_preference_without_querying() -> None:
    def fail_if_queried(_timeout: float) -> RGBColor | None:
        raise AssertionError("explicit themes must not query the terminal")

    dark = resolve_theme("dark", query_background=fail_if_queried)
    light = resolve_theme("light", query_background=fail_if_queried)

    assert (dark.variant, dark.syntax_theme, dark.source) == ("dark", "ansi_dark", "config")
    assert (light.variant, light.syntax_theme, light.source) == ("light", "ansi_light", "config")


def test_safe_osc_query_is_limited_to_recognized_local_terminals() -> None:
    assert _supports_safe_osc_11_query({"TERM_PROGRAM": "vscode"}) is True
    assert _supports_safe_osc_11_query({"KITTY_WINDOW_ID": "1"}) is True
    assert _supports_safe_osc_11_query({"TERM_PROGRAM": "vscode", "SSH_TTY": "/dev/pts/1"}) is False
    assert _supports_safe_osc_11_query({"TERM": "xterm-256color"}) is False


def test_resolve_auto_theme_prefers_osc_11_background() -> None:
    theme = resolve_theme(
        "auto",
        environ={"COLORFGBG": "15;0"},
        query_background=lambda _timeout: RGBColor(248, 248, 248),
    )

    assert theme.variant == "light"
    assert theme.syntax_theme == "ansi_light"
    assert theme.source == "osc11"


def test_resolve_auto_theme_falls_back_to_colorfgbg() -> None:
    dark = resolve_theme(
        "auto",
        environ={"COLORFGBG": "15;0"},
        query_background=lambda _timeout: None,
    )
    light = resolve_theme(
        "auto",
        environ={"COLORFGBG": "0;15"},
        query_background=lambda _timeout: None,
    )

    assert (dark.variant, dark.source) == ("dark", "colorfgbg")
    assert (light.variant, light.source) == ("light", "colorfgbg")


def test_resolve_auto_theme_defaults_to_dark() -> None:
    theme = resolve_theme("auto", environ={}, query_background=lambda _timeout: None)

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


def test_prompt_toolkit_rules_change_light_surface_colors() -> None:
    dark = resolve_theme("dark")
    light = resolve_theme("light")

    dark_rules = prompt_toolkit_style_rules(dark)
    light_rules = prompt_toolkit_style_rules(light)

    assert dark_rules["model-selector"] == "bg:ansibrightblack fg:ansiwhite"
    assert light_rules["model-selector"] == "bg:ansiwhite fg:ansiblack"
