"""Tests for YAACLI model profile helpers."""

from __future__ import annotations

from pathlib import Path

from yaacli.config import GeneralConfig, ModelProfileConfig, YaacliConfig
from yaacli.model_profiles import (
    DEFAULT_MODEL_PROFILE_ID,
    build_model_profiles,
    get_startup_model_profile,
    load_state,
    save_selected_model_profile_id,
)


def test_build_model_profiles_includes_default_and_configured_profiles() -> None:
    """Profile list contains [general] default plus configured alternatives."""
    config = YaacliConfig(
        general=GeneralConfig(
            model="anthropic:claude-sonnet-4-5",
            model_settings="anthropic_adaptive_high",
            model_cfg="claude_200k",
        ),
        model_profiles={
            "fast": ModelProfileConfig(
                label="Fast",
                model="openai-responses:gpt-5-mini",
                model_settings="openai_responses_low",
                model_cfg="gpt5_270k",
            ),
        },
    )

    profiles = build_model_profiles(config)

    assert [profile.id for profile in profiles] == [DEFAULT_MODEL_PROFILE_ID, "fast"]
    assert profiles[0].label == "Default"
    assert profiles[0].is_default is True
    assert profiles[1].label == "Fast"
    assert profiles[1].model == "openai-responses:gpt-5-mini"


def test_save_and_load_selected_model_profile_id(tmp_path: Path) -> None:
    """Last selected profile id is persisted in state.json."""
    save_selected_model_profile_id(tmp_path, "fast")

    state = load_state(tmp_path)

    assert state.model_profile.selected_profile_id == "fast"
    assert (tmp_path / "state.json").exists()


def test_startup_profile_uses_persisted_selection_when_available(tmp_path: Path) -> None:
    """Startup profile restores the persisted selection when config still has it."""
    config = YaacliConfig(
        general=GeneralConfig(model="anthropic:claude-sonnet-4-5"),
        model_profiles={
            "fast": ModelProfileConfig(label="Fast", model="openai-responses:gpt-5-mini"),
        },
    )
    save_selected_model_profile_id(tmp_path, "fast")

    profile = get_startup_model_profile(config, tmp_path)

    assert profile is not None
    assert profile.id == "fast"
    assert profile.model == "openai-responses:gpt-5-mini"


def test_startup_profile_falls_back_to_default_for_stale_selection(tmp_path: Path) -> None:
    """Startup profile falls back to [general] when state points to a stale id."""
    config = YaacliConfig(
        general=GeneralConfig(model="anthropic:claude-sonnet-4-5"),
        model_profiles={
            "fast": ModelProfileConfig(label="Fast", model="openai-responses:gpt-5-mini"),
        },
    )
    save_selected_model_profile_id(tmp_path, "stale")

    profile = get_startup_model_profile(config, tmp_path)

    assert profile is not None
    assert profile.id == DEFAULT_MODEL_PROFILE_ID
    assert profile.model == "anthropic:claude-sonnet-4-5"


def test_websocket_responses_provider_uses_openai_response_presets() -> None:
    from yaacli.cli import PROVIDER_ENV_VARS, PROVIDER_MODEL_CFG, PROVIDER_MODEL_SETTINGS

    for provider in ("openai-responses-rs", "openai-responses-ws"):
        assert PROVIDER_ENV_VARS[provider] == ("OPENAI_API_KEY", "OPENAI_BASE_URL")
        assert PROVIDER_MODEL_SETTINGS[provider] == "openai_responses_default"
        assert PROVIDER_MODEL_CFG[provider] == "gpt5_270k"


def test_xai_provider_uses_grok_4_5_presets() -> None:
    from yaacli.cli import PROVIDER_ENV_VARS, PROVIDER_MODEL_CFG, PROVIDER_MODEL_SETTINGS

    assert PROVIDER_ENV_VARS["xai"] == ("XAI_API_KEY", None)
    assert PROVIDER_MODEL_SETTINGS["xai"] == "grok_4_5_default"
    assert PROVIDER_MODEL_CFG["xai"] == "grok_4_5_500k"
