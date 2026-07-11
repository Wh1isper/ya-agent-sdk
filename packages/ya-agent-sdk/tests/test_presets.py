"""Tests for subagents.presets module."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from ya_agent_sdk.presets import (
    ANTHROPIC_1M_CM_DEFAULT,
    ANTHROPIC_1M_CM_DEFAULT_INTERLEAVED_THINKING,
    ANTHROPIC_1M_CM_HIGH,
    ANTHROPIC_1M_CM_HIGH_INTERLEAVED_THINKING,
    ANTHROPIC_1M_CM_LOW,
    ANTHROPIC_1M_CM_LOW_INTERLEAVED_THINKING,
    ANTHROPIC_1M_CM_MEDIUM,
    ANTHROPIC_1M_CM_MEDIUM_INTERLEAVED_THINKING,
    ANTHROPIC_1M_CM_OFF,
    ANTHROPIC_1M_CM_OFF_INTERLEAVED_THINKING,
    ANTHROPIC_1M_DEFAULT,
    ANTHROPIC_1M_DEFAULT_INTERLEAVED_THINKING,
    ANTHROPIC_1M_HIGH,
    ANTHROPIC_1M_HIGH_INTERLEAVED_THINKING,
    ANTHROPIC_1M_LOW,
    ANTHROPIC_1M_LOW_INTERLEAVED_THINKING,
    ANTHROPIC_1M_MEDIUM,
    ANTHROPIC_1M_MEDIUM_INTERLEAVED_THINKING,
    ANTHROPIC_1M_OFF,
    ANTHROPIC_1M_OFF_INTERLEAVED_THINKING,
    ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT,
    ANTHROPIC_ADAPTIVE_1M_CM_HIGH,
    ANTHROPIC_ADAPTIVE_1M_CM_LOW,
    ANTHROPIC_ADAPTIVE_1M_CM_MEDIUM,
    ANTHROPIC_ADAPTIVE_1M_CM_XHIGH,
    ANTHROPIC_ADAPTIVE_1M_DEFAULT,
    ANTHROPIC_ADAPTIVE_1M_HIGH,
    ANTHROPIC_ADAPTIVE_1M_LOW,
    ANTHROPIC_ADAPTIVE_1M_MEDIUM,
    ANTHROPIC_ADAPTIVE_1M_XHIGH,
    ANTHROPIC_ADAPTIVE_CM_DEFAULT,
    ANTHROPIC_ADAPTIVE_CM_HIGH,
    ANTHROPIC_ADAPTIVE_CM_LOW,
    ANTHROPIC_ADAPTIVE_CM_MEDIUM,
    ANTHROPIC_ADAPTIVE_CM_XHIGH,
    ANTHROPIC_ADAPTIVE_DEFAULT,
    ANTHROPIC_ADAPTIVE_HIGH,
    ANTHROPIC_ADAPTIVE_LOW,
    ANTHROPIC_ADAPTIVE_MEDIUM,
    ANTHROPIC_ADAPTIVE_XHIGH,
    ANTHROPIC_CM_DEFAULT,
    ANTHROPIC_CM_DEFAULT_INTERLEAVED_THINKING,
    ANTHROPIC_CM_HIGH,
    ANTHROPIC_CM_HIGH_INTERLEAVED_THINKING,
    ANTHROPIC_CM_LOW,
    ANTHROPIC_CM_LOW_INTERLEAVED_THINKING,
    ANTHROPIC_CM_MEDIUM,
    ANTHROPIC_CM_MEDIUM_INTERLEAVED_THINKING,
    ANTHROPIC_CM_OFF,
    ANTHROPIC_CM_OFF_INTERLEAVED_THINKING,
    ANTHROPIC_CONTEXT_MANAGEMENT_BETA,
    ANTHROPIC_DEFAULT,
    ANTHROPIC_DEFAULT_INTERLEAVED_THINKING,
    ANTHROPIC_HIGH,
    ANTHROPIC_HIGH_INTERLEAVED_THINKING,
    ANTHROPIC_LOW,
    ANTHROPIC_LOW_INTERLEAVED_THINKING,
    ANTHROPIC_MEDIUM,
    ANTHROPIC_MEDIUM_INTERLEAVED_THINKING,
    ANTHROPIC_OFF,
    ANTHROPIC_OFF_INTERLEAVED_THINKING,
    DEEPSEEK_V4_DEFAULT,
    DEEPSEEK_V4_HIGH,
    DEEPSEEK_V4_MAX,
    DEEPSEEK_V4_OFF,
    GROK_4_5_DEFAULT,
    GROK_4_5_HIGH,
    GROK_4_5_LOW,
    GROK_4_5_MEDIUM,
    INHERIT,
    MIMO_V2_5_DEFAULT,
    MIMO_V2_5_PRO_DEFAULT,
    OPENAI_DEFAULT,
    OPENAI_HIGH,
    OPENAI_LOW,
    OPENAI_MAX,
    OPENAI_MEDIUM,
    OPENAI_RESPONSES_DEFAULT,
    OPENAI_RESPONSES_DEFAULT_FAST,
    OPENAI_RESPONSES_HIGH,
    OPENAI_RESPONSES_HIGH_FAST,
    OPENAI_RESPONSES_LOW,
    OPENAI_RESPONSES_LOW_FAST,
    OPENAI_RESPONSES_MAX,
    OPENAI_RESPONSES_MAX_FAST,
    OPENAI_RESPONSES_MEDIUM,
    OPENAI_RESPONSES_MEDIUM_FAST,
    OPENAI_RESPONSES_PRO,
    OPENAI_RESPONSES_PRO_HIGH,
    OPENAI_RESPONSES_PRO_LOW,
    OPENAI_RESPONSES_PRO_MAX,
    OPENAI_RESPONSES_PRO_MEDIUM,
    OPENAI_RESPONSES_PRO_XHIGH,
    OPENAI_RESPONSES_XHIGH,
    OPENAI_RESPONSES_XHIGH_FAST,
    OPENAI_XHIGH,
    ModelConfigPreset,
    ModelSettingsPreset,
    build_context_management,
    get_model_cfg,
    get_model_settings,
    list_model_cfg_presets,
    list_presets,
    resolve_model_cfg,
    resolve_model_settings,
    with_context_management,
)


def _assert_anthropic_adaptive_preset(
    preset: dict[str, object],
    *,
    effort: str,
    has_1m_beta: bool = False,
    has_context_management: bool = False,
) -> None:
    assert preset["anthropic_thinking"] == {"type": "adaptive", "display": "summarized"}
    assert preset["anthropic_effort"] == effort
    assert preset["anthropic_cache_instructions"] is True
    assert preset["anthropic_cache_tool_definitions"] is True
    assert preset["anthropic_cache_messages"] is True

    beta_header = preset.get("extra_headers", {}).get("anthropic-beta", "")
    assert ("context-1m" in beta_header) is has_1m_beta
    assert ("context-management" in beta_header) is has_context_management
    assert "interleaved-thinking" not in beta_header

    if has_context_management:
        assert "extra_body" in preset
        assert "context_management" in preset["extra_body"]
    else:
        assert "extra_body" not in preset


def test_anthropic_presets_structure() -> None:
    """Test that Anthropic compatibility presets resolve to adaptive thinking."""
    _assert_anthropic_adaptive_preset(ANTHROPIC_DEFAULT, effort="high")
    _assert_anthropic_adaptive_preset(ANTHROPIC_HIGH, effort="high")
    _assert_anthropic_adaptive_preset(ANTHROPIC_MEDIUM, effort="medium")
    _assert_anthropic_adaptive_preset(ANTHROPIC_LOW, effort="low")

    assert ANTHROPIC_DEFAULT == ANTHROPIC_ADAPTIVE_DEFAULT
    assert ANTHROPIC_HIGH == ANTHROPIC_ADAPTIVE_HIGH
    assert ANTHROPIC_MEDIUM == ANTHROPIC_ADAPTIVE_MEDIUM
    assert ANTHROPIC_LOW == ANTHROPIC_ADAPTIVE_LOW

    assert ANTHROPIC_OFF["anthropic_thinking"]["type"] == "disabled"
    assert "extra_headers" not in ANTHROPIC_OFF


def test_anthropic_presets_cache_tool_definitions() -> None:
    """Test that Anthropic cache presets also cache tool definitions."""
    for preset in [
        ANTHROPIC_DEFAULT,
        ANTHROPIC_HIGH,
        ANTHROPIC_MEDIUM,
        ANTHROPIC_LOW,
        ANTHROPIC_OFF,
        ANTHROPIC_1M_DEFAULT,
        ANTHROPIC_1M_HIGH,
        ANTHROPIC_1M_MEDIUM,
        ANTHROPIC_1M_LOW,
        ANTHROPIC_1M_OFF,
        ANTHROPIC_CM_DEFAULT,
        ANTHROPIC_CM_HIGH,
        ANTHROPIC_CM_MEDIUM,
        ANTHROPIC_CM_LOW,
        ANTHROPIC_CM_OFF,
        ANTHROPIC_1M_CM_DEFAULT,
        ANTHROPIC_1M_CM_HIGH,
        ANTHROPIC_1M_CM_MEDIUM,
        ANTHROPIC_1M_CM_LOW,
        ANTHROPIC_1M_CM_OFF,
    ]:
        assert preset["anthropic_cache_instructions"] is True
        assert preset["anthropic_cache_tool_definitions"] is True
        assert preset["anthropic_cache_messages"] is True


def test_anthropic_1m_presets_structure() -> None:
    """Test that Anthropic 1M compatibility presets resolve to adaptive 1M presets."""
    _assert_anthropic_adaptive_preset(ANTHROPIC_1M_DEFAULT, effort="high", has_1m_beta=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_1M_HIGH, effort="high", has_1m_beta=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_1M_MEDIUM, effort="medium", has_1m_beta=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_1M_LOW, effort="low", has_1m_beta=True)

    assert ANTHROPIC_1M_DEFAULT == ANTHROPIC_ADAPTIVE_1M_DEFAULT
    assert ANTHROPIC_1M_HIGH == ANTHROPIC_ADAPTIVE_1M_HIGH
    assert ANTHROPIC_1M_MEDIUM == ANTHROPIC_ADAPTIVE_1M_MEDIUM
    assert ANTHROPIC_1M_LOW == ANTHROPIC_ADAPTIVE_1M_LOW

    assert ANTHROPIC_1M_OFF["anthropic_thinking"]["type"] == "disabled"
    assert "context-1m" in ANTHROPIC_1M_OFF["extra_headers"]["anthropic-beta"]


def test_anthropic_cm_presets_structure() -> None:
    """Test that Anthropic CM compatibility presets resolve to adaptive CM presets."""
    _assert_anthropic_adaptive_preset(ANTHROPIC_CM_DEFAULT, effort="high", has_context_management=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_CM_HIGH, effort="high", has_context_management=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_CM_MEDIUM, effort="medium", has_context_management=True)
    _assert_anthropic_adaptive_preset(ANTHROPIC_CM_LOW, effort="low", has_context_management=True)

    assert ANTHROPIC_CM_DEFAULT == ANTHROPIC_ADAPTIVE_CM_DEFAULT
    assert ANTHROPIC_CM_HIGH == ANTHROPIC_ADAPTIVE_CM_HIGH
    assert ANTHROPIC_CM_MEDIUM == ANTHROPIC_ADAPTIVE_CM_MEDIUM
    assert ANTHROPIC_CM_LOW == ANTHROPIC_ADAPTIVE_CM_LOW

    assert ANTHROPIC_CM_OFF["anthropic_thinking"]["type"] == "disabled"
    assert "context-management" in ANTHROPIC_CM_OFF["extra_headers"]["anthropic-beta"]
    assert len(ANTHROPIC_CM_OFF["extra_body"]["context_management"]["edits"]) == 0


def test_anthropic_1m_cm_presets_structure() -> None:
    """Test that Anthropic 1M CM compatibility presets resolve to adaptive 1M CM presets."""
    _assert_anthropic_adaptive_preset(
        ANTHROPIC_1M_CM_DEFAULT, effort="high", has_1m_beta=True, has_context_management=True
    )
    _assert_anthropic_adaptive_preset(
        ANTHROPIC_1M_CM_HIGH, effort="high", has_1m_beta=True, has_context_management=True
    )
    _assert_anthropic_adaptive_preset(
        ANTHROPIC_1M_CM_MEDIUM, effort="medium", has_1m_beta=True, has_context_management=True
    )
    _assert_anthropic_adaptive_preset(ANTHROPIC_1M_CM_LOW, effort="low", has_1m_beta=True, has_context_management=True)

    assert ANTHROPIC_1M_CM_DEFAULT == ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT
    assert ANTHROPIC_1M_CM_HIGH == ANTHROPIC_ADAPTIVE_1M_CM_HIGH
    assert ANTHROPIC_1M_CM_MEDIUM == ANTHROPIC_ADAPTIVE_1M_CM_MEDIUM
    assert ANTHROPIC_1M_CM_LOW == ANTHROPIC_ADAPTIVE_1M_CM_LOW

    assert ANTHROPIC_1M_CM_OFF["anthropic_thinking"]["type"] == "disabled"
    beta_off = ANTHROPIC_1M_CM_OFF["extra_headers"]["anthropic-beta"]
    assert "context-1m" in beta_off
    assert "context-management" in beta_off


def test_anthropic_interleaved_compatibility_aliases() -> None:
    """Test that legacy interleaved preset names resolve to adaptive presets."""
    assert ANTHROPIC_DEFAULT_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_DEFAULT
    assert ANTHROPIC_HIGH_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_HIGH
    assert ANTHROPIC_MEDIUM_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_MEDIUM
    assert ANTHROPIC_LOW_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_LOW
    assert ANTHROPIC_OFF_INTERLEAVED_THINKING == ANTHROPIC_OFF

    assert ANTHROPIC_1M_DEFAULT_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_DEFAULT
    assert ANTHROPIC_1M_HIGH_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_HIGH
    assert ANTHROPIC_1M_MEDIUM_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_MEDIUM
    assert ANTHROPIC_1M_LOW_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_LOW
    assert ANTHROPIC_1M_OFF_INTERLEAVED_THINKING == ANTHROPIC_1M_OFF

    assert ANTHROPIC_CM_DEFAULT_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_CM_DEFAULT
    assert ANTHROPIC_CM_HIGH_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_CM_HIGH
    assert ANTHROPIC_CM_MEDIUM_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_CM_MEDIUM
    assert ANTHROPIC_CM_LOW_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_CM_LOW
    assert ANTHROPIC_CM_OFF_INTERLEAVED_THINKING == ANTHROPIC_CM_OFF

    assert ANTHROPIC_1M_CM_DEFAULT_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT
    assert ANTHROPIC_1M_CM_HIGH_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_CM_HIGH
    assert ANTHROPIC_1M_CM_MEDIUM_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_CM_MEDIUM
    assert ANTHROPIC_1M_CM_LOW_INTERLEAVED_THINKING == ANTHROPIC_ADAPTIVE_1M_CM_LOW
    assert ANTHROPIC_1M_CM_OFF_INTERLEAVED_THINKING == ANTHROPIC_1M_CM_OFF


def test_anthropic_legacy_aliases_point_to_adaptive_presets() -> None:
    """Test that Anthropic legacy preset names remain stable compatibility aliases."""
    legacy_aliases = [
        (ANTHROPIC_DEFAULT, ANTHROPIC_ADAPTIVE_DEFAULT),
        (ANTHROPIC_HIGH, ANTHROPIC_ADAPTIVE_HIGH),
        (ANTHROPIC_MEDIUM, ANTHROPIC_ADAPTIVE_MEDIUM),
        (ANTHROPIC_LOW, ANTHROPIC_ADAPTIVE_LOW),
        (ANTHROPIC_1M_DEFAULT, ANTHROPIC_ADAPTIVE_1M_DEFAULT),
        (ANTHROPIC_1M_HIGH, ANTHROPIC_ADAPTIVE_1M_HIGH),
        (ANTHROPIC_1M_MEDIUM, ANTHROPIC_ADAPTIVE_1M_MEDIUM),
        (ANTHROPIC_1M_LOW, ANTHROPIC_ADAPTIVE_1M_LOW),
        (ANTHROPIC_CM_DEFAULT, ANTHROPIC_ADAPTIVE_CM_DEFAULT),
        (ANTHROPIC_CM_HIGH, ANTHROPIC_ADAPTIVE_CM_HIGH),
        (ANTHROPIC_CM_MEDIUM, ANTHROPIC_ADAPTIVE_CM_MEDIUM),
        (ANTHROPIC_CM_LOW, ANTHROPIC_ADAPTIVE_CM_LOW),
        (ANTHROPIC_1M_CM_DEFAULT, ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT),
        (ANTHROPIC_1M_CM_HIGH, ANTHROPIC_ADAPTIVE_1M_CM_HIGH),
        (ANTHROPIC_1M_CM_MEDIUM, ANTHROPIC_ADAPTIVE_1M_CM_MEDIUM),
        (ANTHROPIC_1M_CM_LOW, ANTHROPIC_ADAPTIVE_1M_CM_LOW),
    ]
    for legacy, adaptive in legacy_aliases:
        assert legacy == adaptive


def test_anthropic_adaptive_presets_structure() -> None:
    """Test that Anthropic adaptive presets have expected structure."""
    for preset in [
        ANTHROPIC_ADAPTIVE_DEFAULT,
        ANTHROPIC_ADAPTIVE_XHIGH,
        ANTHROPIC_ADAPTIVE_HIGH,
        ANTHROPIC_ADAPTIVE_MEDIUM,
        ANTHROPIC_ADAPTIVE_LOW,
    ]:
        # Adaptive thinking config
        assert preset["anthropic_thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in preset["anthropic_thinking"]
        # Effort level
        assert "anthropic_effort" in preset
        assert preset["anthropic_effort"] in ("low", "medium", "high", "xhigh", "max")
        # Caching enabled
        assert preset["anthropic_cache_instructions"] is True
        assert preset["anthropic_cache_tool_definitions"] is True
        assert preset["anthropic_cache_messages"] is True
        # No beta headers needed (adaptive auto-enables interleaved)
        assert "extra_headers" not in preset
        # No extra_body (no context management)
        assert "extra_body" not in preset

    # Verify effort levels
    assert ANTHROPIC_ADAPTIVE_XHIGH["anthropic_effort"] == "xhigh"
    assert ANTHROPIC_ADAPTIVE_HIGH["anthropic_effort"] == "high"
    assert ANTHROPIC_ADAPTIVE_MEDIUM["anthropic_effort"] == "medium"
    assert ANTHROPIC_ADAPTIVE_LOW["anthropic_effort"] == "low"
    assert ANTHROPIC_ADAPTIVE_DEFAULT["anthropic_effort"] == "high"  # API default


def test_anthropic_adaptive_1m_presets_structure() -> None:
    """Test that Anthropic adaptive + 1M presets have 1M beta but no interleaved beta."""
    for preset in [
        ANTHROPIC_ADAPTIVE_1M_DEFAULT,
        ANTHROPIC_ADAPTIVE_1M_XHIGH,
        ANTHROPIC_ADAPTIVE_1M_HIGH,
        ANTHROPIC_ADAPTIVE_1M_MEDIUM,
        ANTHROPIC_ADAPTIVE_1M_LOW,
    ]:
        assert preset["anthropic_thinking"]["type"] == "adaptive"
        assert "anthropic_effort" in preset
        assert "extra_headers" in preset
        assert "context-1m" in preset["extra_headers"]["anthropic-beta"]
        # Should NOT have interleaved thinking beta (adaptive auto-enables it)
        assert "interleaved-thinking" not in preset["extra_headers"]["anthropic-beta"]
        assert "extra_body" not in preset


def test_anthropic_adaptive_cm_presets_structure() -> None:
    """Test that Anthropic adaptive + CM presets have CM beta and extra_body."""
    for preset in [
        ANTHROPIC_ADAPTIVE_CM_DEFAULT,
        ANTHROPIC_ADAPTIVE_CM_XHIGH,
        ANTHROPIC_ADAPTIVE_CM_HIGH,
        ANTHROPIC_ADAPTIVE_CM_MEDIUM,
        ANTHROPIC_ADAPTIVE_CM_LOW,
    ]:
        assert preset["anthropic_thinking"]["type"] == "adaptive"
        assert "anthropic_effort" in preset
        assert "extra_headers" in preset
        assert "context-management" in preset["extra_headers"]["anthropic-beta"]
        assert "interleaved-thinking" not in preset["extra_headers"]["anthropic-beta"]
        assert "extra_body" in preset
        assert "context_management" in preset["extra_body"]


def test_anthropic_adaptive_1m_cm_presets_structure() -> None:
    """Test that Anthropic adaptive + 1M + CM presets have both betas and extra_body."""
    for preset in [
        ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT,
        ANTHROPIC_ADAPTIVE_1M_CM_XHIGH,
        ANTHROPIC_ADAPTIVE_1M_CM_HIGH,
        ANTHROPIC_ADAPTIVE_1M_CM_MEDIUM,
        ANTHROPIC_ADAPTIVE_1M_CM_LOW,
    ]:
        assert preset["anthropic_thinking"]["type"] == "adaptive"
        assert "anthropic_effort" in preset
        beta_header = preset["extra_headers"]["anthropic-beta"]
        assert "context-1m" in beta_header
        assert "context-management" in beta_header
        assert "interleaved-thinking" not in beta_header
        assert "extra_body" in preset
        assert "context_management" in preset["extra_body"]


def test_anthropic_adaptive_max_tokens_ordering() -> None:
    """Test that adaptive preset max_tokens decrease with lower effort."""
    xhigh_tokens = ANTHROPIC_ADAPTIVE_XHIGH["max_tokens"]
    high_tokens = ANTHROPIC_ADAPTIVE_HIGH["max_tokens"]
    medium_tokens = ANTHROPIC_ADAPTIVE_MEDIUM["max_tokens"]
    low_tokens = ANTHROPIC_ADAPTIVE_LOW["max_tokens"]
    assert xhigh_tokens > high_tokens > medium_tokens > low_tokens


def test_openai_chat_presets_structure() -> None:
    """Test that OpenAI Chat presets have expected structure."""
    for preset in [OPENAI_DEFAULT, OPENAI_MAX, OPENAI_XHIGH, OPENAI_HIGH, OPENAI_MEDIUM, OPENAI_LOW]:
        assert "openai_reasoning_effort" in preset
        assert "max_tokens" in preset

    assert OPENAI_MAX["openai_reasoning_effort"] == "max"
    assert OPENAI_XHIGH["openai_reasoning_effort"] == "xhigh"
    assert OPENAI_MAX["max_tokens"] == OPENAI_XHIGH["max_tokens"]
    assert OPENAI_XHIGH["max_tokens"] > OPENAI_HIGH["max_tokens"]


def test_openai_responses_presets_structure() -> None:
    """Test that OpenAI Responses presets have expected structure."""
    responses_presets = [
        OPENAI_RESPONSES_DEFAULT,
        OPENAI_RESPONSES_MAX,
        OPENAI_RESPONSES_XHIGH,
        OPENAI_RESPONSES_HIGH,
        OPENAI_RESPONSES_MEDIUM,
        OPENAI_RESPONSES_LOW,
        OPENAI_RESPONSES_PRO,
        OPENAI_RESPONSES_PRO_MAX,
        OPENAI_RESPONSES_PRO_XHIGH,
        OPENAI_RESPONSES_PRO_HIGH,
        OPENAI_RESPONSES_PRO_MEDIUM,
        OPENAI_RESPONSES_PRO_LOW,
        OPENAI_RESPONSES_DEFAULT_FAST,
        OPENAI_RESPONSES_MAX_FAST,
        OPENAI_RESPONSES_XHIGH_FAST,
        OPENAI_RESPONSES_HIGH_FAST,
        OPENAI_RESPONSES_MEDIUM_FAST,
        OPENAI_RESPONSES_LOW_FAST,
    ]
    for preset in responses_presets:
        assert "openai_reasoning_effort" in preset
        assert "openai_reasoning_summary" in preset

    for preset in [
        OPENAI_RESPONSES_DEFAULT_FAST,
        OPENAI_RESPONSES_MAX_FAST,
        OPENAI_RESPONSES_XHIGH_FAST,
        OPENAI_RESPONSES_HIGH_FAST,
        OPENAI_RESPONSES_MEDIUM_FAST,
        OPENAI_RESPONSES_LOW_FAST,
    ]:
        assert preset["openai_service_tier"] == "priority"

    pro_presets = [
        OPENAI_RESPONSES_PRO,
        OPENAI_RESPONSES_PRO_MAX,
        OPENAI_RESPONSES_PRO_XHIGH,
        OPENAI_RESPONSES_PRO_HIGH,
        OPENAI_RESPONSES_PRO_MEDIUM,
        OPENAI_RESPONSES_PRO_LOW,
    ]
    for preset in pro_presets:
        assert preset["openai_reasoning_mode"] == "pro"
        assert "extra_body" not in preset

    assert OPENAI_RESPONSES_PRO == OPENAI_RESPONSES_PRO_MEDIUM
    assert OPENAI_RESPONSES_PRO_MAX["openai_reasoning_effort"] == "max"
    assert OPENAI_RESPONSES_PRO_XHIGH["openai_reasoning_effort"] == "xhigh"
    assert OPENAI_RESPONSES_PRO_HIGH["openai_reasoning_effort"] == "high"
    assert OPENAI_RESPONSES_PRO_MEDIUM["openai_reasoning_effort"] == "medium"
    assert OPENAI_RESPONSES_PRO_LOW["openai_reasoning_effort"] == "low"
    assert OPENAI_RESPONSES_MAX["openai_reasoning_effort"] == "max"
    assert OPENAI_RESPONSES_MAX_FAST["openai_reasoning_effort"] == "max"
    assert OPENAI_RESPONSES_XHIGH["openai_reasoning_effort"] == "xhigh"
    assert OPENAI_RESPONSES_XHIGH_FAST["openai_reasoning_effort"] == "xhigh"
    assert OPENAI_RESPONSES_HIGH_FAST["openai_reasoning_effort"] == "high"
    assert OPENAI_RESPONSES_MEDIUM_FAST["openai_reasoning_effort"] == "medium"
    assert OPENAI_RESPONSES_LOW_FAST["openai_reasoning_effort"] == "low"
    assert OPENAI_RESPONSES_MAX["max_tokens"] == OPENAI_RESPONSES_XHIGH["max_tokens"]
    assert OPENAI_RESPONSES_XHIGH["max_tokens"] > OPENAI_RESPONSES_HIGH["max_tokens"]


def test_deepseek_v4_presets_structure() -> None:
    """Test that DeepSeek V4 presets have expected structure."""
    for preset in [DEEPSEEK_V4_DEFAULT, DEEPSEEK_V4_HIGH, DEEPSEEK_V4_MAX]:
        assert preset["extra_body"] == {"thinking": {"type": "enabled"}}
        assert preset["openai_reasoning_effort"] in {"high", "max"}
        assert "max_tokens" in preset

    assert DEEPSEEK_V4_DEFAULT == DEEPSEEK_V4_HIGH
    assert DEEPSEEK_V4_HIGH["openai_reasoning_effort"] == "high"
    assert DEEPSEEK_V4_MAX["openai_reasoning_effort"] == "max"
    assert DEEPSEEK_V4_OFF["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "openai_reasoning_effort" not in DEEPSEEK_V4_OFF


def test_mimo_v2_5_presets_structure() -> None:
    """Test that MiMo V2.5 presets have expected structure."""
    for preset in [MIMO_V2_5_DEFAULT, MIMO_V2_5_PRO_DEFAULT]:
        assert preset["extra_body"] == {"thinking": {"type": "enabled"}}
        assert "openai_reasoning_effort" not in preset
        assert "max_tokens" not in preset


def test_get_model_settings_by_enum() -> None:
    """Test getting model settings by enum."""
    settings = get_model_settings(ModelSettingsPreset.ANTHROPIC_HIGH)
    assert settings == ANTHROPIC_HIGH

    # Test 1M preset
    settings_1m = get_model_settings(ModelSettingsPreset.ANTHROPIC_1M_HIGH)
    assert settings_1m == ANTHROPIC_1M_HIGH

    settings_xhigh = get_model_settings(ModelSettingsPreset.ANTHROPIC_ADAPTIVE_XHIGH)
    assert settings_xhigh == ANTHROPIC_ADAPTIVE_XHIGH

    settings_interleaved = get_model_settings(ModelSettingsPreset.ANTHROPIC_HIGH_INTERLEAVED_THINKING)
    assert settings_interleaved == ANTHROPIC_HIGH_INTERLEAVED_THINKING

    settings_1m_interleaved = get_model_settings(ModelSettingsPreset.ANTHROPIC_1M_HIGH_INTERLEAVED_THINKING)
    assert settings_1m_interleaved == ANTHROPIC_1M_HIGH_INTERLEAVED_THINKING

    settings_openai_max = get_model_settings(ModelSettingsPreset.OPENAI_MAX)
    assert settings_openai_max == OPENAI_MAX

    settings_openai_xhigh = get_model_settings(ModelSettingsPreset.OPENAI_XHIGH)
    assert settings_openai_xhigh == OPENAI_XHIGH

    settings_openai_responses_max = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_MAX)
    assert settings_openai_responses_max == OPENAI_RESPONSES_MAX

    settings_openai_responses_xhigh = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_XHIGH)
    assert settings_openai_responses_xhigh == OPENAI_RESPONSES_XHIGH

    settings_openai_responses_max_fast = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_MAX_FAST)
    assert settings_openai_responses_max_fast == OPENAI_RESPONSES_MAX_FAST

    settings_openai_responses_pro = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_PRO)
    assert settings_openai_responses_pro == OPENAI_RESPONSES_PRO

    settings_openai_responses_pro_high = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_PRO_HIGH)
    assert settings_openai_responses_pro_high == OPENAI_RESPONSES_PRO_HIGH

    settings_openai_responses_high_fast = get_model_settings(ModelSettingsPreset.OPENAI_RESPONSES_HIGH_FAST)
    assert settings_openai_responses_high_fast == OPENAI_RESPONSES_HIGH_FAST

    settings_deepseek = get_model_settings(ModelSettingsPreset.DEEPSEEK_V4_MAX)
    assert settings_deepseek == DEEPSEEK_V4_MAX

    settings_mimo = get_model_settings(ModelSettingsPreset.MIMO_V2_5_PRO_DEFAULT)
    assert settings_mimo == MIMO_V2_5_PRO_DEFAULT


def test_get_model_settings_by_string() -> None:
    """Test getting model settings by string name."""
    settings = get_model_settings("anthropic_high")
    assert settings == ANTHROPIC_HIGH

    settings_xhigh = get_model_settings("anthropic_adaptive_xhigh")
    assert settings_xhigh == ANTHROPIC_ADAPTIVE_XHIGH

    # Test 1M preset
    settings_1m = get_model_settings("anthropic_1m_high")
    assert settings_1m == ANTHROPIC_1M_HIGH

    settings_openai_max = get_model_settings("openai_max")
    assert settings_openai_max == OPENAI_MAX

    settings_openai_xhigh = get_model_settings("openai_xhigh")
    assert settings_openai_xhigh == OPENAI_XHIGH

    settings_openai_responses_max = get_model_settings("openai_responses_max")
    assert settings_openai_responses_max == OPENAI_RESPONSES_MAX

    settings_openai_responses_xhigh = get_model_settings("openai_responses_xhigh")
    assert settings_openai_responses_xhigh == OPENAI_RESPONSES_XHIGH

    settings_openai_responses_max_fast = get_model_settings("openai_responses_max_fast")
    assert settings_openai_responses_max_fast == OPENAI_RESPONSES_MAX_FAST

    settings_openai_responses_pro = get_model_settings("openai_responses_pro")
    assert settings_openai_responses_pro == OPENAI_RESPONSES_PRO

    settings_openai_responses_pro_high = get_model_settings("openai_responses_pro_high")
    assert settings_openai_responses_pro_high == OPENAI_RESPONSES_PRO_HIGH

    settings_openai_responses_high_fast = get_model_settings("openai_responses_high_fast")
    assert settings_openai_responses_high_fast == OPENAI_RESPONSES_HIGH_FAST

    settings_deepseek = get_model_settings("deepseek_v4_max")
    assert settings_deepseek == DEEPSEEK_V4_MAX

    settings_mimo = get_model_settings("mimo_v2_5_pro")
    assert settings_mimo == MIMO_V2_5_PRO_DEFAULT

    settings_grok = get_model_settings("grok_4_5_high")
    assert settings_grok == GROK_4_5_HIGH


def test_grok_4_5_presets_structure() -> None:
    """Test Grok 4.5 reasoning effort presets."""
    assert GROK_4_5_DEFAULT == GROK_4_5_HIGH
    assert GROK_4_5_DEFAULT["xai_reasoning_effort"] == "high"
    assert GROK_4_5_DEFAULT["xai_include_encrypted_content"] is True
    assert GROK_4_5_DEFAULT["max_tokens"] == 32 * 1024
    assert GROK_4_5_MEDIUM["xai_reasoning_effort"] == "medium"
    assert GROK_4_5_MEDIUM["max_tokens"] == 16 * 1024
    assert GROK_4_5_LOW["xai_reasoning_effort"] == "low"
    assert GROK_4_5_LOW["max_tokens"] == 8 * 1024


def test_get_model_settings_by_alias() -> None:
    """Test getting model settings by alias."""
    settings = get_model_settings("anthropic")
    assert settings == ANTHROPIC_DEFAULT

    # Test 1M alias
    settings_1m = get_model_settings("anthropic_1m")
    assert settings_1m == ANTHROPIC_1M_DEFAULT

    settings_interleaved = get_model_settings("anthropic_interleaved")
    assert settings_interleaved == ANTHROPIC_DEFAULT_INTERLEAVED_THINKING

    settings_1m_interleaved = get_model_settings("anthropic_1m_interleaved")
    assert settings_1m_interleaved == ANTHROPIC_1M_DEFAULT_INTERLEAVED_THINKING

    settings_cm = get_model_settings("anthropic_cm")
    assert settings_cm == ANTHROPIC_CM_DEFAULT

    settings_1m_cm = get_model_settings("anthropic_1m_cm")
    assert settings_1m_cm == ANTHROPIC_1M_CM_DEFAULT

    settings_cm_interleaved = get_model_settings("anthropic_cm_interleaved")
    assert settings_cm_interleaved == ANTHROPIC_CM_DEFAULT_INTERLEAVED_THINKING

    settings_1m_cm_interleaved = get_model_settings("anthropic_1m_cm_interleaved")
    assert settings_1m_cm_interleaved == ANTHROPIC_1M_CM_DEFAULT_INTERLEAVED_THINKING

    settings_adaptive = get_model_settings("anthropic_adaptive")
    assert settings_adaptive == ANTHROPIC_ADAPTIVE_DEFAULT

    settings_adaptive_1m = get_model_settings("anthropic_adaptive_1m")
    assert settings_adaptive_1m == ANTHROPIC_ADAPTIVE_1M_DEFAULT

    settings_adaptive_cm = get_model_settings("anthropic_adaptive_cm")
    assert settings_adaptive_cm == ANTHROPIC_ADAPTIVE_CM_DEFAULT

    settings_adaptive_1m_cm = get_model_settings("anthropic_adaptive_1m_cm")
    assert settings_adaptive_1m_cm == ANTHROPIC_ADAPTIVE_1M_CM_DEFAULT

    settings = get_model_settings("openai")
    assert settings == OPENAI_DEFAULT

    settings = get_model_settings("openai_responses_standard")
    assert settings == OPENAI_RESPONSES_DEFAULT

    settings = get_model_settings("openai_responses_gpt5_6_pro")
    assert settings == OPENAI_RESPONSES_PRO

    settings = get_model_settings("openai_responses_gpt56_pro")
    assert settings == OPENAI_RESPONSES_PRO

    settings = get_model_settings("openai_responses_gpt5_6_sol")
    assert settings == OPENAI_RESPONSES_MAX

    settings = get_model_settings("openai_responses_gpt56_sol")
    assert settings == OPENAI_RESPONSES_MAX

    settings = get_model_settings("openai_responses_sol")
    assert settings == OPENAI_RESPONSES_MAX

    settings = get_model_settings("openai_responses_terra")
    assert settings == OPENAI_RESPONSES_MEDIUM

    settings = get_model_settings("openai_responses_luna")
    assert settings == OPENAI_RESPONSES_LOW

    settings = get_model_settings("deepseek")
    assert settings == DEEPSEEK_V4_DEFAULT

    settings = get_model_settings("deepseek_v4")
    assert settings == DEEPSEEK_V4_DEFAULT

    settings = get_model_settings("mimo")
    assert settings == MIMO_V2_5_PRO_DEFAULT

    settings = get_model_settings("mimo_v2.5")
    assert settings == MIMO_V2_5_DEFAULT

    settings = get_model_settings("mimo_v2.5_pro")
    assert settings == MIMO_V2_5_PRO_DEFAULT

    settings = get_model_settings("xai")
    assert settings == GROK_4_5_DEFAULT

    settings = get_model_settings("grok")
    assert settings == GROK_4_5_DEFAULT

    settings = get_model_settings("grok_4.5")
    assert settings == GROK_4_5_DEFAULT

    settings = get_model_settings("grok_4_5_latest")
    assert settings == GROK_4_5_DEFAULT


def test_get_model_settings_invalid() -> None:
    """Test that invalid preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset"):
        get_model_settings("invalid_preset_name")


def test_resolve_model_settings_none() -> None:
    """Test that None returns None."""
    result = resolve_model_settings(None)
    assert result is None


def test_resolve_model_settings_dict() -> None:
    """Test that dict is returned as-is."""
    custom = {"temperature": 0.5, "max_tokens": 1000}
    result = resolve_model_settings(custom)
    assert result == custom


def test_resolve_model_settings_string() -> None:
    """Test that string is resolved to preset."""
    result = resolve_model_settings("anthropic_medium")
    assert result == ANTHROPIC_MEDIUM

    # Test 1M preset
    result_1m = resolve_model_settings("anthropic_1m_medium")
    assert result_1m == ANTHROPIC_1M_MEDIUM


def test_list_presets() -> None:
    """Test list_presets returns all available presets."""
    presets = list_presets()

    assert presets == snapshot([
        "anthropic",
        "anthropic_1m",
        "anthropic_1m_cm",
        "anthropic_1m_cm_default",
        "anthropic_1m_cm_default_interleaved_thinking",
        "anthropic_1m_cm_high",
        "anthropic_1m_cm_high_interleaved_thinking",
        "anthropic_1m_cm_interleaved",
        "anthropic_1m_cm_low",
        "anthropic_1m_cm_low_interleaved_thinking",
        "anthropic_1m_cm_medium",
        "anthropic_1m_cm_medium_interleaved_thinking",
        "anthropic_1m_cm_off",
        "anthropic_1m_cm_off_interleaved_thinking",
        "anthropic_1m_default",
        "anthropic_1m_default_interleaved_thinking",
        "anthropic_1m_high",
        "anthropic_1m_high_interleaved_thinking",
        "anthropic_1m_interleaved",
        "anthropic_1m_low",
        "anthropic_1m_low_interleaved_thinking",
        "anthropic_1m_medium",
        "anthropic_1m_medium_interleaved_thinking",
        "anthropic_1m_off",
        "anthropic_1m_off_interleaved_thinking",
        "anthropic_adaptive",
        "anthropic_adaptive_1m",
        "anthropic_adaptive_1m_cm",
        "anthropic_adaptive_1m_cm_default",
        "anthropic_adaptive_1m_cm_high",
        "anthropic_adaptive_1m_cm_low",
        "anthropic_adaptive_1m_cm_medium",
        "anthropic_adaptive_1m_cm_xhigh",
        "anthropic_adaptive_1m_default",
        "anthropic_adaptive_1m_high",
        "anthropic_adaptive_1m_low",
        "anthropic_adaptive_1m_medium",
        "anthropic_adaptive_1m_xhigh",
        "anthropic_adaptive_cm",
        "anthropic_adaptive_cm_default",
        "anthropic_adaptive_cm_high",
        "anthropic_adaptive_cm_low",
        "anthropic_adaptive_cm_medium",
        "anthropic_adaptive_cm_xhigh",
        "anthropic_adaptive_default",
        "anthropic_adaptive_high",
        "anthropic_adaptive_low",
        "anthropic_adaptive_medium",
        "anthropic_adaptive_xhigh",
        "anthropic_cm",
        "anthropic_cm_default",
        "anthropic_cm_default_interleaved_thinking",
        "anthropic_cm_high",
        "anthropic_cm_high_interleaved_thinking",
        "anthropic_cm_interleaved",
        "anthropic_cm_low",
        "anthropic_cm_low_interleaved_thinking",
        "anthropic_cm_medium",
        "anthropic_cm_medium_interleaved_thinking",
        "anthropic_cm_off",
        "anthropic_cm_off_interleaved_thinking",
        "anthropic_default",
        "anthropic_default_interleaved_thinking",
        "anthropic_high",
        "anthropic_high_interleaved_thinking",
        "anthropic_interleaved",
        "anthropic_low",
        "anthropic_low_interleaved_thinking",
        "anthropic_medium",
        "anthropic_medium_interleaved_thinking",
        "anthropic_off",
        "anthropic_off_interleaved_thinking",
        "deepseek",
        "deepseek_v4",
        "deepseek_v4_default",
        "deepseek_v4_high",
        "deepseek_v4_max",
        "deepseek_v4_off",
        "gemini",
        "gemini_2.5",
        "gemini_3",
        "gemini_thinking_budget_default",
        "gemini_thinking_budget_high",
        "gemini_thinking_budget_low",
        "gemini_thinking_budget_medium",
        "gemini_thinking_level_default",
        "gemini_thinking_level_high",
        "gemini_thinking_level_low",
        "gemini_thinking_level_medium",
        "gemini_thinking_level_minimal",
        "grok",
        "grok_4.5",
        "grok_4.5_latest",
        "grok_4_5",
        "grok_4_5_default",
        "grok_4_5_high",
        "grok_4_5_latest",
        "grok_4_5_low",
        "grok_4_5_medium",
        "high",
        "low",
        "medium",
        "mimo",
        "mimo_v2.5",
        "mimo_v2.5_pro",
        "mimo_v2_5",
        "mimo_v2_5_pro",
        "openai",
        "openai_default",
        "openai_high",
        "openai_low",
        "openai_max",
        "openai_medium",
        "openai_responses",
        "openai_responses_default",
        "openai_responses_default_fast",
        "openai_responses_gpt56_pro",
        "openai_responses_gpt56_sol",
        "openai_responses_gpt5_6_pro",
        "openai_responses_gpt5_6_sol",
        "openai_responses_high",
        "openai_responses_high_fast",
        "openai_responses_low",
        "openai_responses_low_fast",
        "openai_responses_luna",
        "openai_responses_max",
        "openai_responses_max_fast",
        "openai_responses_medium",
        "openai_responses_medium_fast",
        "openai_responses_pro",
        "openai_responses_pro_high",
        "openai_responses_pro_low",
        "openai_responses_pro_max",
        "openai_responses_pro_medium",
        "openai_responses_pro_xhigh",
        "openai_responses_sol",
        "openai_responses_standard",
        "openai_responses_terra",
        "openai_responses_xhigh",
        "openai_responses_xhigh_fast",
        "openai_xhigh",
        "xai",
    ])


# =============================================================================
# build_context_management Tests
# =============================================================================


def test_build_context_management_defaults() -> None:
    """Test build_context_management with default parameters (thinking only, keep all)."""
    cm = build_context_management()
    assert "edits" in cm
    edits = cm["edits"]
    assert len(edits) == 1
    # Only thinking clearing with keep all
    assert edits[0]["type"] == "clear_thinking_20251015"
    assert edits[0]["keep"] == "all"


def test_build_context_management_with_tool_uses() -> None:
    """Test build_context_management with both thinking and tool use clearing."""
    cm = build_context_management(clear_tool_uses=True, thinking_keep_turns=2)
    edits = cm["edits"]
    assert len(edits) == 2
    # Thinking clearing must come first
    assert edits[0]["type"] == "clear_thinking_20251015"
    assert edits[0]["keep"] == {"type": "thinking_turns", "value": 2}
    # Tool use clearing second
    assert edits[1]["type"] == "clear_tool_uses_20250919"
    assert edits[1]["trigger"] == {"type": "input_tokens", "value": 100_000}
    assert edits[1]["keep"] == {"type": "tool_uses", "value": 3}
    assert edits[1]["clear_at_least"] == {"type": "input_tokens", "value": 20_000}


def test_build_context_management_tool_uses_only() -> None:
    """Test build_context_management with only tool use clearing."""
    cm = build_context_management(clear_thinking=False, clear_tool_uses=True)
    edits = cm["edits"]
    assert len(edits) == 1
    assert edits[0]["type"] == "clear_tool_uses_20250919"


def test_build_context_management_thinking_keep_turns() -> None:
    """Test build_context_management with specific thinking keep turns."""
    cm = build_context_management(thinking_keep_turns=3)
    edits = cm["edits"]
    assert edits[0]["keep"] == {"type": "thinking_turns", "value": 3}


def test_build_context_management_custom_tool_params() -> None:
    """Test build_context_management with custom tool use parameters."""
    cm = build_context_management(
        clear_thinking=False,
        clear_tool_uses=True,
        tool_use_trigger_tokens=50_000,
        tool_use_keep=5,
        tool_use_clear_at_least=None,
        tool_use_clear_inputs=True,
        tool_use_exclude_tools=["web_search"],
    )
    edits = cm["edits"]
    assert len(edits) == 1
    tool_edit = edits[0]
    assert tool_edit["trigger"]["value"] == 50_000
    assert tool_edit["keep"]["value"] == 5
    assert "clear_at_least" not in tool_edit
    assert tool_edit["clear_tool_inputs"] is True
    assert tool_edit["exclude_tools"] == ["web_search"]


# =============================================================================
# with_context_management Tests
# =============================================================================


def test_with_context_management_on_standard_preset() -> None:
    """Test with_context_management adds beta header and extra_body."""
    result = with_context_management(ANTHROPIC_HIGH)
    # Original should be unmodified
    assert "extra_headers" not in ANTHROPIC_HIGH
    assert "extra_body" not in ANTHROPIC_HIGH
    # Result should have context management
    assert ANTHROPIC_CONTEXT_MANAGEMENT_BETA in result["extra_headers"]["anthropic-beta"]
    assert "context_management" in result["extra_body"]
    # Original thinking settings preserved
    assert result["anthropic_thinking"] == ANTHROPIC_HIGH["anthropic_thinking"]


def test_with_context_management_on_1m_preset() -> None:
    """Test with_context_management merges beta header with existing ones."""
    result = with_context_management(ANTHROPIC_1M_HIGH)
    beta_header = result["extra_headers"]["anthropic-beta"]
    # Should have both 1M and context management betas
    assert "context-1m" in beta_header
    assert "context-management" in beta_header


def test_with_context_management_custom_kwargs() -> None:
    """Test with_context_management forwards kwargs to build_context_management."""
    result = with_context_management(
        ANTHROPIC_MEDIUM,
        clear_tool_uses=True,
        tool_use_trigger_tokens=50_000,
        thinking_keep_turns=3,
    )
    cm = result["extra_body"]["context_management"]
    edits = cm["edits"]
    assert edits[0]["type"] == "clear_thinking_20251015"
    assert edits[0]["keep"] == {"type": "thinking_turns", "value": 3}
    assert edits[1]["trigger"]["value"] == 50_000


def test_with_context_management_prebuilt_config() -> None:
    """Test with_context_management with a pre-built context_management config."""
    cm = build_context_management(clear_thinking=False, clear_tool_uses=True, tool_use_keep=10)
    result = with_context_management(ANTHROPIC_DEFAULT, context_management=cm)
    edits = result["extra_body"]["context_management"]["edits"]
    assert len(edits) == 1  # Only tool use, no thinking
    assert edits[0]["keep"]["value"] == 10


# =============================================================================
# ModelConfigPreset Tests
# =============================================================================


def test_model_cfg_presets_structure() -> None:
    """Test that ModelConfig presets have expected structure."""
    cfg = get_model_cfg("claude_200k")
    assert cfg["context_window"] == 200_000
    assert cfg["max_videos"] == 0  # Claude doesn't support video
    assert "max_images" in cfg
    assert "split_large_images" in cfg
    assert "image_split_max_height" in cfg
    assert "image_split_overlap" in cfg
    assert "capabilities" in cfg

    cfg_400k = get_model_cfg("claude_400k")
    assert cfg_400k["context_window"] == 400_000
    assert cfg_400k["max_videos"] == 0  # Claude doesn't support video
    assert cfg_400k["split_large_images"] is True
    assert cfg_400k["image_split_max_height"] == 4096
    assert cfg_400k["image_split_overlap"] == 50

    cfg_1m = get_model_cfg("claude_1m")
    assert cfg_1m["context_window"] == 1_000_000
    assert cfg_1m["max_videos"] == 0  # Claude doesn't support video
    assert cfg_1m["split_large_images"] is True
    assert cfg_1m["image_split_max_height"] == 4096
    assert cfg_1m["image_split_overlap"] == 50

    cfg_gpt_350k = get_model_cfg("gpt5_350k")
    assert cfg_gpt_350k["context_window"] == 350_000
    assert cfg_gpt_350k["max_videos"] == 0  # GPT doesn't support video
    assert cfg_gpt_350k["split_large_images"] is True
    assert cfg_gpt_350k["image_split_max_height"] == 4096
    assert cfg_gpt_350k["image_split_overlap"] == 50

    cfg_gpt_1m = get_model_cfg("gpt5_1m")
    assert cfg_gpt_1m["context_window"] == 922_000
    assert cfg_gpt_1m["max_videos"] == 0  # GPT doesn't support video
    assert cfg_gpt_1m["split_large_images"] is True
    assert cfg_gpt_1m["image_split_max_height"] == 4096
    assert cfg_gpt_1m["image_split_overlap"] == 50

    # DeepSeek V4 400K
    cfg_deepseek_400k = get_model_cfg("deepseek_v4_400k")
    assert cfg_deepseek_400k["context_window"] == 400_000
    assert cfg_deepseek_400k["max_images"] == 0
    assert cfg_deepseek_400k["max_videos"] == 0
    assert cfg_deepseek_400k["split_large_images"] is False
    assert cfg_deepseek_400k["image_split_max_height"] == 4096
    assert cfg_deepseek_400k["image_split_overlap"] == 50

    cfg_deepseek_v4_1m = get_model_cfg("deepseek_v4_1m")
    assert cfg_deepseek_v4_1m["context_window"] == 1_000_000
    assert cfg_deepseek_v4_1m["max_images"] == 0
    assert cfg_deepseek_v4_1m["max_videos"] == 0
    assert cfg_deepseek_v4_1m["split_large_images"] is False
    assert cfg_deepseek_v4_1m["image_split_max_height"] == 4096
    assert cfg_deepseek_v4_1m["image_split_overlap"] == 50

    cfg_mimo_pro = get_model_cfg("mimo_v2_5_pro_1m")
    assert cfg_mimo_pro["context_window"] == 1_000_000
    assert cfg_mimo_pro["max_images"] == 0
    assert cfg_mimo_pro["max_videos"] == 0
    assert cfg_mimo_pro["split_large_images"] is False
    assert cfg_mimo_pro["image_split_max_height"] == 4096
    assert cfg_mimo_pro["image_split_overlap"] == 50


def test_model_cfg_capabilities() -> None:
    """Test that ModelConfig presets have correct capabilities."""
    from ya_agent_sdk.context import ModelCapability

    # Claude: vision + document, no video
    cfg_claude = get_model_cfg("claude_200k")
    assert ModelCapability.vision in cfg_claude["capabilities"]
    assert ModelCapability.document_understanding in cfg_claude["capabilities"]
    assert ModelCapability.video_understanding not in cfg_claude["capabilities"]

    # GPT-5: vision only
    cfg_gpt = get_model_cfg("gpt5_270k")
    assert ModelCapability.vision in cfg_gpt["capabilities"]
    assert ModelCapability.video_understanding not in cfg_gpt["capabilities"]

    cfg_gpt_350k = get_model_cfg("gpt5_350k")
    assert ModelCapability.vision in cfg_gpt_350k["capabilities"]
    assert ModelCapability.video_understanding not in cfg_gpt_350k["capabilities"]

    cfg_gpt_1m = get_model_cfg("gpt5_1m")
    assert ModelCapability.vision in cfg_gpt_1m["capabilities"]
    assert ModelCapability.video_understanding not in cfg_gpt_1m["capabilities"]

    # DeepSeek V4: reasoning round-trip required
    cfg_deepseek_400k = get_model_cfg("deepseek_v4_400k")
    assert cfg_deepseek_400k["capabilities"] == {ModelCapability.reasoning_required}

    cfg_deepseek_v4 = get_model_cfg("deepseek_v4_1m")
    assert cfg_deepseek_v4["capabilities"] == {ModelCapability.reasoning_required}

    # MiMo V2.5: reasoning round-trip required and foreign thinking stripped
    cfg_mimo = get_model_cfg("mimo_v2_5")
    assert cfg_mimo["capabilities"] == {
        ModelCapability.reasoning_required,
        ModelCapability.reasoning_foreign_incompatible,
    }

    cfg_mimo_pro = get_model_cfg("mimo_v2_5_pro")
    assert cfg_mimo_pro["capabilities"] == {
        ModelCapability.reasoning_required,
        ModelCapability.reasoning_foreign_incompatible,
    }

    # Grok 4.5: vision, no video
    cfg_grok = get_model_cfg("grok_4_5_500k")
    assert cfg_grok["context_window"] == 500_000
    assert cfg_grok["max_images"] == 20
    assert cfg_grok["max_videos"] == 0
    assert cfg_grok["capabilities"] == {ModelCapability.vision}

    # Gemini: vision + video + document
    cfg_gemini = get_model_cfg("gemini_1m")
    assert ModelCapability.vision in cfg_gemini["capabilities"]
    assert ModelCapability.video_understanding in cfg_gemini["capabilities"]
    assert ModelCapability.youtube_url in cfg_gemini["capabilities"]
    assert ModelCapability.document_understanding in cfg_gemini["capabilities"]


def test_get_model_cfg_by_enum() -> None:
    """Test getting model config by enum."""
    cfg = get_model_cfg(ModelConfigPreset.CLAUDE_200K)
    assert cfg["context_window"] == 200_000

    cfg_400k = get_model_cfg(ModelConfigPreset.CLAUDE_400K)
    assert cfg_400k["context_window"] == 400_000

    cfg_gpt_350k = get_model_cfg(ModelConfigPreset.GPT5_350K)
    assert cfg_gpt_350k["context_window"] == 350_000
    assert cfg_gpt_350k["max_videos"] == 0  # GPT doesn't support video

    cfg_gpt = get_model_cfg(ModelConfigPreset.GPT5_1M)
    assert cfg_gpt["context_window"] == 922_000
    assert cfg_gpt["max_videos"] == 0  # GPT doesn't support video

    # DeepSeek V4 400K by enum
    cfg_deepseek_400k = get_model_cfg(ModelConfigPreset.DEEPSEEK_V4_400K)
    assert cfg_deepseek_400k["context_window"] == 400_000
    assert cfg_deepseek_400k["max_videos"] == 0

    cfg_deepseek_v4 = get_model_cfg(ModelConfigPreset.DEEPSEEK_V4_1M)
    assert cfg_deepseek_v4["context_window"] == 1_000_000
    assert cfg_deepseek_v4["max_videos"] == 0

    cfg_mimo_pro = get_model_cfg(ModelConfigPreset.MIMO_V2_5_PRO_1M)
    assert cfg_mimo_pro["context_window"] == 1_000_000
    assert cfg_mimo_pro["max_videos"] == 0

    cfg_grok = get_model_cfg(ModelConfigPreset.GROK_4_5_500K)
    assert cfg_grok["context_window"] == 500_000
    assert cfg_grok["max_videos"] == 0

    cfg_gemini = get_model_cfg(ModelConfigPreset.GEMINI_1M)
    assert cfg_gemini["context_window"] == 1_000_000
    assert cfg_gemini["max_videos"] == 1  # Gemini supports video


def test_get_model_cfg_by_string() -> None:
    """Test getting model config by string name."""
    cfg = get_model_cfg("claude_200k")
    assert cfg["context_window"] == 200_000

    cfg_400k = get_model_cfg("claude_400k")
    assert cfg_400k["context_window"] == 400_000

    cfg_gpt = get_model_cfg("gpt5_270k")
    assert cfg_gpt["context_window"] == 270_000
    assert cfg_gpt["max_videos"] == 0  # GPT doesn't support video

    cfg_gpt_350k = get_model_cfg("gpt5_350k")
    assert cfg_gpt_350k["context_window"] == 350_000
    assert cfg_gpt_350k["max_videos"] == 0  # GPT doesn't support video

    cfg_gpt_1m = get_model_cfg("gpt5_1m")
    assert cfg_gpt_1m["context_window"] == 922_000
    assert cfg_gpt_1m["max_videos"] == 0  # GPT doesn't support video

    # DeepSeek V4 400K by string
    cfg_deepseek_400k = get_model_cfg("deepseek_v4_400k")
    assert cfg_deepseek_400k["context_window"] == 400_000
    assert cfg_deepseek_400k["max_videos"] == 0

    cfg_deepseek_v4 = get_model_cfg("deepseek_v4_1m")
    assert cfg_deepseek_v4["context_window"] == 1_000_000
    assert cfg_deepseek_v4["max_videos"] == 0

    cfg_mimo = get_model_cfg("mimo_v2_5_1m")
    assert cfg_mimo["context_window"] == 1_000_000
    assert cfg_mimo["max_videos"] == 0

    cfg_mimo_pro = get_model_cfg("mimo_v2_5_pro_1m")
    assert cfg_mimo_pro["context_window"] == 1_000_000
    assert cfg_mimo_pro["max_videos"] == 0

    cfg_grok = get_model_cfg("grok_4_5_500k")
    assert cfg_grok["context_window"] == 500_000
    assert cfg_grok["max_videos"] == 0


def test_get_model_cfg_by_alias() -> None:
    """Test getting model config by alias."""
    cfg = get_model_cfg("claude")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("anthropic")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("anthropic_400k")
    assert cfg["context_window"] == 400_000

    cfg = get_model_cfg("openai")
    assert cfg["context_window"] == 270_000  # GPT-5 series

    cfg = get_model_cfg("gemini")
    assert cfg["context_window"] == 200_000  # Default to 200K (cheaper)

    # DeepSeek 400k alias
    cfg = get_model_cfg("deepseek_400k")
    assert cfg["context_window"] == 400_000

    cfg = get_model_cfg("deepseek")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("deepseek_v4")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("mimo")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("mimo_v2.5")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("mimo_v2.5_pro")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("mimo_v2_5")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("mimo_v2_5_pro")
    assert cfg["context_window"] == 1_000_000

    cfg = get_model_cfg("xai")
    assert cfg["context_window"] == 500_000

    cfg = get_model_cfg("grok")
    assert cfg["context_window"] == 500_000

    cfg = get_model_cfg("grok_4.5_latest")
    assert cfg["context_window"] == 500_000

    cfg = get_model_cfg("grok_4_5_latest")
    assert cfg["context_window"] == 500_000


def test_get_model_cfg_invalid() -> None:
    """Test that invalid preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown ModelConfig preset"):
        get_model_cfg("invalid_preset_name")


def test_resolve_model_cfg_none() -> None:
    """Test that None returns None (inherit)."""
    result = resolve_model_cfg(None)
    assert result is None


def test_resolve_model_cfg_inherit() -> None:
    """Test that 'inherit' returns None."""
    result = resolve_model_cfg(INHERIT)
    assert result is None
    result = resolve_model_cfg("inherit")
    assert result is None


def test_resolve_model_cfg_dict() -> None:
    """Test that dict is returned as-is."""
    custom = {"context_window": 100000, "max_images": 10}
    result = resolve_model_cfg(custom)
    assert result == custom


def test_resolve_model_cfg_string() -> None:
    """Test that string is resolved to preset."""
    result = resolve_model_cfg("claude_200k")
    assert result is not None
    assert result["context_window"] == 200_000

    result_400k = resolve_model_cfg("anthropic_400k")
    assert result_400k is not None
    assert result_400k["context_window"] == 400_000


def test_list_model_cfg_presets() -> None:
    """Test list_model_cfg_presets returns all available presets."""
    presets = list_model_cfg_presets()

    assert presets == snapshot([
        "anthropic",
        "anthropic_400k",
        "claude",
        "claude_1m",
        "claude_200k",
        "claude_400k",
        "deepseek",
        "deepseek_400k",
        "deepseek_v4",
        "deepseek_v4_1m",
        "deepseek_v4_400k",
        "gemini",
        "gemini_1m",
        "gemini_200k",
        "gpt5",
        "gpt5_1m",
        "gpt5_270k",
        "gpt5_350k",
        "grok",
        "grok_4.5",
        "grok_4.5_latest",
        "grok_4_5",
        "grok_4_5_500k",
        "grok_4_5_latest",
        "mimo",
        "mimo_v2.5",
        "mimo_v2.5_pro",
        "mimo_v2_5",
        "mimo_v2_5_1m",
        "mimo_v2_5_pro",
        "mimo_v2_5_pro_1m",
        "openai",
        "xai",
    ])
