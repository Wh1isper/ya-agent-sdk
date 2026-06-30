"""Tests for gateway model inference helpers."""

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from ya_agent_sdk.agents.models.gateway import (
    _is_deepseek_model,
    _is_mimo_model,
    _supports_required_tool_choice,
    infer_model,
)
from ya_agent_sdk.agents.models.websocket import WebsocketResponsesModel


def test_deepseek_v4_model_detection() -> None:
    """Should patch DeepSeek V4 models and chat aliases."""
    assert _is_deepseek_model("deepseek-v4-pro")
    assert _is_deepseek_model("deepseek_v4_lite")
    assert _is_deepseek_model("deepseek-chat")


def test_deepseek_r1_model_detection_excluded() -> None:
    """Should keep R1 on pydantic-ai's built-in DeepSeek profile path."""
    assert not _is_deepseek_model("deepseek-reasoner")
    assert not _is_deepseek_model("deepseek-r1")


def test_mimo_v2_5_model_detection() -> None:
    """Should patch MiMo V2.5 models."""
    assert _is_mimo_model("MiMo-V2.5")
    assert _is_mimo_model("MiMo-V2.5-Pro")
    assert _is_mimo_model("mimo_v2_5_pro")
    assert _is_mimo_model("mimo-v2-5")


def test_deepseek_required_tool_choice_detection() -> None:
    """Should detect DeepSeek required tool choice support."""
    assert not _supports_required_tool_choice("deepseek-chat")
    assert not _supports_required_tool_choice("deepseek-v4-pro")
    assert not _supports_required_tool_choice("deepseek-reasoner")
    assert _supports_required_tool_choice("gpt-4o")
    assert _supports_required_tool_choice("MiMo-V2.5-Pro")


def test_infer_gateway_deepseek_v4_uses_reasoning_content_profile(monkeypatch) -> None:
    """Should build OpenAIChatModel with field-mode reasoning_content for DeepSeek V4."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    model = infer_model("gateway", "openai-chat:deepseek-v4-pro")

    assert isinstance(model, OpenAIChatModel)
    profile = model.profile
    assert model.model_name == "deepseek-v4-pro"
    assert profile.get("supports_thinking") is True
    assert profile.get("thinking_always_enabled") is True
    assert profile.get("openai_chat_thinking_field") == "reasoning_content"
    assert profile.get("openai_chat_send_back_thinking_parts") == "field"
    assert profile.get("openai_supports_tool_choice_required") is False


def test_infer_gateway_mimo_v2_5_uses_reasoning_content_profile(monkeypatch) -> None:
    """Should build OpenAIChatModel with field-mode reasoning_content for MiMo V2.5."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    model = infer_model("gateway", "openai-chat:MiMo-V2.5-Pro")

    assert isinstance(model, OpenAIChatModel)
    profile = model.profile
    assert model.model_name == "MiMo-V2.5-Pro"
    assert profile.get("supports_thinking") is True
    assert profile.get("thinking_always_enabled") is True
    assert profile.get("openai_chat_thinking_field") == "reasoning_content"
    assert profile.get("openai_chat_send_back_thinking_parts") == "field"


def test_infer_gateway_deepseek_r1_uses_tool_choice_profile_patch(monkeypatch) -> None:
    """Should patch DeepSeek R1 required tool choice support."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    model = infer_model("gateway", "openai-chat:deepseek-reasoner")

    assert isinstance(model, OpenAIChatModel)
    profile = model.profile
    assert model.model_name == "deepseek-reasoner"
    assert profile.get("openai_chat_thinking_field") is None
    assert profile.get("openai_chat_send_back_thinking_parts") == "auto"
    assert profile.get("openai_supports_tool_choice_required") is False


@pytest.mark.parametrize("provider_prefix", ["openai", "chat", "responses"])
def test_infer_gateway_rejects_openai_provider_aliases(provider_prefix: str, monkeypatch) -> None:
    """Should require explicit OpenAI API selection for gateways."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    with pytest.raises(ValueError, match=r"openai-chat.*openai-responses"):
        infer_model("gateway", f"{provider_prefix}:gpt-4o")


def test_infer_gateway_uses_google_provider_for_google(monkeypatch) -> None:
    """Should route the canonical Gemini API provider to GoogleProvider."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    model = infer_model("gateway", "google:gemini-2.5-pro")

    assert model.provider.name == "google"
    assert model.model_name == "gemini-2.5-pro"


@pytest.mark.parametrize("provider_prefix", ["google-cloud", "google-gla", "google-vertex", "google-custom"])
def test_infer_gateway_uses_google_cloud_provider_for_google_prefixes(provider_prefix: str, monkeypatch) -> None:
    """Should route google-* provider prefixes to GoogleCloudProvider."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")

    model = infer_model("gateway", f"{provider_prefix}:gemini-2.5-pro")

    assert model.provider.name == "google-cloud"
    assert model.model_name == "gemini-2.5-pro"


def test_infer_gateway_responses_websocket_aliases_use_websocket_model(monkeypatch) -> None:
    """Gateway mode should route Responses WebSocket aliases through the SDK WebSocket transport."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")
    monkeypatch.delenv("YA_AGENT_OPENAI_RESPONSES_WEBSOCKET_MODE", raising=False)

    for provider_prefix in ("openai-responses-rs", "openai-responses-ws"):
        model = infer_model("gateway", f"{provider_prefix}:gpt-5")
        assert isinstance(model, WebsocketResponsesModel)
        assert model.provider.name == "openai"
        assert model.model_name == "gpt-5"
        assert model.websocket_fallback_state.mode == "auto"
