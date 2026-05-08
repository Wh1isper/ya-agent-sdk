"""Tests for gateway model inference helpers."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from ya_agent_sdk.agents.models.gateway import (
    _is_deepseek_model,
    _is_mimo_model,
    _supports_required_tool_choice,
    infer_model,
)


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
    monkeypatch.setenv("COLORIST_API_KEY", "test-key")
    monkeypatch.setenv("COLORIST_BASE_URL", "https://example.com/v1")

    model = infer_model("colorist", "openai:deepseek-v4-pro")

    assert isinstance(model, OpenAIChatModel)
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert model.model_name == "deepseek-v4-pro"
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True
    assert profile.openai_chat_thinking_field == "reasoning_content"
    assert profile.openai_chat_send_back_thinking_parts == "field"
    assert profile.openai_supports_tool_choice_required is False


def test_infer_gateway_mimo_v2_5_uses_reasoning_content_profile(monkeypatch) -> None:
    """Should build OpenAIChatModel with field-mode reasoning_content for MiMo V2.5."""
    monkeypatch.setenv("COLORIST_API_KEY", "test-key")
    monkeypatch.setenv("COLORIST_BASE_URL", "https://example.com/v1")

    model = infer_model("colorist", "openai:MiMo-V2.5-Pro")

    assert isinstance(model, OpenAIChatModel)
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert model.model_name == "MiMo-V2.5-Pro"
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is True
    assert profile.openai_chat_thinking_field == "reasoning_content"
    assert profile.openai_chat_send_back_thinking_parts == "field"


def test_infer_gateway_deepseek_r1_uses_tool_choice_profile_patch(monkeypatch) -> None:
    """Should patch DeepSeek R1 required tool choice support."""
    monkeypatch.setenv("COLORIST_API_KEY", "test-key")
    monkeypatch.setenv("COLORIST_BASE_URL", "https://example.com/v1")

    model = infer_model("colorist", "openai:deepseek-reasoner")

    assert isinstance(model, OpenAIChatModel)
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert model.model_name == "deepseek-reasoner"
    assert profile.openai_chat_thinking_field is None
    assert profile.openai_chat_send_back_thinking_parts == "auto"
    assert profile.openai_supports_tool_choice_required is False
