from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models import infer_model as legacy_infer_model
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider

from ya_agent_sdk.agents.models.utils import create_async_http_client

REQUIRED_TOOL_CHOICE_UNSUPPORTED_MODEL_KEYWORDS: tuple[str, ...] = ("deepseek",)


def _supports_required_tool_choice(model_name: str) -> bool:
    """Return whether an OpenAI-compatible model supports tool_choice=required."""
    lower = model_name.lower()
    return not any(keyword in lower for keyword in REQUIRED_TOOL_CHOICE_UNSUPPORTED_MODEL_KEYWORDS)


def _request_hook(api_key: str) -> Callable[[httpx.Request], Awaitable[httpx.Request]]:
    """Request hook for the gateway provider.

    It adds the `"Authorization"` header to the request.
    """

    async def _hook(request: httpx.Request) -> httpx.Request:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {api_key}"

        return request

    return _hook


# DeepSeek V4 and MiMo V2.5 thinking models return reasoning tokens through the
# OpenAI-compatible `reasoning_content` field. The chat alias routes to the
# current DeepSeek chat model family, so it receives the same profile patch.
# R1/deepseek-reasoner use a different strict input contract and remain on
# pydantic-ai's built-in DeepSeek profile unless another gateway patch applies.
_DEEPSEEK_V4_MODEL_KEYWORDS: tuple[str, ...] = (
    "deepseek-v4",
    "deepseek_v4",
    "deepseek-chat",
)
_DEEPSEEK_EXCLUDED_MODEL_KEYWORDS: tuple[str, ...] = (
    "deepseek-reasoner",
    "deepseek_reasoner",
    "deepseek-r1",
    "deepseek_r1",
)
_MIMO_V2_5_MODEL_KEYWORDS: tuple[str, ...] = (
    "mimo-v2.5",
    "mimo_v2.5",
    "mimo-v2-5",
    "mimo_v2_5",
)


def _is_deepseek_model(model_name: str) -> bool:
    """Return whether ``model_name`` should use the DeepSeek V4 profile patch."""
    lower = model_name.lower()
    if any(keyword in lower for keyword in _DEEPSEEK_EXCLUDED_MODEL_KEYWORDS):
        return False
    return any(keyword in lower for keyword in _DEEPSEEK_V4_MODEL_KEYWORDS)


def _is_mimo_model(model_name: str) -> bool:
    """Return whether ``model_name`` should use the MiMo V2.5 profile patch."""
    lower = model_name.lower()
    return any(keyword in lower for keyword in _MIMO_V2_5_MODEL_KEYWORDS)


def _requires_reasoning_content_profile(model_name: str) -> bool:
    """Return whether ``model_name`` needs field-mode reasoning round-tripping."""
    return _is_deepseek_model(model_name) or _is_mimo_model(model_name)


def _build_openai_chat_profile(model_name: str) -> OpenAIModelProfile | None:
    """Build OpenAI profile patches needed by OpenAI-compatible gateways.

    DeepSeek V4 and MiMo V2.5 thinking modes emit reasoning through the
    OpenAI-compatible ``reasoning_content`` field, and assistant messages that
    performed tool calls must send that field back in subsequent requests.

    Setting ``openai_chat_thinking_field`` lets ``OpenAIChatModel`` read incoming
    reasoning from ``reasoning_content``. Setting
    ``openai_chat_send_back_thinking_parts='field'`` sends historical
    ``ThinkingPart`` values back through the same field instead of embedding them
    in ``content`` as ``<think>`` tags.

    DeepSeek's OpenAI-compatible chat endpoint also rejects
    ``tool_choice=required``. Pydantic AI consults
    ``openai_supports_tool_choice_required`` before emitting that value for tool
    output.
    """
    requires_reasoning_content = _requires_reasoning_content_profile(model_name)
    supports_required_tool_choice = _supports_required_tool_choice(model_name)
    if not requires_reasoning_content and supports_required_tool_choice:
        return None

    return OpenAIModelProfile(
        supports_thinking=requires_reasoning_content,
        thinking_always_enabled=requires_reasoning_content,
        ignore_streamed_leading_whitespace=requires_reasoning_content,
        openai_chat_thinking_field="reasoning_content" if requires_reasoning_content else None,
        openai_chat_send_back_thinking_parts="field" if requires_reasoning_content else "auto",
        openai_supports_tool_choice_required=supports_required_tool_choice,
    )


def _build_openai_chat_model(model_name: str, provider: Provider[Any]) -> Model:
    """Construct an OpenAIChatModel with profile patches when needed."""
    from pydantic_ai.models.openai import OpenAIChatModel

    return OpenAIChatModel(model_name=model_name, provider=provider, profile=_build_openai_chat_profile(model_name))


def make_gateway_provider(
    gateway_name: str,
    extra_headers: dict[str, str] | None = None,
) -> Callable[[str], Provider[Any]]:
    """Create a gateway_provider function with optional extra headers.

    Args:
        extra_headers: Additional HTTP headers to include in all requests.
            Useful for sticky routing via x-session-id header.

    Returns:
        A gateway_provider function that can be passed to legacy_infer_model.

    Usage:
        # With extra headers for sticky routing
        model = infer_model("google-gla:...", extra_headers={"x-session-id": session_id})

        # Without extra headers
        model = infer_model("google-gla:...")
    """
    gateway_prefix = gateway_name.upper()
    api_key_env_var = f"{gateway_prefix}_API_KEY"
    base_url_env_var = f"{gateway_prefix}_BASE_URL"

    def gateway_provider(provider_name: str) -> Provider[Any]:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise KeyError(f"API key not found, check environment variable: {api_key_env_var}.")

        base_url = os.getenv(base_url_env_var)
        if not base_url:
            raise KeyError(f"Gateway URL not found, check environment variable: {base_url_env_var}.")

        # Only google-gla/bedrock need extra_headers via http_client (their providers don't support direct header injection)
        needs_extra_headers_patch = provider_name in ("google-vertex", "google-gla", "bedrock", "converse")

        if extra_headers and needs_extra_headers_patch:
            http_client = create_async_http_client(extra_headers=extra_headers)
        else:
            http_client = create_async_http_client()

        http_client.event_hooks = {"request": [_request_hook(api_key)]}

        if provider_name in (
            "openai",
            "openai-chat",
            "openai-responses",
            "chat",
            "responses",
        ):
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIProvider(api_key=api_key, base_url=base_url, http_client=http_client)
        elif provider_name == "groq":
            from pydantic_ai.providers.groq import GroqProvider

            return GroqProvider(api_key=api_key, base_url=base_url, http_client=http_client)
        elif provider_name == "anthropic":
            from anthropic import AsyncAnthropic  # pyright: ignore[reportMissingImports]
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=http_client)
            )
        elif provider_name in ("bedrock", "converse"):
            from pydantic_ai.providers.bedrock import BedrockProvider

            return BedrockProvider(
                api_key=api_key,
                base_url=base_url,
                region_name=gateway_name,  # Fake region name to avoid NoRegionError
            )
        elif provider_name in ("google-vertex", "google-gla"):
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleProvider(vertexai=True, api_key=api_key, base_url=base_url, http_client=http_client)
        else:
            raise KeyError(f"Unknown upstream provider: {provider_name}")

    return gateway_provider


def _split_provider_and_model(model: str) -> tuple[str | None, str]:
    """Split a ``provider:model_name`` string into ``(provider, model_name)``."""
    if ":" not in model:
        return None, model
    provider, _, model_name = model.partition(":")
    return provider, model_name


def infer_model(gateway_name: str, model: str, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer model from string, optionally with extra HTTP headers.

    Args:
        gateway_name: Gateway name used for env var lookup.
        model: Model string in format "provider:model_name".
        extra_headers: Optional dict of extra headers to send with each request.
            Useful for sticky routing via x-session-id header.

    Returns:
        The inferred Model instance.

    DeepSeek V4 / MiMo V2.5:
        When ``model`` looks like a thinking model that emits reasoning through
        ``reasoning_content`` and uses an OpenAI-compatible chat provider, the
        gateway constructs the ``OpenAIChatModel`` directly with a corrected
        ``OpenAIModelProfile``. This preserves reasoning round-tripping for
        tool-call turns.
    """
    provider_factory = make_gateway_provider(gateway_name, extra_headers)

    provider_prefix, model_name = _split_provider_and_model(model)
    if provider_prefix in ("openai", "openai-chat", "chat"):
        profile = _build_openai_chat_profile(model_name)
        if profile is not None:
            provider = provider_factory(provider_prefix)
            return _build_openai_chat_model(model_name, provider)

    return legacy_infer_model(model, provider_factory)
