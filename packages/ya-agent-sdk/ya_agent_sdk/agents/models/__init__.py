from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from pydantic_ai.models import (
    Model,
    OpenAIChatCompatibleProvider,
    parse_model_id,
)
from pydantic_ai.models import infer_model as legacy_infer_model
from pydantic_ai.providers import Provider, infer_provider

from ya_agent_sdk.agents.models.gateway import infer_model as infer_gateway_model
from ya_agent_sdk.agents.models.gateway import normalize_legacy_provider_alias
from ya_agent_sdk.agents.models.utils import create_async_http_client

__all__ = ["Model", "infer_model"]


_OPENAI_PROVIDER_ERROR = (
    "Model provider 'openai:' is ambiguous. Use 'openai-chat:<model>' for Chat Completions "
    "or 'openai-responses:<model>' for the Responses API."
)

_OPENAI_RESPONSES_WEBSOCKET_PREFIXES = ("openai-responses-rs:", "openai-responses-ws:")
_HTTPX_RETRY_PROVIDER_NAMES = frozenset({
    "alibaba",
    "anthropic",
    "azure",
    "cerebras",
    "deepseek",
    "fireworks",
    "github",
    "google",
    "google-cloud",
    "heroku",
    "litellm",
    "moonshotai",
    "nebius",
    "ollama",
    "openai-chat",
    "openai-responses",
    "openrouter",
    "ovhcloud",
    "sambanova",
    "together",
    "vercel",
}) | frozenset(get_args(OpenAIChatCompatibleProvider.__value__))


def _raise_for_ambiguous_openai_provider(model: str) -> None:
    if model.startswith("openai:"):
        raise ValueError(_OPENAI_PROVIDER_ERROR)


def infer_model(model: str | Model, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer model from string or return Model instance.

    Args:
        model: Model string or Model instance.
        extra_headers: Optional dict of extra HTTP headers for gateway-backed providers.
            Useful for sticky routing via x-session-id header.
            Applies to gateway-backed model strings such as gateway@provider:model.

    Returns:
        The inferred Model instance.
    """
    if not isinstance(model, str):
        return legacy_infer_model(model)
    _raise_for_ambiguous_openai_provider(model)
    if model.startswith(_OPENAI_RESPONSES_WEBSOCKET_PREFIXES):
        _, model_name = model.split(":", 1)
        if not model_name:
            raise ValueError("OpenAI Responses WebSocket model strings must use format openai-responses-rs:<model>")
        from ya_agent_sdk.agents.models.websocket import build_openai_responses_websocket_model

        return build_openai_responses_websocket_model(model_name)
    if model.startswith("oauth@"):
        provider_name, _, model_name = model.removeprefix("oauth@").partition(":")
        if not provider_name or not model_name:
            raise ValueError("OAuth model strings must use format oauth@provider:model")
        try:
            from ya_oauth_provider import infer_oauth_model
        except ImportError as exc:
            raise ImportError(
                "OAuth-backed models require ya-oauth-provider. Install ya-agent-sdk[oauth] or ya-oauth-provider."
            ) from exc
        return infer_oauth_model(provider_name, model_name, extra_headers=extra_headers)
    if "@" in model:
        gateway_name, model_name = model.split("@", 1)
        return infer_gateway_model(gateway_name, model_name, extra_headers=extra_headers)
    normalized_model = normalize_legacy_provider_alias(model)
    return legacy_infer_model(normalized_model, _retrying_provider_factory_for_model(normalized_model))


def _retrying_provider_factory_for_model(model: str) -> Callable[[str], Provider[Any]]:
    parsed_provider_name, _model_name = parse_model_id(model)

    def provider_factory(provider_name: str) -> Provider[Any]:
        if provider_name != parsed_provider_name or provider_name not in _HTTPX_RETRY_PROVIDER_NAMES:
            return infer_provider(provider_name)
        return _infer_retrying_provider(provider_name)

    return provider_factory


def _infer_retrying_provider(provider_name: str) -> Provider[Any]:
    provider_kwargs: dict[str, Any] = {"http_client": create_async_http_client()}
    if provider_name == "google-cloud":
        return _build_google_cloud_provider(provider_kwargs)

    from pydantic_ai.providers import infer_provider_class

    return infer_provider_class(provider_name)(**provider_kwargs)


def _build_google_cloud_provider(provider_kwargs: dict[str, Any]) -> Provider[Any]:
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    return GoogleCloudProvider(**provider_kwargs)
