from pydantic_ai.models import (
    Model,
)
from pydantic_ai.models import infer_model as legacy_infer_model

from ya_agent_sdk.agents.models.gateway import infer_model as infer_gateway_model
from ya_agent_sdk.agents.models.gateway import normalize_legacy_provider_alias

__all__ = ["Model", "infer_model"]


_OPENAI_PROVIDER_ERROR = (
    "Model provider 'openai:' is ambiguous. Use 'openai-chat:<model>' for Chat Completions "
    "or 'openai-responses:<model>' for the Responses API."
)

_OPENAI_RESPONSES_WEBSOCKET_PREFIXES = ("openai-responses-rs:", "openai-responses-ws:")


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
    return legacy_infer_model(normalize_legacy_provider_alias(model))
