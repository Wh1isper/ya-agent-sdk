"""Provider session routing shared by root and child agent construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.models import KnownModelName, Model

from ya_agent_sdk.context import AgentContext, ModelConfig, ModelFeature

_CONTEXT_HEADER_MODEL_PREFIXES = ("oauth@codex:", "openai-responses-rs:", "openai-responses-ws:")
_GATEWAY_CONTEXT_HEADER_UPSTREAM_PREFIXES = (
    "openai-responses:",
    "openai-responses-rs:",
    "openai-responses-ws:",
)
_PROVIDER_CONTEXT_HEADER_NAMES = frozenset({
    "session_id",
    "session-id",
    "x-session-id",
    "thread_id",
    "thread-id",
    "x-client-request-id",
})


def model_uses_context_headers(model: Model | KnownModelName | str | None) -> bool:
    """Return whether an SDK model route accepts provider context headers."""
    if not isinstance(model, str):
        return False
    if model.startswith(_CONTEXT_HEADER_MODEL_PREFIXES):
        return True
    gateway_name, separator, upstream_model = model.partition("@")
    return bool(gateway_name and separator and upstream_model.startswith(_GATEWAY_CONTEXT_HEADER_UPSTREAM_PREFIXES))


def patch_provider_session_settings(
    model_cfg: ModelConfig,
    model_settings: Mapping[str, Any] | None,
    model_extra_headers: Mapping[str, str],
) -> ModelSettings:
    """Bind one model request to the provider session carried by its current context."""
    patched_settings: dict[str, Any] = dict(model_settings or {})
    configured_extra_headers = patched_settings.get("extra_headers")
    patched_extra_headers = (
        {
            name: value
            for name, value in configured_extra_headers.items()
            if not isinstance(name, str) or name.lower() not in _PROVIDER_CONTEXT_HEADER_NAMES
        }
        if isinstance(configured_extra_headers, Mapping)
        else {}
    )
    patched_extra_headers.update(model_extra_headers)
    patched_settings["extra_headers"] = patched_extra_headers

    if model_cfg.has_capability(ModelFeature.openai_prompt_cache_key):
        configured_extra_body = patched_settings.get("extra_body")
        if isinstance(configured_extra_body, Mapping):
            patched_settings["extra_body"] = {
                name: value for name, value in configured_extra_body.items() if name != "prompt_cache_key"
            }
        patched_settings["openai_prompt_cache_key"] = model_extra_headers["x-session-id"]
    return cast(ModelSettings, patched_settings)


def patch_prompt_cache_key(
    model_cfg: ModelConfig,
    model_settings: ModelSettings | None,
    model_extra_headers: dict[str, str] | None,
) -> ModelSettings | None:
    """Bind capable model requests to the same session used by their headers."""
    if not model_cfg.has_capability(ModelFeature.openai_prompt_cache_key) or model_extra_headers is None:
        return model_settings
    return patch_provider_session_settings(model_cfg, model_settings, model_extra_headers)


@dataclass(slots=True)
class ProviderSessionSettingsCapability(AbstractCapability[AgentContext]):
    """Resolve provider routing headers from the active run context on every request."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    def get_model_settings(self) -> Callable[[RunContext[AgentContext]], ModelSettings]:
        def settings(ctx: RunContext[AgentContext]) -> ModelSettings:
            return patch_provider_session_settings(
                ctx.deps.model_cfg,
                ctx.model_settings,
                ctx.deps.get_model_extra_headers(),
            )

        return settings
