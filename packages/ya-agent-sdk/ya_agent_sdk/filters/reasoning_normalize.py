"""Reasoning content normalization filter for DeepSeek-style providers.

Different LLM providers handle assistant ``reasoning_content`` differently. DeepSeek
V4 thinking mode requires assistant messages that performed tool calls to pass
``reasoning_content`` back in subsequent requests. Plain assistant messages can omit
reasoning content because DeepSeek ignores it for regular multi-turn context.

This filter consults the active model's capabilities and rewrites history in place:

- ``reasoning_required``:
  - Tool-call assistant messages: ensure a ``ThinkingPart`` with
    ``id='reasoning_content'`` exists. A missing reasoning value gets an empty
    placeholder so pydantic-ai can serialize it into the OpenAI-compatible
    ``reasoning_content`` field.
  - Plain assistant messages: drop ``ThinkingPart`` entries to keep the payload
    lean and compatible with stricter reasoning providers.
- ``reasoning_foreign_incompatible``: drop ``ThinkingPart`` entries whose
  ``provider_name`` differs from the active provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart, ToolCallPart

from ya_agent_sdk._logger import logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.context.agent import ModelCapability, ModelConfig

_SYNTHETIC_REASONING_CONTENT = ""
"""Placeholder text for synthesized ThinkingPart.

DeepSeek V4's validator requires the ``reasoning_content`` field to be present for
assistant messages with tool calls. Empty string preserves the field without
fabricating reasoning content.
"""

_REASONING_FIELD_ID = "reasoning_content"
"""``ThinkingPart.id`` value matching the DeepSeek OpenAI-compatible field name."""


@dataclass
class _ReasoningNormalizeStats:
    synthesized: int = 0
    dropped_foreign: int = 0
    dropped_plain: int = 0

    @property
    def changed(self) -> bool:
        """Return whether normalization changed any message."""
        return bool(self.synthesized or self.dropped_foreign or self.dropped_plain)


def normalize_reasoning_for_model(
    ctx: RunContext[AgentContext],
    history: list[ModelMessage],
) -> list[ModelMessage]:
    """Reshape ModelResponse history to satisfy the active provider's reasoning policy.

    Args:
        ctx: pydantic-ai run context. ``ctx.deps.model_cfg.capabilities`` drives
            the rewrite policy, and ``ctx.model`` provides the active provider tag.
        history: Message history to reshape. Messages are mutated in place.

    Returns:
        The same message history, possibly mutated.
    """
    if not history:
        return history

    policy = _reasoning_policy(ctx.deps.model_cfg)
    if policy is None:
        return history

    requires, strict_provider = policy
    if not requires and not strict_provider:
        return history

    current_provider = _provider_tag(ctx)
    stats = _ReasoningNormalizeStats()

    for msg in history:
        if isinstance(msg, ModelResponse):
            _normalize_response(msg, current_provider, requires, strict_provider, stats)

    if stats.changed:
        logger.debug(
            "Reasoning normalize: synthesized=%d dropped_foreign=%d dropped_plain=%d provider=%s",
            stats.synthesized,
            stats.dropped_foreign,
            stats.dropped_plain,
            current_provider,
        )
    return history


def _reasoning_policy(model_cfg: object) -> tuple[bool, bool] | None:
    """Return reasoning-required and strict-provider flags for a model config."""
    if not isinstance(model_cfg, ModelConfig):
        return None
    caps = model_cfg.capabilities
    return (
        ModelCapability.reasoning_required in caps,
        ModelCapability.reasoning_foreign_incompatible in caps,
    )


def _normalize_response(
    msg: ModelResponse,
    current_provider: str,
    requires: bool,
    strict_provider: bool,
    stats: _ReasoningNormalizeStats,
) -> None:
    """Normalize a single assistant response in place."""
    if strict_provider:
        stats.dropped_foreign += _drop_foreign_thinking(msg, current_provider)

    if requires:
        _normalize_required_reasoning(msg, current_provider, stats)


def _drop_foreign_thinking(msg: ModelResponse, current_provider: str) -> int:
    """Drop ThinkingPart entries whose provider differs from the active provider."""
    original_count = len(msg.parts)
    msg.parts = [
        part
        for part in msg.parts
        if not isinstance(part, ThinkingPart) or (part.provider_name or "") == current_provider
    ]
    return original_count - len(msg.parts)


def _normalize_required_reasoning(
    msg: ModelResponse,
    current_provider: str,
    stats: _ReasoningNormalizeStats,
) -> None:
    """Apply DeepSeek V4-style required reasoning rules to a response."""
    if _has_tool_call(msg):
        if not _has_thinking(msg):
            msg.parts = [_synthesize_thinking(current_provider), *msg.parts]
            stats.synthesized += 1
        return

    stats.dropped_plain += _drop_all_thinking(msg)


def _has_tool_call(msg: ModelResponse) -> bool:
    """Return whether a response includes a tool call."""
    return any(isinstance(part, ToolCallPart) for part in msg.parts)


def _has_thinking(msg: ModelResponse) -> bool:
    """Return whether a response includes reasoning content."""
    return any(isinstance(part, ThinkingPart) for part in msg.parts)


def _drop_all_thinking(msg: ModelResponse) -> int:
    """Drop all ThinkingPart entries from a response."""
    original_count = len(msg.parts)
    msg.parts = [part for part in msg.parts if not isinstance(part, ThinkingPart)]
    return original_count - len(msg.parts)


def _synthesize_thinking(provider: str) -> ThinkingPart:
    """Create an empty placeholder ThinkingPart for the reasoning_content field."""
    return ThinkingPart(
        content=_SYNTHETIC_REASONING_CONTENT,
        id=_REASONING_FIELD_ID,
        provider_name=provider,
    )


def _provider_tag(ctx: RunContext[AgentContext]) -> str:
    """Return the active provider tag for ``ThinkingPart.provider_name``.

    Pydantic AI models expose ``system`` as the provider identifier, and provider
    filtering for thinking parts compares against that value.
    """
    return ctx.model.system
