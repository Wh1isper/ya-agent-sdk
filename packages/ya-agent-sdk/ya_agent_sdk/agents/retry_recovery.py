"""Error-aware retry recovery for streamed agent runs.

This module contains durable message-history healing steps used by
``stream_agent(..., resume_on_error=True)`` before the next retry attempt.
The recovery runs after tool-call closure and before the resume prompt is
added, so the next model request and any persisted recoverable history share
the repaired messages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.filters.cold_start import _trim_tool_returns

logger = get_logger(__name__)

_MEDIA_REMOVED_REMINDER = (
    "<system-reminder>Media content was removed during retry recovery because the previous "
    "request exceeded the model context limit. If the media is still needed, ask the user "
    "to attach it again or inspect it with a focused tool call.</system-reminder>"
)

_RESPONSE_MEDIA_REMOVED_TEXT = (
    "<system-reminder>Assistant media content was removed during retry recovery because the "
    "previous request exceeded the model context limit.</system-reminder>"
)

_CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded",
    "maximum context length",
    "max context length",
    "context window",
    "context limit",
    "context too long",
    "prompt is too long",
    "prompt too long",
    "too many tokens",
    "token count exceeds maximum",
    "exceeds maximum token",
    "exceed the maximum number of tokens",
    "input is too long",
    "input too long",
    "reduce the length of the messages",
    "reduce the size of your message",
    "messages resulted in",
    "requested tokens",
)

_OPENAI_REFERENCE_PATTERNS = (
    "invalid_encrypted_content",
    "encrypted_content",
    "item_not_found",
    "item not found",
    "no item with id",
    "could not find item",
    "was provided without its required following item",
    "required following item",
    "previous_response_id",
    "previous response",
)

_OPENAI_ITEM_REFERENCE_RE = re.compile(r"item ['\"]?[^'\"\s]+['\"]?.{0,80}(not found|required following item)")


@dataclass(frozen=True)
class RetryRecoveryResult:
    """Result of retry message recovery."""

    history: list[ModelMessage]
    changed: bool = False
    reasons: tuple[str, ...] = ()


def recover_retry_message_history(
    exc: BaseException,
    history: Sequence[ModelMessage],
    ctx: AgentContext,
) -> RetryRecoveryResult:
    """Apply built-in retry recovery policies to message history.

    Args:
        exc: Exception raised by the previous attempt.
        history: Recovered message history from the previous attempt.
        ctx: Active agent context.

    Returns:
        Recovered history plus change metadata.
    """
    recovered = list(history)
    if not recovered:
        return RetryRecoveryResult(history=recovered)

    error_text = _exception_text(exc)
    reasons: list[str] = []
    changed = False

    if _is_openai_item_reference_error(error_text):
        recovered, item_changed = heal_openai_item_reference_history(recovered)
        if item_changed:
            changed = True
            reasons.append("openai_item_reference")

    if _is_context_overflow_error(error_text):
        recovered, overflow_changed = heal_context_overflow_history(recovered, ctx)
        if overflow_changed:
            changed = True
            reasons.append("context_overflow")

    if changed:
        logger.info("Retry recovery changed message history reasons=%s", ",".join(reasons))

    return RetryRecoveryResult(history=recovered, changed=changed, reasons=tuple(reasons))


def heal_openai_item_reference_history(history: Sequence[ModelMessage]) -> tuple[list[ModelMessage], bool]:
    """Remove OpenAI Responses server-side item references from history.

    OpenAI Responses can reject replayed reasoning, message, function-call, or
    previous-response references after provider-side state changes, API gateway
    routing changes, or stale encrypted reasoning content. This recovery keeps
    the semantic text/tool-call history while dropping server-bound IDs and
    encrypted reasoning payloads so the next attempt sends a fresh stateless
    replay.
    """
    recovered: list[ModelMessage] = []
    changed = False
    tool_call_id_map: dict[str, str] = {}

    for message in history:
        if isinstance(message, ModelResponse):
            new_message, message_changed, message_id_map = _heal_openai_response(message)
            recovered.append(new_message)
            tool_call_id_map.update(message_id_map)
            changed = changed or message_changed
            continue

        if isinstance(message, ModelRequest):
            new_message, message_changed = _heal_openai_request_tool_call_ids(message, tool_call_id_map)
            recovered.append(new_message)
            changed = changed or message_changed
            continue

        recovered.append(message)

    return recovered, changed


def heal_context_overflow_history(
    history: Sequence[ModelMessage], ctx: AgentContext
) -> tuple[list[ModelMessage], bool]:
    """Trim large tool returns and remove image/video media for overflow retries."""
    recovered = list(history)
    changed = False

    trimmed_count = _trim_tool_returns(recovered)
    if trimmed_count:
        changed = True
        logger.info("Retry recovery trimmed %d tool return parts after context overflow", trimmed_count)

    media_recovered, media_changed = _strip_image_video_media(recovered)
    if media_changed:
        changed = True
        logger.info("Retry recovery removed image/video media after context overflow")

    return media_recovered, changed


def _heal_openai_response(message: ModelResponse) -> tuple[ModelResponse, bool, dict[str, str]]:
    new_parts: list[ModelResponsePart] = []
    changed = False
    tool_call_id_map: dict[str, str] = {}

    for part in message.parts:
        new_part, part_changed = _heal_openai_response_part(part)
        new_parts.append(new_part)
        changed = changed or part_changed
        if (
            isinstance(part, ToolCallPart | NativeToolCallPart)
            and isinstance(new_part, ToolCallPart | NativeToolCallPart)
            and part.tool_call_id != new_part.tool_call_id
        ):
            tool_call_id_map[part.tool_call_id] = new_part.tool_call_id

    provider_details, details_changed = _drop_keys(
        message.provider_details,
        {
            "conversation_id",
            "encrypted_content",
            "previous_response_id",
            "response_id",
        },
    )
    changed = changed or details_changed

    if message.provider_response_id is not None or message.conversation_id is not None or changed:
        return (
            replace(
                message,
                parts=new_parts,
                provider_response_id=None,
                conversation_id=None,
                provider_details=provider_details,
            ),
            True,
            tool_call_id_map,
        )

    return message, False, tool_call_id_map


def _heal_openai_request_tool_call_ids(
    message: ModelRequest,
    tool_call_id_map: dict[str, str],
) -> tuple[ModelRequest, bool]:
    if not tool_call_id_map:
        return message, False

    new_parts: list[ModelRequestPart] = []
    changed = False
    for part in message.parts:
        if isinstance(part, ToolReturnPart | NativeToolReturnPart | RetryPromptPart):
            new_tool_call_id = tool_call_id_map.get(part.tool_call_id)
            if new_tool_call_id is not None and new_tool_call_id != part.tool_call_id:
                new_parts.append(replace(part, tool_call_id=new_tool_call_id))
                changed = True
                continue
        new_parts.append(part)

    if changed:
        return replace(message, parts=new_parts), True
    return message, False


def _heal_openai_response_part(part: ModelResponsePart) -> tuple[ModelResponsePart, bool]:
    if isinstance(part, ThinkingPart):
        return _heal_openai_thinking_part(part)

    if isinstance(part, TextPart | CompactionPart):
        return _heal_part_id_and_details(part)

    if isinstance(part, ToolCallPart | NativeToolCallPart):
        return _heal_tool_call_part(part)

    if isinstance(part, FilePart):
        return _heal_part_id_and_details(part)

    return _heal_generic_provider_details(part)


def _heal_openai_thinking_part(part: ThinkingPart) -> tuple[ThinkingPart, bool]:
    provider_details, details_changed = _drop_keys(part.provider_details, {"raw_content", "encrypted_content"})
    if part.id is not None or part.signature is not None or details_changed:
        return replace(part, id=None, signature=None, provider_details=provider_details), True
    return part, False


def _heal_tool_call_part(
    part: ToolCallPart | NativeToolCallPart,
) -> tuple[ToolCallPart | NativeToolCallPart, bool]:
    provider_details, details_changed = _drop_keys(part.provider_details, {"encrypted_content"})
    call_id = part.tool_call_id
    new_call_id = call_id.split("|", 1)[0] if "|" in call_id else call_id
    if part.id is not None or details_changed or new_call_id != call_id:
        return replace(part, id=None, provider_details=provider_details, tool_call_id=new_call_id), True
    return part, False


def _heal_part_id_and_details(
    part: TextPart | CompactionPart | ToolCallPart | NativeToolCallPart | FilePart,
) -> tuple[TextPart | CompactionPart | ToolCallPart | NativeToolCallPart | FilePart, bool]:
    provider_details, details_changed = _drop_keys(part.provider_details, {"encrypted_content"})
    if part.id is not None or details_changed:
        return replace(part, id=None, provider_details=provider_details), True
    return part, False


def _heal_generic_provider_details(part: ModelResponsePart) -> tuple[ModelResponsePart, bool]:
    provider_details = getattr(part, "provider_details", None)
    if not isinstance(provider_details, dict) or "encrypted_content" not in provider_details:
        return part, False
    new_provider_details, _ = _drop_keys(provider_details, {"encrypted_content"})
    try:
        return replace(part, provider_details=new_provider_details), True
    except TypeError:
        return part, False


def _drop_keys(details: dict[str, Any] | None, keys: set[str]) -> tuple[dict[str, Any] | None, bool]:
    if not details:
        return details, False
    new_details = {key: value for key, value in details.items() if key not in keys}
    if len(new_details) == len(details):
        return details, False
    return new_details or None, True


def _strip_image_video_media(history: Sequence[ModelMessage]) -> tuple[list[ModelMessage], bool]:
    recovered: list[ModelMessage] = []
    changed = False

    for message in history:
        if isinstance(message, ModelRequest):
            new_message, message_changed = _strip_media_from_request(message)
            recovered.append(new_message)
            changed = changed or message_changed
            continue

        if isinstance(message, ModelResponse):
            new_message, message_changed = _strip_media_from_response(message)
            recovered.append(new_message)
            changed = changed or message_changed
            continue

        recovered.append(message)

    return recovered, changed


def _strip_media_from_request(message: ModelRequest) -> tuple[ModelRequest, bool]:
    new_parts: list[ModelRequestPart] = []
    changed = False

    for part in message.parts:
        if isinstance(part, UserPromptPart):
            if isinstance(part.content, str):
                new_parts.append(part)
                continue
            new_content, content_changed = _replace_media_content(part.content)
            new_parts.append(replace(part, content=_coerce_user_prompt_content(new_content)))
            changed = changed or content_changed
            continue

        if isinstance(part, ToolReturnPart | NativeToolReturnPart):
            new_content, content_changed = _replace_media_content(part.content)
            new_parts.append(replace(part, content=new_content))
            changed = changed or content_changed
            continue

        new_parts.append(part)

    if changed:
        return replace(message, parts=new_parts), True
    return message, False


def _strip_media_from_response(message: ModelResponse) -> tuple[ModelResponse, bool]:
    new_parts: list[ModelResponsePart] = []
    changed = False

    for part in message.parts:
        if isinstance(part, FilePart) and _is_image_or_video_content(part.content):
            new_parts.append(TextPart(content=_RESPONSE_MEDIA_REMOVED_TEXT))
            changed = True
            continue
        new_parts.append(part)

    if not changed:
        return message, False

    return replace(message, parts=new_parts), True


def _replace_media_content(content: object) -> tuple[object, bool]:
    if _is_image_or_video_content(content):
        return _MEDIA_REMOVED_REMINDER, True

    if isinstance(content, list):
        changed = False
        new_list_items: list[object] = []
        for item in content:
            new_item, item_changed = _replace_media_content(item)
            new_list_items.append(new_item)
            changed = changed or item_changed
        return new_list_items, changed

    if isinstance(content, tuple):
        changed = False
        new_tuple_items: list[object] = []
        for item in content:
            new_item, item_changed = _replace_media_content(item)
            new_tuple_items.append(new_item)
            changed = changed or item_changed
        return tuple(new_tuple_items), changed

    if isinstance(content, dict):
        changed = False
        new_dict_items: dict[object, object] = {}
        for key, value in content.items():
            new_value, value_changed = _replace_media_content(value)
            new_dict_items[key] = new_value
            changed = changed or value_changed
        return new_dict_items, changed

    return content, False


def _coerce_user_prompt_content(content: object) -> str | Sequence[UserContent]:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        return list(content)  # type: ignore[list-item]
    return [content]  # type: ignore[list-item]


def _is_image_or_video_content(content: object) -> bool:
    if isinstance(content, ImageUrl | VideoUrl):
        return True
    if isinstance(content, UploadedFile) and isinstance(content.media_type, str):
        return content.media_type.startswith(("image/", "video/"))
    return isinstance(content, BinaryContent) and content.media_type.startswith(("image/", "video/"))


def _is_openai_item_reference_error(error_text: str) -> bool:
    lowered = error_text.lower()
    if any(pattern in lowered for pattern in _OPENAI_REFERENCE_PATTERNS):
        return True
    return bool(_OPENAI_ITEM_REFERENCE_RE.search(lowered))


def _is_context_overflow_error(error_text: str) -> bool:
    lowered = error_text.lower()
    if not any(pattern in lowered for pattern in _CONTEXT_OVERFLOW_PATTERNS):
        return False
    token_or_context = (
        "token" in lowered or "context" in lowered or "prompt" in lowered or "message" in lowered or "input" in lowered
    )
    return token_or_context


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    _collect_exception_text(exc, parts, seen)
    return "\n".join(parts)


def _collect_exception_text(exc: BaseException, parts: list[str], seen: set[int]) -> None:
    if id(exc) in seen:
        return
    seen.add(id(exc))

    parts.append(type(exc).__name__)
    try:
        parts.append(str(exc))
    except Exception:
        parts.append(repr(exc))

    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(_safe_json_text(body))

    args = getattr(exc, "args", None)
    if args:
        parts.append(_safe_json_text(args))

    if exc.__cause__ is not None:
        _collect_exception_text(exc.__cause__, parts, seen)
    if exc.__context__ is not None:
        _collect_exception_text(exc.__context__, parts, seen)


def _safe_json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)
