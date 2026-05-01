"""Public trim-mode history helpers for compact and memory workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic_ai import UserContent
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)

from ya_agent_sdk.context.agent import ENVIRONMENT_CONTEXT_TAG, RUNTIME_CONTEXT_TAG

KEEP_TAG_KEYS = ("keep", "ya_keep")


@dataclass(frozen=True)
class TrimHistoryOptions:
    """Options for trim-mode history preparation."""

    preserve_last_turn: bool = False
    injected_context_tags: tuple[str, ...] = (RUNTIME_CONTEXT_TAG, ENVIRONMENT_CONTEXT_TAG)
    max_tool_return_chars: int = 500
    tool_return_keep_head: int = 200
    tool_return_keep_tail: int = 200
    strip_media: bool = True
    strip_injected_context: bool = True
    preserve_keep_tagged_messages: bool = True


@dataclass(frozen=True)
class TrimHistoryResult:
    """Result of trim-mode history preparation."""

    messages: list[ModelMessage]
    original_message_count: int
    trimmed_message_count: int
    removed_part_count: int = 0
    truncated_tool_return_count: int = 0
    stripped_media_count: int = 0
    stripped_injected_context_count: int = 0


def _truncate_str(content: str, options: TrimHistoryOptions) -> str:
    if len(content) <= options.max_tool_return_chars:
        return content
    head = content[: options.tool_return_keep_head]
    tail = content[-options.tool_return_keep_tail :]
    truncated_count = len(content) - options.tool_return_keep_head - options.tool_return_keep_tail
    return f"{head}\n[... {truncated_count} chars truncated ...]\n{tail}"


def _is_media_content(item: object) -> bool:
    if isinstance(item, (ImageUrl, VideoUrl)):
        return True
    return isinstance(item, BinaryContent) and (
        item.media_type.startswith("image/") or item.media_type.startswith("video/")
    )


def _media_to_placeholder(item: object) -> str:
    if isinstance(item, ImageUrl):
        return f"[image: {item.url}]"
    if isinstance(item, VideoUrl):
        return f"[video: {item.url}]"
    if isinstance(item, BinaryContent):
        return f"[{item.media_type} binary content removed]"
    return "[media content removed]"


def _has_keep_tag(message: ModelMessage) -> bool:
    metadata = getattr(message, "metadata", None)
    return isinstance(metadata, dict) and any(isinstance(metadata.get(key), str) for key in KEEP_TAG_KEYS)


def _build_tag_regex(tags: tuple[str, ...]) -> re.Pattern[str] | None:
    if not tags:
        return None
    alternatives = "|".join(re.escape(tag) for tag in tags)
    return re.compile(rf"<({alternatives})(?:\s[^>]*)?>.*?</\1>", re.DOTALL)


def _build_tag_prefixes(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"<{tag}" for tag in tags)


def _find_last_user_turn_index(message_history: list[ModelMessage]) -> int | None:
    for i in range(len(message_history) - 1, -1, -1):
        msg = message_history[i]
        if isinstance(msg, ModelRequest) and not any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in msg.parts
        ):
            return i
    return None


def _trim_tool_return(part: ToolReturnPart, options: TrimHistoryOptions) -> tuple[ToolReturnPart, bool]:
    content_str = part.model_response_str()
    if len(content_str) <= options.max_tool_return_chars:
        return part, False
    return replace(part, content=_truncate_str(content_str, options)), True


def _strip_media(part: UserPromptPart) -> tuple[UserPromptPart, bool]:
    content = part.content
    if _is_media_content(content):
        return replace(part, content=_media_to_placeholder(content)), True
    if not isinstance(content, Sequence) or isinstance(content, str):
        return part, False
    has_media = any(_is_media_content(item) for item in content)
    if not has_media:
        return part, False
    replaced: list[UserContent] = [_media_to_placeholder(item) if _is_media_content(item) else item for item in content]
    return replace(part, content=replaced), True


def _strip_injected_context_text(content: str, tags: tuple[str, ...]) -> tuple[str | None, bool]:
    tag_re = _build_tag_regex(tags)
    cleaned = tag_re.sub("", content) if tag_re else content
    cleaned = cleaned.strip()
    if not cleaned:
        return None, cleaned != content
    if cleaned != content:
        return cleaned, True
    return content, False


def _strip_injected_context(
    part: UserPromptPart,
    tags: tuple[str, ...],
) -> tuple[UserPromptPart | None, bool]:
    content = part.content
    if isinstance(content, str):
        cleaned, changed = _strip_injected_context_text(content, tags)
        if cleaned is None:
            return None, changed
        if changed:
            return replace(part, content=cleaned), True
        return part, False

    if isinstance(content, Sequence):
        prefixes = _build_tag_prefixes(tags)
        filtered = [
            item
            for item in content
            if not (isinstance(item, str) and any(item.lstrip().startswith(prefix) for prefix in prefixes))
        ]
        if not filtered:
            return None, len(filtered) != len(content)
        if len(filtered) != len(content):
            return replace(part, content=filtered), True
    return part, False


def trim_history_for_summary(  # noqa: C901
    message_history: Sequence[ModelMessage],
    options: TrimHistoryOptions | None = None,
) -> TrimHistoryResult:
    """Prepare message history for summary or memory extraction.

    The returned history is optimized for secondary summarization tasks. It can
    strip per-turn injected context, replace media with text placeholders, and
    truncate large tool returns.
    """

    effective_options = options or TrimHistoryOptions()
    messages = list(message_history)
    last_user_turn_idx = _find_last_user_turn_index(messages) if effective_options.preserve_last_turn else None

    trimmed: list[ModelMessage] = []
    removed_part_count = 0
    truncated_tool_return_count = 0
    stripped_media_count = 0
    stripped_injected_context_count = 0

    for index, message in enumerate(messages):
        if effective_options.preserve_keep_tagged_messages and _has_keep_tag(message):
            trimmed.append(message)
            continue

        if isinstance(message, ModelResponse):
            trimmed.append(message)
            continue

        if not isinstance(message, ModelRequest):
            trimmed.append(message)
            continue

        is_in_last_turn = last_user_turn_idx is not None and index >= last_user_turn_idx
        new_parts = []
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                new_part, changed = _trim_tool_return(part, effective_options)
                if changed:
                    truncated_tool_return_count += 1
                new_parts.append(new_part)
                continue

            if isinstance(part, UserPromptPart):
                new_part = part
                if effective_options.strip_media:
                    new_part, changed = _strip_media(new_part)
                    if changed:
                        stripped_media_count += 1
                if effective_options.strip_injected_context and not is_in_last_turn:
                    stripped_part, changed = _strip_injected_context(
                        new_part,
                        effective_options.injected_context_tags,
                    )
                    if changed:
                        stripped_injected_context_count += 1
                    if stripped_part is None:
                        removed_part_count += 1
                        continue
                    new_part = stripped_part
                new_parts.append(new_part)
                continue

            new_parts.append(part)

        if not new_parts:
            continue
        trimmed.append(replace(message, parts=new_parts))

    return TrimHistoryResult(
        messages=trimmed,
        original_message_count=len(messages),
        trimmed_message_count=len(trimmed),
        removed_part_count=removed_part_count,
        truncated_tool_return_count=truncated_tool_return_count,
        stripped_media_count=stripped_media_count,
        stripped_injected_context_count=stripped_injected_context_count,
    )
