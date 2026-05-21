from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from ya_claw.bridge.context_snapshot import BridgePreviousMessageSnapshotItem, BridgeSnapshotSpeaker


def sort_snapshot_items(items: list[BridgePreviousMessageSnapshotItem]) -> list[BridgePreviousMessageSnapshotItem]:
    return sorted(items, key=lambda item: int_value(item.create_time) or 0)


def speaker_for_lark_sender(
    *,
    sender_id: str | None,
    sender_type: str | None,
    app_id: str | None,
) -> BridgeSnapshotSpeaker:
    normalized_sender_type = sender_type.strip().lower() if isinstance(sender_type, str) else None
    if normalized_sender_type in {"app", "bot"}:
        return "self"
    if isinstance(sender_id, str) and isinstance(app_id, str) and sender_id.strip() == app_id.strip():
        return "self"
    if sender_id is not None or sender_type is not None:
        return "external_user"
    return "unknown"


def lark_message_content_text(lark_module: Any, *, message_type: str | None, raw_content: str | None) -> str | None:
    if raw_content is None:
        return placeholder_for_message_type(message_type)
    parsed = _parse_content(lark_module, raw_content)
    if message_type == "text":
        text = parsed.get("text") if isinstance(parsed, dict) else None
        return text if isinstance(text, str) else raw_content
    if message_type == "post":
        title = parsed.get("title") if isinstance(parsed, dict) else None
        content = parsed.get("content") if isinstance(parsed, dict) else None
        parts: list[str] = []
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())
        parts.extend(_flatten_post_content(content))
        return "\n".join(parts) if parts else raw_content
    placeholder = placeholder_for_message_type(message_type)
    if placeholder is not None:
        return placeholder
    return raw_content


def placeholder_for_message_type(message_type: str | None) -> str | None:
    if message_type in {"image", "img"}:
        return "[image message]"
    if message_type == "file":
        return "[file message]"
    if message_type in {"interactive", "card"}:
        return "[interactive card message]"
    if message_type == "audio":
        return "[audio message]"
    if message_type == "media":
        return "[media message]"
    if message_type == "sticker":
        return "[sticker message]"
    return None


def string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def int_value(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_content(lark_module: Any, raw_content: str) -> object:
    with suppress(Exception):
        parsed = lark_module.JSON.unmarshal(raw_content, dict)
        if isinstance(parsed, dict):
            return parsed
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content


def _flatten_post_content(content: object) -> list[str]:
    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            parts.extend(_flatten_post_content(item))
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
        else:
            for value in content.values():
                parts.extend(_flatten_post_content(value))
    return parts
