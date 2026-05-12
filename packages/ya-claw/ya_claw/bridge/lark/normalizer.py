from __future__ import annotations

import json
from typing import Any

from ya_claw.bridge.models import BridgeAdapterType, BridgeInboundAction, BridgeInboundMessage

_MESSAGE_RECEIVE_EVENT = "im.message.receive_v1"
_DRIVE_EVENT_PREFIX = "drive."


def normalize_lark_action(raw_event: dict[str, Any]) -> BridgeInboundAction | None:
    header = _dict_value(raw_event.get("header"))
    event = _dict_value(raw_event.get("event")) or raw_event
    action = _dict_value(event.get("action"))
    value = _dict_value(action.get("value")) or _dict_value(event.get("value"))
    raw_action = _string_value(value.get("action") or action.get("action") or event.get("action"))
    event_id = _string_value(header.get("event_id") or raw_event.get("event_id") or raw_event.get("uuid"))
    tenant_key = _string_value(header.get("tenant_key") or raw_event.get("tenant_key")) or "default"
    action_id = _string_value(action.get("tag") or action.get("name")) or raw_action or "bridge_action"

    interaction_token = _string_value(value.get("interaction_token") or event.get("interaction_token"))
    if interaction_token is not None and raw_action in {"approve", "deny"}:
        return BridgeInboundAction(
            adapter=BridgeAdapterType.LARK,
            tenant_key=tenant_key,
            event_id=event_id or interaction_token,
            action_id=action_id,
            action_type="hitl_respond",
            token=interaction_token,
            approved=raw_action == "approve",
            raw_event=raw_event,
            metadata={"action": raw_action},
        )

    recovery_token = _string_value(value.get("recovery_token") or event.get("recovery_token"))
    if recovery_token is not None and raw_action in {"retry", "reset_and_retry"}:
        return BridgeInboundAction(
            adapter=BridgeAdapterType.LARK,
            tenant_key=tenant_key,
            event_id=event_id or recovery_token,
            action_id=action_id,
            action_type="session_recovery",
            token=recovery_token,
            raw_event=raw_event,
            metadata={"action": raw_action},
        )

    return None


def normalize_lark_event(raw_event: dict[str, Any]) -> BridgeInboundMessage | None:
    header = _dict_value(raw_event.get("header"))
    event = _dict_value(raw_event.get("event"))
    event_type = _string_value(header.get("event_type") or raw_event.get("type")) or _MESSAGE_RECEIVE_EVENT
    event_id = _string_value(header.get("event_id") or raw_event.get("event_id") or raw_event.get("uuid"))

    if event_type == _MESSAGE_RECEIVE_EVENT:
        return _normalize_message_receive(raw_event, header, event, event_type, event_id)
    return _normalize_generic_event(raw_event, header, event, event_type, event_id)


def normalize_lark_compact_event(raw_event: dict[str, Any]) -> BridgeInboundMessage | None:
    message_id = _string_value(raw_event.get("message_id") or raw_event.get("id"))
    chat_id = _string_value(raw_event.get("chat_id"))
    event_id = _string_value(raw_event.get("event_id") or message_id)
    if event_id is None:
        return None
    event_type = _string_value(raw_event.get("type")) or _MESSAGE_RECEIVE_EVENT
    if message_id is None:
        message_id = event_id
    if chat_id is None:
        chat_id = _fallback_conversation_key(event_type, event_id)
    content = raw_event.get("content")
    content_text = content if isinstance(content, str) else None
    return BridgeInboundMessage(
        adapter=BridgeAdapterType.LARK,
        tenant_key=_string_value(raw_event.get("tenant_key")) or "default",
        event_id=event_id,
        message_id=message_id,
        root_id=_string_value(raw_event.get("root_id")),
        parent_id=_string_value(raw_event.get("parent_id")),
        thread_id=_string_value(raw_event.get("thread_id")),
        chat_id=chat_id,
        event_type=event_type,
        sender_id=_string_value(raw_event.get("sender_id")),
        sender_type=_string_value(raw_event.get("sender_type")),
        chat_type=_string_value(raw_event.get("chat_type")),
        message_type=_string_value(raw_event.get("message_type")) or _message_type_from_event_type(event_type),
        content_text=content_text,
        content_json={"text": content_text} if isinstance(content_text, str) else None,
        create_time=_string_value(raw_event.get("create_time") or raw_event.get("timestamp")),
        raw_event=raw_event,
    )


def _normalize_message_receive(
    raw_event: dict[str, Any],
    header: dict[str, Any],
    event: dict[str, Any],
    event_type: str,
    event_id: str | None,
) -> BridgeInboundMessage | None:
    message = _dict_value(event.get("message"))
    sender = _dict_value(event.get("sender"))
    sender_id = _dict_value(sender.get("sender_id"))

    message_id = _string_value(message.get("message_id") or raw_event.get("message_id") or raw_event.get("id"))
    chat_id = _string_value(message.get("chat_id") or raw_event.get("chat_id"))
    resolved_event_id = event_id or message_id
    if resolved_event_id is None:
        return None
    if message_id is None:
        message_id = resolved_event_id
    if chat_id is None:
        chat_id = _fallback_conversation_key(event_type, resolved_event_id)

    content_json = _parse_content(message.get("content") or raw_event.get("content"))
    message_type = _string_value(message.get("message_type") or raw_event.get("message_type")) or "text"
    return BridgeInboundMessage(
        adapter=BridgeAdapterType.LARK,
        tenant_key=_string_value(header.get("tenant_key") or raw_event.get("tenant_key")) or "default",
        event_id=resolved_event_id,
        message_id=message_id,
        root_id=_string_value(message.get("root_id") or raw_event.get("root_id")),
        parent_id=_string_value(message.get("parent_id") or raw_event.get("parent_id")),
        thread_id=_string_value(message.get("thread_id") or raw_event.get("thread_id")),
        chat_id=chat_id,
        event_type=event_type,
        sender_id=_string_value(sender_id.get("open_id") or sender_id.get("user_id") or raw_event.get("sender_id")),
        sender_type=_string_value(sender.get("sender_type") or raw_event.get("sender_type")),
        chat_type=_string_value(message.get("chat_type") or raw_event.get("chat_type")),
        message_type=message_type,
        content_text=_content_text(message_type, content_json, raw_event.get("content")),
        content_json=content_json,
        create_time=_string_value(
            message.get("create_time") or header.get("create_time") or raw_event.get("create_time")
        ),
        raw_event=raw_event,
    )


def _normalize_generic_event(
    raw_event: dict[str, Any],
    header: dict[str, Any],
    event: dict[str, Any],
    event_type: str,
    event_id: str | None,
) -> BridgeInboundMessage | None:
    resolved_event_id = event_id or _string_value(_find_first_key(event, ("event_id", "id")))
    if resolved_event_id is None:
        return None

    message_id = (
        _string_value(
            _find_first_key(
                event,
                (
                    "message_id",
                    "comment_id",
                    "reply_id",
                    "file_token",
                    "obj_token",
                    "token",
                    "id",
                ),
            )
        )
        or resolved_event_id
    )
    chat_id = _resolve_conversation_key(event_type, event, raw_event, resolved_event_id)
    sender = _dict_value(event.get("sender"))
    operator = _dict_value(event.get("operator_id"))
    user_id = _dict_value(event.get("user_id"))

    return BridgeInboundMessage(
        adapter=BridgeAdapterType.LARK,
        tenant_key=_string_value(header.get("tenant_key") or raw_event.get("tenant_key")) or "default",
        event_id=resolved_event_id,
        message_id=message_id,
        root_id=_string_value(_find_first_key(event, ("root_id",))),
        parent_id=_string_value(_find_first_key(event, ("parent_id",))),
        thread_id=_string_value(_find_first_key(event, ("thread_id",))),
        chat_id=chat_id,
        event_type=event_type,
        sender_id=_string_value(
            sender.get("open_id")
            or sender.get("user_id")
            or operator.get("open_id")
            or operator.get("user_id")
            or user_id.get("open_id")
            or user_id.get("user_id")
            or _find_first_key(event, ("open_id", "user_id", "union_id"))
        ),
        sender_type=_string_value(event.get("sender_type") or raw_event.get("sender_type")),
        chat_type=_string_value(event.get("chat_type") or raw_event.get("chat_type")),
        message_type=_message_type_from_event_type(event_type),
        content_text=_generic_content_text(event_type, event),
        content_json=event,
        create_time=_string_value(
            header.get("create_time") or event.get("create_time") or raw_event.get("create_time")
        ),
        raw_event=raw_event,
        metadata={"lark_event_type": event_type},
    )


def _resolve_conversation_key(
    event_type: str,
    event: dict[str, Any],
    raw_event: dict[str, Any],
    event_id: str,
) -> str:
    chat_id = _string_value(_find_first_key(event, ("chat_id", "open_chat_id")) or raw_event.get("chat_id"))
    if chat_id is not None:
        return chat_id

    drive_token = _string_value(
        _find_first_key(event, ("file_token", "obj_token", "node_token", "document_id", "file_id", "token"))
    )
    if drive_token is not None and event_type.startswith(_DRIVE_EVENT_PREFIX):
        return f"drive/{drive_token}"

    return _fallback_conversation_key(event_type, event_id)


def _fallback_conversation_key(event_type: str, event_id: str) -> str:
    return f"event/{event_type}/{event_id}"


def _message_type_from_event_type(event_type: str) -> str:
    if event_type == _MESSAGE_RECEIVE_EVENT:
        return "text"
    return "event"


def _generic_content_text(event_type: str, event: dict[str, Any]) -> str:
    return json.dumps({"event_type": event_type, "event": event}, ensure_ascii=False)


def _find_first_key(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if _string_value(candidate) is not None:
                return candidate
        for candidate in value.values():
            found = _find_first_key(candidate, keys)
            if _string_value(found) is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_first_key(item, keys)
            if _string_value(found) is not None:
                return found
    return None


def _parse_content(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"text": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return None


def _content_text(message_type: str, content: dict[str, Any] | None, fallback: Any) -> str | None:
    if content is None:
        return fallback if isinstance(fallback, str) else None
    if message_type == "text" and isinstance(content.get("text"), str):
        return content["text"]
    if message_type == "post":
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content.get("text"), str):
        return content["text"]
    return json.dumps(content, ensure_ascii=False)


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
