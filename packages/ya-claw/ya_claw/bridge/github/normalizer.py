from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from ya_claw.bridge.github.models import (
    GitHubNotification,
    GitHubNotificationResource,
    GitHubNotificationSource,
    GitHubUser,
)
from ya_claw.bridge.models import BridgeAdapterType, BridgeInboundMessage

_ATTRIBUTABLE_SUBJECT_REASONS = frozenset({"mention", "team_mention"})
_MAX_SOURCE_NOTIFICATION_DELAY = timedelta(minutes=1)


def resolve_notification_resource(notification: GitHubNotification) -> GitHubNotificationResource | None:
    subject_type = notification.subject.type.casefold()
    if subject_type == "issue":
        kind = "issue"
        html_segment = "issues"
    elif subject_type in {"pullrequest", "pull_request"}:
        kind = "pull"
        html_segment = "pull"
    else:
        return None

    subject_url = notification.subject.url
    if not isinstance(subject_url, str):
        return None
    number = _resource_number(subject_url)
    if number is None:
        return None
    repository = notification.repository
    return GitHubNotificationResource(
        kind=kind,
        number=number,
        chat_id=f"github:{repository.id}:{kind}:{number}",
        html_url=f"{repository.html_url.rstrip('/')}/{html_segment}/{number}",
    )


def source_sender_is_attributable(
    notification: GitHubNotification,
    payload: dict[str, Any] | None,
    *,
    source_is_comment: bool,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if not source_is_comment and notification.reason.casefold() not in _ATTRIBUTABLE_SUBJECT_REASONS:
        return False
    created_at = _payload_timestamp(payload.get("created_at"))
    updated_at = _payload_timestamp(payload.get("updated_at"))
    if created_at is None or updated_at is None:
        return False
    if not source_is_comment and created_at != updated_at:
        return False
    source_at = updated_at if source_is_comment else created_at
    notification_delay = _as_utc(notification.updated_at) - source_at
    return timedelta(0) <= notification_delay <= _MAX_SOURCE_NOTIFICATION_DELAY


def resolve_notification_source(
    payload: dict[str, Any] | None,
    *,
    sender_is_attributable: bool,
) -> GitHubNotificationSource:
    sender = _source_user(payload) if sender_is_attributable else None
    return GitHubNotificationSource(
        payload=payload,
        sender=sender,
        content_text=_source_content(payload),
    )


def build_inbound_message(
    notification: GitHubNotification,
    *,
    tenant_key: str,
    resource: GitHubNotificationResource,
    source: GitHubNotificationSource,
) -> BridgeInboundMessage:
    event_id = notification_event_id(notification)
    sender_login = source.sender.login if source.sender is not None else None
    source_payload = source.payload
    notification_payload = notification.model_dump(mode="json")
    content_json: dict[str, Any] = {
        "notification": notification_payload,
        "resource": {
            "kind": resource.kind,
            "number": resource.number,
            "html_url": resource.html_url,
        },
        "sender_login": sender_login,
        "source": source_payload,
    }
    raw_event = dict(notification_payload)
    if source_payload is not None:
        raw_event["resolved_source"] = source_payload

    return BridgeInboundMessage(
        adapter=BridgeAdapterType.GITHUB,
        tenant_key=tenant_key,
        event_id=event_id,
        message_id=event_id,
        root_id=notification.subject.url,
        parent_id=notification.subject.latest_comment_url,
        thread_id=notification.id,
        chat_id=resource.chat_id,
        event_type="github.notification",
        sender_id=sender_login,
        sender_type=source.sender.type if source.sender is not None else None,
        chat_type=resource.kind,
        message_type="notification",
        content_text=_notification_content(notification, resource, source),
        content_json=content_json,
        create_time=_github_timestamp(notification.updated_at),
        raw_event=raw_event,
        metadata={
            "github": {
                "notification_id": notification.id,
                "reason": notification.reason,
                "repository_id": notification.repository.id,
                "repository": notification.repository.full_name,
                "resource_kind": resource.kind,
                "resource_number": resource.number,
                "resource_url": resource.html_url,
                "subject_title": notification.subject.title,
                "subject_url": notification.subject.url,
                "latest_comment_url": notification.subject.latest_comment_url,
                "sender_login": sender_login,
            }
        },
    )


def notification_event_id(notification: GitHubNotification) -> str:
    return f"github:{notification.id}:{_github_timestamp(notification.updated_at)}"


def _notification_content(
    notification: GitHubNotification,
    resource: GitHubNotificationResource,
    source: GitHubNotificationSource,
) -> str:
    sender = source.sender.login if source.sender is not None else "unknown"
    lines = [
        "GitHub notification update.",
        f"Repository: {notification.repository.full_name}",
        f"Resource: {resource.kind} #{resource.number}",
        f"Title: {notification.subject.title}",
        f"Reason: {notification.reason}",
        f"Sender: {sender}",
        f"Updated at: {_github_timestamp(notification.updated_at)}",
        f"URL: {resource.html_url}",
    ]
    if isinstance(source.content_text, str) and source.content_text.strip() != "":
        lines.extend(("Latest source content:", source.content_text.strip()))
    return "\n".join(lines)


def _source_user(payload: dict[str, Any] | None) -> GitHubUser | None:
    if not isinstance(payload, dict):
        return None
    for key in ("user", "actor", "sender", "author"):
        candidate = payload.get(key)
        if not isinstance(candidate, dict):
            continue
        try:
            return GitHubUser.model_validate(candidate)
        except ValueError:
            continue
    return None


def _source_content(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    body = payload.get("body")
    title = payload.get("title")
    if isinstance(title, str) and title.strip() and isinstance(body, str) and body.strip():
        return f"{title.strip()}\n\n{body.strip()}"
    if isinstance(body, str) and body.strip():
        return body.strip()
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _resource_number(url: str) -> int | None:
    path = urlparse(url).path.rstrip("/")
    raw_number = path.rsplit("/", 1)[-1]
    try:
        number = int(raw_number)
    except ValueError:
        return None
    return number if number > 0 else None


def _payload_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _github_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")
