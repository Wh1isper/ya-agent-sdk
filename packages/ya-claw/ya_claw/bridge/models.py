from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ya_claw.controller.models import ActiveInteraction


class BridgeAdapterType(StrEnum):
    LARK = "lark"


class BridgeDispatchMode(StrEnum):
    EMBEDDED = "embedded"
    MANUAL = "manual"


class BridgeEventStatus(StrEnum):
    RECEIVED = "received"
    QUEUED = "queued"
    SUBMITTED = "submitted"
    STEERED = "steered"
    DEFERRED = "deferred"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class BridgeInboundAction(BaseModel):
    adapter: BridgeAdapterType
    tenant_key: str = "default"
    event_id: str
    action_id: str = "hitl_respond"
    action_type: str = "hitl_respond"
    token: str | None = None
    approved: bool = False
    reason: str | None = None
    raw_event: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BridgeInboundMessage(BaseModel):
    adapter: BridgeAdapterType
    tenant_key: str = "default"
    event_id: str
    message_id: str
    root_id: str | None = None
    parent_id: str | None = None
    thread_id: str | None = None
    chat_id: str
    event_type: str = "im.message.receive_v1"
    sender_id: str | None = None
    sender_type: str | None = None
    chat_type: str | None = None
    message_type: str = "text"
    content_text: str | None = None
    content_json: dict[str, Any] | None = None
    create_time: str | None = None
    raw_event: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BridgeDispatchResult(BaseModel):
    status: BridgeEventStatus
    adapter: BridgeAdapterType
    event_id: str
    message_id: str | None = None
    chat_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    duplicate: bool = False
    queued_count: int | None = None
    remaining_interaction_count: int | None = None
    current_interaction: ActiveInteraction | None = None
    error_message: str | None = None


class BridgeConversationSummary(BaseModel):
    id: str
    adapter: BridgeAdapterType
    tenant_key: str
    external_chat_id: str
    session_id: str
    profile_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    active_run_id: str | None = None
    event_count: int = 0
    latest_event_status: BridgeEventStatus | None = None
    created_at: datetime
    updated_at: datetime
    last_event_at: datetime | None = None


class BridgeConversationListResponse(BaseModel):
    conversations: list[BridgeConversationSummary] = Field(default_factory=list)


class BridgeEventSummary(BaseModel):
    id: str
    adapter: BridgeAdapterType
    tenant_key: str
    event_id: str
    external_message_id: str | None = None
    external_chat_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    event_type: str
    status: BridgeEventStatus
    error_message: str | None = None
    raw_event: dict[str, Any] = Field(default_factory=dict)
    normalized_event: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class BridgeEventListResponse(BaseModel):
    events: list[BridgeEventSummary] = Field(default_factory=list)
