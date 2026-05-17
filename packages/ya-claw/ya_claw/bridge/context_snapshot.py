from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BridgeSnapshotSpeaker = Literal["self", "external_user", "unknown"]
BridgeSnapshotRelation = Literal["parent", "thread", "chat_recent"]
BridgeSnapshotSource = Literal["lark"]


class BridgePreviousMessageSnapshotItem(BaseModel):
    source: BridgeSnapshotSource = "lark"
    speaker: BridgeSnapshotSpeaker = "unknown"
    relation: BridgeSnapshotRelation
    message_id: str | None = None
    sender_id: str | None = None
    sender_type: str | None = None
    message_type: str | None = None
    create_time: str | None = None
    content_text: str
    truncated: bool = False


class BridgePreviousMessagesSnapshot(BaseModel):
    source: Literal["lark"] = "lark"
    items: list[BridgePreviousMessageSnapshotItem] = Field(default_factory=list)
    truncated: bool = False
    self_identity_label: str = "this Lark bridge bot/app"
