from __future__ import annotations

from ya_claw.bridge.lark.normalizer import normalize_lark_event


def test_normalize_lark_message_receive_event() -> None:
    message = normalize_lark_event({
        "header": {
            "event_id": "event-1",
            "event_type": "im.message.receive_v1",
            "tenant_key": "tenant-1",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_1"}, "sender_type": "user"},
            "message": {
                "message_id": "om_1",
                "root_id": "om_root",
                "parent_id": "om_parent",
                "thread_id": "omt_1",
                "chat_id": "oc_1",
                "chat_type": "group",
                "message_type": "text",
                "content": '{"text":"hello"}',
            },
        },
    })

    assert message is not None
    assert message.event_id == "event-1"
    assert message.message_id == "om_1"
    assert message.root_id == "om_root"
    assert message.parent_id == "om_parent"
    assert message.thread_id == "omt_1"
    assert message.chat_id == "oc_1"
    assert message.content_text == "hello"


def test_normalize_lark_chat_member_event_uses_chat_conversation() -> None:
    message = normalize_lark_event({
        "header": {
            "event_id": "event-2",
            "event_type": "im.chat.member.user.added_v1",
            "tenant_key": "tenant-1",
        },
        "event": {
            "chat_id": "oc_1",
            "operator_id": {"open_id": "ou_operator"},
            "users": [{"name": "Alice", "user_id": {"open_id": "ou_2"}}],
        },
    })

    assert message is not None
    assert message.event_id == "event-2"
    assert message.message_id == "event-2"
    assert message.chat_id == "oc_1"
    assert message.message_type == "event"
    assert message.content_json is not None
    assert message.content_json["chat_id"] == "oc_1"


def test_normalize_lark_drive_comment_event_uses_drive_conversation() -> None:
    message = normalize_lark_event({
        "header": {
            "event_id": "event-3",
            "event_type": "drive.notice.comment_add_v1",
            "tenant_key": "tenant-1",
        },
        "event": {
            "file_token": "doccn_1",
            "comment_id": "comment-1",
            "operator_id": {"open_id": "ou_1"},
            "comment": {"content": "please review"},
        },
    })

    assert message is not None
    assert message.event_id == "event-3"
    assert message.message_id == "comment-1"
    assert message.chat_id == "drive/doccn_1"
    assert message.sender_id == "ou_1"
    assert message.message_type == "event"
