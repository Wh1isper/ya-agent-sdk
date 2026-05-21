from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from ya_claw.bridge.base import BridgeMessageHandler
from ya_claw.bridge.lark.adapter import LarkBridgeAdapter
from ya_claw.bridge.models import BridgeInboundMessage
from ya_claw.config import ClawSettings


class _FakeHandler(BridgeMessageHandler):
    async def handle_message(self, message: BridgeInboundMessage) -> Any:
        return None

    async def handle_action(self, action: Any) -> Any:
        return None


class _SnapshotTestAdapter(LarkBridgeAdapter):
    def __init__(self, *, client: object, settings: ClawSettings) -> None:
        super().__init__(settings=settings, handler=_FakeHandler())
        self._client_override = client
        self._app_id = settings.bridge_lark_app_id
        self._app_secret = settings.bridge_lark_app_secret_value

    def _openapi_client(self, lark_module: object) -> object:
        return self._client_override


class _FakeJson:
    @staticmethod
    def unmarshal(value: str | bytes, target: object) -> object:
        raw = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(raw)


class _FakeLarkModule:
    JSON = _FakeJson()


class _FakeResponse:
    def __init__(self, items: list[object]) -> None:
        self.data = SimpleNamespace(items=items)

    def success(self) -> bool:
        return True


class _FakeMessageApi:
    def __init__(self, *, parent: object, chat_items: list[object]) -> None:
        self.parent = parent
        self.chat_items = chat_items
        self.get_requests: list[object] = []
        self.list_requests: list[object] = []

    def get(self, request: object) -> _FakeResponse:
        self.get_requests.append(request)
        return _FakeResponse([self.parent])

    def list(self, request: object) -> _FakeResponse:
        self.list_requests.append(request)
        return _FakeResponse(self.chat_items)


class _FakeClient:
    def __init__(self, *, parent: object, chat_items: list[object]) -> None:
        self.message_api = _FakeMessageApi(parent=parent, chat_items=chat_items)
        self.im = SimpleNamespace(v1=SimpleNamespace(message=self.message_api))


def _message(
    *,
    message_id: str,
    content: str,
    create_time: int,
    sender_id: str = "ou_user",
    sender_type: str = "user",
    msg_type: str = "text",
) -> object:
    return SimpleNamespace(
        message_id=message_id,
        msg_type=msg_type,
        create_time=create_time,
        sender=SimpleNamespace(id=sender_id, sender_type=sender_type),
        body=SimpleNamespace(content=json.dumps({"text": content})),
    )


def test_lark_bridge_adapter_builds_remote_previous_messages_snapshot() -> None:
    parent = _message(
        message_id="om_parent",
        content="scheduled task prompt",
        create_time=1000,
        sender_id="cli_app",
        sender_type="app",
    )
    chat_items = [
        _message(message_id="om_current", content="approved", create_time=1200),
        _message(message_id="om_recent", content="recent context", create_time=1100),
    ]
    client = _FakeClient(parent=parent, chat_items=chat_items)
    adapter = _SnapshotTestAdapter(
        client=client,
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_lark_app_id="cli_app",
            bridge_lark_app_secret="test-secret",  # noqa: S106
            bridge_lark_previous_messages_limit=3,
            _env_file=None,
        ),
    )

    snapshot = adapter._build_remote_previous_messages_snapshot(
        _FakeLarkModule(),
        BridgeInboundMessage(
            adapter="lark",
            event_id="event-1",
            message_id="om_current",
            parent_id="om_parent",
            chat_id="oc_1",
            create_time="1200",
            content_text="approved",
        ),
    )

    assert snapshot is not None
    assert [item.message_id for item in snapshot.items] == ["om_parent", "om_recent"]
    assert snapshot.items[0].speaker == "self"
    assert snapshot.items[0].relation == "parent"
    assert snapshot.items[0].content_text == "scheduled task prompt"
    assert snapshot.items[1].speaker == "external_user"
    assert snapshot.items[1].relation == "chat_recent"


def test_lark_bridge_adapter_keeps_full_remote_snapshot_message_content() -> None:
    content = "x" * 50
    parent = _message(
        message_id="om_parent",
        content=content,
        create_time=1000,
        sender_id="cli_app",
        sender_type="app",
    )
    client = _FakeClient(parent=parent, chat_items=[])
    adapter = _SnapshotTestAdapter(
        client=client,
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_lark_app_id="cli_app",
            bridge_lark_app_secret="test-secret",  # noqa: S106
            _env_file=None,
        ),
    )

    snapshot = adapter._build_remote_previous_messages_snapshot(
        _FakeLarkModule(),
        BridgeInboundMessage(
            adapter="lark",
            event_id="event-1",
            message_id="om_current",
            parent_id="om_parent",
            chat_id="oc_1",
            create_time="1200",
            content_text="approved",
        ),
    )

    assert snapshot is not None
    assert snapshot.truncated is False
    assert snapshot.items[0].truncated is False
    assert snapshot.items[0].content_text == content
