from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine
from ya_claw.bridge.base import BridgeMessageHandler
from ya_claw.bridge.github.adapter import GitHubBridgeAdapter
from ya_claw.bridge.github.client import GitHubApiError, GitHubRestClient
from ya_claw.bridge.github.models import GitHubNotification, GitHubNotificationPage, GitHubUser
from ya_claw.bridge.github.normalizer import (
    build_inbound_message,
    notification_event_id,
    resolve_notification_resource,
    resolve_notification_source,
)
from ya_claw.bridge.models import (
    BridgeAdapterType,
    BridgeDispatchResult,
    BridgeEventStatus,
    BridgeInboundMessage,
)
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.tables import BridgeCursorRecord


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[..., None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'github-bridge.sqlite3').resolve()}"
    initialize_sqlite_database(database_url, profile_names=("default",))
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


class _RecordingHandler(BridgeMessageHandler):
    def __init__(self) -> None:
        self.messages: list[BridgeInboundMessage] = []
        self.called = asyncio.Event()

    async def handle_message(self, message: BridgeInboundMessage) -> BridgeDispatchResult:
        self.messages.append(message)
        self.called.set()
        return BridgeDispatchResult(
            status=BridgeEventStatus.QUEUED,
            adapter=message.adapter,
            event_id=message.event_id,
            message_id=message.message_id,
            chat_id=message.chat_id,
            session_id="session-1",
            run_id="run-1",
        )


class _FailOnceHandler(BridgeMessageHandler):
    def __init__(self) -> None:
        self.call_count = 0
        self.succeeded = asyncio.Event()

    async def handle_message(self, message: BridgeInboundMessage) -> BridgeDispatchResult:
        self.call_count += 1
        if self.call_count == 1:
            return BridgeDispatchResult(
                status=BridgeEventStatus.FAILED,
                adapter=message.adapter,
                event_id=message.event_id,
                message_id=message.message_id,
                chat_id=message.chat_id,
                error_message="transient dispatch failure",
            )
        self.succeeded.set()
        return BridgeDispatchResult(
            status=BridgeEventStatus.QUEUED,
            adapter=message.adapter,
            event_id=message.event_id,
            message_id=message.message_id,
            chat_id=message.chat_id,
            session_id="session-1",
            run_id="run-1",
        )


class _FakeGitHubClient:
    def __init__(
        self,
        *,
        notifications: list[GitHubNotification],
        sources: dict[str, dict[str, Any]],
        expected_mark_count: int,
    ) -> None:
        self.notifications = notifications
        self.sources = sources
        self.expected_mark_count = expected_mark_count
        self.list_since: list[datetime] = []
        self.marked_read: list[str] = []
        self.marked = asyncio.Event()
        self.closed = False

    async def get_authenticated_user(self) -> GitHubUser:
        return GitHubUser(login="ya-claw-bot", id=100, type="User")

    async def list_notifications(self, *, since: datetime) -> GitHubNotificationPage:
        self.list_since.append(since)
        return GitHubNotificationPage(notifications=self.notifications, poll_interval_seconds=75)

    async def get_json(self, url: str) -> dict[str, Any]:
        return self.sources[url]

    async def mark_thread_read(self, thread_id: str) -> None:
        self.marked_read.append(thread_id)
        if len(self.marked_read) >= self.expected_mark_count:
            self.marked.set()

    async def close(self) -> None:
        self.closed = True


def _notification(
    *,
    thread_id: str,
    subject_type: str = "Issue",
    reason: str = "mention",
    repository_id: int = 42,
    number: int = 7,
    updated_at: datetime | None = None,
    latest_comment: bool = True,
) -> GitHubNotification:
    resource_segment = "pulls" if subject_type == "PullRequest" else "issues"
    comment_url = f"https://api.github.com/repos/acme/widgets/issues/comments/{thread_id}"
    return GitHubNotification.model_validate({
        "id": thread_id,
        "unread": True,
        "reason": reason,
        "updated_at": (updated_at or datetime(2026, 8, 26, 8, 0, tzinfo=UTC)).isoformat(),
        "last_read_at": None,
        "subject": {
            "title": "Fix widget",
            "url": f"https://api.github.com/repos/acme/widgets/{resource_segment}/{number}",
            "latest_comment_url": comment_url if latest_comment else None,
            "type": subject_type,
        },
        "repository": {
            "id": repository_id,
            "full_name": "acme/widgets",
            "html_url": "https://github.com/acme/widgets",
        },
    })


async def test_github_adapter_filters_senders_routes_issue_and_persists_cursor(db_engine: AsyncEngine) -> None:
    allowed = _notification(thread_id="101")
    denied = _notification(thread_id="102")
    unsupported = _notification(thread_id="103", subject_type="Release")
    client = _FakeGitHubClient(
        notifications=[denied, unsupported, allowed],
        sources={
            allowed.subject.latest_comment_url or "": {
                "id": 501,
                "body": "@ya-claw-bot please investigate",
                "created_at": "2026-08-26T08:00:00Z",
                "updated_at": "2026-08-26T08:00:00Z",
                "user": {"login": "Alice", "id": 1, "type": "User"},
            },
            denied.subject.latest_comment_url or "": {
                "id": 502,
                "body": "not allowed",
                "created_at": "2026-08-26T08:00:00Z",
                "updated_at": "2026-08-26T08:00:00Z",
                "user": {"login": "mallory", "id": 2, "type": "User"},
            },
        },
        expected_mark_count=3,
    )
    handler = _RecordingHandler()
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_github_token="github-token",  # noqa: S106
        bridge_github_allowed_senders="alice",
        bridge_github_initial_lookback_seconds=120,
        bridge_github_poll_interval_seconds=300,
        _env_file=None,
    )
    adapter = GitHubBridgeAdapter(
        settings=settings,
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )
    started_at = datetime.now(UTC)

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert len(handler.messages) == 1
    message = handler.messages[0]
    assert message.adapter == BridgeAdapterType.GITHUB
    assert message.chat_id == "github:42:issue:7"
    assert message.sender_id == "Alice"
    assert message.event_id == "github:101:2026-08-26T08:00:00Z"
    assert message.message_id == message.event_id
    assert message.thread_id == "101"
    assert message.metadata["github"]["repository"] == "acme/widgets"
    assert client.marked_read == ["101", "102", "103"]
    assert len(client.list_since) == 1
    assert started_at - timedelta(seconds=125) <= client.list_since[0] <= started_at
    assert client.closed is False

    async with create_session_factory(db_engine)() as db_session:
        cursor = (
            await db_session.execute(
                select(BridgeCursorRecord).where(BridgeCursorRecord.adapter == BridgeAdapterType.GITHUB)
            )
        ).scalar_one()
    assert cursor.tenant_key == "github:api.github.com:100"
    assert cursor.cursor_key == "notifications"
    assert datetime.fromisoformat(cursor.cursor_value.replace("Z", "+00:00")) >= started_at


async def test_github_adapter_retries_failed_dispatch_without_advancing_cursor_or_marking_read(
    db_engine: AsyncEngine,
) -> None:
    notification = _notification(thread_id="151")
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            notification.subject.latest_comment_url or "": {
                "id": 503,
                "body": "@ya-claw-bot please retry",
                "created_at": "2026-08-26T08:00:00Z",
                "updated_at": "2026-08-26T08:00:00Z",
                "user": {"login": "alice", "id": 1, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _FailOnceHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="alice",
            bridge_github_poll_interval_seconds=1,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(handler.succeeded.wait(), timeout=5)
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert handler.call_count == 2
    assert client.marked_read == ["151"]
    assert len(client.list_since) == 2
    assert client.list_since[1] == client.list_since[0]


async def test_github_adapter_rejects_stale_comment_actor_and_authenticated_account(
    db_engine: AsyncEngine,
) -> None:
    stale = _notification(thread_id="161", reason="assign")
    own = _notification(thread_id="162")
    client = _FakeGitHubClient(
        notifications=[stale, own],
        sources={
            stale.subject.latest_comment_url or "": {
                "id": 504,
                "body": "older allowed comment",
                "created_at": "2026-08-26T07:58:30Z",
                "updated_at": "2026-08-26T07:58:30Z",
                "user": {"login": "alice", "id": 1, "type": "User"},
            },
            own.subject.latest_comment_url or "": {
                "id": 505,
                "body": "self-triggered comment",
                "created_at": "2026-08-26T08:00:00Z",
                "updated_at": "2026-08-26T08:00:00Z",
                "user": {"login": "ya-claw-bot", "id": 100, "type": "User"},
            },
        },
        expected_mark_count=2,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="alice,ya-claw-bot",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert handler.messages == []
    assert client.marked_read == ["161", "162"]


async def test_github_adapter_attributes_new_subject_mention_to_subject_author(db_engine: AsyncEngine) -> None:
    notification = _notification(thread_id="171", reason="mention", latest_comment=False)
    assert notification.subject.url is not None
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            notification.subject.url: {
                "number": 7,
                "title": "Fix widget",
                "body": "@ya-claw-bot please investigate",
                "created_at": "2026-08-26T08:00:00Z",
                "updated_at": "2026-08-26T08:00:00Z",
                "user": {"login": "alice", "id": 1, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="alice",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(handler.called.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert [message.sender_id for message in handler.messages] == ["alice"]


async def test_github_adapter_attributes_delayed_new_issue_mention_when_latest_comment_is_subject(
    db_engine: AsyncEngine,
) -> None:
    notification = _notification(
        thread_id="25310933989",
        reason="mention",
        updated_at=datetime(2026, 8, 26, 13, 47, 29, tzinfo=UTC),
        latest_comment=False,
    )
    assert notification.subject.url is not None
    notification = notification.model_copy(
        update={"subject": notification.subject.model_copy(update={"latest_comment_url": notification.subject.url})}
    )
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            notification.subject.url: {
                "number": 7,
                "title": "Testing",
                "body": "@ya-claw-bot Can you see this?",
                "created_at": "2026-08-26T13:47:08Z",
                "updated_at": "2026-08-26T13:47:08Z",
                "user": {"login": "Wh1isper", "id": 43375501, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="wh1isper",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert [message.sender_id for message in handler.messages] == ["Wh1isper"]


async def test_github_adapter_attributes_delayed_latest_comment_sender(db_engine: AsyncEngine) -> None:
    notification = _notification(
        thread_id="25310933989",
        reason="mention",
        updated_at=datetime(2026, 8, 26, 15, 21, 3, tzinfo=UTC),
    )
    assert notification.subject.latest_comment_url is not None
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            notification.subject.latest_comment_url: {
                "id": 5427431414,
                "body": "@ya-claw-bot hi",
                "created_at": "2026-08-26T15:20:40Z",
                "updated_at": "2026-08-26T15:20:40Z",
                "user": {"login": "Wh1isper", "id": 43375501, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="wh1isper",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert [message.sender_id for message in handler.messages] == ["Wh1isper"]


async def test_github_adapter_treats_equivalent_subject_url_as_subject_for_sender_attribution(
    db_engine: AsyncEngine,
) -> None:
    notification = _notification(
        thread_id="25310933990",
        reason="assign",
        updated_at=datetime(2026, 8, 26, 13, 47, 8, tzinfo=UTC),
        latest_comment=False,
    )
    assert notification.subject.url is not None
    equivalent_subject_url = f"{notification.subject.url}/"
    notification = notification.model_copy(
        update={"subject": notification.subject.model_copy(update={"latest_comment_url": equivalent_subject_url})}
    )
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            equivalent_subject_url: {
                "number": 7,
                "title": "Testing",
                "body": "@ya-claw-bot Can you see this?",
                "created_at": "2026-08-26T13:47:08Z",
                "updated_at": "2026-08-26T13:47:08Z",
                "user": {"login": "Wh1isper", "id": 43375501, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="wh1isper",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(client.marked.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert handler.messages == []


async def test_github_adapter_wildcard_accepts_unattributable_pull_request(db_engine: AsyncEngine) -> None:
    notification = _notification(
        thread_id="201",
        subject_type="PullRequest",
        reason="review_requested",
        number=9,
        latest_comment=False,
    )
    assert notification.subject.url is not None
    client = _FakeGitHubClient(
        notifications=[notification],
        sources={
            notification.subject.url: {
                "number": 9,
                "title": "Improve widgets",
                "body": "Please review",
                "user": {"login": "pull-author", "id": 3, "type": "User"},
            }
        },
        expected_mark_count=1,
    )
    handler = _RecordingHandler()
    adapter = GitHubBridgeAdapter(
        settings=ClawSettings(
            api_token="test-token",  # noqa: S106
            bridge_github_token="github-token",  # noqa: S106
            bridge_github_allowed_senders="*",
            bridge_github_poll_interval_seconds=300,
            _env_file=None,
        ),
        handler=handler,
        session_factory=create_session_factory(db_engine),
        client=client,
    )

    task = asyncio.create_task(adapter.run())
    await asyncio.wait_for(handler.called.wait(), timeout=5)
    await adapter.stop()
    await asyncio.wait_for(task, timeout=5)

    assert len(handler.messages) == 1
    message = handler.messages[0]
    assert message.chat_id == "github:42:pull:9"
    assert message.sender_id is None
    assert "Improve widgets" in (message.content_text or "")


def test_github_normalizer_versions_notification_threads() -> None:
    first = _notification(thread_id="301", updated_at=datetime(2026, 8, 26, 8, 0, tzinfo=UTC))
    second = _notification(thread_id="301", updated_at=datetime(2026, 8, 26, 8, 1, tzinfo=UTC))
    resource = resolve_notification_resource(first)
    assert resource is not None
    source = resolve_notification_source(
        {"body": "hello", "user": {"login": "alice", "id": 1, "type": "User"}},
        sender_is_attributable=True,
    )

    first_message = build_inbound_message(first, tenant_key="github:test:100", resource=resource, source=source)

    assert notification_event_id(first) == "github:301:2026-08-26T08:00:00Z"
    assert notification_event_id(second) == "github:301:2026-08-26T08:01:00Z"
    assert first_message.message_id == first_message.event_id
    assert first_message.root_id == first.subject.url


async def test_github_rest_client_lists_pages_and_marks_read() -> None:
    requests: list[httpx.Request] = []
    notification_payload = _notification(thread_id="401").model_dump(mode="json")

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer github-token"
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "ya-claw-bot", "id": 100, "type": "User"})
        if request.url.path == "/notifications" and request.url.params.get("page") == "2":
            return httpx.Response(200, json=[], headers={"X-Poll-Interval": "90"})
        if request.url.path == "/notifications":
            query = parse_qs(request.url.query.decode())
            assert query["all"] == ["true"]
            assert query["per_page"] == ["50"]
            assert query["since"] == ["2026-08-26T08:00:00Z"]
            assert "If-Modified-Since" in request.headers
            return httpx.Response(
                200,
                json=[notification_payload],
                headers={
                    "Date": "Wed, 26 Aug 2026 08:00:30 GMT",
                    "Link": '<https://api.github.com/notifications?page=2>; rel="next"',
                    "X-Poll-Interval": "60",
                },
            )
        if request.url.path == "/notifications/threads/401":
            assert request.method == "PATCH"
            return httpx.Response(205)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = GitHubRestClient(
        token="github-token",  # noqa: S106
        transport=httpx.MockTransport(handle),
    )
    try:
        user = await client.get_authenticated_user()
        page = await client.list_notifications(since=datetime(2026, 8, 26, 8, 0, tzinfo=UTC))
        await client.mark_thread_read("401")
    finally:
        await client.close()

    assert user.login == "ya-claw-bot"
    assert [item.id for item in page.notifications] == ["401"]
    assert page.poll_interval_seconds == 90
    assert page.response_date == datetime(2026, 8, 26, 8, 0, 30, tzinfo=UTC)
    assert len(requests) == 4


def test_github_rest_client_rejects_insecure_api_url() -> None:
    with pytest.raises(ValueError, match="must use https"):
        GitHubRestClient(token="github-token", api_url="http://api.github.test")  # noqa: S106


async def test_github_rest_client_honors_poll_interval_on_not_modified() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/notifications"
        return httpx.Response(304, headers={"X-Poll-Interval": "120"})

    client = GitHubRestClient(
        token="github-token",  # noqa: S106
        transport=httpx.MockTransport(handle),
    )
    try:
        page = await client.list_notifications(since=datetime(2026, 8, 26, 8, 0, tzinfo=UTC))
    finally:
        await client.close()

    assert page.notifications == []
    assert page.poll_interval_seconds == 120


async def test_github_rest_client_rejects_foreign_source_url() -> None:
    client = GitHubRestClient(
        token="github-token",  # noqa: S106
        transport=httpx.MockTransport(lambda _request: None),
    )
    try:
        with pytest.raises(GitHubApiError, match="foreign API origin"):
            await client.get_json("https://example.com/repos/acme/widgets/issues/1")
    finally:
        await client.close()
