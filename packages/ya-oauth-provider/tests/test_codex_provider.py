from __future__ import annotations

import json

import anyio
import httpx
import pytest
from pydantic_ai.exceptions import UserError
from ya_oauth.types import OAuthAccount, TokenSnapshot
from ya_oauth_provider.codex import (
    CodexResponsesModel,
    CodexWebsocketResponsesModel,
    build_codex_model,
    build_session_headers,
)
from ya_oauth_provider.http import (
    CODEX_WEBSOCKET_BETA,
    OAuthBearerAuth,
    build_codex_headers,
    build_codex_websocket_headers,
)
from ya_oauth_provider.websocket_model import normalize_codex_responses_payload, responses_websocket_url

ACCESS_TOKEN_OLD = "fixture-access-token-old"  # noqa: S105
ACCESS_TOKEN_NEW = "fixture-access-token-new"  # noqa: S105


class FakeTokenSource:
    def __init__(self) -> None:
        self.refresh_count = 0

    async def get_token(self) -> TokenSnapshot:
        return TokenSnapshot(
            provider_name="codex",
            access_token=ACCESS_TOKEN_OLD,
            account=OAuthAccount(chatgpt_account_id="acct_123", chatgpt_account_is_fedramp=True),
        )

    async def refresh_token(self) -> TokenSnapshot:
        self.refresh_count += 1
        return TokenSnapshot(
            provider_name="codex",
            access_token=ACCESS_TOKEN_NEW,
            account=OAuthAccount(chatgpt_account_id="acct_456"),
        )


def test_build_codex_headers() -> None:
    headers = build_codex_headers(
        OAuthAccount(chatgpt_account_id="acct_123", chatgpt_account_is_fedramp=True),
        extra_headers={"session_id": "s1", "thread-id": "t1", "x-client-request-id": "t1"},
    )

    assert "Authorization" not in headers
    assert headers["ChatGPT-Account-ID"] == "acct_123"
    assert headers["X-OpenAI-Fedramp"] == "true"
    assert headers["originator"] == "ya_agent_sdk"
    assert "version" not in headers
    assert headers["session_id"] == "s1"
    assert headers["thread-id"] == "t1"
    assert headers["x-client-request-id"] == "t1"


def test_build_codex_headers_omits_version_by_default() -> None:
    headers = build_codex_headers(OAuthAccount())

    assert headers["originator"] == "ya_agent_sdk"
    assert "version" not in headers


def test_build_codex_headers_rejects_reserved_extra_headers() -> None:
    with pytest.raises(ValueError, match="reserved OAuth/Codex header"):
        build_codex_headers(OAuthAccount(), extra_headers={"Authorization": "Bearer other"})


def test_build_session_headers_uses_both_variants() -> None:
    assert build_session_headers("session", "thread") == {
        "session_id": "session",
        "session-id": "session",
        "thread_id": "thread",
        "thread-id": "thread",
        "x-client-request-id": "thread",
    }


def test_build_codex_model_defaults_to_websocket_auto() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())

    assert isinstance(model, CodexWebsocketResponsesModel)
    assert model.websocket_fallback_state.mode == "auto"


def test_build_codex_model_can_force_http() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource(), websocket_mode="http")

    assert isinstance(model, CodexResponsesModel)
    assert not isinstance(model, CodexWebsocketResponsesModel)


def test_build_codex_model_requires_streaming_for_non_stream_request() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())

    assert isinstance(model, CodexWebsocketResponsesModel)
    with pytest.raises(UserError, match="requires streaming"):
        anyio.run(model.request, [], None, None)  # type: ignore[arg-type]


def test_responses_websocket_url() -> None:
    assert responses_websocket_url("https://chatgpt.com/backend-api/codex") == (
        "wss://chatgpt.com/backend-api/codex/responses"
    )
    assert responses_websocket_url("http://localhost:8080/v1") == "ws://localhost:8080/v1/responses"


def test_normalize_codex_responses_payload() -> None:
    assert normalize_codex_responses_payload({
        "model": "gpt-5.5",
        "instructions": None,
        "max_tokens": 1,
        "max_completion_tokens": 2,
        "max_output_tokens": 3,
    }) == {"model": "gpt-5.5", "instructions": "", "store": False}


@pytest.mark.asyncio
async def test_build_codex_websocket_headers() -> None:
    headers = await build_codex_websocket_headers(
        FakeTokenSource(),
        extra_headers={"session-id": "s1", "thread-id": "t1", "x-client-request-id": "t1"},
    )

    assert headers["Authorization"] == f"Bearer {ACCESS_TOKEN_OLD}"
    assert headers["ChatGPT-Account-ID"] == "acct_123"
    assert headers["X-OpenAI-Fedramp"] == "true"
    assert headers["originator"] == "ya_agent_sdk"
    assert headers["OpenAI-Beta"] == CODEX_WEBSOCKET_BETA
    assert headers["session-id"] == "s1"
    assert headers["thread-id"] == "t1"
    assert headers["x-client-request-id"] == "t1"


@pytest.mark.asyncio
async def test_oauth_bearer_auth_fills_codex_responses_instructions() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        assert "version" not in request.headers
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    )

    await client.post("https://chatgpt.com/backend-api/codex/responses", json={"model": "gpt-5.5"})
    await client.post(
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5", "instructions": None},
    )
    await client.aclose()

    assert seen == [
        {"model": "gpt-5.5", "instructions": "", "store": False},
        {"model": "gpt-5.5", "instructions": "", "store": False},
    ]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_strips_codex_response_token_limits() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    )

    await client.post(
        "https://chatgpt.com/backend-api/codex/responses",
        json={
            "model": "gpt-5.5",
            "max_tokens": 4096,
            "max_completion_tokens": 4096,
            "max_output_tokens": 4096,
        },
    )
    await client.aclose()

    assert seen == [{"model": "gpt-5.5", "instructions": "", "store": False}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_keeps_token_limits_for_non_codex_response_requests() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    )

    await client.post("https://example.com/v1/responses", json={"model": "gpt-5.5", "max_tokens": 4096})
    await client.aclose()

    assert seen == [{"model": "gpt-5.5", "max_tokens": 4096}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_refreshes_once_on_401() -> None:
    source = FakeTokenSource()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            return httpx.Response(401, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex", extra_headers={"session_id": "s1"}),
    )

    response = await client.get("https://example.com/test")
    await client.aclose()

    assert response.status_code == 200
    assert seen == [f"Bearer {ACCESS_TOKEN_OLD}", f"Bearer {ACCESS_TOKEN_NEW}"]
    assert source.refresh_count == 1


@pytest.mark.asyncio
async def test_websocket_response_stream_filters_codex_events_and_closes_on_terminal() -> None:
    from ya_oauth_provider.websocket_model import _WebsocketResponseStream

    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self._messages = iter([
                json.dumps({"type": "codex.rate_limits", "rate_limits": []}),
                json.dumps({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_1",
                        "created_at": 1,
                        "model": "gpt-5.5",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "in_progress",
                    },
                }),
                json.dumps({
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "id": "resp_1",
                        "created_at": 1,
                        "model": "gpt-5.5",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "status": "completed",
                    },
                }),
            ])

        def __aiter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def close(self) -> None:
            self.closed = True

    fake = FakeConnection()

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    stream = _WebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        connect=connect,
    )

    async with stream:
        events = [event async for event in stream]

    assert [event.type for event in events] == ["response.created", "response.completed"]
    assert fake.closed is True
    assert fake.sent == ['{"type":"response.create"}']


@pytest.mark.asyncio
async def test_websocket_model_falls_back_to_http_before_first_event(monkeypatch) -> None:
    from contextlib import asynccontextmanager

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from ya_oauth_provider.websocket_model import WebsocketResponsesModel

    class DummyStreamedResponse(StreamedResponse):
        @property
        def model_name(self) -> str:
            return "dummy"

        @property
        def provider_name(self) -> str:
            return "dummy"

        @property
        def provider_url(self) -> str:
            return "https://example.test"

        @property
        def timestamp(self):  # type: ignore[no-untyped-def]
            from datetime import UTC, datetime

            return datetime.now(UTC)

        async def _get_event_iterator(self):  # type: ignore[no-untyped-def]
            if False:
                yield None

    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())
    assert isinstance(model, WebsocketResponsesModel)

    async def fail_create_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("ws connect failed")

    fallback_called = False

    @asynccontextmanager
    async def fake_http_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal fallback_called
        fallback_called = True
        yield DummyStreamedResponse(ModelRequestParameters())

    monkeypatch.setattr(model, "_create_websocket_stream", fail_create_stream)
    monkeypatch.setattr(model, "_request_stream_http", fake_http_stream)

    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content="hello")])],
        None,
        ModelRequestParameters(),
    ) as response:
        assert isinstance(response, DummyStreamedResponse)

    assert fallback_called is True
    assert model.websocket_fallback_state.failure_count == 1
    assert model.websocket_fallback_state.last_error is not None


def test_websocket_payload_maps_responses_model_settings() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())
    assert isinstance(model, CodexWebsocketResponsesModel)

    payload = anyio.run(
        model._build_websocket_payload,
        [ModelRequest(parts=[UserPromptPart(content="hello")])],
        {
            "max_tokens": 123,
            "openai_service_tier": "flex",
            "openai_store": True,
            "openai_reasoning_effort": "high",
            "openai_user": "user-1",
            "openai_top_logprobs": 2,
            "openai_logprobs": True,
            "extra_body": {"metadata": {"source": "test"}},
            "temperature": 0.7,
        },
        ModelRequestParameters(),
    )

    assert payload["type"] == "response.create"
    assert payload["stream"] is True
    assert payload["service_tier"] == "flex"
    assert payload["store"] is False
    assert payload["user"] == "user-1"
    assert "top_logprobs" not in payload
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["metadata"] == {"source": "test"}
    assert payload["instructions"] == ""
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert "message.output_text.logprobs" not in payload["include"]
