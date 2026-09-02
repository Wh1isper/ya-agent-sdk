from __future__ import annotations

import json

import anyio
import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage
from websockets.datastructures import Headers
from ya_oauth.types import OAuthAccount, TokenSnapshot
from ya_oauth_provider.codex import (
    CodexResponsesModel,
    CodexWebsocketResponsesModel,
    _CodexRequestHeaders,
    _CodexWebsocketResponseStream,
    build_codex_model,
    build_session_headers,
    normalize_codex_responses_payload,
)
from ya_oauth_provider.http import (
    CODEX_ROUTING_HINT_HEADER,
    CODEX_TURN_STATE_HEADER,
    CODEX_WEBSOCKET_BETA,
    OAuthBearerAuth,
    build_codex_headers,
    build_codex_websocket_headers,
)

ACCESS_TOKEN_OLD = "fixture-access-token-old"  # noqa: S105
ACCESS_TOKEN_NEW = "fixture-access-token-new"  # noqa: S105


def _response(status: str) -> dict[str, object]:
    return {
        "id": "resp_1",
        "created_at": 1,
        "model": "gpt-5.5",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }


def _responses_sse() -> str:
    events = [
        {"type": "response.created", "sequence_number": 0, "response": _response("in_progress")},
        {"type": "response.completed", "sequence_number": 1, "response": _response("completed")},
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


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
            account=OAuthAccount(),
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
    assert isinstance(model.provider.client._client, httpx2.AsyncClient)
    assert model.websocket_fallback_state.mode == "auto"


def test_build_codex_model_can_force_http() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource(), websocket_mode="http")

    assert isinstance(model, CodexResponsesModel)
    assert not isinstance(model, CodexWebsocketResponsesModel)


@pytest.mark.asyncio
async def test_codex_http_round_trips_turn_state_and_builds_routing_hint() -> None:
    request_headers: list[httpx2.Headers] = []
    response_states = iter(("turn-one", "ignored-replacement", "turn-two"))

    async def handler(request: httpx2.Request) -> httpx2.Response:
        request_headers.append(request.headers)
        return httpx2.Response(
            200,
            request=request,
            headers={
                "content-type": "text/event-stream",
                CODEX_TURN_STATE_HEADER: next(response_states),
            },
            content=_responses_sse(),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        model = CodexResponsesModel(
            "gpt-5.5",
            provider=OpenAIProvider(
                openai_client=AsyncOpenAI(api_key="test-key", http_client=http_client),
            ),
        )
        usage_by_run = {"run-one": RunUsage(), "run-two": RunUsage()}
        for run_id in ("run-one", "run-one", "run-two"):
            run_context = RunContext(
                deps=object(),
                model=model,
                usage=usage_by_run[run_id],
                run_id=run_id,
            )
            async with model.request_stream(
                [],
                {
                    "openai_service_tier": "priority",
                    "extra_headers": {
                        "X-Codex-Turn-State": "caller-stale-state",
                        "X-Codex-Routing-Hint": "caller-stale-hint",
                    },
                },
                ModelRequestParameters(),
                run_context,
            ) as response:
                async for _event in response:
                    pass

    assert CODEX_TURN_STATE_HEADER not in request_headers[0]
    assert request_headers[0][CODEX_ROUTING_HINT_HEADER] == "model=gpt-5.5;tier=priority"
    assert request_headers[1][CODEX_TURN_STATE_HEADER] == "turn-one"
    assert request_headers[1][CODEX_ROUTING_HINT_HEADER] == "model=gpt-5.5;tier=priority"
    assert CODEX_TURN_STATE_HEADER not in request_headers[2]


def test_codex_turn_state_accepts_header_array_and_resets_for_reused_run_id() -> None:
    request_headers = _CodexRequestHeaders("gpt-5.5")
    first_context = RunContext(
        deps=object(),
        model=None,  # type: ignore[arg-type]
        usage=RunUsage(),
        run_id="run-one",
    )
    with request_headers.scope(first_context):
        request_headers.capture({CODEX_TURN_STATE_HEADER: ["array-state", "ignored"]})
        first_settings = request_headers.apply({})

    reused_context = RunContext(
        deps=object(),
        model=None,  # type: ignore[arg-type]
        usage=RunUsage(),
        run_id="run-one",
    )
    with request_headers.scope(reused_context):
        reused_settings = request_headers.apply({})

    assert first_settings["extra_headers"][CODEX_TURN_STATE_HEADER] == "array-state"
    assert CODEX_TURN_STATE_HEADER not in reused_settings["extra_headers"]


@pytest.mark.asyncio
async def test_codex_websocket_captures_handshake_state_before_send_failure() -> None:
    request_headers = _CodexRequestHeaders("gpt-5.5")
    run_context = RunContext(
        deps=object(),
        model=None,  # type: ignore[arg-type]
        usage=RunUsage(),
        run_id="run-one",
    )

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = Headers([
                ("set-cookie", "first=value"),
                ("set-cookie", "second=value"),
                (CODEX_TURN_STATE_HEADER, "handshake-state"),
            ])

    class FakeConnection:
        def __init__(self) -> None:
            self.response = FakeResponse()

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def messages():  # type: ignore[no-untyped-def]
                if False:
                    yield ""

            return messages()

        async def send(self, message: str) -> None:
            raise OSError("send failed")

        async def close(self, code: int = 1000, reason: str = "") -> None:
            pass

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeConnection()

    stream = _CodexWebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        connect=connect,
        request_headers=request_headers,
    )
    with request_headers.scope(run_context):
        with pytest.raises(OSError, match="send failed"):
            await stream.__aenter__()
        fallback_settings = request_headers.apply({})

    assert fallback_settings["extra_headers"][CODEX_TURN_STATE_HEADER] == "handshake-state"


@pytest.mark.parametrize(
    ("handshake_state", "expected_state"),
    [(None, "websocket-state"), ("handshake-state", "handshake-state")],
)
@pytest.mark.asyncio
async def test_codex_websocket_round_trips_turn_state_and_builds_routing_hint(
    handshake_state: str | None,
    expected_state: str,
) -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())
    assert isinstance(model, CodexWebsocketResponsesModel)
    run_context = RunContext(
        deps=object(),
        model=model,
        usage=RunUsage(),
        run_id="run-one",
    )

    class FakeResponse:
        def __init__(self, state: str) -> None:
            self.headers = {CODEX_TURN_STATE_HEADER: state}

    class FakeConnection:
        def __init__(self) -> None:
            self.response = FakeResponse(handshake_state) if handshake_state is not None else None
            self.sent: list[str] = []

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def messages():  # type: ignore[no-untyped-def]
                yield json.dumps({
                    "type": "response.metadata",
                    "headers": {"X-Codex-Turn-State": "websocket-state"},
                })

            return messages()

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def close(self, code: int = 1000, reason: str = "") -> None:
            pass

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeConnection()

    settings = {
        "service_tier": "flex",
        "extra_headers": {
            "X-Codex-Turn-State": "caller-stale-state",
            "X-Codex-Routing-Hint": "caller-stale-hint",
        },
    }
    with model._codex_request_headers.scope(run_context):
        first_stream = await model._create_websocket_stream(
            [],
            settings,  # type: ignore[arg-type]
            ModelRequestParameters(),
            payload={"type": "response.create"},
        )
        first_stream._connect = connect
        assert CODEX_TURN_STATE_HEADER not in first_stream.headers
        assert first_stream.headers[CODEX_ROUTING_HINT_HEADER] == "model=gpt-5.5;tier=flex"
        async with first_stream:
            async for _event in first_stream:
                pass

    with model._codex_request_headers.scope(run_context):
        second_stream = await model._create_websocket_stream(
            [],
            settings,  # type: ignore[arg-type]
            ModelRequestParameters(),
            payload={"type": "response.create"},
        )

    assert CODEX_TURN_STATE_HEADER not in second_stream.headers
    assert second_stream.payload["client_metadata"] == {CODEX_TURN_STATE_HEADER: expected_state}
    assert second_stream.headers[CODEX_ROUTING_HINT_HEADER] == "model=gpt-5.5;tier=flex"


@pytest.mark.asyncio
async def test_build_codex_model_owned_httpx2_client_closes_and_recreates() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())
    assert isinstance(model, CodexWebsocketResponsesModel)
    first_client = model.provider.client._client

    async with model:
        assert first_client.is_closed is False
    assert first_client.is_closed is True

    async with model:
        second_client = model.provider.client._client
        assert second_client is not first_client
        assert second_client.is_closed is False
    assert second_client.is_closed is True


def test_build_codex_model_requires_streaming_for_non_stream_request() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())

    assert isinstance(model, CodexWebsocketResponsesModel)
    with pytest.raises(UserError, match="requires streaming"):
        anyio.run(model.request, [], None, None)  # type: ignore[arg-type]


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
    assert "OpenAI-Beta" not in headers
    assert "openai-beta" not in headers
    assert headers["session-id"] == "s1"
    assert headers["thread-id"] == "t1"
    assert headers["x-client-request-id"] == "t1"


@pytest.mark.asyncio
async def test_codex_websocket_model_adds_core_beta_header_once() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())
    assert isinstance(model, CodexWebsocketResponsesModel)

    headers = await model._build_websocket_headers({"openai-beta": CODEX_WEBSOCKET_BETA})

    assert headers["openai-beta"] == CODEX_WEBSOCKET_BETA
    assert "OpenAI-Beta" not in headers


@pytest.mark.asyncio
async def test_oauth_bearer_auth_fills_codex_responses_instructions() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(json.loads(request.content)))
        assert "version" not in request.headers
        return httpx2.Response(200, json={"ok": True}, request=request)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
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
async def test_oauth_bearer_auth_normalizes_async_stream_body() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    async def body():
        yield b'{"model":"gpt-5.5","max_output_tokens":4096}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(json.loads(request.content)))
        assert "transfer-encoding" not in request.headers
        assert request.headers["content-length"] == str(len(request.content))
        return httpx2.Response(200, json={"ok": True}, request=request)

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    ) as client:
        await client.post(
            "https://chatgpt.com/backend-api/codex/responses",
            content=body(),
            headers={"Content-Type": "application/json"},
        )

    assert seen == [{"model": "gpt-5.5", "instructions": "", "store": False}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_strips_codex_response_token_limits() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx2.Response(200, json={"ok": True}, request=request)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
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

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx2.Response(200, json={"ok": True}, request=request)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    )

    await client.post("https://example.com/v1/responses", json={"model": "gpt-5.5", "max_tokens": 4096})
    await client.aclose()

    assert seen == [{"model": "gpt-5.5", "max_tokens": 4096}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_refreshes_once_on_401() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, str | None]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({
            "authorization": request.headers.get("Authorization"),
            "account_id": request.headers.get("ChatGPT-Account-ID"),
            "fedramp": request.headers.get("X-OpenAI-Fedramp"),
        })
        if len(seen) == 1:
            return httpx2.Response(401, request=request)
        return httpx2.Response(200, json={"ok": True}, request=request)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex", extra_headers={"session_id": "s1"}),
    )

    response = await client.get("https://example.com/test")
    await client.aclose()

    assert response.status_code == 200
    assert seen == [
        {
            "authorization": f"Bearer {ACCESS_TOKEN_OLD}",
            "account_id": "acct_123",
            "fedramp": "true",
        },
        {
            "authorization": f"Bearer {ACCESS_TOKEN_NEW}",
            "account_id": None,
            "fedramp": None,
        },
    ]
    assert source.refresh_count == 1


@pytest.mark.asyncio
async def test_oauth_bearer_auth_normalizes_and_replays_async_stream_body_on_401() -> None:
    source = FakeTokenSource()
    seen_bodies: list[dict[str, object]] = []

    async def body():
        yield b'{"model":"gpt-5.5","max_output_tokens":4096}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen_bodies.append(dict(json.loads(request.content)))
        status_code = 401 if len(seen_bodies) == 1 else 200
        return httpx2.Response(status_code, json={"ok": True}, request=request)

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    ) as client:
        response = await client.post(
            "https://chatgpt.com/backend-api/codex/responses",
            content=body(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert seen_bodies == [
        {"model": "gpt-5.5", "instructions": "", "store": False},
        {"model": "gpt-5.5", "instructions": "", "store": False},
    ]
    assert source.refresh_count == 1


@pytest.mark.asyncio
async def test_websocket_model_falls_back_to_http_before_first_event(monkeypatch) -> None:
    from contextlib import asynccontextmanager

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from ya_agent_sdk.agents.models.websocket import WebsocketResponsesModel

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
