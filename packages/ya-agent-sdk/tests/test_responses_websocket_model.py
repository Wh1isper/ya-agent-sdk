from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from ya_agent_sdk.agents.models.utils import ModelRequestRetryOptions
from ya_agent_sdk.agents.models.websocket import (
    DEFAULT_WEBSOCKET_BETA,
    DEFAULT_WEBSOCKET_MAX_SIZE,
    WebsocketResponsesModel,
    _WebsocketResponseStream,
    build_openai_responses_websocket_model,
    responses_websocket_url,
)
from ya_agent_sdk.presets import OPENAI_RESPONSES_PRO


def test_responses_websocket_url() -> None:
    assert responses_websocket_url("https://api.openai.com/v1") == "wss://api.openai.com/v1/responses"
    assert responses_websocket_url("http://localhost:8080/v1") == "ws://localhost:8080/v1/responses"
    assert responses_websocket_url("wss://example.test/custom", path="events") == "wss://example.test/custom/events"


@pytest.mark.asyncio
async def test_openai_responses_pro_preset_reaches_http_request_body() -> None:
    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": "stop after capture", "type": "test_error"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncOpenAI(api_key="test-key", http_client=http_client)
        model = OpenAIResponsesModel(
            "gpt-5.6",
            provider=OpenAIProvider(openai_client=client),
        )
        with pytest.raises(ModelHTTPError):
            await model.request([], OPENAI_RESPONSES_PRO.copy(), ModelRequestParameters())

    assert captured_body["reasoning"] == {
        "mode": "pro",
        "effort": "medium",
        "summary": "auto",
    }


@pytest.mark.asyncio
async def test_openai_responses_pro_preset_reaches_websocket_payload() -> None:
    model = WebsocketResponsesModel("gpt-5.6", provider=OpenAIProvider(api_key="test-key"))

    payload = await model._build_websocket_payload(
        [],
        OPENAI_RESPONSES_PRO.copy(),  # type: ignore[arg-type]
        ModelRequestParameters(),
    )

    assert payload["reasoning"] == {
        "mode": "pro",
        "effort": "medium",
        "summary": "auto",
    }


@pytest.mark.asyncio
async def test_websocket_model_adds_default_beta_header() -> None:
    model = WebsocketResponsesModel("gpt-5", provider=OpenAIProvider(api_key="test-key"))

    headers = await model._build_websocket_headers({})

    assert headers["OpenAI-Beta"] == DEFAULT_WEBSOCKET_BETA


@pytest.mark.asyncio
async def test_websocket_model_deduplicates_beta_header_case_insensitively() -> None:
    model = WebsocketResponsesModel("gpt-5", provider=OpenAIProvider(api_key="test-key"))

    headers = await model._build_websocket_headers({"openai-beta": DEFAULT_WEBSOCKET_BETA})

    assert headers["openai-beta"] == DEFAULT_WEBSOCKET_BETA
    assert "OpenAI-Beta" not in headers


@pytest.mark.asyncio
async def test_build_openai_responses_websocket_model_adds_extra_headers_to_handshake(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = build_openai_responses_websocket_model(
        "gpt-5",
        extra_headers={"x-session-id": "session-1", "session-id": "session-1"},
    )

    headers = await model._build_websocket_headers({"thread-id": "thread-1"})

    assert headers["x-session-id"] == "session-1"
    assert headers["session-id"] == "session-1"
    assert headers["thread-id"] == "thread-1"
    assert headers["OpenAI-Beta"] == DEFAULT_WEBSOCKET_BETA


@pytest.mark.asyncio
async def test_websocket_model_passes_max_size_to_stream(monkeypatch) -> None:
    model = WebsocketResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key="test-key"),
        websocket_max_size=123,
    )

    async def build_payload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"type": "response.create"}

    async def build_headers(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {}

    monkeypatch.setattr(model, "_build_websocket_payload", build_payload)
    monkeypatch.setattr(model, "_build_request_options", lambda *_args, **_kwargs: ({}, None))
    monkeypatch.setattr(model, "_build_websocket_headers", build_headers)

    stream = await model._create_websocket_stream([], {}, ModelRequestParameters())  # type: ignore[arg-type]

    assert stream.max_size == 123


@pytest.mark.asyncio
async def test_websocket_response_stream_filters_non_response_events_and_closes_on_terminal() -> None:
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    fake = FakeConnection()
    connect_kwargs = {}

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        connect_kwargs.update(kwargs)
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
    assert fake.close_code == 1000
    assert fake.close_reason == "responses stream completed"
    assert fake.sent == ['{"type":"response.create"}']
    assert connect_kwargs["max_size"] == DEFAULT_WEBSOCKET_MAX_SIZE


@pytest.mark.asyncio
async def test_websocket_response_stream_ignores_unsupported_response_events() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self._messages = iter([
                json.dumps({
                    "type": "response.metadata",
                    "sequence_number": 0,
                    "metadata": {"tool_call": {}, "tool_response": {}},
                }),
                json.dumps({
                    "type": "response.preflight",
                    "sequence_number": 0,
                    "metadata": {"accepted": True},
                }),
                json.dumps({
                    "type": "response.created",
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
                        "status": "in_progress",
                    },
                }),
                json.dumps({
                    "type": "response.completed",
                    "sequence_number": 2,
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

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
    assert stream.events_seen == 2
    assert fake.closed is True
    assert fake.close_code == 1000
    assert fake.close_reason == "responses stream completed"


@pytest.mark.asyncio
async def test_websocket_response_stream_sends_cancel_on_early_scope_exit() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self.close_code: int | None = None
            self.close_reason: str | None = None
            self._messages = iter([
                json.dumps({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_early",
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
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hello",
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

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
        async for event in stream:
            assert event.type == "response.created"
            break

    assert fake.closed is True
    assert fake.close_code == 1000
    assert fake.close_reason == "client exited response stream"
    assert fake.sent == [
        '{"type":"response.create"}',
        '{"type":"response.cancel","response_id":"resp_early"}',
    ]


@pytest.mark.asyncio
async def test_websocket_model_can_fallback_after_upgrade_before_first_response_event() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def __aiter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def close(self, code: int = 1000, reason: str = "") -> None:
            pass

    fake = FakeConnection()

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    stream = _WebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        connect=connect,
    )
    model = WebsocketResponsesModel("gpt-5", provider=OpenAIProvider(api_key="test-key"))

    async with stream:
        assert stream.websocket_upgraded is True
        assert stream.events_seen == 0
        assert model._should_fallback_from(UnexpectedModelBehavior("parse failed"), stream) is True


@pytest.mark.asyncio
async def test_websocket_model_falls_back_to_http_for_upstream_connect_error_after_gateway_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self.close_code: int | None = None
            self.close_reason: str | None = None
            self._messages = iter([
                json.dumps({
                    "type": "error",
                    "message": "Failed to connect to upstream Responses WebSocket",
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    fake = FakeConnection()

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    stream = _WebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        connect=connect,
    )
    model = WebsocketResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key="test-key"),
        websocket_retry_options=ModelRequestRetryOptions(enabled=False),
    )
    http_response = object()
    http_calls = 0

    async def create_websocket_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        return stream

    @asynccontextmanager
    async def request_stream_http(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal http_calls
        http_calls += 1
        yield http_response

    monkeypatch.setattr(model, "_create_websocket_stream", create_websocket_stream)
    monkeypatch.setattr(model, "_request_stream_http", request_stream_http)

    async with model.request_stream([], {}, ModelRequestParameters()) as response:  # type: ignore[arg-type]
        assert response is http_response

    assert http_calls == 1
    assert model.websocket_fallback_state.failure_count == 1
    assert model.websocket_fallback_state.disabled_until is not None
    assert model.websocket_fallback_state.last_error is not None
    assert "Failed to connect to upstream Responses WebSocket" in model.websocket_fallback_state.last_error
    assert stream.websocket_upgraded is True
    assert stream.events_seen == 0
    assert fake.closed is True
    assert fake.close_code == 1000
    assert fake.close_reason == "client exited response stream"
    assert fake.sent == [
        '{"type":"response.create"}',
        '{"type":"response.cancel"}',
    ]


@pytest.mark.asyncio
async def test_websocket_response_stream_cleanup_timeout_does_not_block_scope_exit() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.close_started = False
            self.closed = False
            self._messages = iter([
                json.dumps({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_timeout",
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.close_started = True
            await asyncio.sleep(10)
            self.closed = True

    fake = FakeConnection()

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    stream = _WebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        cleanup_timeout=0.01,
        connect=connect,
    )

    async with stream:
        async for event in stream:
            assert event.type == "response.created"
            break

    assert fake.sent == [
        '{"type":"response.create"}',
        '{"type":"response.cancel","response_id":"resp_timeout"}',
    ]
    assert fake.close_started is True
    assert fake.closed is False


@pytest.mark.asyncio
async def test_websocket_response_stream_closes_when_create_send_fails() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.closed = False
            self.close_code: int | None = None
            self.close_reason: str | None = None

        async def send(self, message: str) -> None:
            raise OSError("send failed")

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    fake = FakeConnection()

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake

    stream = _WebsocketResponseStream(
        url="wss://example.test/responses",
        headers={},
        payload={"type": "response.create"},
        connect=connect,
    )

    with pytest.raises(OSError, match="send failed"):
        async with stream:
            pass

    assert fake.closed is True
    assert fake.close_code == 1011
    assert fake.close_reason == "failed to send response.create"


@pytest.mark.asyncio
async def test_websocket_response_stream_restores_function_call_done_name_from_output_item() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self._messages = iter([
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
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "sample_tool",
                        "arguments": "",
                        "status": "in_progress",
                    },
                }),
                json.dumps({
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item_id": "fc_1",
                    "delta": '{"value": 1}',
                }),
                json.dumps({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": 3,
                    "output_index": 0,
                    "item_id": "fc_1",
                    "arguments": '{"value": 1}',
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

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

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

    done_event = events[-1]
    assert done_event.type == "response.function_call_arguments.done"
    assert done_event.name == "sample_tool"


@pytest.mark.asyncio
async def test_websocket_model_retries_recoverable_connect_errors_before_http_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    http_calls = 0
    http_response = object()
    model = WebsocketResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key="test-key"),
        websocket_retry_options=ModelRequestRetryOptions(attempts=3, max_wait_seconds=0),
    )

    async def create_websocket_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise TimeoutError("temporary connect timeout")

    @asynccontextmanager
    async def request_stream_http(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal http_calls
        http_calls += 1
        yield http_response

    monkeypatch.setattr(model, "_create_websocket_stream", create_websocket_stream)
    monkeypatch.setattr(model, "_request_stream_http", request_stream_http)

    async with model.request_stream([], {}, ModelRequestParameters()) as response:  # type: ignore[arg-type]
        assert response is http_response

    assert attempts == 3
    assert http_calls == 1
    assert model.websocket_fallback_state.failure_count == 1


@pytest.mark.asyncio
async def test_websocket_model_retries_upstream_timeout_before_first_response_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def __init__(self, messages: list[str]) -> None:
            self.sent: list[str] = []
            self.closed = False
            self.close_code: int | None = None
            self.close_reason: str | None = None
            self._messages = iter(messages)

        def __aiter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    attempts = 0
    http_calls = 0
    first_connection = FakeConnection([
        json.dumps({
            "type": "error",
            "message": "Upstream WebSocket connection timed out",
        }),
    ])
    second_connection = FakeConnection([
        json.dumps({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_retry_success",
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
    ])
    connections = [first_connection, second_connection]
    model = WebsocketResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key="test-key"),
        websocket_retry_options=ModelRequestRetryOptions(attempts=3, max_wait_seconds=0),
    )

    async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        return connections.pop(0)

    async def create_websocket_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        return _WebsocketResponseStream(
            url="wss://example.test/responses",
            headers={},
            payload={"type": "response.create"},
            connect=connect,
        )

    @asynccontextmanager
    async def request_stream_http(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal http_calls
        http_calls += 1
        yield object()

    monkeypatch.setattr(model, "_create_websocket_stream", create_websocket_stream)
    monkeypatch.setattr(model, "_request_stream_http", request_stream_http)

    async with model.request_stream([], {}, ModelRequestParameters()) as response:  # type: ignore[arg-type]
        assert response is not None

    assert attempts == 2
    assert http_calls == 0
    assert model.websocket_fallback_state.failure_count == 0
    assert first_connection.closed is True
    assert first_connection.close_code == 1000
    assert first_connection.close_reason == "client exited response stream"
    assert first_connection.sent == [
        '{"type":"response.create"}',
        '{"type":"response.cancel"}',
    ]
    assert second_connection.closed is True


@pytest.mark.asyncio
async def test_websocket_model_can_disable_pre_stream_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    http_response = object()
    model = WebsocketResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key="test-key"),
        websocket_retry_options=ModelRequestRetryOptions(enabled=False),
    )

    async def create_websocket_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        raise TimeoutError("temporary connect timeout")

    @asynccontextmanager
    async def request_stream_http(*args, **kwargs):  # type: ignore[no-untyped-def]
        yield http_response

    monkeypatch.setattr(model, "_create_websocket_stream", create_websocket_stream)
    monkeypatch.setattr(model, "_request_stream_http", request_stream_http)

    async with model.request_stream([], {}, ModelRequestParameters()) as response:  # type: ignore[arg-type]
        assert response is http_response

    assert attempts == 1
