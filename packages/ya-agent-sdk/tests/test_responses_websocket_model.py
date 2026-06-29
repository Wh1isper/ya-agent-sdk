from __future__ import annotations

import json

import pytest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.providers.openai import OpenAIProvider
from ya_agent_sdk.agents.models.websocket import (
    DEFAULT_WEBSOCKET_BETA,
    DEFAULT_WEBSOCKET_MAX_SIZE,
    WebsocketResponsesModel,
    _WebsocketResponseStream,
    responses_websocket_url,
)


def test_responses_websocket_url() -> None:
    assert responses_websocket_url("https://api.openai.com/v1") == "wss://api.openai.com/v1/responses"
    assert responses_websocket_url("http://localhost:8080/v1") == "ws://localhost:8080/v1/responses"
    assert responses_websocket_url("wss://example.test/custom", path="events") == "wss://example.test/custom/events"


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

        async def close(self) -> None:
            self.closed = True

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
    assert fake.sent == ['{"type":"response.create"}']
    assert connect_kwargs["max_size"] == DEFAULT_WEBSOCKET_MAX_SIZE


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

    done_event = events[-1]
    assert done_event.type == "response.function_call_arguments.done"
    assert done_event.name == "sample_tool"
