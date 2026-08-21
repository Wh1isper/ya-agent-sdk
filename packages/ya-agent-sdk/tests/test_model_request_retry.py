from __future__ import annotations

import httpx
import httpx2
import pytest
from openai import APIStatusError
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.agents.models.utils import (
    DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES,
    ModelRequestRetryOptions,
    create_async_http_client,
    create_model_request_retry_transport,
    env_model_request_retry_options,
    is_retryable_model_stream_exception,
    validate_model_retry_response,
)


def test_env_model_request_retry_options_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "YA_AGENT_MODEL_REQUEST_RETRY_ENABLED",
        "YA_AGENT_MODEL_REQUEST_RETRY_ATTEMPTS",
        "YA_AGENT_MODEL_REQUEST_RETRY_BACKOFF_MULTIPLIER",
        "YA_AGENT_MODEL_REQUEST_RETRY_MAX_WAIT_SECONDS",
        "YA_AGENT_MODEL_REQUEST_RETRY_AFTER_MAX_WAIT_SECONDS",
        "YA_AGENT_MODEL_REQUEST_RETRY_STATUS_CODES",
    ):
        monkeypatch.delenv(name, raising=False)

    options = env_model_request_retry_options()

    assert options.enabled is True
    assert options.attempts == 5
    assert options.status_codes == DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES


def test_env_model_request_retry_options_can_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_AGENT_MODEL_REQUEST_RETRY_ENABLED", "false")

    options = env_model_request_retry_options()

    assert options.enabled is False
    assert create_model_request_retry_transport(retry_options=options) is None


def test_env_model_request_retry_options_parses_status_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_AGENT_MODEL_REQUEST_RETRY_STATUS_CODES", "429, 502")

    options = env_model_request_retry_options()

    assert options.status_codes == frozenset({429, 502})


def test_validate_model_retry_response_only_raises_retryable_statuses() -> None:
    request = httpx2.Request("POST", "https://example.test/model")
    options = ModelRequestRetryOptions(status_codes=frozenset({429}))

    validate_model_retry_response(httpx2.Response(500, request=request), options)
    with pytest.raises(httpx2.HTTPStatusError):
        validate_model_retry_response(httpx2.Response(429, request=request), options)


@pytest.mark.parametrize("transport_error_type", [httpx.RemoteProtocolError, httpx2.RemoteProtocolError])
def test_model_stream_retry_recognizes_wrapped_incomplete_chunked_read(transport_error_type) -> None:
    transport_error = transport_error_type("peer closed connection without sending complete message body")
    provider_error = RuntimeError("model provider stream failed")
    provider_error.__cause__ = transport_error

    assert is_retryable_model_stream_exception(provider_error) is True


def test_model_stream_retry_recognizes_official_provider_status_error() -> None:
    request = httpx2.Request("POST", "https://example.test/model")
    response = httpx2.Response(503, request=request)
    error = APIStatusError("temporarily unavailable", response=response, body={})

    assert is_retryable_model_stream_exception(error) is True


def test_model_stream_retry_recognizes_provider_retry_signal() -> None:
    error = UnexpectedModelBehavior("An error occurred while processing your request. You can retry your request.")

    assert is_retryable_model_stream_exception(error) is True


def test_model_stream_retry_rejects_permanent_unexpected_model_behavior() -> None:
    error = UnexpectedModelBehavior("Cannot apply a text delta to an existing tool call part")

    assert is_retryable_model_stream_exception(error) is False


def test_model_stream_retry_rejects_unexpected_behavior_wrapping_permanent_status() -> None:
    error = UnexpectedModelBehavior("An error occurred while processing your request. You can retry your request.")
    error.__cause__ = ModelHTTPError(400, "openai-responses:gpt-5", body={"error": "invalid request"})

    assert is_retryable_model_stream_exception(error) is False


def test_model_stream_retry_rejects_permanent_status_despite_transport_context() -> None:
    request = httpx2.Request("POST", "https://example.test/model")
    error = httpx2.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx2.Response(400, request=request),
    )
    error.__context__ = OSError("earlier websocket failure")

    assert is_retryable_model_stream_exception(error) is False


def test_model_stream_retry_requires_every_exception_group_branch_to_be_transient() -> None:
    mixed_error = ExceptionGroup(
        "model stream and host failure",
        [httpx2.RemoteProtocolError("incomplete chunked read"), ValueError("invalid event")],
    )
    transport_error = ExceptionGroup(
        "model transport failures",
        [httpx2.RemoteProtocolError("incomplete chunked read"), TimeoutError("timed out")],
    )

    assert is_retryable_model_stream_exception(mixed_error) is False
    assert is_retryable_model_stream_exception(transport_error) is True


@pytest.mark.asyncio
async def test_httpx2_retry_transport_replays_byte_request_body() -> None:
    attempts: list[bytes] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(await request.aread())
        status_code = 503 if len(attempts) == 1 else 200
        return httpx2.Response(status_code, request=request)

    options = ModelRequestRetryOptions(
        attempts=2,
        backoff_multiplier=0,
        max_wait_seconds=0,
        retry_after_max_wait_seconds=0,
    )
    transport = create_model_request_retry_transport(
        retry_options=options,
        wrapped=httpx2.MockTransport(handler),
    )
    assert transport is not None

    async with httpx2.AsyncClient(transport=transport) as client:
        response = await client.post("https://example.test/model", content=b'{"prompt":"hello"}')

    assert response.status_code == 200
    assert attempts == [b'{"prompt":"hello"}', b'{"prompt":"hello"}']


def test_create_async_http_client_uses_retry_transport_by_default() -> None:
    client = create_async_http_client()

    try:
        assert isinstance(client._transport, AsyncHTTPX2TenacityTransport)  # pyright: ignore[reportPrivateUsage]
    finally:
        # The default transport has no network resources until opened.
        pass


def test_create_async_http_client_can_disable_retry_transport() -> None:
    client = create_async_http_client(retry_options=ModelRequestRetryOptions(enabled=False))

    assert not isinstance(client._transport, AsyncHTTPX2TenacityTransport)  # pyright: ignore[reportPrivateUsage]


def test_infer_model_openai_chat_uses_retrying_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = infer_model("openai-chat:gpt-4o")

    assert isinstance(model.provider, OpenAIProvider)
    assert isinstance(model.provider.client._client._transport, AsyncHTTPX2TenacityTransport)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_github_provider_keeps_legacy_httpx_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_API_KEY", "test-key")
    with pytest.warns(UserWarning, match="GitHubProvider.*deprecated"):
        model = infer_model("github:gpt-4o")
    first_client = model.provider.client._client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(first_client, httpx.AsyncClient)

    async with model:
        assert first_client.is_closed is False
    assert first_client.is_closed is True

    async with model:
        second_client = model.provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert second_client is not first_client
        assert isinstance(second_client, httpx.AsyncClient)
        assert second_client.is_closed is False
    assert second_client.is_closed is True


@pytest.mark.asyncio
async def test_infer_model_owned_httpx2_client_closes_and_recreates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = infer_model("openai-chat:gpt-4o")
    assert isinstance(model.provider, OpenAIProvider)
    first_client = model.provider.client._client  # pyright: ignore[reportPrivateUsage]

    async with model:
        assert first_client.is_closed is False
    assert first_client.is_closed is True

    async with model:
        second_client = model.provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert second_client is not first_client
        assert second_client.is_closed is False
    assert second_client.is_closed is True
