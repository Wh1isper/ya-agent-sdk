from __future__ import annotations

import httpx
import pytest
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.agents.models.utils import (
    DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES,
    ModelRequestRetryOptions,
    create_async_http_client,
    create_model_request_retry_transport,
    env_model_request_retry_options,
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
    request = httpx.Request("POST", "https://example.test/model")
    options = ModelRequestRetryOptions(status_codes=frozenset({429}))

    validate_model_retry_response(httpx.Response(500, request=request), options)
    with pytest.raises(httpx.HTTPStatusError):
        validate_model_retry_response(httpx.Response(429, request=request), options)


def test_create_async_http_client_uses_retry_transport_by_default() -> None:
    client = create_async_http_client()

    try:
        assert isinstance(client._transport, AsyncTenacityTransport)  # pyright: ignore[reportPrivateUsage]
    finally:
        # The default transport has no network resources until opened.
        pass


def test_create_async_http_client_can_disable_retry_transport() -> None:
    client = create_async_http_client(retry_options=ModelRequestRetryOptions(enabled=False))

    assert not isinstance(client._transport, AsyncTenacityTransport)  # pyright: ignore[reportPrivateUsage]


def test_infer_model_openai_chat_uses_retrying_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = infer_model("openai-chat:gpt-4o")

    assert isinstance(model.provider, OpenAIProvider)
    assert isinstance(model.provider.client._client._transport, AsyncTenacityTransport)  # pyright: ignore[reportPrivateUsage]
