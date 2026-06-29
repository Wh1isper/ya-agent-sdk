from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args
from urllib.parse import urlparse, urlunparse

import httpx
import websockets
from openai import APIStatusError
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse, check_allow_model_requests
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openai import _drop_sampling_params_for_reasoning as _openai_drop_sampling_params_for_reasoning
from pydantic_ai.models.openai import _drop_unsupported_params as _openai_drop_unsupported_params
from pydantic_ai.models.openai import _resolve_openai_service_tier as _openai_resolve_service_tier
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)

ResponsesWebsocketMode = Literal["auto", "websocket", "http"]

_RESPONSES_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)
_RESPONSE_CREATE_TYPE = "response.create"
DEFAULT_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
_WS_DISABLE_TTL_SECONDS = 300.0
_RECOVERABLE_HTTP_STATUS_CODES = {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504}

HeaderBuilder = Callable[[Mapping[str, str]], Awaitable[dict[str, str]]]
PayloadNormalizer = Callable[[dict[str, Any]], dict[str, Any]]
WebsocketConnect = Callable[..., Awaitable[Any]]


@dataclass
class ResponsesWebsocketFallbackState:
    """Small per-model state for WebSocket auto fallback."""

    mode: ResponsesWebsocketMode = "auto"
    disabled_ttl_seconds: float = _WS_DISABLE_TTL_SECONDS
    disabled_until: float | None = None
    last_error: str | None = None
    failure_count: int = 0

    def should_use_websocket(self) -> bool:
        if self.mode == "http":
            return False
        if self.mode == "websocket":
            return True
        if self.disabled_until is None:
            return True
        return time.monotonic() >= self.disabled_until

    def mark_success(self) -> None:
        self.disabled_until = None
        self.last_error = None

    def mark_failure(self, exc: BaseException) -> None:
        self.failure_count += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.mode == "auto":
            self.disabled_until = time.monotonic() + self.disabled_ttl_seconds


def resolve_websocket_mode(value: str | None, *, default: ResponsesWebsocketMode = "auto") -> ResponsesWebsocketMode:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized not in get_args(ResponsesWebsocketMode):
        raise ValueError("Responses websocket mode must be one of: auto, websocket, http")
    return cast(ResponsesWebsocketMode, normalized)


def env_responses_websocket_mode(env_name: str, *, default: ResponsesWebsocketMode = "auto") -> ResponsesWebsocketMode:
    return resolve_websocket_mode(os.getenv(env_name), default=default)


def responses_websocket_url(base_url: str, *, path: str = "responses") -> str:
    parsed = urlparse(str(base_url).rstrip("/"))
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    else:
        raise ValueError(f"Unsupported Responses WebSocket base URL scheme: {parsed.scheme!r}")
    response_path = "/".join(part.strip("/") for part in (parsed.path, path) if part.strip("/"))
    return urlunparse((scheme, parsed.netloc, f"/{response_path}", "", parsed.query, parsed.fragment))


def _json_compatible(value: Any) -> Any:
    if _is_omitted(value):
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items() if not _is_omitted(v)}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value if not _is_omitted(item)]
    return value


def _is_omitted(value: Any) -> bool:
    return value.__class__.__name__ in {"Omit", "NotGiven"}


def normalize_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Responses payload for JSON websocket transport."""
    normalized = {key: _json_compatible(value) for key, value in payload.items() if not _is_omitted(value)}
    return {key: value for key, value in normalized.items() if value is not None}


class _WebsocketResponseStream:
    """Async stream wrapper compatible with Pydantic AI's OpenAI Responses stream adapter."""

    def __init__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        open_timeout: float = 10.0,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        connect: WebsocketConnect = websockets.connect,
    ) -> None:
        self.url = url
        self.headers = dict(headers)
        self.payload = dict(payload)
        self.open_timeout = open_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._connect = connect
        self._connection: Any | None = None
        self._events_seen = 0
        self._function_call_names_by_item_id: dict[str, str] = {}

    @property
    def events_seen(self) -> int:
        return self._events_seen

    async def __aenter__(self) -> _WebsocketResponseStream:
        connection = await self._connect(
            self.url,
            additional_headers=self.headers,
            user_agent_header=None,
            open_timeout=self.open_timeout,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
        )
        self._connection = connection
        await connection.send(json.dumps(self.payload, separators=(",", ":")))
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[ResponseStreamEvent]:
        if self._connection is None:
            raise RuntimeError("WebSocket response stream has not been entered")
        async for raw_message in self._connection:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            data = json.loads(raw_message)
            if not isinstance(data, dict):
                raise UnexpectedModelBehavior("Responses WebSocket event must be a JSON object")
            event_type = data.get("type")
            if event_type == "error":
                raise _websocket_error_from_event(data)
            if isinstance(event_type, str) and not event_type.startswith("response."):
                logger.debug("Ignoring non-Responses WebSocket event: %s", event_type)
                continue
            data = self._normalize_stream_event(data)
            event = _RESPONSES_STREAM_EVENT_ADAPTER.validate_python(data)
            self._events_seen += 1
            yield event
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                await self.close()
                return

    def _normalize_stream_event(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize gateway WebSocket event shapes to OpenAI SDK canonical events."""
        event_type = data.get("type")
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._record_function_call_name(data.get("item"))
        elif event_type == "response.function_call_arguments.done" and "name" not in data:
            item_id = data.get("item_id")
            if isinstance(item_id, str):
                name = self._function_call_names_by_item_id.get(item_id)
                if name:
                    data = dict(data)
                    data["name"] = name
        return data

    def _record_function_call_name(self, item: object) -> None:
        if not isinstance(item, Mapping):
            return
        if item.get("type") != "function_call":
            return
        item_id = item.get("id")
        name = item.get("name")
        if isinstance(item_id, str) and isinstance(name, str):
            self._function_call_names_by_item_id[item_id] = name

    async def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            await connection.close()


@dataclass(init=False)
class WebsocketResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model with YA-owned WebSocket transport and HTTP fallback."""

    _websocket_base_url: str = field(repr=False)
    _websocket_headers_builder: HeaderBuilder | None = field(default=None, repr=False)
    _payload_normalizer: PayloadNormalizer = field(default=normalize_responses_payload, repr=False)
    _fallback_state: ResponsesWebsocketFallbackState = field(repr=False)
    _websocket_beta: str | None = field(default=None, repr=False)
    _websocket_open_timeout: float = field(default=10.0, repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAIProvider,
        profile: OpenAIModelProfile | None = None,
        websocket_base_url: str | None = None,
        websocket_headers_builder: HeaderBuilder | None = None,
        payload_normalizer: PayloadNormalizer = normalize_responses_payload,
        websocket_mode: ResponsesWebsocketMode = "auto",
        websocket_beta: str | None = DEFAULT_WEBSOCKET_BETA,
        websocket_open_timeout: float = 10.0,
    ) -> None:
        super().__init__(model_name, provider=provider, profile=profile)
        base_url = websocket_base_url or str(self._provider.base_url)
        self._websocket_base_url = base_url
        self._websocket_headers_builder = websocket_headers_builder
        self._payload_normalizer = payload_normalizer
        self._fallback_state = ResponsesWebsocketFallbackState(mode=websocket_mode)
        self._websocket_beta = websocket_beta
        self._websocket_open_timeout = websocket_open_timeout

    @property
    def websocket_fallback_state(self) -> ResponsesWebsocketFallbackState:
        return self._fallback_state

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Non-streaming WS is deliberately handled by the stable HTTP path for now.
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        prepared_settings, prepared_parameters = self.prepare_request(model_settings, model_request_parameters)
        settings = cast(OpenAIResponsesModelSettings, prepared_settings or {})
        if not self._fallback_state.should_use_websocket():
            async with self._request_stream_http(messages, settings, prepared_parameters, run_context) as response:
                yield response
            return

        stream: _WebsocketResponseStream | None = None
        try:
            stream = await self._create_websocket_stream(
                cast(list[ModelRequest | ModelResponse], messages),
                settings,
                prepared_parameters,
            )
            async with stream:
                response = await self._process_streamed_response(
                    cast(Any, stream),
                    settings,
                    prepared_parameters,
                )
                self._fallback_state.mark_success()
                yield response
        except BaseException as exc:
            if self._should_fallback_from(exc, stream):
                self._fallback_state.mark_failure(exc)
                logger.warning("Responses WebSocket failed before streaming; falling back to HTTP: %s", exc)
                async with self._request_stream_http(messages, settings, prepared_parameters, run_context) as response:
                    yield response
                return
            if stream is not None and stream.events_seen > 0:
                self._fallback_state.mark_failure(exc)
            raise

    @asynccontextmanager
    async def _request_stream_http(
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None,
    ) -> AsyncIterator[StreamedResponse]:
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as response:
            yield response

    async def _create_websocket_stream(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> _WebsocketResponseStream:
        payload = await self._build_websocket_payload(messages, model_settings, model_request_parameters)
        extra_headers, timeout = self._build_request_options(model_settings)
        headers = await self._build_websocket_headers(extra_headers)
        if timeout is not None and not _is_omitted(timeout):
            # The WebSocket API has handshake timeout only; per-response timeout is not safely supported here.
            pass
        return _WebsocketResponseStream(
            url=responses_websocket_url(self._websocket_base_url),
            headers=headers,
            payload=payload,
            open_timeout=self._websocket_open_timeout,
        )

    async def _build_websocket_payload(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> dict[str, Any]:
        profile = self.profile
        request_params = await self._build_responses_request_params(
            messages, model_settings, model_request_parameters, profile
        )
        _openai_drop_sampling_params_for_reasoning(profile, model_settings, model_request_parameters)
        _openai_drop_unsupported_params(profile, model_settings)
        include: list[str] = []
        if profile.get("openai_supports_encrypted_reasoning_content", False):
            include.append("reasoning.encrypted_content")
        if model_settings.get("openai_include_code_execution_outputs"):
            include.append("code_interpreter_call.outputs")
        if model_settings.get("openai_include_web_search_sources"):
            include.append("web_search_call.action.sources")
        if model_settings.get("openai_include_file_search_results"):
            include.append("file_search_call.results")
        if model_settings.get("openai_logprobs"):
            include.append("message.output_text.logprobs")

        payload = {
            "type": _RESPONSE_CREATE_TYPE,
            "model": request_params.model,
            "input": request_params.input,
            "instructions": request_params.instructions,
            "parallel_tool_calls": request_params.parallel_tool_calls,
            "tools": request_params.tools,
            "tool_choice": request_params.tool_choice,
            "previous_response_id": request_params.previous_response_id,
            "reasoning": request_params.reasoning,
            "text": request_params.text,
            "truncation": request_params.truncation,
            "context_management": request_params.context_management,
            "max_output_tokens": model_settings.get("max_tokens"),
            "stream": True,
            "temperature": model_settings.get("temperature"),
            "top_p": model_settings.get("top_p"),
            "service_tier": _openai_resolve_service_tier(model_settings),
            "conversation": request_params.conversation,
            "top_logprobs": model_settings.get("openai_top_logprobs"),
            "store": model_settings.get("openai_store"),
            "user": model_settings.get("openai_user"),
            "include": include or None,
            "prompt_cache_key": model_settings.get("openai_prompt_cache_key"),
            "prompt_cache_retention": model_settings.get("openai_prompt_cache_retention"),
        }
        extra_body = model_settings.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        return self._payload_normalizer(payload)

    async def _build_websocket_headers(self, extra_headers: Mapping[str, str]) -> dict[str, str]:
        if self._websocket_headers_builder is not None:
            headers = await self._websocket_headers_builder(extra_headers)
        else:
            headers = self._default_websocket_headers(extra_headers)
        if self._websocket_beta:
            headers = _merge_openai_beta_header(headers, self._websocket_beta)
        return headers

    def _default_websocket_headers(self, extra_headers: Mapping[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {}
        default_headers = getattr(self.client, "default_headers", {})
        if isinstance(default_headers, Mapping):
            for key, value in default_headers.items():
                if value is None or _is_omitted(value):
                    continue
                headers[str(key)] = str(value)
        headers.update(extra_headers)
        return headers

    def _should_fallback_from(self, exc: BaseException, stream: _WebsocketResponseStream | None) -> bool:
        if self._fallback_state.mode != "auto":
            return False
        if stream is not None and stream.events_seen > 0:
            return False
        return is_recoverable_websocket_error(exc)


def _merge_openai_beta_header(headers: dict[str, str], beta: str) -> dict[str, str]:
    existing_key = next((key for key in headers if key.lower() == "openai-beta"), "OpenAI-Beta")
    existing_beta = headers.get(existing_key)
    if not existing_beta:
        headers[existing_key] = beta
        return headers
    existing_parts = {part.strip() for part in existing_beta.split(",")}
    if beta not in existing_parts:
        headers[existing_key] = f"{existing_beta},{beta}"
    return headers


def is_recoverable_websocket_error(exc: BaseException) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _RECOVERABLE_HTTP_STATUS_CODES
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RECOVERABLE_HTTP_STATUS_CODES
    return isinstance(exc, (TimeoutError, OSError, httpx.HTTPError, websockets.WebSocketException))


def _websocket_error_from_event(data: Mapping[str, Any]) -> BaseException:
    status = data.get("status") or data.get("status_code") or data.get("code")
    message = data.get("message") or data.get("error") or data.get("type") or "Responses WebSocket error"
    error_obj = data.get("error")
    if isinstance(error_obj, Mapping):
        message = str(error_obj.get("message") or message)
        status = error_obj.get("status") or error_obj.get("status_code") or status
    if isinstance(status, int):
        return ModelHTTPError(status_code=status, model_name="responses-websocket", body=data)
    return UnexpectedModelBehavior(str(message))


def build_openai_responses_websocket_model(
    model_name: str,
    *,
    websocket_mode: ResponsesWebsocketMode | None = None,
) -> WebsocketResponsesModel:
    from pydantic_ai.providers.openai import OpenAIProvider

    mode = websocket_mode or env_responses_websocket_mode("YA_AGENT_OPENAI_RESPONSES_WEBSOCKET_MODE", default="auto")
    return WebsocketResponsesModel(
        model_name,
        provider=OpenAIProvider(),
        websocket_mode=mode,
    )


__all__ = [
    "DEFAULT_WEBSOCKET_BETA",
    "ResponsesWebsocketFallbackState",
    "ResponsesWebsocketMode",
    "WebsocketResponsesModel",
    "build_openai_responses_websocket_model",
    "env_responses_websocket_mode",
    "is_recoverable_websocket_error",
    "normalize_responses_payload",
    "resolve_websocket_mode",
    "responses_websocket_url",
]
