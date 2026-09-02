from __future__ import annotations

import json
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from openai import AsyncStream
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.agents.models.websocket import (
    ResponsesWebsocketMode,
    WebsocketResponsesModel,
    _WebsocketResponseStream,
    env_responses_websocket_mode,
    normalize_responses_payload,
)
from ya_oauth.codex import CODEX_BASE_URL, create_codex_token_source
from ya_oauth.types import OAuthTokenSource

from ya_oauth_provider.http import (
    CODEX_RESPONSE_TOKEN_LIMIT_FIELDS,
    CODEX_ROUTING_HINT_HEADER,
    CODEX_TURN_STATE_HEADER,
    OAuthBearerAuth,
    build_codex_websocket_headers,
)


@dataclass
class _CodexTurnState:
    value: str | None = None

    def capture(self, headers: Mapping[str, Any]) -> None:
        if self.value is not None:
            return
        for name in headers:
            if str(name).lower() != CODEX_TURN_STATE_HEADER:
                continue
            try:
                value = headers[name]
            except LookupError:
                return
            if isinstance(value, str):
                normalized = value.strip()
            elif isinstance(value, list) and value and isinstance(value[0], str):
                normalized = value[0].strip()
            else:
                return
            if normalized:
                self.value = normalized
            return


class _CodexRequestHeaders:
    """Codex-only dynamic request headers, scoped by Pydantic AI run ID."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._turn_states: dict[str, tuple[weakref.ReferenceType[RunUsage], _CodexTurnState]] = {}
        self._active_turn_state: ContextVar[_CodexTurnState | None] = ContextVar(
            f"ya_oauth_provider.codex_turn_state.{id(self)}",
            default=None,
        )

    @contextmanager
    def scope(self, run_context: RunContext[Any] | None):
        state = self._state_for_run(run_context)
        token = self._active_turn_state.set(state)
        try:
            yield
        finally:
            self._active_turn_state.reset(token)

    def apply(
        self,
        model_settings: OpenAIResponsesModelSettings,
        *,
        include_turn_state: bool = True,
    ) -> OpenAIResponsesModelSettings:
        settings = OpenAIResponsesModelSettings(**model_settings)
        configured_headers = settings.get("extra_headers")
        headers = (
            {
                str(name): str(value)
                for name, value in configured_headers.items()
                if str(name).lower() not in {CODEX_TURN_STATE_HEADER, CODEX_ROUTING_HINT_HEADER}
            }
            if isinstance(configured_headers, Mapping)
            else {}
        )
        headers[CODEX_ROUTING_HINT_HEADER] = _build_codex_routing_hint(self._model_name, settings)
        state = self.active_turn_state()
        if include_turn_state and state is not None and state.value is not None:
            headers[CODEX_TURN_STATE_HEADER] = state.value
        settings["extra_headers"] = headers
        return settings

    def capture(self, headers: Mapping[str, Any]) -> None:
        state = self.active_turn_state()
        if state is not None:
            state.capture(headers)

    def active_turn_state(self) -> _CodexTurnState | None:
        return self._active_turn_state.get()

    def _state_for_run(self, run_context: RunContext[Any] | None) -> _CodexTurnState | None:
        if not isinstance(run_context, RunContext) or not run_context.run_id:
            return None
        run_id = run_context.run_id
        existing = self._turn_states.get(run_id)
        if existing is not None and existing[0]() is run_context.usage:
            return existing[1]

        state = _CodexTurnState()
        owner_ref = weakref.ref(self)

        def remove_state(usage_ref: weakref.ReferenceType[RunUsage]) -> None:
            owner = owner_ref()
            if owner is None:
                return
            current = owner._turn_states.get(run_id)
            if current is not None and current[0] is usage_ref:
                owner._turn_states.pop(run_id, None)

        usage_ref = weakref.ref(run_context.usage, remove_state)
        self._turn_states[run_id] = (usage_ref, state)
        return state


def _build_codex_routing_hint(model_name: str, model_settings: Mapping[str, Any]) -> str:
    service_tier = model_settings.get("openai_service_tier") or model_settings.get("service_tier")
    if isinstance(service_tier, str) and service_tier:
        return f"model={model_name};tier={service_tier}"
    return f"model={model_name}"


class _CodexMetadataCapturingConnection:
    def __init__(self, connection: Any, request_headers: _CodexRequestHeaders) -> None:
        self._connection = connection
        self._iterator = connection.__aiter__()
        self._request_headers = request_headers

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self

    async def __anext__(self) -> str | bytes:
        raw_message = await anext(self._iterator)
        self._capture_metadata(raw_message)
        return raw_message

    async def send(self, message: str) -> None:
        await self._connection.send(message)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        await self._connection.close(code=code, reason=reason)

    def _capture_metadata(self, raw_message: str | bytes) -> None:
        try:
            text = raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(data, Mapping) or data.get("type") != "response.metadata":
            return
        headers = data.get("headers")
        if isinstance(headers, Mapping):
            self._request_headers.capture(headers)


class _CodexWebsocketResponseStream(_WebsocketResponseStream):
    def __init__(self, *, request_headers: _CodexRequestHeaders, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._codex_request_headers = request_headers

    async def __aenter__(self) -> _CodexWebsocketResponseStream:
        if self._connection is not None:
            return self
        connection = await self._connect(
            self.url,
            additional_headers=self.headers,
            user_agent_header=None,
            open_timeout=self.open_timeout,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=self.max_size,
        )
        self._websocket_upgraded = True
        response = getattr(connection, "response", None)
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            self._codex_request_headers.capture(headers)
        self._connection = _CodexMetadataCapturingConnection(connection, self._codex_request_headers)
        try:
            await self._connection.send(self._payload_json)
            self._create_sent = True
        except BaseException:
            await self._best_effort_cleanup(
                self.close(code=1011, reason="failed to send response.create"),
                action="close Codex Responses WebSocket after failed response.create",
            )
            raise
        return self


def infer_oauth_model(provider_name: str, model_name: str, *, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer an OAuth-backed model from `oauth@provider:model` parts."""
    if provider_name == "codex":
        return build_codex_model(model_name, extra_headers=extra_headers)
    raise KeyError(f"Unknown OAuth provider: {provider_name}")


def build_codex_model(
    model_name: str,
    *,
    token_source: OAuthTokenSource | None = None,
    extra_headers: dict[str, str] | None = None,
    base_url: str = CODEX_BASE_URL,
    websocket_mode: ResponsesWebsocketMode | None = None,
) -> Model:
    """Build a Codex OAuth-backed OpenAI Responses model."""
    import httpx2
    from pydantic_ai.models import get_user_agent
    from ya_agent_sdk.agents.models.utils import (
        create_model_request_retry_transport,
        create_owned_httpx2_provider,
    )

    source = token_source or create_codex_token_source()

    def create_http_client() -> httpx2.AsyncClient:
        transport = create_model_request_retry_transport()
        auth = OAuthBearerAuth(source, provider_name="codex", extra_headers=extra_headers)
        headers = {"User-Agent": get_user_agent()}
        timeout = httpx2.Timeout(timeout=900, connect=5, read=300)
        if transport is None:
            return httpx2.AsyncClient(auth=auth, headers=headers, timeout=timeout)
        return httpx2.AsyncClient(auth=auth, headers=headers, timeout=timeout, transport=transport)

    provider = create_owned_httpx2_provider(
        lambda http_client: OpenAIProvider(api_key="oauth-managed", base_url=base_url, http_client=http_client),
        http_client_factory=create_http_client,
    )
    mode = websocket_mode or env_responses_websocket_mode("YA_OAUTH_CODEX_RESPONSES_TRANSPORT", default="auto")
    if mode == "http":
        return CodexResponsesModel(model_name, provider=provider, profile=_codex_profile())
    return CodexWebsocketResponsesModel(
        model_name,
        provider=provider,
        profile=_codex_profile(),
        token_source=source,
        extra_headers=extra_headers,
        base_url=base_url,
        websocket_mode=mode,
    )


class CodexResponsesModel(OpenAIResponsesModel):
    """Codex Responses API model with OAuth-only routing and per-turn state headers."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAIProvider,
        profile: OpenAIModelProfile | None = None,
    ) -> None:
        super().__init__(model_name, provider=provider, profile=profile)
        self._codex_request_headers = _CodexRequestHeaders(model_name)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise UserError(
            "Codex OAuth Responses API requires streaming. "
            "Use agent.run_stream(), agent.iter(), or ya_agent_sdk.stream_agent()."
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        with self._codex_request_headers.scope(run_context):
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response:
                yield response

    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        response = await super()._responses_create(
            messages,
            stream,
            self._codex_request_headers.apply(model_settings),
            model_request_parameters,
        )
        if isinstance(response, AsyncStream):
            self._codex_request_headers.capture(response.response.headers)
        return response

    async def _responses_retrieve(
        self,
        response_id: str,
        model_settings: OpenAIResponsesModelSettings,
        *,
        stream: bool = False,
        starting_after: int | None = None,
    ) -> Any:
        response = await super()._responses_retrieve(
            response_id,
            self._codex_request_headers.apply(model_settings),
            stream=stream,
            starting_after=starting_after,
        )
        if isinstance(response, AsyncStream):
            self._codex_request_headers.capture(response.response.headers)
        return response


class CodexWebsocketResponsesModel(WebsocketResponsesModel):
    """Codex Responses model that prefers WebSocket and falls back to HTTP."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAIProvider,
        profile: OpenAIModelProfile,
        token_source: OAuthTokenSource,
        extra_headers: Mapping[str, str] | None = None,
        base_url: str = CODEX_BASE_URL,
        websocket_mode: ResponsesWebsocketMode = "auto",
    ) -> None:
        self._codex_token_source = token_source
        self._codex_extra_headers = dict(extra_headers or {})
        self._codex_request_headers = _CodexRequestHeaders(model_name)
        super().__init__(
            model_name,
            provider=provider,
            profile=profile,
            websocket_base_url=base_url,
            websocket_headers_builder=self._build_codex_websocket_headers,
            payload_normalizer=normalize_codex_responses_payload,
            websocket_mode=websocket_mode,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise UserError(
            "Codex OAuth Responses API requires streaming. "
            "Use agent.run_stream(), agent.iter(), or ya_agent_sdk.stream_agent()."
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        with self._codex_request_headers.scope(run_context):
            async with super().request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response:
                yield response

    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        response = await super()._responses_create(
            messages,
            stream,
            self._codex_request_headers.apply(model_settings),
            model_request_parameters,
        )
        if isinstance(response, AsyncStream):
            self._codex_request_headers.capture(response.response.headers)
        return response

    async def _responses_retrieve(
        self,
        response_id: str,
        model_settings: OpenAIResponsesModelSettings,
        *,
        stream: bool = False,
        starting_after: int | None = None,
    ) -> Any:
        response = await super()._responses_retrieve(
            response_id,
            self._codex_request_headers.apply(model_settings),
            stream=stream,
            starting_after=starting_after,
        )
        if isinstance(response, AsyncStream):
            self._codex_request_headers.capture(response.response.headers)
        return response

    async def _create_websocket_stream(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> _WebsocketResponseStream:
        request_settings = self._codex_request_headers.apply(model_settings, include_turn_state=False)
        request_payload = dict(payload) if payload is not None else None
        turn_state = self._codex_request_headers.active_turn_state()
        if request_payload is not None and turn_state is not None and turn_state.value is not None:
            configured_metadata = request_payload.get("client_metadata")
            client_metadata = dict(configured_metadata) if isinstance(configured_metadata, Mapping) else {}
            client_metadata[CODEX_TURN_STATE_HEADER] = turn_state.value
            request_payload["client_metadata"] = client_metadata
        stream = await super()._create_websocket_stream(
            messages,
            request_settings,
            model_request_parameters,
            payload=request_payload,
        )
        return _CodexWebsocketResponseStream(
            url=stream.url,
            headers=stream.headers,
            payload=stream.payload,
            open_timeout=stream.open_timeout,
            ping_interval=stream.ping_interval,
            ping_timeout=stream.ping_timeout,
            max_size=stream.max_size,
            cleanup_timeout=stream.cleanup_timeout,
            connect=stream._connect,
            request_headers=self._codex_request_headers,
        )

    async def _build_codex_websocket_headers(self, extra_headers: Mapping[str, str]) -> dict[str, str]:
        merged = {
            name: value
            for name, value in self._codex_extra_headers.items()
            if name.lower() not in {CODEX_TURN_STATE_HEADER, CODEX_ROUTING_HINT_HEADER}
        }
        merged.update(extra_headers)
        return await build_codex_websocket_headers(self._codex_token_source, extra_headers=merged)


def normalize_codex_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Align Codex Responses payload requirements for both HTTP and WebSocket transports."""
    normalized = normalize_responses_payload(payload)
    if not normalized.get("instructions"):
        normalized["instructions"] = ""
    normalized["store"] = False
    for field_name in CODEX_RESPONSE_TOKEN_LIMIT_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def _codex_profile() -> OpenAIModelProfile:
    return OpenAIModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_thinking=True,
        thinking_always_enabled=True,
        openai_supports_reasoning=True,
        openai_supports_encrypted_reasoning_content=True,
        openai_supports_strict_tool_definition=True,
        openai_responses_requires_function_call_status_none=True,
    )


def build_session_headers(session_id: str | None, thread_id: str | None) -> dict[str, str]:
    """Build Codex session/thread headers with underscore and hyphen variants."""
    headers: dict[str, str] = {}
    if session_id:
        headers["session_id"] = session_id
        headers["session-id"] = session_id
    if thread_id:
        headers["thread_id"] = thread_id
        headers["thread-id"] = thread_id
        headers["x-client-request-id"] = thread_id
    return headers
