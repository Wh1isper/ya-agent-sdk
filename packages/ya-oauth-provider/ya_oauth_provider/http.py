from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx2
from pydantic_ai.models import get_user_agent
from ya_agent_sdk.agents.models.websocket import DEFAULT_WEBSOCKET_BETA
from ya_oauth.types import OAuthAccount, OAuthTokenSource, TokenSnapshot

_OAUTH_MANAGED_HEADERS = (
    "authorization",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "originator",
)

_RESERVED_EXTRA_HEADERS = {
    "authorization",
    "proxy-authorization",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "originator",
    "version",
}

CODEX_ORIGINATOR = "ya_agent_sdk"
CODEX_WEBSOCKET_BETA = DEFAULT_WEBSOCKET_BETA
CODEX_RESPONSE_TOKEN_LIMIT_FIELDS = frozenset({
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
})


class OAuthBearerAuth(httpx2.Auth):
    """httpx2 auth flow that attaches OAuth bearer headers and refreshes once on 401."""

    def __init__(
        self, token_source: OAuthTokenSource, *, provider_name: str, extra_headers: dict[str, str] | None = None
    ) -> None:
        self.token_source = token_source
        self.provider_name = provider_name
        self.extra_headers = _safe_extra_headers(extra_headers)

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        # Buffer once so Codex payload normalization and a possible 401 replay see
        # the same complete, replayable request body even for async input streams.
        await request.aread()
        snapshot = await self.token_source.get_token()
        self._prepare_request(request)
        self._apply_headers(request, snapshot)
        response = yield request
        if response.status_code != 401:
            return

        # Release the rejected response before token refresh performs I/O.
        await response.aread()
        await response.aclose()
        refreshed = await self.token_source.refresh_token()
        retry = _clone_request(request)
        self._prepare_request(retry)
        self._apply_headers(retry, refreshed)
        yield retry

    def _prepare_request(self, request: httpx2.Request) -> None:
        if self.provider_name == "codex":
            _ensure_codex_responses_instructions(request)

    def _apply_headers(self, request: httpx2.Request, snapshot: TokenSnapshot) -> None:
        for header_name in _OAUTH_MANAGED_HEADERS:
            request.headers.pop(header_name, None)
        request.headers.update(
            build_oauth_headers(snapshot, provider_name=self.provider_name, extra_headers=self.extra_headers)
        )


def build_oauth_headers(
    snapshot: TokenSnapshot,
    *,
    provider_name: str,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build OAuth request headers for HTTP or WebSocket transports."""
    headers: dict[str, str] = {"Authorization": f"Bearer {snapshot.access_token}"}
    if provider_name == "codex":
        headers.update(build_codex_headers(snapshot.account, extra_headers=extra_headers))
    else:
        headers.update(_safe_extra_headers(extra_headers))
    return headers


def build_codex_headers(
    account: OAuthAccount,
    *,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build Codex-compatible request headers."""
    headers: dict[str, str] = {
        "originator": CODEX_ORIGINATOR,
    }
    if account.chatgpt_account_id:
        headers["ChatGPT-Account-ID"] = account.chatgpt_account_id
    if account.chatgpt_account_is_fedramp:
        headers["X-OpenAI-Fedramp"] = "true"
    headers.update(_safe_extra_headers(extra_headers))
    return headers


async def build_codex_websocket_headers(
    token_source: OAuthTokenSource,
    *,
    extra_headers: Mapping[str, str] | None = None,
    refresh: bool = False,
) -> dict[str, str]:
    """Build Codex Responses WebSocket handshake headers."""
    snapshot = await (token_source.refresh_token() if refresh else token_source.get_token())
    headers = build_oauth_headers(snapshot, provider_name="codex", extra_headers=extra_headers)
    headers.setdefault("User-Agent", get_user_agent())
    return headers


def _safe_extra_headers(extra_headers: Mapping[str, str] | None) -> dict[str, str]:
    safe_headers: dict[str, str] = {}
    for key, value in (extra_headers or {}).items():
        if key.lower() in _RESERVED_EXTRA_HEADERS:
            raise ValueError(f"extra_headers may not override reserved OAuth/Codex header: {key}")
        safe_headers[key] = value
    return safe_headers


def _ensure_codex_responses_instructions(request: httpx2.Request) -> None:
    """Align Codex Responses API request body requirements."""
    if request.method.upper() != "POST" or request.url.path.rstrip("/") != "/backend-api/codex/responses":
        return

    try:
        body = json.loads(request.content or b"{}")
    except (json.JSONDecodeError, httpx2.RequestNotRead):
        return
    if not isinstance(body, dict):
        return

    normalized = normalize_codex_http_payload(body)
    if normalized != body:
        _replace_json_body(request, normalized)


def normalize_codex_http_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Codex Responses HTTP payload consistently with WebSocket payloads."""
    normalized = dict(body)
    if not normalized.get("instructions"):
        normalized["instructions"] = ""
    normalized["store"] = False
    for field in CODEX_RESPONSE_TOKEN_LIMIT_FIELDS:
        normalized.pop(field, None)
    return normalized


def _replace_json_body(request: httpx2.Request, body: dict[str, Any]) -> None:
    content = json.dumps(body, separators=(",", ":")).encode()
    request.stream = httpx2.ByteStream(content)
    request._content = content
    request.headers.pop("Transfer-Encoding", None)
    request.headers["Content-Length"] = str(len(content))


def _clone_request(request: httpx2.Request) -> httpx2.Request:
    return httpx2.Request(
        method=request.method,
        url=request.url,
        headers=request.headers.copy(),
        content=request.content,
        extensions=request.extensions.copy(),
    )
