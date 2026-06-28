from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx
from pydantic_ai.models import get_user_agent
from ya_oauth.types import OAuthAccount, OAuthTokenSource, TokenSnapshot

_RESERVED_EXTRA_HEADERS = {
    "authorization",
    "proxy-authorization",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "originator",
    "version",
}

CODEX_ORIGINATOR = "ya_agent_sdk"
CODEX_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
CODEX_RESPONSE_TOKEN_LIMIT_FIELDS = frozenset({
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
})


class OAuthBearerAuth(httpx.Auth):
    """httpx auth flow that attaches OAuth bearer headers and refreshes once on 401."""

    requires_response_body = True

    def __init__(
        self, token_source: OAuthTokenSource, *, provider_name: str, extra_headers: dict[str, str] | None = None
    ) -> None:
        self.token_source = token_source
        self.provider_name = provider_name
        self.extra_headers = _safe_extra_headers(extra_headers)

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        snapshot = await self.token_source.get_token()
        self._prepare_request(request)
        self._apply_headers(request, snapshot)
        response = yield request
        if response.status_code != 401:
            return
        refreshed = await self.token_source.refresh_token()
        retry = _clone_request(request)
        self._prepare_request(retry)
        self._apply_headers(retry, refreshed)
        yield retry

    def _prepare_request(self, request: httpx.Request) -> None:
        if self.provider_name == "codex":
            _ensure_codex_responses_instructions(request)

    def _apply_headers(self, request: httpx.Request, snapshot: TokenSnapshot) -> None:
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
    existing_beta = headers.get("OpenAI-Beta") or headers.get("openai-beta")
    headers["OpenAI-Beta"] = CODEX_WEBSOCKET_BETA if not existing_beta else f"{existing_beta},{CODEX_WEBSOCKET_BETA}"
    return headers


def _safe_extra_headers(extra_headers: Mapping[str, str] | None) -> dict[str, str]:
    safe_headers: dict[str, str] = {}
    for key, value in (extra_headers or {}).items():
        if key.lower() in _RESERVED_EXTRA_HEADERS:
            raise ValueError(f"extra_headers may not override reserved OAuth/Codex header: {key}")
        safe_headers[key] = value
    return safe_headers


def _ensure_codex_responses_instructions(request: httpx.Request) -> None:
    """Align Codex Responses API request body requirements."""
    if request.method.upper() != "POST" or request.url.path.rstrip("/") != "/backend-api/codex/responses":
        return

    try:
        body = json.loads(request.content or b"{}")
    except (json.JSONDecodeError, httpx.RequestNotRead):
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


def _replace_json_body(request: httpx.Request, body: dict[str, Any]) -> None:
    content = json.dumps(body, separators=(",", ":")).encode()
    request.stream = httpx.ByteStream(content)
    request._content = content
    request.headers["Content-Length"] = str(len(content))


def _clone_request(request: httpx.Request) -> httpx.Request:
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=request.headers.copy(),
        content=request.content,
        extensions=request.extensions.copy(),
    )
