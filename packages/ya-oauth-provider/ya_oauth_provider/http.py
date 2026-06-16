from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
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
        request.headers["Authorization"] = f"Bearer {snapshot.access_token}"
        if self.provider_name == "codex":
            request.headers.update(build_codex_headers(snapshot.account, extra_headers=self.extra_headers))
        else:
            request.headers.update(self.extra_headers)


def build_codex_headers(
    account: OAuthAccount,
    *,
    extra_headers: dict[str, str] | None = None,
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


def _safe_extra_headers(extra_headers: dict[str, str] | None) -> dict[str, str]:
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

    changed = False
    instructions = body.get("instructions")
    if not instructions:
        body["instructions"] = ""
        changed = True
    if body.get("store") is not False:
        body["store"] = False
        changed = True
    for field in CODEX_RESPONSE_TOKEN_LIMIT_FIELDS:
        if field in body:
            del body[field]
            changed = True
    if changed:
        _replace_json_body(request, body)


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
