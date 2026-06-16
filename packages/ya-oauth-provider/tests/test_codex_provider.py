from __future__ import annotations

import json

import anyio
import httpx
import pytest
from pydantic_ai.exceptions import UserError
from ya_oauth.types import OAuthAccount, TokenSnapshot
from ya_oauth_provider.codex import CodexResponsesModel, build_codex_model, build_session_headers
from ya_oauth_provider.http import OAuthBearerAuth, build_codex_headers

ACCESS_TOKEN_OLD = "fixture-access-token-old"  # noqa: S105
ACCESS_TOKEN_NEW = "fixture-access-token-new"  # noqa: S105


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
            account=OAuthAccount(chatgpt_account_id="acct_456"),
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


def test_build_codex_model_requires_streaming_for_non_stream_request() -> None:
    model = build_codex_model("gpt-5.5", token_source=FakeTokenSource())

    assert isinstance(model, CodexResponsesModel)
    with pytest.raises(UserError, match="requires streaming"):
        anyio.run(model.request, [], None, None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_fills_codex_responses_instructions() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        assert "version" not in request.headers
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
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
async def test_oauth_bearer_auth_strips_codex_response_token_limits() -> None:
    source = FakeTokenSource()
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
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

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(json.loads(request.content)))
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex"),
    )

    await client.post("https://example.com/v1/responses", json={"model": "gpt-5.5", "max_tokens": 4096})
    await client.aclose()

    assert seen == [{"model": "gpt-5.5", "max_tokens": 4096}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_refreshes_once_on_401() -> None:
    source = FakeTokenSource()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            return httpx.Response(401, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=OAuthBearerAuth(source, provider_name="codex", extra_headers={"session_id": "s1"}),
    )

    response = await client.get("https://example.com/test")
    await client.aclose()

    assert response.status_code == 200
    assert seen == [f"Bearer {ACCESS_TOKEN_OLD}", f"Bearer {ACCESS_TOKEN_NEW}"]
    assert source.refresh_count == 1
