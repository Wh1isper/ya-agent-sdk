from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime

import httpx2 as httpx
from ya_oauth.codex import CODEX_CLIENT_ID, CODEX_TOKEN_ENDPOINT, CodexOAuthClient
from ya_oauth.jwt import account_from_id_token
from ya_oauth.store import OAuthStore
from ya_oauth.types import OAuthAccount, OAuthProviderRecord, OAuthTokens

ACCESS_TOKEN = "fixture-access-token"  # noqa: S105
REFRESH_TOKEN = "fixture-refresh-token"  # noqa: S105
OLD_ACCESS_TOKEN = "fixture-old-access-token"  # noqa: S105
OLD_REFRESH_TOKEN = "fixture-old-refresh-token"  # noqa: S105
NEW_ACCESS_TOKEN = "fixture-new-access-token"  # noqa: S105
OLD_ID_TOKEN = "fixture-old-id-token"  # noqa: S105


def _jwt(payload: dict[str, object]) -> str:
    def enc(data: dict[str, object]) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc(payload)}.sig"


def test_account_from_id_token_parses_codex_claims() -> None:
    token = _jwt({
        "email": "top@example.com",
        "https://api.openai.com/auth": {
            "chatgpt_plan_type": "plus",
            "chatgpt_user_id": "user_123",
            "chatgpt_account_id": "acct_123",
            "chatgpt_account_is_fedramp": True,
        },
    })

    account = account_from_id_token(token)

    assert account.email == "top@example.com"
    assert account.chatgpt_plan_type == "plus"
    assert account.chatgpt_user_id == "user_123"
    assert account.chatgpt_account_id == "acct_123"
    assert account.chatgpt_account_is_fedramp is True


def test_codex_device_code_login_requests_match_reference() -> None:
    id_token = _jwt({
        "https://api.openai.com/profile": {"email": "dev@example.com"},
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123", "chatgpt_plan_type": "pro"},
    })
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/accounts/deviceauth/usercode":
            assert json.loads(request.content) == {"client_id": CODEX_CLIENT_ID}
            return httpx.Response(200, json={"device_auth_id": "dev_1", "user_code": "ABCD", "interval": "1"})
        if request.url.path == "/api/accounts/deviceauth/token":
            assert json.loads(request.content) == {"device_auth_id": "dev_1", "user_code": "ABCD"}
            return httpx.Response(
                200,
                json={"authorization_code": "auth_code", "code_challenge": "challenge", "code_verifier": "verifier"},
            )
        if request.url.path == "/oauth/token":
            body = dict(pair.split("=") for pair in request.content.decode().split("&"))
            assert body["grant_type"] == "authorization_code"
            assert body["code"] == "auth_code"
            assert body["client_id"] == CODEX_CLIENT_ID
            assert body["code_verifier"] == "verifier"
            return httpx.Response(
                200, json={"id_token": id_token, "access_token": ACCESS_TOKEN, "refresh_token": REFRESH_TOKEN}
            )
        return httpx.Response(404)

    client = CodexOAuthClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    device_code, record = client.login_device_code(timeout_seconds=1)

    assert device_code.user_code == "ABCD"
    assert record.tokens.access_token == ACCESS_TOKEN
    assert record.tokens.refresh_token == REFRESH_TOKEN
    assert record.account.email == "dev@example.com"
    assert record.account.chatgpt_account_id == "acct_123"
    assert [request.url.path for request in seen] == [
        "/api/accounts/deviceauth/usercode",
        "/api/accounts/deviceauth/token",
        "/oauth/token",
    ]


def test_codex_refresh_preserves_omitted_token_fields(tmp_path) -> None:
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_new"}})

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CODEX_TOKEN_ENDPOINT
        assert json.loads(request.content) == {
            "client_id": CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": OLD_REFRESH_TOKEN,
        }
        return httpx.Response(200, json={"id_token": id_token, "access_token": NEW_ACCESS_TOKEN})

    store = OAuthStore(tmp_path / "auth.json")
    client = CodexOAuthClient(store=store, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    record = OAuthProviderRecord(
        issuer="https://auth.openai.com",
        client_id=CODEX_CLIENT_ID,
        token_endpoint=CODEX_TOKEN_ENDPOINT,
        tokens=OAuthTokens(
            id_token=OLD_ID_TOKEN,
            access_token=OLD_ACCESS_TOKEN,
            refresh_token=OLD_REFRESH_TOKEN,
        ),
        last_refresh_at=datetime.now(UTC),
    )

    refreshed = client.refresh_record(record)

    assert refreshed.tokens.id_token == id_token
    assert refreshed.tokens.access_token == NEW_ACCESS_TOKEN
    assert refreshed.tokens.refresh_token == OLD_REFRESH_TOKEN
    assert refreshed.account.chatgpt_account_id == "acct_new"


def test_codex_refresh_rejects_account_mismatch(tmp_path) -> None:
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_new"}})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id_token": id_token, "access_token": NEW_ACCESS_TOKEN})

    store = OAuthStore(tmp_path / "auth.json")
    client = CodexOAuthClient(store=store, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    record = OAuthProviderRecord(
        issuer="https://auth.openai.com",
        client_id=CODEX_CLIENT_ID,
        token_endpoint=CODEX_TOKEN_ENDPOINT,
        tokens=OAuthTokens(access_token=OLD_ACCESS_TOKEN, refresh_token=OLD_REFRESH_TOKEN),
        account=OAuthAccount(chatgpt_account_id="acct_old"),
    )

    import pytest

    with pytest.raises(RuntimeError, match="different ChatGPT account"):
        client.refresh_record(record)


def test_store_permissions(tmp_path) -> None:
    store = OAuthStore(tmp_path / ".yaai" / "auth.json")
    store.save(store.load())

    if os.name == "posix":
        assert oct(store.path.parent.stat().st_mode & 0o777) == "0o700"
        assert oct(store.path.stat().st_mode & 0o777) == "0o600"
