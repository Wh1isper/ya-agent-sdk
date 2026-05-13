from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import AliasChoices, BaseModel, Field

from ya_oauth.jwt import account_from_id_token
from ya_oauth.store import OAuthStore, StoreBackedTokenSource
from ya_oauth.types import OAuthAccount, OAuthProviderRecord, OAuthTokens

CODEX_ISSUER = "https://auth.openai.com"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"  # noqa: S105
CODEX_REVOKE_ENDPOINT = "https://auth.openai.com/oauth/revoke"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
CODEX_SCOPES = [
    "openid",
    "profile",
    "email",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
]


@dataclass(frozen=True)
class CodexOAuthProfile:
    issuer: str = CODEX_ISSUER
    client_id: str = CODEX_CLIENT_ID
    token_endpoint: str = CODEX_TOKEN_ENDPOINT
    revoke_endpoint: str = CODEX_REVOKE_ENDPOINT
    base_url: str = CODEX_BASE_URL
    scopes: tuple[str, ...] = tuple(CODEX_SCOPES)

    @property
    def device_user_code_endpoint(self) -> str:
        return f"{self.issuer.rstrip('/')}/api/accounts/deviceauth/usercode"

    @property
    def device_token_endpoint(self) -> str:
        return f"{self.issuer.rstrip('/')}/api/accounts/deviceauth/token"

    @property
    def verification_url(self) -> str:
        return f"{self.issuer.rstrip('/')}/codex/device"

    @property
    def device_redirect_uri(self) -> str:
        return f"{self.issuer.rstrip('/')}/deviceauth/callback"


CODEX_PROFILE = CodexOAuthProfile()


class DeviceCode(BaseModel):
    verification_url: str
    user_code: str
    device_auth_id: str
    interval: int = 5


class _UserCodeResponse(BaseModel):
    device_auth_id: str
    user_code: str = Field(validation_alias=AliasChoices("user_code", "usercode"))
    interval: int | str = 5

    def to_device_code(self, profile: CodexOAuthProfile) -> DeviceCode:
        return DeviceCode(
            verification_url=profile.verification_url,
            user_code=self.user_code,
            device_auth_id=self.device_auth_id,
            interval=int(self.interval),
        )


class _DeviceTokenResponse(BaseModel):
    authorization_code: str
    code_challenge: str
    code_verifier: str


class _TokenResponse(BaseModel):
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None


class CodexOAuthClient:
    """Codex OAuth device-code login and refresh client aligned with OpenAI Codex."""

    def __init__(
        self,
        *,
        profile: CodexOAuthProfile = CODEX_PROFILE,
        store: OAuthStore | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.profile = profile
        self.store = store or OAuthStore()
        self.http_client = http_client or httpx.Client(timeout=30)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def request_device_code(self) -> DeviceCode:
        response = self.http_client.post(
            self.profile.device_user_code_endpoint,
            json={"client_id": self.profile.client_id},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return _UserCodeResponse.model_validate(response.json()).to_device_code(self.profile)

    def poll_device_token(self, device_code: DeviceCode, *, timeout_seconds: int = 15 * 60) -> _DeviceTokenResponse:
        monotonic = __import__("time").monotonic
        end_at = monotonic() + timeout_seconds
        while True:
            response = self.http_client.post(
                self.profile.device_token_endpoint,
                json={"device_auth_id": device_code.device_auth_id, "user_code": device_code.user_code},
                headers={"Content-Type": "application/json"},
            )
            if response.is_success:
                return _DeviceTokenResponse.model_validate(response.json())
            if response.status_code in (403, 404) and monotonic() < end_at:
                sleep_for = min(device_code.interval, max(0.0, end_at - monotonic()))
                __import__("time").sleep(sleep_for)
                continue
            response.raise_for_status()
            raise RuntimeError("Codex device authorization failed")

    def exchange_device_code(self, code_response: _DeviceTokenResponse) -> OAuthProviderRecord:
        response = self.http_client.post(
            self.profile.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code_response.authorization_code,
                "redirect_uri": self.profile.device_redirect_uri,
                "client_id": self.profile.client_id,
                "code_verifier": code_response.code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token_response = _TokenResponse.model_validate(response.json())
        return self._record_from_token_response(token_response)

    def login_device_code(self, *, timeout_seconds: int = 15 * 60) -> tuple[DeviceCode, OAuthProviderRecord]:
        device_code = self.request_device_code()
        token_code = self.poll_device_token(device_code, timeout_seconds=timeout_seconds)
        return device_code, self.exchange_device_code(token_code)

    def refresh_record(self, record: OAuthProviderRecord) -> OAuthProviderRecord:
        refresh_token = record.tokens.refresh_token
        if not refresh_token:
            raise RuntimeError("Codex refresh token is missing; run `ya-oauth login codex` again.")
        response = self.http_client.post(
            self.profile.token_endpoint,
            json={
                "client_id": self.profile.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        token_response = _TokenResponse.model_validate(response.json())
        account = account_from_id_token(token_response.id_token) if token_response.id_token else record.account
        _validate_same_account(record.account, account)
        return record.with_refreshed_tokens(
            id_token=token_response.id_token,
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            account=account,
        )

    def revoke_record(self, record: OAuthProviderRecord) -> None:
        token = record.tokens.refresh_token or record.tokens.access_token
        if not token or not self.profile.revoke_endpoint:
            return
        response = self.http_client.post(
            self.profile.revoke_endpoint,
            data={"client_id": self.profile.client_id, "token": token},
        )
        if response.status_code < 400:
            return
        response.raise_for_status()

    def make_token_source(self) -> StoreBackedTokenSource:
        return StoreBackedTokenSource("codex", store=self.store, refresh_provider=self.refresh_record)

    def _record_from_token_response(self, token_response: _TokenResponse) -> OAuthProviderRecord:
        if not token_response.access_token:
            raise RuntimeError("Codex token response did not include access_token")
        account = OAuthAccount()
        if token_response.id_token:
            account = account_from_id_token(token_response.id_token)
        return OAuthProviderRecord(
            issuer=self.profile.issuer,
            client_id=self.profile.client_id,
            token_endpoint=self.profile.token_endpoint,
            revoke_endpoint=self.profile.revoke_endpoint,
            base_url=self.profile.base_url,
            scopes=list(self.profile.scopes),
            tokens=OAuthTokens(
                id_token=token_response.id_token,
                access_token=token_response.access_token,
                refresh_token=token_response.refresh_token,
            ),
            account=account,
            last_refresh_at=datetime.now(UTC),
        )


def _validate_same_account(old: OAuthAccount, new: OAuthAccount) -> None:
    if old.chatgpt_account_id and new.chatgpt_account_id and old.chatgpt_account_id != new.chatgpt_account_id:
        raise RuntimeError("Codex refresh returned a different ChatGPT account; run `ya-oauth login codex` again.")
    if old.chatgpt_user_id and new.chatgpt_user_id and old.chatgpt_user_id != new.chatgpt_user_id:
        raise RuntimeError("Codex refresh returned a different ChatGPT user; run `ya-oauth login codex` again.")


def create_codex_token_source(*, store: OAuthStore | None = None) -> StoreBackedTokenSource:
    return CodexOAuthClient(store=store).make_token_source()


def redact_record(record: OAuthProviderRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        for key in list(tokens):
            if tokens[key]:
                tokens[key] = "<redacted>"
    return data
