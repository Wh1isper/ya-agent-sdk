from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class OAuthTokens(BaseModel):
    """OAuth token material stored for a provider."""

    id_token: str | None = None
    access_token: str
    refresh_token: str | None = None


class OAuthAccount(BaseModel):
    """Account metadata derived from provider tokens."""

    email: str | None = None
    chatgpt_user_id: str | None = None
    chatgpt_account_id: str | None = None
    chatgpt_plan_type: str | None = None
    chatgpt_account_is_fedramp: bool = False


class OAuthProviderRecord(BaseModel):
    """Stored OAuth configuration and credential record for one provider."""

    type: str = "oauth2"
    issuer: str
    client_id: str
    token_endpoint: str
    revoke_endpoint: str | None = None
    base_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    tokens: OAuthTokens
    account: OAuthAccount = Field(default_factory=OAuthAccount)
    last_refresh_at: datetime | None = None

    def with_refreshed_tokens(
        self,
        *,
        id_token: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        account: OAuthAccount | None = None,
    ) -> OAuthProviderRecord:
        """Return an updated copy, preserving refresh response fields Codex omits."""
        return self.model_copy(
            update={
                "tokens": self.tokens.model_copy(
                    update={
                        "id_token": id_token if id_token is not None else self.tokens.id_token,
                        "access_token": access_token if access_token is not None else self.tokens.access_token,
                        "refresh_token": refresh_token if refresh_token is not None else self.tokens.refresh_token,
                    }
                ),
                "account": account if account is not None else self.account,
                "last_refresh_at": datetime.now(UTC),
            }
        )


class AuthFile(BaseModel):
    """On-disk auth file schema for ~/.yaai/auth.json."""

    version: int = 1
    providers: dict[str, OAuthProviderRecord] = Field(default_factory=dict)


class TokenSnapshot(BaseModel):
    """Provider token state safe for request construction."""

    provider_name: str
    access_token: str
    account: OAuthAccount = Field(default_factory=OAuthAccount)
    base_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OAuthTokenSource(Protocol):
    """Async token source consumed by OAuth-backed model providers."""

    async def get_token(self) -> TokenSnapshot: ...

    async def refresh_token(self) -> TokenSnapshot: ...
