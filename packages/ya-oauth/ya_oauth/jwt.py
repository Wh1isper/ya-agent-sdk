from __future__ import annotations

import base64
import json
from typing import Any

from ya_oauth.types import OAuthAccount


def decode_jwt_payload(jwt: str) -> dict[str, Any]:
    """Decode a JWT payload without signature validation for local metadata extraction."""
    parts = jwt.split(".")
    if len(parts) != 3 or not parts[1]:
        raise ValueError("invalid JWT format")
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
    data = json.loads(decoded.decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("invalid JWT payload")
    return data


def account_from_id_token(id_token: str) -> OAuthAccount:
    """Extract Codex-compatible ChatGPT account metadata from an ID token."""
    claims = decode_jwt_payload(id_token)
    profile = claims.get("https://api.openai.com/profile")
    auth = claims.get("https://api.openai.com/auth")
    profile_data = profile if isinstance(profile, dict) else {}
    auth_data = auth if isinstance(auth, dict) else {}
    return OAuthAccount(
        email=_string_or_none(claims.get("email")) or _string_or_none(profile_data.get("email")),
        chatgpt_user_id=_string_or_none(auth_data.get("chatgpt_user_id")) or _string_or_none(auth_data.get("user_id")),
        chatgpt_account_id=_string_or_none(auth_data.get("chatgpt_account_id")),
        chatgpt_plan_type=_plan_type(auth_data.get("chatgpt_plan_type")),
        chatgpt_account_is_fedramp=bool(auth_data.get("chatgpt_account_is_fedramp", False)),
    )


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _plan_type(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = value.get("raw_value") or value.get("value") or value.get("name")
        return _string_or_none(raw)
    return None
