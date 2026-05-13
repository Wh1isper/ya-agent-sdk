# ya-oauth-provider

Pydantic AI model/provider helpers that consume OAuth token sources from `ya-oauth`.

## Codex model string

YA Agent SDK loads this package for model strings such as:

```text
oauth@codex:gpt-5.5
```

The provider attaches Codex-compatible bearer, account, originator, session, and thread headers. It omits the Codex `version` header by default to avoid coupling YA package versions to Codex CLI release gates.

## Proactive refresh

`ya_oauth_provider.OAuthRefreshSupervisor` refreshes configured OAuth token sources on startup and on a background interval. Runtime packages can use `create_oauth_refresh_supervisor_for_models(...)` to detect `oauth@provider:model` strings and maintain logged-in providers before the first model request needs a token refresh.
