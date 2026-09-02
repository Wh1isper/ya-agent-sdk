# ya-oauth-provider

Pydantic AI model/provider helpers that consume OAuth token sources from `ya-oauth`.

## Codex model string

YA Agent SDK loads this package for model strings such as:

```text
oauth@codex:gpt-5.5
```

The provider owns Codex request authentication and header alignment. It attaches the
bearer token, ChatGPT account ID, optional FedRAMP marker, originator, underscore and
hyphen variants of session/thread headers, SDK-provided `x-session-id`, and
`x-client-request-id`. It omits the Codex
`version` header by default to avoid coupling YA package versions to Codex CLI release
gates.

Generic OpenAI Responses WebSocket transport remains in `ya-agent-sdk` as
`WebsocketResponsesModel`. This package adds only Codex-specific authentication,
headers, token refresh, and payload normalization, and refreshes once on HTTP 401
through the configured token source before retrying. Its OpenAI client, OAuth auth
flow, and retry transport all use HTTPX2; callers must not inject a legacy
`httpx.AsyncClient`. The SDK-created client closes with the model context and is
recreated on re-entry. OAuth authentication buffers the request body once so Codex
payload normalization and a possible 401 replay operate on identical bytes.

The OAuth Codex models also follow Codex's backend-routing contracts. Every request
sets `x-codex-routing-hint` to `model=<model>` and appends `;tier=<service-tier>` when
a service tier is selected. The first `x-codex-turn-state` returned during a Pydantic
AI run is retained without replacement and replayed only within that run. HTTP sends
it as a request header; WebSocket sends it in
`client_metadata["x-codex-turn-state"]`, matching the Codex WebSocket protocol. A
new Pydantic AI run starts with no turn state. These behaviors are deliberately
limited to `oauth@codex:*`; ordinary `openai-responses:*` and
`openai-responses-ws:*` models remain generic.

## Proactive refresh

`ya_oauth_provider.OAuthRefreshSupervisor` refreshes configured OAuth token sources on startup and on a background interval. Runtime packages can use `create_oauth_refresh_supervisor_for_models(...)` to detect `oauth@provider:model` strings and maintain logged-in providers before the first model request needs a token refresh.
