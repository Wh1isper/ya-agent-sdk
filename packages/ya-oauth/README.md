# ya-oauth

OAuth login, refresh, logout, token storage, and CLI for YA model providers.

## Codex login

```bash
ya-oauth login codex
ya-oauth status codex
ya-oauth refresh codex
```

`ya-oauth login codex` follows the OpenAI Codex device-code flow and preserves Codex
token refresh semantics.

Credentials are stored in `~/.yaai/auth.json`. The store uses process-safe locking and
atomic replacement, creates its directory with mode `0700`, and writes the credential
file with mode `0600`.
