# ya-oauth

OAuth login, refresh, logout, token storage, and CLI for YA model providers.

## Codex login

```bash
ya-oauth login codex
ya-oauth status codex
ya-oauth refresh codex
```

Credentials are stored in `~/.yaai/auth.json` with directory mode `0700` and file mode `0600`.
