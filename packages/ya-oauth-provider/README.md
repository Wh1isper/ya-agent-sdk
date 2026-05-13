# ya-oauth-provider

Pydantic AI model/provider helpers that consume OAuth token sources from `ya-oauth`.

## Codex model string

YA Agent SDK loads this package for model strings such as:

```text
oauth@codex:gpt-5.5
```

The provider attaches Codex-compatible bearer, account, originator, version, session, and thread headers.
