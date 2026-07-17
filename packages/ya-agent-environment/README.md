# ya-agent-environment

Environment abstractions for general agents.

`ya-agent-environment` provides the shared base interfaces used by YA agents:

- `Environment`
- `FileOperator`
- `Shell`
- `ResourceRegistry`
- resumable resources
- `TmpFileOperator`

The Python import package is `ya_agent_environment`.

## Shell Backend Contract

Custom `Shell` backends implement `_create_process()` and return an `ExecutionHandle` with stream, wait, and kill callbacks. `Shell.execute()` is the final public foreground boundary and must not be overridden: the base class owns process creation, timeout, cancellation, foreground/background registration, and session-reset cleanup. The foreground timeout budget begins before process creation, but shell-owned creation still resolves and terminates any eventual handle before reporting timeout. Class creation raises `TypeError` when final MRO resolution replaces `execute()`, so direct and mixin-based legacy overrides fail fast instead of silently bypassing lifecycle ownership; migrate their process creation into `_create_process()`. A wait callback must not report natural completion while owned descendants remain, and a kill callback must not return until backend execution is confirmed terminated; raising preserves the handle for a later reset retry. Backends must retain a stable execution identity through termination: a POSIX numeric PGID is no longer an ownership handle after its leader is reaped, so residual-group cleanup requires a live guardian or an equivalent platform-owned primitive.

## Development

This package is maintained as a workspace member in `ya-mono`.

```bash
uv run python -m pytest packages/ya-agent-environment/tests -vv
uv run python -m pyright
uv build --package ya-agent-environment -o dist
```
