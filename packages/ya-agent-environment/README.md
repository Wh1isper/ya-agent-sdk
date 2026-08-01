# ya-agent-environment

Environment abstractions for general agents.

`ya-agent-environment` provides the shared base interfaces used by YA agents:

- `Environment`
- `FileOperator`
- `Shell`
- `ResourceRegistry`
- resumable resources
- environment-owned temporary storage via `tmp_dir` and `resolve_tmp_path()`

The Python import package is `ya_agent_environment`.

## Environment and File Backend Contracts

`Environment` owns temporary-directory configuration and lifecycle. While entered,
`env.tmp_dir` exposes the agent-facing root (or `None`) and
`env.resolve_tmp_path(relative_path)` safely resolves a contained path, rejecting
absolute paths and parent traversal. Temporary paths are handled through the normal
`FileOperator` methods; there is no separate temporary-file operator or routing API.
Environment cleanup closes registered resources, the shell, and the file operator
before `_teardown()` releases backend/container and temporary-directory resources.
The same ordering is preserved after partial setup failures and during cancellation. The base class does not assume
a filesystem backing store and never deletes `_tmp_dir` itself: custom environments
must allocate their agent-facing temporary backend in `_setup()` and remove only owned
storage in `_teardown()`.

A `FileOperator` implementation exposes one logical path space by implementing its
public abstract methods directly. `read_bytes_stream()` returns an `AsyncIterator`
directly, so consume the returned iterator without awaiting the method call.

Workspace-backed concrete environments allocate owned temporary instances below
`.tmp/ya-agent-<id>`. Each instance contains a self-ignoring `.gitignore`, and teardown
removes only that owned instance. Custom environments remain responsible for allocating
and tearing down their own temporary backend.

`LocalFileOperator.walk_files()` returns paths relative to `default_path` when possible
and directly reusable absolute paths for explicitly allowed roots outside it. `glob` and
`grep` may search those roots explicitly; `root="."` does not implicitly include
temporary storage.

## Bounded Output

Shared output-bounding policy lives in `ya_agent_environment.output`. It provides
head/tail budget splitting, character and UTF-8 byte truncation, and incremental bounded
text accumulation. Shell ingestion and SDK previews should reuse these helpers instead
of defining independent truncation policies.

## Shell Backend Contract

Custom `Shell` backends implement `_create_process()` and return an `ExecutionHandle` with stream, wait, and kill callbacks. `Shell.execute()` is the final public foreground boundary and must not be overridden: the base class owns process creation, timeout, cancellation, foreground/background registration, and session-reset cleanup. The foreground timeout budget begins before process creation, but shell-owned creation still resolves and terminates any eventual handle before reporting timeout. Class creation raises `TypeError` when final MRO resolution replaces `execute()`, so direct and mixin-based legacy overrides fail fast instead of silently bypassing lifecycle ownership; migrate their process creation into `_create_process()`. A wait callback must not report natural completion while owned descendants remain, and a kill callback must not return until backend execution is confirmed terminated; raising preserves the handle for a later reset retry. Backends must retain a stable execution identity through termination: a POSIX numeric PGID is no longer an ownership handle after its leader is reaped, so residual-group cleanup requires a live guardian or an equivalent platform-owned primitive.

## Development

This package is maintained as a workspace member in `ya-mono`.

```bash
uv run python -m pytest packages/ya-agent-environment/tests -vv
uv run python -m pyright
uv build --package ya-agent-environment -o dist
```
