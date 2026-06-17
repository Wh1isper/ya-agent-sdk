# Contributing to `ya-agent-environment`

`ya-agent-environment` is maintained inside the `ya-mono` workspace.

## Local Development

```bash
git clone https://github.com/wh1isper/ya-mono.git
cd ya-mono
make install
```

Run package tests:

```bash
uv run python -m pytest packages/ya-agent-environment/tests -vv
```

Run repository checks before a pull request:

```bash
make lint
make check
make test
```

## Guidelines

- Keep the import package name `ya_agent_environment` for compatibility.
- Update tests and docs when changing public protocols, environment interfaces, shell behavior, file operations, or resource state.
