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

Relay protocol work lives in the sibling [`ya-environment-relay`](../ya-environment-relay) package.

## Development

This package is maintained as a workspace member in `ya-mono`.

```bash
uv run python -m pytest packages/ya-agent-environment/tests -vv
uv run python -m pyright
uv build --package ya-agent-environment -o dist
```
