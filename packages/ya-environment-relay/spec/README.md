# YA Environment Relay Spec

YA Environment Relay is a provider-neutral protocol for connecting agent runtimes to external execution environments. It is designed to build on top of `ya-agent-sdk` Environment abstractions and expose remote or local capabilities as file operators, shells, resources, and toolsets.

YA Environment Relay is a general protocol. YA Desktop is one important relay client, but the protocol should also support headless relay agents, server-side workers, browser sandboxes, VM sandboxes, and custom tool hosts.

## Goals

- Let an agent runtime call capabilities that live outside the runtime process.
- Represent remote capabilities as SDK Environment components.
- Support file operations, shell execution, custom tools, resources, artifacts, and computer use.
- Use one bidirectional transport for request/response, streaming, cancellation, and provider events.
- Keep tool schemas compatible with JSON Schema and model tool calling.
- Preserve runtime-owned tracing, approvals, and artifact persistence.

## Package Direction

Initial work is spec-only. Future implementation can become a workspace package:

```text
packages/ya-environment-relay/
  spec/
  pyproject.toml
  ya_environment_relay/
    protocol.py
    client.py
    server.py
    environment.py
    providers/
```

The first Python implementation should integrate with `ya-agent-sdk` and expose:

- `RelayEnvironment`
- `RelayFileOperator`
- `RelayShell`
- `RelayToolset`
- `RelayResourceRegistry`
- protocol models for `ya-environment-relay.v1`

## Section Map

| Section | Document                                               | Topic                                                                  |
| ------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| 01      | [01-overview.md](01-overview.md)                       | goals, parties, capability model, relationship to SDK and Claw         |
| 02      | [02-protocol.md](02-protocol.md)                       | WebSocket frames, request/response, streaming, cancellation, errors    |
| 03      | [03-environment.md](03-environment.md)                 | SDK Environment mapping, file operator, shell, resources, custom tools |
| 04      | [04-security-and-policy.md](04-security-and-policy.md) | authentication, grants, policy, approvals, artifact safety             |
| 05      | [05-implementation-plan.md](05-implementation-plan.md) | MVP phases and package layout                                          |

## Relationship to Products

- `ya-agent-sdk` provides the Environment concepts that YA Environment Relay implements remotely.
- `ya-claw` can host a relay server and route agent tool calls to connected providers.
- `ya-desktop` can act as a relay client for local files, shell, computer use, and custom tools.
- Future services can implement relay clients for sandboxes, browsers, VMs, and specialized tools.
