# YA Environment Protocol Spec

YA Environment Protocol is YA's own provider-neutral protocol for connecting agent runtimes to execution environments. Any language can implement a provider when it follows the JSON payload contract.

The protocol lets a runtime expose external capabilities as `ya-agent-sdk` Environment components:

- file operations
- stateless shell execution
- background processes
- stateful shell sessions
- provider resources
- custom tools
- artifacts
- computer use

The protocol string is `ya-environment-protocol.v1`.

`ya-envd` is the official daemon implementation. It is expected to be a Rust binary that can run as a child process over stdio, as a long-lived local daemon over a socket or named pipe, or behind a remote transport. The daemon is an implementation of the protocol, not the protocol itself.

## Goals

- Support pluggable environments owned by the agent, a user device, a workspace daemon, a cloud worker, or a custom provider.
- Keep the protocol language independent by defining JSON payloads, JSON Schema models, and request/response semantics.
- Keep the protocol transport independent by separating message shape from stdio, socket, named pipe, WebSocket, or future transports.
- Let user-mounted environments go offline and come back online without blocking agent-owned detached execution.
- Make environment readiness observable so applications can decide whether to activate an agent when a provider resumes.
- Keep file and shell views aligned through shared mounts and execution targets.
- Model shell state explicitly instead of hiding it behind plain command execution.

## Package Direction

Initial work is spec-first. The current package path remains `packages/ya-environment-relay` while the protocol is being designed. Future implementation can either keep this package as the Python protocol/adapter package or rename it once downstream impact is clear.

Recommended repository shape:

```text
crates/
  ya-env-protocol/
  ya-envd-core/
  ya-envd/
  ya-envd-client/

packages/ya-environment-relay/
  spec/
  ya_environment_relay/
    protocol.py
    client.py
    environment.py
    transports/
```

The Python package should expose SDK adapters:

- `DaemonEnvironment`
- `DaemonFileOperator`
- `DaemonShell`
- `DaemonResourceRegistry`
- `DaemonToolset`
- protocol models for `ya-environment-protocol.v1`

## Section Map

| Section | Document                                                           | Topic                                                                     |
| ------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 01      | [01-overview.md](01-overview.md)                                   | goals, parties, provider model, capability model, detached execution      |
| 02      | [02-protocol.md](02-protocol.md)                                   | JSON-RPC envelope, transports, lifecycle, methods, events, versioning     |
| 03      | [03-environment.md](03-environment.md)                             | SDK Environment mapping, mounts, fileops, shell, sessions, resources      |
| 04      | [04-security-and-policy.md](04-security-and-policy.md)             | authentication, grants, path safety, shell safety, audit, revocation      |
| 05      | [05-implementation-plan.md](05-implementation-plan.md)             | Rust daemon, Python adapter, Claw/Desktop integration phases, test matrix |
| 06      | [06-desktop-local-pc-provider.md](06-desktop-local-pc-provider.md) | Desktop local PC provider, online/offline lifecycle, activation events    |
| 07      | [07-ya-envd.md](07-ya-envd.md)                                     | official `ya-envd` daemon architecture, process/session state, transports |

## Relationship to Products

- `ya-agent-sdk` provides the Environment concepts that protocol adapters implement.
- `ya-claw` can bind sessions to one or more providers and decide when provider events should activate an agent.
- `ya-desktop` can supervise a user-owned `ya-envd` and expose local files, shell, computer use, and Desktop tools.
- Agent-owned detached workers can supervise their own `ya-envd` and continue running when user devices are offline.
- Custom services can implement the same protocol in any language.
