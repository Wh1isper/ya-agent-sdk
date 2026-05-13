# YA Claw Architecture Spec

YA Claw is a workspace-native single-node runtime web service built on top of `ya-agent-sdk`.

This spec set defines one execution shape:

- one runtime service
- one in-process execution supervisor
- one in-process runtime state view for active runs and event delivery
- one SQLite database by default
- optional PostgreSQL for deployments that prefer an external relational store
- one local data root for committed continuity blobs
- one bundled web shell

## Design Principles

- **Single-node first**: one machine, one runtime, one operational context
- **Workspace-binding based**: a configured default workspace and optional session mount sets bind runtime file and shell access
- **SDK-aligned**: `ya-agent-sdk` stays responsible for agent execution primitives
- **Queued-run execution model**: API ingress, schedules, heartbeat, and bridges all create durable queued runs before execution starts
- **Explicit runtime assembly**: workspace resolution, environment construction, context construction, and agent runtime construction each have their own boundary
- **Profile-driven configuration**: reusable execution profiles live in the relational store and can be seeded from YAML
- **Durable and practical**: SQLite is the default durable store, PostgreSQL is an optional backend, and the local filesystem stores committed session state

## Section Map

| Section | Document                                                                                 | Topic                                                                        |
| ------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 00      | [00-overview.md](00-overview.md)                                                         | runtime definition, top-level architecture, and core runtime objects         |
| 01      | [01-configuration-and-workspace-provider.md](01-configuration-and-workspace-provider.md) | configuration layers, profiles, workspace binding, and runtime assembly      |
| 02      | [02-execution-and-session.md](02-execution-and-session.md)                               | queued runs, supervisor/coordinator layering, session/run lifecycle          |
| 03      | [03-storage-and-streaming.md](03-storage-and-streaming.md)                               | relational store, run store, runtime state, and event delivery               |
| 04      | [04-api.md](04-api.md)                                                                   | HTTP API surface and queued-run API semantics                                |
| 05      | [05-web-ui-and-operations.md](05-web-ui-and-operations.md)                               | web shell, runtime operations, schedules, and bridge usage                   |
| 06      | [06-runtime-assembly.md](06-runtime-assembly.md)                                         | `WorkspaceBinding -> Environment -> ClawAgentContext -> AgentRuntime`        |
| 07      | [07-async-subagents.md](07-async-subagents.md)                                           | non-blocking background subagent dispatch, spawn_delegate, steer             |
| 08      | [08-schedules-and-heartbeat.md](08-schedules-and-heartbeat.md)                           | schedule modes, heartbeat, agent schedule toolset, and timer dispatch        |
| 09      | [09-session-memory.md](09-session-memory.md)                                             | workspace memory files, internal memory sessions, extract and summary agents |
| 10      | [10-workspace-mount-sets.md](10-workspace-mount-sets.md)                                 | session-scoped workspace mount sets and multi-folder execution               |
| 11      | [11-session-agency.md](11-session-agency.md)                                             | session agency, paired agency sessions, signals, steering, and action loops  |

## Out of Scope

- multi-tenant control plane
- distributed worker pools and cross-node scheduling
- runtime-managed project catalogs or project CRUD
- hosted operator portal
- full database schema freeze before implementation
