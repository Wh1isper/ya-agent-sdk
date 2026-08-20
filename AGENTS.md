# Repository Guide

`ya-mono` is a workspace-first monorepo managed with `uv`. Python packages target
Python 3.11+; pure-Python packages use Hatchling and `ya-ripgrep-core` uses Maturin.
Frontend applications use Vite, React, and TypeScript.

Most architecture work centers on `packages/ya-agent-sdk` and `packages/ya-claw`.

## Package Documentation

Package behavior, architecture, and maintainer contracts belong in the package README
or spec set. Read those documents before changing a package and update them with any
behavioral or architectural change.

| Package                             | Role                                                                        | Canonical documentation                                                                       |
| ----------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `packages/ya-agent-environment`     | Shared environment, file, shell, lifecycle, and bounded-output abstractions | [README](packages/ya-agent-environment/README.md)                                             |
| `packages/ya-agent-sdk`             | SDK for building and streaming Pydantic AI agents                           | [README](packages/ya-agent-sdk/README.md), [spec index](packages/ya-agent-sdk/spec/README.md) |
| `packages/ya-agent-stream-protocol` | Shared AGUI adaptation, replay, validation, and SSE helpers                 | [README](packages/ya-agent-stream-protocol/README.md)                                         |
| `packages/ya-ripgrep-core`          | Native filesystem search bindings                                           | [README](packages/ya-ripgrep-core/README.md)                                                  |
| `packages/ya-oauth`                 | OAuth login, refresh, storage, and CLI                                      | [README](packages/ya-oauth/README.md)                                                         |
| `packages/ya-oauth-provider`        | OAuth-backed Pydantic AI provider integration                               | [README](packages/ya-oauth-provider/README.md)                                                |
| `packages/yaacli`                   | TUI reference application built on the SDK                                  | [README](packages/yaacli/README.md), [spec index](packages/yaacli/spec/00-overview.md)        |
| `packages/ya-claw`                  | Workspace-native single-node runtime web service                            | [README](packages/ya-claw/README.md), [spec index](packages/ya-claw/spec/README.md)           |
| `packages/ya-agent-platform`        | WIP stateless agent service                                                 | [README](packages/ya-agent-platform/README.md)                                                |

## Runtime Architecture

- `packages/ya-agent-sdk` composes agent behavior only through `capabilities=`; do not
  reintroduce public `tools=`, `toolsets=`, MessageBus, or generated-subagent
  compatibility layers.
- Runtime input uses Pydantic AI native enqueue/deferred mechanisms. Durable hosts own
  persisted inboxes, execution records, and delivery/application state.
- Subagents execute and persist as strict portable `SubagentSpec` documents resolved
  into explicit plans. Trusted hosts may normalize human-facing Markdown definitions
  into that boundary before resolution; they must not restore implicit inheritance or
  alternate delegation runtimes. YAACLI and YA Claw provide restart-durable execution
  drivers rather than alternate model-facing delegation APIs.
- The SDK execution harness owns one stateless native agent segment only. YAACLI and YA
  Claw compose it with host-owned coordinators and stores; active process-owned work is
  interrupted rather than replayed after a crash. YA Claw profiles use strict native
  schema v2; old profile data is handled only by explicit Alembic
  migration, never by runtime compatibility parsing.
- User configuration is strict. Preserve an old key only when it normalizes exactly to
  one current behavior; otherwise remove it from code, templates, specs, and examples.

## Shared Repository Areas

- `apps/` — frontend applications and user-facing shells
- `skills/` — canonical skill sources and reference material
- `examples/` — runnable SDK examples
- `scripts/` — repository automation
- `.github/` — CI and release workflows
- `Dockerfile.ya-claw` — YA Claw service image
- `Dockerfile.ya-claw-workspace` — official YA Claw workspace image
- `Dockerfile.ya-agent-platform` — YA Agent Platform image

## Development Workflow

After changing code, run:

1. `make lint`
2. `make check`
3. `make test`

Use narrower package tests while iterating, but run the repository checks before
finalizing a broad change.

Useful commands:

| Command                            | Description                               |
| ---------------------------------- | ----------------------------------------- |
| `make run-claw`                    | Run the YA Claw backend                   |
| `make web-dev`                     | Run the YA Claw web app                   |
| `make build-claw`                  | Build the `ya-claw` package               |
| `make build-platform`              | Build the WIP `ya-agent-platform` package |
| `make docker-build-claw`           | Build the YA Claw service image           |
| `make docker-build-claw-workspace` | Build the YA Claw workspace image         |
| `make docker-build-platform`       | Build the YA Agent Platform image         |

## Environment Configuration

Environment variables are loaded with `pydantic-settings` from the process environment
or `.env` files.

| Scope    | Example file                         | Prefix            |
| -------- | ------------------------------------ | ----------------- |
| SDK      | `packages/ya-agent-sdk/.env.example` | `YA_AGENT_`       |
| YAACLI   | `packages/yaacli/.env.example`       | `YAACLI_`         |
| YA Claw  | `packages/ya-claw/.env.example`      | `YA_CLAW_`        |
| Examples | `examples/.env.example`              | varies by example |

Keep the affected example environment file current when adding or changing a setting.

## Cross-Repository Changes

When changing workspace, package, release, or deployment metadata, update all affected
surfaces together. Check the root and package `pyproject.toml` files, `uv.lock`,
`pnpm-workspace.yaml`, `Makefile`, CI workflows, Dockerfiles, `.dockerignore`, package
README/spec files, canonical skills, and skill sync scripts as applicable.
