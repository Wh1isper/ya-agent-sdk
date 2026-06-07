# ya-mono

> Yet Another Agents

[![Release](https://img.shields.io/github/v/release/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/wh1isper/ya-mono/main.yml?branch=main)](https://github.com/wh1isper/ya-mono/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/wh1isper/ya-mono/graph/badge.svg?token=UOM3ONfEb4)](https://codecov.io/gh/wh1isper/ya-mono)
[![License](https://img.shields.io/github/license/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/blob/main/LICENSE)

## Packages

- [`packages/ya-agent-environment`](packages/ya-agent-environment) — Environment abstractions for general agents
- [`packages/ya-environment-relay`](packages/ya-environment-relay) — Provider-neutral relay protocol for external Environment capabilities
- [`packages/ya-agent-sdk`](packages/ya-agent-sdk) — Python SDK for building AI agents with Pydantic AI
- [`packages/ya-agent-stream-protocol`](packages/ya-agent-stream-protocol) — shared stream protocol adapters between `ya-agent-sdk` and applications
- [`packages/ya-oauth`](packages/ya-oauth) — OAuth login, refresh, logout, token storage, and CLI for subscription-backed providers
- [`packages/ya-oauth-provider`](packages/ya-oauth-provider) — Pydantic AI provider helpers for OAuth-backed model access
- [`packages/yaacli`](packages/yaacli) — TUI reference implementation built on top of `ya-agent-sdk`
- [`packages/ya-claw`](packages/ya-claw) — workspace-native single-node runtime web service with `WorkspaceProvider`, in-process runtime state, schedules, bridges, and SQLite-first storage
- [`packages/ya-agent-platform`](packages/ya-agent-platform) — WIP stateless agent service with TBD scope

## Apps

- [`apps/ya-claw-web`](apps/ya-claw-web) — Vite + React web shell for YA Claw

## Repository Layout

- [`packages/`](packages/) — publishable workspace members
- [`apps/`](apps/) — frontend applications and user-facing shells
- [`skills/`](skills/) — canonical skill sources and reference material
- [`examples/`](examples/) — runnable SDK examples
- [`scripts/`](scripts/) — repository automation scripts
- [`.github/`](.github/) — CI and release workflows

## Primary Skill Sources

- [`skills/agent-builder/`](skills/agent-builder/) — source of truth for the `agent-builder` skill bundled into YAACLI as `building-agents`
- [`skills/ya-claw-deploy/`](skills/ya-claw-deploy/) — source of truth for the YA Claw deployment skill published as `YA_CLAW_DEPLOY_SKILL.zip`

## Installation

Prerequisites:

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- `make`
- Node.js with `corepack` available for the web app

Clone the repository and install the workspace environment:

```bash
git clone https://github.com/wh1isper/ya-mono.git
cd ya-mono
make install
```

Install the canonical skills into `~/.agents/skills`:

```bash
make install-skills
```

This installs `agent-builder` and `ya-claw-deploy` for local agent use.

Install published tools with native filesystem search enabled:

```bash
uv tool install 'yaacli[rs]'
uv tool install 'ya-claw[rs]'
```

`uv tool install` accepts extras in the package requirement. The equivalent extra-dependency form is `uv tool install yaacli --with ya-ripgrep-core` or `uv tool install ya-claw --with ya-ripgrep-core`. `ya-ripgrep-core` is a library dependency, so `--with` is the matching uv form; `--with-executables-from` applies to companion packages that also expose CLI executables.

## Getting Started

### SDK and CLI

Recommended starting points:

- Read [`skills/agent-builder/SKILL.md`](skills/agent-builder/SKILL.md) for the main agent-building workflow
- Read [`skills/agent-builder/README.md`](skills/agent-builder/README.md) for the skill file map
- Copy [`examples/.env.example`](examples/.env.example) to `examples/.env` when running SDK examples
- Copy [`packages/ya-agent-sdk/.env.example`](packages/ya-agent-sdk/.env.example) to `packages/ya-agent-sdk/.env` when developing the SDK or configuring SDK/tool variables for workspace apps
- Copy [`packages/yaacli/.env.example`](packages/yaacli/.env.example) to `packages/yaacli/.env` or your current working directory `.env` when developing YAACLI
- Run examples from [`examples/`](examples/) for end-to-end usage patterns

### OAuth-backed Codex

Run Codex through your ChatGPT subscription with a local OAuth credential store:

```bash
uv run --package ya-oauth ya-oauth login codex
```

Then use `oauth@codex:gpt-5.5` in SDK, YAACLI, or YA Claw profiles. Credentials live in `~/.yaai/auth.json`.

### YA Claw

Read the deployment and runtime docs:

- [`skills/ya-claw-deploy/SKILL.md`](skills/ya-claw-deploy/SKILL.md)
- [`skills/ya-claw-deploy/references/docker.md`](skills/ya-claw-deploy/references/docker.md)
- [`packages/ya-claw/spec/README.md`](packages/ya-claw/spec/README.md)
- [`packages/ya-claw/spec/00-overview.md`](packages/ya-claw/spec/00-overview.md)
- [`packages/ya-claw/spec/01-configuration-and-workspace-provider.md`](packages/ya-claw/spec/01-configuration-and-workspace-provider.md)

Prepare the runtime environment and start the service:

```bash
cp packages/ya-claw/.env.example packages/ya-claw/.env
make run-claw
```

SDK and tool variables for YA Claw workflows live in [`packages/ya-agent-sdk/.env.example`](packages/ya-agent-sdk/.env.example).

Run the web shell:

```bash
make web-dev
```

Bridge commands live under the main YA Claw CLI:

```bash
uv run --package ya-claw ya-claw bridge ls
uv run --package ya-claw ya-claw bridge run lark
uv run --package ya-claw ya-claw bridge serve lark
```

Build the images:

```bash
make docker-build-claw
make docker-build-claw-workspace
make docker-build-platform
```

## Dockerfiles

- `Dockerfile.ya-claw` — YA Claw image build
- `Dockerfile.ya-claw-workspace` — official YA Claw Docker workspace image build
- `Dockerfile.ya-agent-platform` — YA Agent Platform image build

## Development

Run repository checks:

```bash
make lint
make check
make test
```

## Package Guides

- [ya-agent-environment README](packages/ya-agent-environment/README.md)
- [ya-environment-relay README](packages/ya-environment-relay/README.md)
- [ya-environment-relay spec](packages/ya-environment-relay/spec/README.md)
- [ya-agent-sdk README](packages/ya-agent-sdk/README.md)
- [ya-oauth README](packages/ya-oauth/README.md)
- [ya-oauth-provider README](packages/ya-oauth-provider/README.md)
- [yaacli README](packages/yaacli/README.md)
- [ya-claw README](packages/ya-claw/README.md)
- [ya-agent-platform README](packages/ya-agent-platform/README.md)
- [agent-builder skill](skills/agent-builder/SKILL.md)
- [YA Claw deployment skill](skills/ya-claw-deploy/SKILL.md)
- [Contributing Guide](CONTRIBUTING.md)

## Workspace Commands

| Command                            | Description                                                         |
| ---------------------------------- | ------------------------------------------------------------------- |
| `make install`                     | Install Python dependencies, web dependencies, and pre-commit hooks |
| `make install-skills`              | Install canonical skills into `~/.agents/skills`                    |
| `make lint`                        | Check lock consistency and run pre-commit hooks                     |
| `make check`                       | Run lock validation, lint, pyright, deptry, and web checks          |
| `make test`                        | Run environment, SDK, CLI, YA Claw, and YA Desktop tests            |
| `make run-claw`                    | Run the YA Claw backend                                             |
| `make web-dev`                     | Run the YA Claw web app                                             |
| `make build-claw`                  | Build the `ya-claw` distribution                                    |
| `make build-platform`              | Build the WIP `ya-agent-platform` package                           |
| `make build-all`                   | Build distributions for all workspace packages                      |
| `make docker-build-claw`           | Build the YA Claw Docker image                                      |
| `make docker-build-claw-workspace` | Build the YA Claw workspace Docker image                            |
| `make docker-build-platform`       | Build the YA Agent Platform Docker image                            |

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
