---
name: ya-claw-deploy
description: Deploy, configure, operate, and troubleshoot YA Claw, the single-node runtime web service in ya-mono. Use for production and self-hosted Docker deployment, docker-compose, systemd services, SQLite or PostgreSQL storage, Docker workspace provider configuration, profile seeding, execution coordinator setup, health checks, reverse proxy setup, Lark bridge deployment, upgrades, backup, restore, and production operations for packages/ya-claw.
---

# YA Claw Deployment

Use this skill when deploying, configuring, operating, or troubleshooting YA Claw.

## Recommended Deployment Shape

Use the Docker workspace provider for production-like deployments. Choose the exact shape from the workspace provider matrix.

```mermaid
flowchart LR
    CLIENT["Web/API Client"] --> SVC["YA Claw Service"]
    SVC --> DB[("SQLite or PostgreSQL")]
    SVC --> DATA["Persistent Data Dir"]
    SVC --> PROVIDER["WorkspaceProvider"]
    PROVIDER --> LOCAL["Local shell"]
    PROVIDER --> DOCKER["Docker shell"]
    DOCKER --> WSC["Reusable Workspace Container"]
```

The Docker shell shapes give agents an isolated workspace container with Python, Node.js, `lark-cli`, and bundled workspace skills.

## Start Here

Choose the deployment path:

- Workspace provider matrix: read [`references/workspace-provider/overview.md`](references/workspace-provider/overview.md)
- Service Docker + Docker shell: read [`references/docker.md`](references/docker.md) and [`references/workspace-provider/service-docker-docker-shell.md`](references/workspace-provider/service-docker-docker-shell.md)
- Service local + Docker shell: read [`references/systemd.md`](references/systemd.md) and [`references/workspace-provider/service-local-docker-shell.md`](references/workspace-provider/service-local-docker-shell.md)
- Service local + local shell: read [`references/workspace-provider/service-local-local-shell.md`](references/workspace-provider/service-local-local-shell.md)
- SQLite or PostgreSQL storage: read [`references/database.md`](references/database.md)
- Profile seeding, subscription-backed Codex profiles, and shell review policy: read [`references/profiles.md`](references/profiles.md)
- Bridge deployment: read [`references/bridge/overview.md`](references/bridge/overview.md) and [`references/bridge/lark.md`](references/bridge/lark.md)
- Schedules and heartbeat: read [`references/schedules-heartbeat.md`](references/schedules-heartbeat.md)
- Session agency: read [`references/agency.md`](references/agency.md)
- Session/run pruning: read [`references/operations.md#session-and-run-pruning`](references/operations.md#session-and-run-pruning)
- Health checks, backup, restore, upgrades, and troubleshooting: read [`references/operations.md`](references/operations.md)

## Required Baseline

Every deployment needs:

- `YA_CLAW_API_TOKEN`
- persistent `YA_CLAW_DATA_DIR`
- persistent `YA_CLAW_WORKSPACE_DIR`
- selected workspace backend: `local` or `docker`
- Docker access for the YA Claw service process when backend is `docker`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` when the service runs in Docker and uses Docker shell execution
- model/provider credentials available in the YA Claw service process
- seeded or pre-created AgentProfile rows

## Default Runtime Values

- HTTP port: `9042`
- runtime data dir: `~/.ya-claw/data`
- SQLite path: `~/.ya-claw/ya_claw.sqlite3`
- run store: `~/.ya-claw/data/run-store`
- workspace dir: `~/.ya-claw/data/workspace`
- session agency: enabled by default; inactivity scan after 600 seconds with 1800-second cooldown
- session prune: disabled by default; safe disk-only mode keeps latest 10 runs per session when enabled
- default profile: `default`
- profile shell review threshold default: `extra_high`
- Docker workspace image: `ghcr.io/wh1isper/ya-claw-workspace:latest`
- Docker service host bind: `0.0.0.0`
- service build metadata: `YA_CLAW_SERVICE_VERSION`, `YA_CLAW_SERVICE_COMMIT`, `YA_CLAW_SERVICE_BUILD`, and `YA_CLAW_SERVICE_IMAGE`, exposed through `/api/v1/claw/info` and the web Overview page

## Deployment Checklist

01. Generate a long random `YA_CLAW_API_TOKEN`.
02. Choose SQLite or PostgreSQL.
03. Mount or create persistent data and workspace directories.
04. Choose one workspace provider shape.
05. Grant the service Docker access when the shape uses Docker shell execution.
06. Configure profile seed when packaged profiles should be loaded at startup.
07. Start the service with `ya-claw start`.
08. Verify `/healthz`.
09. Verify authenticated API or web shell access.
10. Start a test session and confirm model credentials, workspace tools, and profile behavior.
11. Verify session agency state or set `YA_CLAW_AGENCY_ENABLED=false` for deployments that want manual rollout.

## Reference Routing

| Topic                     | File                                                                                     | Read when                                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Environment variables     | [`references/environment.md`](references/environment.md)                                 | You need exact `YA_CLAW_*` settings, defaults, or production env shape                                          |
| Docker deployment         | [`references/docker.md`](references/docker.md)                                           | You deploy the YA Claw server as a Docker service                                                               |
| Workspace provider matrix | [`references/workspace-provider/overview.md`](references/workspace-provider/overview.md) | You choose between service local + Docker shell, service Docker + Docker shell, and service local + local shell |
| Docker workspace provider | [`references/workspace-provider/docker.md`](references/workspace-provider/docker.md)     | You configure Docker Engine access, path mapping, workspace mounts, and container reuse                         |
| Host service              | [`references/systemd.md`](references/systemd.md)                                         | You run YA Claw as a supervised host service                                                                    |
| Database                  | [`references/database.md`](references/database.md)                                       | You choose SQLite or PostgreSQL, migrate, backup, or restore storage                                            |
| Profiles                  | [`references/profiles.md`](references/profiles.md)                                       | You seed profiles, manage AgentProfile configuration, or tune shell review `risk_threshold`                     |
| Bridge overview           | [`references/bridge/overview.md`](references/bridge/overview.md)                         | You configure bridge dispatch, adapter enablement, and event-to-run routing                                     |
| Lark bridge               | [`references/bridge/lark.md`](references/bridge/lark.md)                                 | You connect Lark/Feishu events to YA Claw                                                                       |
| Bridge operations         | [`references/bridge/operations.md`](references/bridge/operations.md)                     | You verify embedded bridge startup, Lark ingress, dedupe, profiles, and workspace replies                       |
| Schedules and heartbeat   | [`references/schedules-heartbeat.md`](references/schedules-heartbeat.md)                 | You configure cron schedules, heartbeat guidance, timer dispatchers, and timer operations                       |
| Session agency            | [`references/agency.md`](references/agency.md)                                           | You configure proactive agency, agency signals, paired agency sessions, and agency operations                   |
| Operations                | [`references/operations.md`](references/operations.md)                                   | You need health checks, logs, pruning, upgrades, backup, restore, or troubleshooting                            |

When editing this skill inside the repository, keep `scripts/build-skill-zips.py` aligned so release artifacts include the canonical skill contents.
