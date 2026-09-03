# YA Claw

Workspace-native single-node agent runtime and web service for the `ya-mono` workspace.

## Scope

YA Claw packages a durable runtime shell around `ya-agent-sdk` with:

- one persistent workspace resolved through `WorkspaceProvider`
- reusable agent profiles
- explicitly selected installable capability plugins through an SDK-owned manifest
- resumable sessions and runs
- in-process active handles and live transports over SQL-owned async task state
- session schedules for timed execution
- SQLite-first durable state with optional PostgreSQL
- local filesystem session continuity and exported state
- a bundled web shell for local and self-hosted use
- bridge adapters that connect external event sources to the YA Claw service

## Current Direction

The target single-node shape runs as one web service.
Process memory owns active handles, live delivery transports, and best-effort scheduling coordination. Async-task state, completion intent, parent inbox/continuation targeting, and delivery disposition remain canonical relational state; only the SQL-backed post-run owner dispatches completion, and portable delivery completes only after canonical parent application.
SQLite is the default durable store.
PostgreSQL remains an optional storage backend for deployments that prefer an external relational database.

## Layout

Key areas in this package:

- `.env.example` — runtime environment example
- `spec/` — architecture and runtime design documents
- `tests/` — runtime tests
- `ya_claw/api/` — HTTP API surface
- `ya_claw/bridge/` — external bridge adapters and event handling
- `ya_claw/app.py` and `ya_claw/cli.py` — application entrypoints
- `ya_claw/config.py` — runtime configuration

## Runtime Shape

The runtime shape is:

- one YA Claw web service
- one in-process runtime state manager
- one session scheduler
- one bridge subsystem for external channels
- one shared bearer token for HTTP access
- one SQLite database by default
- optional PostgreSQL
- one runtime data directory for sensitive session continuity
- one persistent workspace directory
- one bundled web shell

## Runtime Architecture Notes

This section is the maintainer index for implementation details that affect code changes across YA Claw.

### Runtime Defaults

- `YA_CLAW_API_TOKEN` is required before service startup.
- `/api/v1/claw/info` exposes service build metadata from `YA_CLAW_SERVICE_VERSION`, `YA_CLAW_SERVICE_COMMIT`, `YA_CLAW_SERVICE_BUILD`, and `YA_CLAW_SERVICE_IMAGE`; Docker builds inject these values for UI display.
- SQLite is the default durable store at `~/.ya-claw/ya_claw.sqlite3`; file-backed SQLite engines use WAL and a 30-second busy timeout for the runtime's concurrent readers and writers.
- `YA_CLAW_DATA_DIR` defaults to `~/.ya-claw/data`.
- `YA_CLAW_WORKSPACE_DIR` defaults to `~/.ya-claw/data/workspace`.
- Browser workspace downloads are capped at 100 MiB by default; configure `YA_CLAW_WORKSPACE_DOWNLOAD_MAX_BYTES` to change the enforced per-file limit. The server enforces the cap both before and during streaming.
- `GET /api/v1/sessions/{session_id}/workspace/files` uses stable case-insensitive name ordering. Send `limit`, then continue with the opaque `next_cursor` while `has_more` is true; the name-key cursor prevents inserts or deletes before the cursor from shifting later pages. `offset`, `next_offset`, and the `truncated` alias remain available for backwards compatibility, but new clients should use `cursor`/`next_cursor`.
- The default Docker workspace image is `ghcr.io/wh1isper/ya-claw-workspace:latest`.
- Session metadata lives in the database; committed continuity blobs live in the local run store.

### Implementation Conventions

- Runtime code is organized around `ya_claw/api/`, `ya_claw/controller/`, and `ya_claw/orm/`.
- Foundational execution modules live under `ya_claw/execution/`.
- Workspace provider modules live under `ya_claw/workspace/`.
- Internal data objects use Pydantic `BaseModel`.
- Code prefers explicit typing and `isinstance` checks.
- The session API is the high-level surface; the run API is the low-level surface.
- SQLite tests use the session-scoped `initialize_sqlite_database` fixture, which copies a schema-only template into each isolated database; avoid per-test `Base.metadata.create_all` calls.

### Session and Run Persistence

- Committed continuity blobs live in `run-store/{run_id}/state.json` and `run-store/{run_id}/message.json`.
- `message.json` stores the compacted replay list of AGUI-aligned events as a top-level JSON array.
- `state.json` stores the bounded latest cumulative usage/cost snapshot. Claw normalizes SDK snapshots to the outer run ID, accumulates deferred-tool continuation segments, and indexes the bounded snapshot in internal run metadata. Run detail may fall back through state and replay for legacy or active runs; paginated summaries never scan full state blobs.
- Input payloads use `input_parts`; run records preserve `input_parts` as original JSON-compatible payloads for replay and UI reconstruction.
- Successful run records store final `output_text` directly in the database.
- Session GET exposes paginated runs with optional raw `input_parts` and compacted message replay lists, returns optional top-level committed state/message from `head_success_run_id`, and derives session status from the latest run.
- Session turns API returns successful completed turns with raw `input_parts` and `output_text`.
- Run GET returns `session + run + optional state + optional message`.
- Run trace API returns compact tool-call/tool-response projections from `message.json`.
- Rerun can explicitly target failed or interrupted runs through `restore_from_run_id`.
- JSON run/session create routes return JSON consistently; foreground SSE creation uses `POST /api/v1/runs:stream`, `POST /api/v1/sessions:stream`, and `POST /api/v1/sessions/{session_id}/runs:stream`.

### Execution Coordination

- Active handles, live events, schedule dispatch, and bridge transports stay in the runtime process; relational rows remain the authority for runs, async tasks, input, and completion delivery.
- Built-in run orchestration lives in `ya_claw/execution/coordinator.py`; it retains SQL scheduling, workspace, HITL, memory, and delivery ownership while composing the SDK `AgentExecutionHarness` for each native segment.
- New ordinary runs and new child spawns resolve model/runtime behavior from the selected execution profile. Existing or resumed async-child executions restore the exact immutable descriptor persisted with their SQL execution record; profile drift, reseeding, or route deletion cannot redefine them.
- `YA_CLAW_DEFAULT_PROFILE` defaults to `default`.
- `YA_CLAW_CAPABILITY_PLUGIN_MANIFEST` optionally names one strict SDK plugin manifest;
  omission means no external plugins, while an explicit missing or invalid file is fatal.
- Runtime instance heartbeat lives in `runtime_instances`.
- Run records carry claim ownership through `claimed_by` and `claimed_at`.
- The built-in `session` toolset lets agents inspect their current session through internal HTTP client tools `list_session_turns` and `get_run_trace`; session ID and bearer token stay inside the client resource.

### Workspace Providers and Docker Runtime

- `LocalWorkspaceProvider` uses `LocalFileOperator` plus policy-driven `LocalShell` over the real workspace path. Claw passes resolved shell sandbox policy for local sandbox execution; raw host shell is controlled by explicit policy.
- `DockerWorkspaceProvider` uses Docker mounts through `SandboxEnvironment`; file operations map the service-visible workspace path to `/workspace`, and Docker shell uses `/workspace`. Managed temporary storage lives at a hidden `.tmp/ya-agent-<id>` path below that shared workspace mount, including for reused containers; each owned instance contains a self-ignoring `.gitignore`, and no per-run bind mount is required.
- Docker shell execution requires a Docker CLI on the service runtime `PATH` plus Docker Engine API access. The official `Dockerfile.ya-claw` image bundles the CLI; custom images and host installs must provide it, and mounting `docker.sock` alone is insufficient.
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` provides the Docker daemon-visible host mount path when the YA Claw service itself runs in Docker.
- Docker workspace containers receive UID/GID envs (`YA_CLAW_WORKSPACE_UID`, `YA_CLAW_WORKSPACE_GID`, `YA_CLAW_HOST_UID`, `YA_CLAW_HOST_GID`) from the service process by default or from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID/GID`.
- `Dockerfile.ya-claw` can drop service execution privileges through `YA_CLAW_RUN_UID` and `YA_CLAW_RUN_GID`.
- The official workspace image defaults to UID/GID 1000 through build args.

### Bridge Runtime

- Bridge adapter types are enumerated through `BridgeAdapterType`; the built-in adapters are `github` and `lark`.
- Bridge deployment dispatch uses `BridgeDispatchMode` (`embedded`, `manual`) and stays separate from run execution dispatch (`queue`, `async`, `stream`).
- `embedded` is the default bridge dispatch mode and runs adapter tasks under `BridgeSupervisor` in the same HTTP server lifespan as `ExecutionSupervisor`.
- `manual` starts the HTTP server with bridge dispatch managed outside the server lifespan.
- GitHub bridge ingress polls every notification reason for an ordinary account with a classic PAT, routes Issue/PR subjects, and applies `YA_CLAW_BRIDGE_GITHUB_ALLOWED_SENDERS` with exact case-insensitive logins or `*`.
- GitHub Issue/PR resources map one-to-one to durable sessions, inherit the configured default workspace, and use a durable timestamp cursor plus versioned notification IDs for restart-safe replay.
- Lark bridge event allowlist comes from `YA_CLAW_BRIDGE_LARK_EVENT_TYPES`; defaults cover `im.chat.member.bot.added_v1`, `im.chat.member.user.added_v1`, `im.message.receive_v1`, and `drive.notice.comment_add_v1`.
- Lark message events map `(adapter, tenant_key, chat_id)` one-to-one to a session.
- Other accepted Lark events use `chat_id` when present and fall back to stable event or Drive conversation keys.
- Each accepted inbound event creates a bridge-triggered run after event/message dedupe.
- GitHub replies/actions are performed by the agent from the workspace with bundled `gh`; `YA_CLAW_BRIDGE_GITHUB_TOKEN` is exposed there as `GH_TOKEN`.
- Lark bridge replies/actions are performed by the agent from the workspace with `lark-cli`.
- Workspace environments receive `GH_TOKEN` from GitHub bridge settings and `LARK_APP_ID` / `LARK_APP_SECRET` from process env or Lark bridge app settings.

### Session Memory

- Session memory is workspace-native.
- Paired internal `session_type="memory"` sessions run background extract/summary jobs with trigger type `memory`.
- Memory jobs share the source workspace sandbox and use the same profile tool surface as the primary agent.
- Memory content lives in workspace files: `memory/MEMORY.md`, `memory/CHANGELOG.md`, and `memory/YYYYMMDD-event.md` files with YAML frontmatter (`name`, `description`).
- `memory/MEMORY.md` is a compact durable brief for stable facts loaded by the main agent. Detailed chronology, file catalogs, and event lists belong in event files and `memory/CHANGELOG.md`.
- Memory extract and summary agents use fixed XML-style prompts from `ya_claw/memory/extract_prompt.py` and `ya_claw/memory/summary_prompt.py`.
- Primary conversation runs load workspace guidance from `AGENTS.md` and inject memory in the system prompt via `WorkspaceMemoryStore`, loading `memory/MEMORY.md` plus event file frontmatter as separate XML-style blocks.
- `memory-context` is registered in `injected_context_tags`, so SDK trim-mode handoff removes historical memory context from prompt history.
- Memory orchestration state lives in `session_memory_states`.
- Session list/detail responses expose `memory_state`.
- Manual endpoints are `memory:extract` and `memory:summarize`.
- File browsing should use workspace filetree APIs.

## Installation

Recommended standalone install with uv:

```bash
uv tool install 'ya-claw[rs]'
ya-clawd version
```

`[rs]` passes `ya-agent-sdk[all,rs]` into the runtime and installs the native Rust filesystem search binding. The equivalent extra-dependency form is:

```bash
uv tool install ya-claw --with ya-ripgrep-core
```

`ya-ripgrep-core` is a library dependency, so `--with` is the matching uv form; `--with-executables-from` applies to companion packages that also expose CLI executables.

Pip can install the same runtime shape:

```bash
pip install 'ya-claw[rs]'
```

## Capability Plugins

Install each trusted plugin distribution into the YA Claw service's Python environment,
not only into a Docker workspace container. For an isolated uv tool installation:

```bash
uv tool install 'ya-claw[rs]' --with acme-agent-plugin
```

Then point YA Claw at the SDK's strict manifest:

```env
YA_CLAW_CAPABILITY_PLUGIN_MANIFEST=/etc/ya-claw/plugins.toml
```

```toml
schema_version = 1
entry_points = ["acme.search"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }
```

`entry_points` explicitly selects installed Python types for the process catalog.
`capabilities` appends ordered grants to every admitted root profile spec before its
immutable descriptor is fingerprinted and persisted. Installation alone grants nothing,
and YA Claw never ambiently loads all installed entry points.

The manifest path is optional; omission produces an empty external catalog. An explicit
missing path, invalid TOML, unknown field, unsupported version, missing or duplicate
entry point, import failure, grant argument signature mismatch, or catalog collision
stops startup. Manifest arguments are durable non-secret configuration; secret-like
keys are rejected recursively. This name-based guard cannot detect a secret stored under
a neutral key. Provide live credentials and authority through typed runtime dependencies
or host APIs instead.

Root manifest grants do not enter named children or self forks. A selected type may be
used by a child only when that child's native `AgentSpec.capabilities` explicitly grants
it. Before a root descriptor is fingerprinted or persisted, YA Claw performs native
static-plan construction with `ClawAgentContext`, the selected custom types, and fresh
base host capabilities; plugin factory values and combined ordering therefore fail at
admission rather than after a run is queued. Runtime entry still validates dynamic
Environment and resource contributions.

YA Claw resolves the file once before application startup and uses the same catalog
snapshot for profile admission, async children, retained plans, recovery, memory runs,
and runtime construction. Restart the service after changing the file or distribution.

The official image does not include third-party plugins. Build a derived service image
that installs the distribution into `/opt/venv`, mount the manifest read-only, and set
its in-container path:

```dockerfile
FROM ghcr.io/wh1isper/ya-claw:latest
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --python /opt/venv/bin/python acme-agent-plugin
```

```yaml
services:
  ya-claw:
    build: .
    environment:
      YA_CLAW_CAPABILITY_PLUGIN_MANIFEST: /etc/ya-claw/plugins.toml
    volumes:
      - ./plugins.toml:/etc/ya-claw/plugins.toml:ro
```

See the SDK [file configuration contract](../ya-agent-sdk/spec/06-capability-plugins/03-file-configuration.md)
and the [installable example](../../examples/capability_plugin/).

## Quick Start

From the workspace root, start the default runtime flow:

```bash
uv sync --all-packages
cp packages/ya-claw/.env.example packages/ya-claw/.env
make run-claw
```

Set `YA_CLAW_API_TOKEN` before starting the service.
The development server listens on `http://127.0.0.1:9042` by default.
YA Claw loads `YA_CLAW_*` settings from `packages/ya-claw/.env` and the process environment.
YA Claw startup also exports provider variables such as `GATEWAY_API_KEY` and `GATEWAY_BASE_URL` from `packages/ya-claw/.env` into the process environment.
Use [`packages/ya-agent-sdk/.env.example`](../ya-agent-sdk/.env.example) for shared SDK and tool environment variables when you want the same keys outside YA Claw startup.
Set `YA_CLAW_PROFILE_SEED_FILE` plus `YA_CLAW_AUTO_SEED_PROFILES=true` when you want packaged profiles to seed into the database on startup. Seeded profiles use create/update semantics: every startup refreshes matching database profiles from the YAML file, including subagent configuration, while profiles absent from the YAML file remain in the database.
Runs auto-dispatch through the built-in coordinator. New ordinary runs and child spawns resolve model/runtime behavior from execution profile rows; existing or resumed async children use their exact persisted immutable descriptor instead. The default profile name is `default`.

Profile, MCP, and coordinator settings:

- `YA_CLAW_PROFILE_SEED_FILE=packages/ya-claw/profiles.yaml`
- `YA_CLAW_AUTO_SEED_PROFILES=true`
- `YA_CLAW_DEFAULT_PROFILE=default`
- `YA_CLAW_CAPABILITY_PLUGIN_MANIFEST=/etc/ya-claw/plugins.toml`
- `YA_CLAW_WORKSPACE_PROVIDER_BACKEND=local|docker`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IMAGE=ghcr.io/wh1isper/ya-claw-workspace:latest`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID=<service process UID>`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID=<service process GID>`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME=/home/claw`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR=~/.ya-claw/data/docker-workspace-containers`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS=/srv/ya-claw/home:/home/claw:rw,/srv/ya-claw/cache:/cache:ro`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_RETENTION_POLICY=stop_on_idle`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IDLE_TTL_SECONDS=3600`
- `YA_CLAW_WORKSPACE_ENV_VARS=MY_TOOL_API_KEY,MY_TOOL_ENDPOINT`
- `YA_CLAW_BRIDGE_DISPATCH_MODE=embedded|manual`
- `YA_CLAW_BRIDGE_ENABLED_ADAPTERS=lark`
- `YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx`
- `YA_CLAW_BRIDGE_LARK_APP_SECRET=...`
- `YA_CLAW_BRIDGE_LARK_DEFAULT_PROFILE=default`
- `YA_CLAW_BRIDGE_LARK_EVENT_TYPES=im.chat.member.bot.added_v1,im.chat.member.user.added_v1,im.message.receive_v1,drive.notice.comment_add_v1`
- `YA_CLAW_BRIDGE_LARK_REPLY_IDENTITY=bot`
- `LARK_APP_ID=cli_xxx`
- `LARK_APP_SECRET=...`
- `LARKSUITE_CLI_BRAND=feishu`
- `LARKSUITE_CLI_DEFAULT_AS=bot`
- `LARKSUITE_CLI_STRICT_MODE=bot`
- `MALLOC_ARENA_MAX=2`
- `MALLOC_TRIM_THRESHOLD_=131072`

The official YA Claw service and workspace Docker images set `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=131072` for long-lived Python workloads. Use the same allocator values for systemd or custom container deployments when memory residency matters.

Profiles use strict schema version 2: a native Pydantic AI `AgentSpec`, a Claw-only host policy, and native SDK `SubagentSpec` children. Portable behavior is granted through `AgentSpec.capabilities`; Claw host groups are limited to session, schedule, workflow, and agency control-plane tools. Host policy owns model runtime configuration, approvals, MCP selection and server definitions, and workspace hints. See [`spec/01-configuration-and-workspace-provider.md`](spec/01-configuration-and-workspace-provider.md).

Codex OAuth profiles use the `oauth@codex:gpt-5.5` model string after the service host has run `ya-oauth login codex`. YA Claw binds `AgentContext.run_id` and provider thread headers to the durable run ID from runtime construction, while provider session headers use the durable YA Claw session ID. The resulting `x-session-id` stays stable across turns in one session, and capable model configurations derive `openai_prompt_cache_key` from that exact value. Every subagent runs in its own durable child session, so its `x-session-id` differs from the main agent and stays stable when the child is resumed with a new run. Docker deployments should mount a persistent host directory to the service user's `~/.yaai`, keep directory mode `0700` and `auth.json` mode `0600`, and keep credentials out of image layers.

Shell review is configured under `host.model_config_override.security.shell_review`. The review model is explicit when enabled, and `model_settings` accepts an SDK preset name or inline settings. Interactive runs can defer for HITL; unattended schedule, workflow, heartbeat, and agency runs convert deferred review to denial.

```yaml
version: 2
profiles:
  - schema_version: 2
    name: default
    agent:
      model: gateway@openai-responses:gpt-5.5
      name: default
      capabilities: [FilesystemCapability, ShellCapability]
    host:
      model_config_preset: gpt5_270k
      model_config_override:
        security:
          shell_review:
            enabled: true
            model: gateway@openai-responses:gpt-5.4-mini
            model_settings: openai_responses_low
            on_needs_approval: defer
            risk_threshold: extra_high
      tool_groups: [session]
    subagents: []
```

Session and run requests can provide `workspace.mounts` with one or more logical workspace folders, one default mount, a default cwd, and `rw` or `ro` access per mount. When requests omit workspace configuration, YA Claw uses the shared workspace configured by `YA_CLAW_WORKSPACE_DIR` and maps it to `/workspace`. Workspace guidance and memory use the default logical mount, and runtime prompts list the resolved mount set.

Workspace environments receive `GH_TOKEN` from `YA_CLAW_BRIDGE_GITHUB_TOKEN`, plus `LARK_APP_ID` and `LARK_APP_SECRET` from explicit process environment values or from the configured Lark bridge app settings. The official Docker workspace entrypoint writes Lark values into `/home/claw/.lark-cli/config.json` for `lark-cli` bot commands, and clears `LARKSUITE_CLI_APP_ID` / `LARKSUITE_CLI_APP_SECRET` in the container runtime environment so `lark-cli` uses the generated config profile. `LARKSUITE_CLI_BRAND`, `LARKSUITE_CLI_DEFAULT_AS`, and `LARKSUITE_CLI_STRICT_MODE` tune that generated profile. `YA_CLAW_WORKSPACE_ENV_VARS` forwards additional comma-separated process environment variable names into workspace environments and may explicitly override built-in aliases. For Docker workspaces, values are passed at session or run workspace container creation time.

The default Docker workspace image is `ghcr.io/wh1isper/ya-claw-workspace:latest`. It is based on Debian stable and includes Python, Node.js, GitHub CLI (`gh`), `lark-cli`, and bundled workspace skills copied into mounted workspaces at container start. Auto-started workspace containers receive `YA_CLAW_WORKSPACE_UID`, `YA_CLAW_WORKSPACE_GID`, `YA_CLAW_HOST_UID`, and `YA_CLAW_HOST_GID`; the default values come from the YA Claw service process UID/GID and can be overridden with `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID` and `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID`. Docker exec uses `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto` by default, which resolves to the configured workspace UID:GID, and sets `HOME` from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME` with default `/home/claw`. `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS` adds comma-separated provider support mounts to Docker workspace containers, with `rw` and `ro` modes. Docker session sandboxes use generation-specific containers named `ya-claw-session-{session_id_short}-gN`, store cache metadata under `~/.ya-claw/data/docker-workspace-containers/sessions/{session_id}/workspace.json`, resolve and store the Docker image digest before reuse, refresh `last_used_at` during active runs and once on run exit, and stop idle containers according to `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_RETENTION_POLICY` and `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IDLE_TTL_SECONDS`. TTL cleanup deletes the session `workspace.json` cache file and uses the same cache-path lock as run startup. A changed image digest causes YA Claw to remove the stale container and start a new one from the current image. Schedule and heartbeat runs use run-scoped containers named `ya-claw-run-{run_id_short}` and cache metadata under `runs/{run_id}/workspace.json`.

Profiles can be managed through:

- REST API: `/api/v1/profiles`
- Seed API: `POST /api/v1/profiles/seed`
- CLI: `ya-claw profiles seed`

Default local paths:

- SQLite database: `~/.ya-claw/ya_claw.sqlite3`
- runtime data root: `~/.ya-claw/data`
- workspace directory: `~/.ya-claw/data/workspace`

## External Database

Set `YA_CLAW_DATABASE_URL` in `packages/ya-claw/.env` when you want an external PostgreSQL database.
The default SQLite file stays at `~/.ya-claw/ya_claw.sqlite3`.

## Database Commands

```bash
uv run --package ya-claw ya-claw db upgrade
uv run --package ya-claw ya-claw db current
uv run --package ya-claw ya-claw db history
uv run --package ya-claw ya-claw db revision "add session tables"
```

## Bridge Commands

The CLI owns a top-level bridge command group.

```bash
uv run --package ya-claw ya-claw bridge ls
uv run --package ya-claw ya-claw bridge run lark
uv run --package ya-claw ya-claw bridge serve lark
```

### Bridge Dispatch

Bridge dispatch controls whether the YA Claw HTTP server starts bridge adapters:

- `embedded` starts enabled adapters inside the YA Claw server lifespan under `BridgeSupervisor`.
- `manual` starts the YA Claw HTTP server without starting `BridgeSupervisor`.

Bridge adapters submit inbound events through the same session/run controller path used by HTTP requests, so bridge ingress behaves as a self-request inside the service process.

The GitHub bridge polls `GET /notifications` with `all=true` from a durable cursor, follows every page, honors `X-Poll-Interval`, and uses a 60-second overlap on established cursors. It accepts Issue and Pull Request subjects from exact case-insensitive sender logins configured in `YA_CLAW_BRIDGE_GITHUB_ALLOWED_SENDERS`; `*` disables the sender gate. Notification thread ID plus `updated_at` forms both the event and message ID. Repository ID plus resource kind and number forms the conversation key, so one Issue/PR reuses one session and the configured default workspace. A classic PAT is required because GitHub Notifications does not support fine-grained PATs. `YA_CLAW_BRIDGE_GITHUB_TOKEN` is also injected into workspaces as `GH_TOKEN` for `gh` and `git` operations. See [GitHub Notification Bridge](spec/14-github-bridge.md).

The Lark bridge reads `YA_CLAW_BRIDGE_LARK_EVENT_TYPES` as a comma-separated event allowlist. The default allowlist covers bot-added-to-chat, user-added-to-chat, message receive, and Drive comment notification events. Message receive events map each `tenant_key + chat_id` pair to one YA Claw session. Other Lark events use `chat_id` when present and fall back to a stable event or Drive conversation key. Every accepted inbound event creates a queued bridge-triggered run, and the agent replies or acts from the workspace with `lark-cli`.

## Web Shell

Run the web shell from the repository root:

```bash
make web-dev
```

## Docker

Build the YA Claw service image from the repository root:

```bash
docker build -f Dockerfile.ya-claw -t ya-claw:dev .
```

Build the official workspace image locally:

```bash
docker build -f Dockerfile.ya-claw-workspace -t ya-claw-workspace:dev .
```

Build the workspace image with a default UID/GID baked in:

```bash
docker build \
  --build-arg WORKSPACE_UID=1000 \
  --build-arg WORKSPACE_GID=1000 \
  -f Dockerfile.ya-claw-workspace \
  -t ya-claw-workspace:dev .
```

Run the YA Claw service image under a specific UID/GID:

```bash
docker run \
  -e YA_CLAW_RUN_UID=1000 \
  -e YA_CLAW_RUN_GID=1000 \
  -e YA_CLAW_API_TOKEN=replace-with-a-long-random-token \
  ya-claw:dev
```

## Initial API Surface

Every HTTP route except `/healthz` expects `Authorization: Bearer <YA_CLAW_API_TOKEN>`.

- `GET /healthz` — service health probe with storage and runtime component status
- `POST /api/v1/sessions` — create a session with optional first queued run and return JSON
- `POST /api/v1/sessions:stream` — create a session with a first run and stream foreground SSE events
- `GET /api/v1/sessions` — list sessions using the backwards-compatible unpaginated response; optional keyset parameters are available for transitional clients
- `GET /api/v1/sessions/page` — list lightweight session pages ordered by `(updated_at, id)`, with `total`, `has_more`, and continuation anchors; this endpoint skips live Docker reconciliation and latest-run output text by default
- `GET /api/v1/sessions/{session_id}` — inspect a session plus paginated runs, top-level committed state, and optional compacted message replay lists; set `include_head_payload=false` when only the run page is needed
- `POST /api/v1/sessions/{session_id}/memory:extract` — enqueue a background memory extract run for the source session
- `POST /api/v1/sessions/{session_id}/memory:summarize` — enqueue a background memory summary run for the source session
- `POST /api/v1/sessions/{session_id}/runs` — create a run under a session and return JSON
- `POST /api/v1/sessions/{session_id}/runs:stream` — create a run under a session and stream foreground SSE events
- `POST /api/v1/sessions/{session_id}/steer` — steer the active run through the session surface
- `POST /api/v1/sessions/{session_id}/interrupt` — interrupt the active run through the session surface
- `POST /api/v1/sessions/{session_id}/cancel` — cancel the active run through the session surface
- `POST /api/v1/runs` — create a run directly through the low-level surface and return JSON
- `POST /api/v1/runs:stream` — create a run directly and stream foreground SSE events
- `GET /api/v1/runs/{run_id}` — inspect a run plus session summary, committed state, and optional compacted message replay list
- `POST /api/v1/runs/{run_id}/steer` — steer a specific active run
- `POST /api/v1/runs/{run_id}/interrupt` — interrupt a specific active run
- `POST /api/v1/runs/{run_id}/cancel` — cancel a specific active run

Direct session/run/async-child steering succeeds only when the addressed logical run is actively accepting input. A queued run, terminal run, or session without an active accepting run returns HTTP 409 before any success response. Success returns the persisted SQL/native receipt fields `input_id`, `input_delivery_key`, `input_disposition`, `input_sdk_id`, and `input_enqueue_id`; an equal idempotent retry returns the same durable receipt. Unified session submit is a separate API and may merge input into a queued run.

Portable SDK subagent spawn is claimed atomically in SQL by owner scope and idempotency key. The task, child session, child run, immutable plan, intent digest, and `applied` initial-input marker commit together before run publication. Concurrent or post-crash retries return that one committed task; a changed intent is rejected and cannot create an orphan child.

## Spec Set

- [`spec/README.md`](spec/README.md)
- [`spec/00-overview.md`](spec/00-overview.md)
- [`spec/01-configuration-and-workspace-provider.md`](spec/01-configuration-and-workspace-provider.md)
- [`spec/02-execution-and-session.md`](spec/02-execution-and-session.md)
- [`spec/03-storage-and-streaming.md`](spec/03-storage-and-streaming.md)
- [`spec/04-api.md`](spec/04-api.md)
- [`spec/05-web-ui-and-operations.md`](spec/05-web-ui-and-operations.md)
