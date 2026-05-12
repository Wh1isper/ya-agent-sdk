# YA Claw

Workspace-native single-node agent runtime and web service for the `ya-mono` workspace.

## Scope

YA Claw packages a durable runtime shell around `ya-agent-sdk` with:

- one persistent workspace resolved through `WorkspaceProvider`
- reusable agent profiles
- resumable sessions and runs
- in-process active state and async task coordination
- session schedules for timed execution
- SQLite-first durable state with optional PostgreSQL
- local filesystem session continuity and exported state
- a bundled web shell for local and self-hosted use
- bridge adapters that connect IM channels to the YA Claw service

## Current Direction

The target single-node shape runs as one web service.
The runtime keeps active session state, live delivery, async tasks, schedule dispatch, and bridge coordination inside one runtime process.
SQLite is the default durable store.
PostgreSQL remains an optional storage backend for deployments that prefer an external relational database.

## Layout

Key areas in this package:

- `.env.example` — runtime environment example
- `spec/` — architecture and runtime design documents
- `tests/` — runtime tests
- `ya_claw/api/` — HTTP API surface
- `ya_claw/bridge/` — IM bridge adapters and relay logic
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
- SQLite is the default durable store at `~/.ya-claw/ya_claw.sqlite3`.
- `YA_CLAW_DATA_DIR` defaults to `~/.ya-claw/data`.
- `YA_CLAW_WORKSPACE_DIR` defaults to `~/.ya-claw/data/workspace`.
- The default Docker workspace image is `ghcr.io/wh1isper/ya-claw-workspace:latest`.
- Session metadata lives in the database; committed continuity blobs live in the local run store.

### Implementation Conventions

- Runtime code is organized around `ya_claw/api/`, `ya_claw/controller/`, and `ya_claw/orm/`.
- Foundational execution modules live under `ya_claw/execution/`.
- Workspace provider modules live under `ya_claw/workspace/`.
- Internal data objects use Pydantic `BaseModel`.
- Code prefers explicit typing and `isinstance` checks.
- The session API is the high-level surface; the run API is the low-level surface.

### Session and Run Persistence

- Committed continuity blobs live in `run-store/{run_id}/state.json` and `run-store/{run_id}/message.json`.
- `message.json` stores the compacted replay list of AGUI-aligned events as a top-level JSON array.
- Input payloads use `input_parts`; run records preserve `input_parts` as original JSON-compatible payloads for replay and UI reconstruction.
- Successful run records store final `output_text` directly in the database and keep `output_summary` for compact displays.
- Session GET exposes paginated runs with optional raw `input_parts` and compacted message replay lists, returns optional top-level committed state/message from `head_success_run_id`, and derives session status from the latest run.
- Session turns API returns successful completed turns with raw `input_parts`, `output_text`, and `output_summary`.
- Run GET returns `session + run + optional state + optional message`.
- Run trace API returns compact tool-call/tool-response projections from `message.json`.
- Rerun can explicitly target failed or interrupted runs through `restore_from_run_id`.
- JSON run/session create routes return JSON consistently; foreground SSE creation uses `POST /api/v1/runs:stream`, `POST /api/v1/sessions:stream`, and `POST /api/v1/sessions/{session_id}/runs:stream`.

### Execution Coordination

- Active session state, live events, async task coordination, schedules, and bridge coordination stay in the runtime process.
- Built-in run orchestration lives in `ya_claw/execution/coordinator.py`.
- Built-in coordinator dispatch resolves model/runtime behavior from `AgentProfile` rows.
- `YA_CLAW_DEFAULT_PROFILE` defaults to `default`.
- Runtime instance heartbeat lives in `runtime_instances`.
- Run records carry claim ownership through `claimed_by` and `claimed_at`.
- The built-in `session` toolset lets agents inspect their current session through internal HTTP client tools `list_session_turns` and `get_run_trace`; session ID and bearer token stay inside the client resource.

### Workspace Providers and Docker Runtime

- `LocalWorkspaceProvider` uses `LocalFileOperator` plus `LocalShell` over the real workspace path.
- `DockerWorkspaceProvider` uses Docker mounts through `SandboxEnvironment`; file operations map the service-visible workspace path to `/workspace`, and Docker shell uses `/workspace`.
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` provides the Docker daemon-visible host mount path when the YA Claw service itself runs in Docker.
- Docker workspace containers receive UID/GID envs (`YA_CLAW_WORKSPACE_UID`, `YA_CLAW_WORKSPACE_GID`, `YA_CLAW_HOST_UID`, `YA_CLAW_HOST_GID`) from the service process by default or from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID/GID`.
- `Dockerfile.ya-claw` can drop service execution privileges through `YA_CLAW_RUN_UID` and `YA_CLAW_RUN_GID`.
- The official workspace image defaults to UID/GID 1000 through build args.

### Bridge Runtime

- Bridge adapter types are enumerated through `BridgeAdapterType`; the current built-in adapter is `lark`.
- Bridge deployment dispatch uses `BridgeDispatchMode` (`embedded`, `manual`) and stays separate from run execution dispatch (`queue`, `async`, `stream`).
- `embedded` is the default bridge dispatch mode and runs adapter tasks under `BridgeSupervisor` in the same HTTP server lifespan as `ExecutionSupervisor`.
- `manual` starts the HTTP server with bridge dispatch managed outside the server lifespan.
- Lark bridge event allowlist comes from `YA_CLAW_BRIDGE_LARK_EVENT_TYPES`; defaults cover `im.chat.member.bot.added_v1`, `im.chat.member.user.added_v1`, `im.message.receive_v1`, and `drive.notice.comment_add_v1`.
- Lark message events map `(adapter, tenant_key, chat_id)` one-to-one to a session.
- Other accepted Lark events use `chat_id` when present and fall back to stable event or Drive conversation keys.
- Each accepted inbound event creates a bridge-triggered run after event/message dedupe.
- Lark bridge replies/actions are performed by the agent from the workspace with `lark-cli`.
- Workspace environments receive `LARK_APP_ID` and `LARK_APP_SECRET` from process env or Lark bridge app settings.

### Session Memory

- Session memory is workspace-native.
- Paired internal `session_type="memory"` sessions run background extract/summary jobs with trigger type `memory`.
- Memory jobs share the source workspace sandbox and use the same profile tool surface as the primary agent.
- Memory content lives in workspace files: `memory/MEMORY.md`, `memory/CHANGELOG.md`, and `memory/YYYYMMDD-event.md` files with YAML frontmatter (`name`, `description`).
- Memory extract and summary agents use fixed XML-style prompts from `ya_claw/memory/extract_prompt.py` and `ya_claw/memory/summary_prompt.py`.
- Primary conversation runs inject memory in the system prompt via `WorkspaceMemoryStore`, loading `memory/MEMORY.md` plus event file frontmatter as separate XML-style blocks.
- Memory orchestration state lives in `session_memory_states`.
- Session list/detail responses expose `memory_state`.
- Manual endpoints are `memory:extract` and `memory:summarize`.
- File browsing should use workspace filetree APIs.

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
Runs auto-dispatch through the built-in coordinator and resolve model/runtime behavior from AgentProfile rows. The default profile name is `default`.

Profile, MCP, and coordinator settings:

- `YA_CLAW_PROFILE_SEED_FILE=packages/ya-claw/profiles.yaml`
- `YA_CLAW_AUTO_SEED_PROFILES=true`
- `YA_CLAW_DEFAULT_PROFILE=default`
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
- `PYTHONMALLOC=malloc`

The official YA Claw service and workspace Docker images set `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=131072`, and `PYTHONMALLOC=malloc` for long-lived Python workloads. Use the same allocator values for systemd or custom container deployments when memory residency matters.

Profiles store model, prompt, model context config, builtin tool groups, subagents, approval policy, security policy, MCP server definitions, and MCP namespace filters. YA Claw accepts profile MCP servers with `streamable_http` transport. Every YA Claw agent runtime receives the profile MCP configuration through `ToolProxyToolset`, and each profile can narrow that surface with `enabled_mcps` and `disabled_mcps`.

Shell command review is configured per profile under `security.shell_review`. The review model is explicit when enabled, and `model_settings` accepts SDK preset names such as `openai_responses_low` or an inline settings object. YA Claw runs shell review in auto-pilot deny mode: commands that reach `risk_threshold` trigger the configured action, and profile values of `on_needs_approval: defer` are coerced to deny at runtime. The default profile risk threshold is `extra_high`.

```yaml
profiles:
  - name: default
    model: gateway@openai-responses:gpt-5.5
    model_settings_preset: openai_responses_high
    model_config_preset: gpt5_270k
    security:
      shell_review:
        enabled: true
        model: gateway@openai-responses:gpt-5.4-mini
        model_settings: openai_responses_low
        on_needs_approval: deny
        risk_threshold: extra_high
```

Session and run requests can provide `workspace.mounts` with one or more logical workspace folders, one default mount, a default cwd, and `rw` or `ro` access per mount. When requests omit workspace configuration, YA Claw uses the shared workspace configured by `YA_CLAW_WORKSPACE_DIR` and maps it to `/workspace`. Workspace guidance and memory use the default logical mount, and runtime prompts list the resolved mount set.

Workspace environments receive `LARK_APP_ID` and `LARK_APP_SECRET` from explicit process environment values or from the configured Lark bridge app settings. The official Docker workspace entrypoint writes these values into `/home/claw/.lark-cli/config.json` for `lark-cli` bot commands, and clears `LARKSUITE_CLI_APP_ID` / `LARKSUITE_CLI_APP_SECRET` in the container runtime environment so `lark-cli` uses the generated config profile. `LARKSUITE_CLI_BRAND`, `LARKSUITE_CLI_DEFAULT_AS`, and `LARKSUITE_CLI_STRICT_MODE` tune that generated profile. `YA_CLAW_WORKSPACE_ENV_VARS` forwards additional comma-separated process environment variable names into workspace environments. For Docker workspaces, forwarded values are passed at session or run workspace container creation time.

The default Docker workspace image is `ghcr.io/wh1isper/ya-claw-workspace:latest`. It is based on Debian stable and includes Python, Node.js, `lark-cli`, and bundled workspace skills copied into mounted workspaces at container start. Auto-started workspace containers receive `YA_CLAW_WORKSPACE_UID`, `YA_CLAW_WORKSPACE_GID`, `YA_CLAW_HOST_UID`, and `YA_CLAW_HOST_GID`; the default values come from the YA Claw service process UID/GID and can be overridden with `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID` and `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID`. Docker exec uses `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto` by default, which resolves to the configured workspace UID:GID, and sets `HOME` from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME` with default `/home/claw`. `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS` adds comma-separated provider support mounts to Docker workspace containers, with `rw` and `ro` modes. Docker session sandboxes use generation-specific containers named `ya-claw-session-{session_id_short}-gN`, store cache metadata under `~/.ya-claw/data/docker-workspace-containers/sessions/{session_id}/workspace.json`, resolve and store the Docker image digest before reuse, refresh `last_used_at` during active runs and once on run exit, and stop idle containers according to `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_RETENTION_POLICY` and `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IDLE_TTL_SECONDS`. TTL cleanup deletes the session `workspace.json` cache file and uses the same cache-path lock as run startup. A changed image digest causes YA Claw to remove the stale container and start a new one from the current image. Schedule and heartbeat runs use run-scoped containers named `ya-claw-run-{run_id_short}` and cache metadata under `runs/{run_id}/workspace.json`.

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

Bridge adapters submit inbound events through the same session/run controller path used by HTTP requests, so bridge ingress behaves as a self-request inside the service process. The Lark bridge reads `YA_CLAW_BRIDGE_LARK_EVENT_TYPES` as a comma-separated event allowlist. The default allowlist covers bot-added-to-chat, user-added-to-chat, message receive, and Drive comment notification events. Message receive events map each `tenant_key + chat_id` pair to one YA Claw session. Other Lark events use `chat_id` when present and fall back to a stable event or Drive conversation key. Every accepted inbound event creates a queued bridge-triggered run, and the agent replies or acts from the workspace with `lark-cli`.

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
- `GET /api/v1/sessions` — list sessions
- `GET /api/v1/sessions/{session_id}` — inspect a session plus paginated runs, top-level committed state, and optional compacted message replay lists
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

## Spec Set

- [`spec/README.md`](spec/README.md)
- [`spec/00-overview.md`](spec/00-overview.md)
- [`spec/01-configuration-and-workspace-provider.md`](spec/01-configuration-and-workspace-provider.md)
- [`spec/02-execution-and-session.md`](spec/02-execution-and-session.md)
- [`spec/03-storage-and-streaming.md`](spec/03-storage-and-streaming.md)
- [`spec/04-api.md`](spec/04-api.md)
- [`spec/05-web-ui-and-operations.md`](spec/05-web-ui-and-operations.md)
