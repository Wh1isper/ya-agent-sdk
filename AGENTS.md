## Repository Overview

`ya-mono` is a workspace-first monorepo managed with `uv`.

Workspace members:

- `packages/ya-agent-sdk` — SDK for building AI agents with Pydantic AI
- `packages/yaacli` — TUI reference implementation built on top of the SDK
- `packages/ya-claw` — workspace-native single-node runtime web service with `WorkspaceProvider`, in-process runtime state, schedules, bridges, and SQLite-first storage
- `packages/ya-agent-platform` — WIP stateless agent service with TBD scope
- `packages/ya-relay` — spec-only for now; provider-neutral relay protocol and SDK Environment mapping for external fileops, shell, resources, custom tools, artifacts, and computer use

Shared repository areas:

- `apps/` — frontend applications and user-facing shells
- `skills/` — canonical skill sources and reference material
- `examples/` — runnable SDK examples
- `scripts/` — repository automation scripts
- `.github/` — CI and release workflows
- `Dockerfile.ya-claw` — YA Claw image build
- `Dockerfile.ya-claw-workspace` — official YA Claw Docker workspace image build
- `Dockerfile.ya-agent-platform` — YA Agent Platform image build
- `.dockerignore` — Docker build context rules

## Primary Package Focus

Most architecture work in this repository targets `packages/ya-agent-sdk` and `packages/ya-claw`.

- **Language**: Python 3.11+
- **Package Manager**: uv
- **Build System**: hatchling
- **Frontend Stack**: Vite + React + TypeScript

## Package Directions

### `packages/ya-agent-sdk`

- SDK for building AI agents with Pydantic AI
- preserves the core execution primitives used across the repository
- changes here should keep examples, skills, and package docs aligned

### `packages/yaacli`

- TUI reference implementation built on top of `ya-agent-sdk`
- runtime-facing CLI behavior belongs here

### `packages/ya-claw`

- active runtime product in this repository
- current delivery target is a single-node runtime
- `WorkspaceProvider` is the core extension boundary
- active session state, live events, async task coordination, schedules, and bridge coordination stay in process
- SQLite is the default durable store
- PostgreSQL is an optional durable store for deployments that prefer an external database
- local filesystem stores committed session continuity data
- requires `YA_CLAW_API_TOKEN` before service startup
- defaults: SQLite at `~/.ya-claw/ya_claw.sqlite3`, runtime data at `~/.ya-claw/data`, workspace root at `~/.ya-claw/workspace`, Docker workspace image `ghcr.io/wh1isper/ya-claw-workspace:latest`
- implementation style: organize runtime code by `api/`, `controller/`, and `orm/`
- internal data objects use Pydantic `BaseModel`
- code prefers explicit typing and `isinstance` checks
- session API is the high-level surface and run API is the low-level surface
- session metadata lives in the database
- committed continuity blobs live in `run-store/{run_id}/state.json` and `run-store/{run_id}/message.json`
- `message.json` stores the compacted replay list of AGUI-aligned events as a top-level JSON array
- session GET exposes paginated runs with optional raw `input_parts` and compacted message replay lists, returns optional top-level committed state/message from `head_success_run_id`, and derives session status from the latest run
- session turns API returns successful completed turns with raw `input_parts`, `output_text`, and `output_summary`
- run GET returns `session + run + optional state + optional message`; run trace API returns compact tool-call/tool-response projections from `message.json`
- built-in `session` toolset lets agents inspect only their current session via internal HTTP client tools `list_session_turns` and `get_run_trace`; session ID and bearer token stay inside the client resource
- runtime instance heartbeat lives in `runtime_instances`; run records carry claim ownership through `claimed_by` and `claimed_at`
- rerun can explicitly target failed or interrupted runs through `restore_from_run_id`
- input payloads use `input_parts` rather than a single `input_text`; run records preserve `input_parts` as original JSON-compatible payloads for replay/UI reconstruction
- successful run records store final `output_text` directly in the database and keep `output_summary` for compact displays
- foundational execution modules live under `ya_claw/execution/`
- workspace provider modules live under `ya_claw/workspace/`
- `LocalWorkspaceProvider` uses `LocalFileOperator` plus `LocalShell` over the real workspace path
- `DockerWorkspaceProvider` uses Docker mounts through `SandboxEnvironment`; file operations map the service-visible workspace path to `/workspace`, and Docker shell uses `/workspace`
- `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` provides the Docker daemon-visible host mount path when the YA Claw service itself runs in Docker
- Docker workspace containers receive UID/GID envs (`YA_CLAW_WORKSPACE_UID`, `YA_CLAW_WORKSPACE_GID`, `YA_CLAW_HOST_UID`, `YA_CLAW_HOST_GID`) from the service process by default or from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID/GID`
- `Dockerfile.ya-claw` can drop service execution privileges through `YA_CLAW_RUN_UID` and `YA_CLAW_RUN_GID`; the official workspace image defaults to UID/GID 1000 through build args
- built-in run orchestration lives in `ya_claw/execution/coordinator.py`
- built-in coordinator dispatch resolves model/runtime behavior from AgentProfile rows; `YA_CLAW_DEFAULT_PROFILE` defaults to `default`
- bridge adapter types are enumerated through `BridgeAdapterType`; current built-in adapter is `lark`
- bridge deployment dispatch uses `BridgeDispatchMode` (`embedded`, `manual`) and stays separate from run execution dispatch (`queue`, `async`, `stream`)
- `embedded` is the default bridge dispatch mode and runs adapter tasks under `BridgeSupervisor` in the same HTTP server lifespan as `ExecutionSupervisor`; `manual` starts the HTTP server without `BridgeSupervisor`
- Lark bridge event allowlist comes from `YA_CLAW_BRIDGE_LARK_EVENT_TYPES`; defaults cover `im.chat.member.bot.added_v1`, `im.chat.member.user.added_v1`, `im.message.receive_v1`, and `drive.notice.comment_add_v1`
- Lark message events map `(adapter, tenant_key, chat_id)` one-to-one to a session; other accepted Lark events use `chat_id` when present and fall back to stable event or Drive conversation keys; each accepted inbound event creates a bridge-triggered run after event/message dedupe
- Lark bridge replies/actions are performed by the agent from the workspace with `lark-cli`; workspace environments receive `LARK_APP_ID` and `LARK_APP_SECRET` from process env or Lark bridge app settings
- JSON run/session create routes return JSON consistently; foreground SSE creation uses `POST /api/v1/runs:stream`, `POST /api/v1/sessions:stream`, and `POST /api/v1/sessions/{session_id}/runs:stream`
- session memory is workspace-native: paired internal `session_type="memory"` sessions run background extract/summary jobs with trigger type `memory`, share the source workspace sandbox, and use the same profile tool surface as the primary agent
- memory agents use fixed XML-style prompts from `ya_claw/memory/extract_prompt.py` and `ya_claw/memory/summary_prompt.py`
- memory content lives in workspace files: `memory/MEMORY.md`, `memory/CHANGELOG.md`, and `memory/YYYYMMDD-event.md` files with YAML frontmatter (`name`, `description`)
- `memory/MEMORY.md` is a compact durable brief for stable facts loaded into the main agent system prompt; detailed chronology, file catalogs, and event lists belong in event files and `memory/CHANGELOG.md`
- primary conversation runs load `AGENTS.md` through workspace guidance and load memory in the system prompt via `WorkspaceMemoryStore` from `memory/MEMORY.md` plus event file frontmatter
- `memory-context` is registered in `injected_context_tags` so SDK trim-mode handoff strips historical memory context from user prompt history
- memory orchestration state lives in `session_memory_states`; memory content lives in workspace files; session list/detail responses expose `memory_state`; file browsing uses workspace filetree APIs and agent filesystem tools; manual endpoints are `memory:extract` and `memory:summarize`

### `packages/ya-agent-platform`

- WIP stateless agent service with TBD scope

## YA Desktop Direction

- `apps/ya-desktop` is a standalone Tauri 2 + TypeScript desktop product, not a direct clone of the Claw Web UI.
- Use a white, ChatGPT-like theme with a more polished desktop feel.
- Keep top-level navigation focused: `Home`, `Chats`, `Board`, `Spaces`, `Inbox`, and bottom `Settings`.
- `Home` owns the command-first experience and recent conversation overview.
- `Chats` owns conversation-first work management; a chat is the primary product object backed by Claw sessions/runs.
- `Board` owns kanban-style organization of chats by status, priority, or workspace.
- `Spaces` owns workspace folders, local/remote/cloud workspace identity, runtime connection, trust, sidecar status, and execution location.
- `Inbox` owns approvals, alerts, failed background work, and user decisions.
- `Settings` owns preferences, hotkeys, notifications, tokens/keychain, advanced runtime, logs, and diagnostics.
- After each YA Desktop product or implementation change, review `apps/ya-desktop/spec/*` and update the specs when the product model, navigation, architecture, or behavior changes.

## Development Workflow

After changing code, run:

1. `make lint`
2. `make check`
3. `make test`

Useful commands:

| Command                            | Description                               |
| ---------------------------------- | ----------------------------------------- |
| `make run-claw`                    | Run the YA Claw backend                   |
| `make web-dev`                     | Run the YA Claw web app                   |
| `make build-claw`                  | Build the `ya-claw` package               |
| `make build-platform`              | Build the WIP `ya-agent-platform` package |
| `make docker-build-claw`           | Build the YA Claw Docker image            |
| `make docker-build-claw-workspace` | Build the YA Claw workspace Docker image  |
| `make docker-build-platform`       | Build the YA Agent Platform Docker image  |

## Environment Configuration

Environment variables are loaded via `pydantic-settings` from the process environment or `.env` files.

- YA Agent SDK example env file: `packages/ya-agent-sdk/.env.example`
- YAACLI example env file: `packages/yaacli/.env.example`
- YA Claw example env file: `packages/ya-claw/.env.example`
- Example runtime env file: `examples/.env.example`
- YAACLI runtime env prefix: `YAACLI_`
- YA Agent SDK runtime env prefix: `YA_AGENT_`
- YA Claw runtime env prefix: `YA_CLAW_`

Keep `packages/ya-agent-sdk/.env.example`, `packages/yaacli/.env.example`, `packages/ya-claw/.env.example`, and `examples/.env.example` updated when environment variables change.

## Notes For Repository Changes

When editing workspace metadata, keep these files aligned:

- `pyproject.toml`
- `packages/ya-agent-sdk/pyproject.toml`
- `packages/yaacli/pyproject.toml`
- `packages/ya-claw/pyproject.toml`
- `packages/ya-agent-platform/pyproject.toml`
- `pnpm-workspace.yaml`
- `Makefile`
- `.github/workflows/*.yml`
- `Dockerfile.ya-claw`
- `Dockerfile.ya-claw-workspace`
- `Dockerfile.ya-agent-platform`
- `.dockerignore`
- `README.md` and package READMEs
- `packages/ya-claw/spec/*`
- `skills/agent-builder/*`
- `scripts/sync-skills.sh`
