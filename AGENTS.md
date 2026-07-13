## Repository Overview

`ya-mono` is a workspace-first monorepo managed with `uv`.

Workspace members:

- `packages/ya-agent-environment` — Environment abstractions for general agents
- `packages/ya-agent-sdk` — SDK for building AI agents with Pydantic AI
- `packages/ya-agent-stream-protocol` — shared stream protocol adapters between `ya-agent-sdk` and applications
- `packages/ya-oauth` — OAuth login, refresh, logout, token storage, and CLI for subscription-backed providers
- `packages/ya-oauth-provider` — Pydantic AI provider helpers for OAuth-backed model access
- `packages/yaacli` — TUI reference implementation built on top of the SDK
- `packages/ya-claw` — workspace-native single-node runtime web service with `WorkspaceProvider`, in-process runtime state, schedules, bridges, and SQLite-first storage
- `packages/ya-agent-platform` — WIP stateless agent service with TBD scope

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

### `packages/ya-agent-environment`

- shared base abstractions for agent environments
- implementation package import name is `ya_agent_environment`
- Environment base definitions live in this package.

### `packages/ya-agent-sdk`

- SDK for building AI agents with Pydantic AI
- preserves the core execution primitives used across the repository
- changes here should keep examples, skills, and package docs aligned
- OAuth-backed model strings use `oauth@provider:model`; Codex currently uses `oauth@codex:gpt-5.5`, uses the `gpt5_350k` model config for its subscription context window, and receives session/thread headers from `AgentContext.get_model_extra_headers()`
- Generic OpenAI Responses WebSocket transport lives in `ya-agent-sdk` under `ya_agent_sdk.agents.models.websocket`; aliases `openai-responses-ws:<model>` and `openai-responses-rs:<model>` are SDK core model strings and use `YA_AGENT_OPENAI_RESPONSES_WEBSOCKET_MODE` for `auto`/`websocket`/`http`
- Skill routing is two-stage: inspect plausible candidates with high recall, then activate only direct scope matches; inspected candidates are non-binding, activated skills are mandatory within scope, and compaction carries forward only activated skills still relevant to unfinished work

### `packages/ya-agent-stream-protocol`

- shared stream protocol layer between `ya-agent-sdk` and applications
- implementation package import name is `ya_agent_stream_protocol`
- owns AGUI event adaptation, compact replay buffers, message validation, and SSE framing helpers shared by YAACLI and YA Claw
- applications configure their own event namespaces through `AguiAdapterConfig`

### `packages/ya-oauth`

- CLI and storage package for OAuth-backed providers
- stores credentials in `~/.yaai/auth.json` with locked atomic writes, directory mode `0700`, and file mode `0600`
- `ya-oauth login codex` follows OpenAI Codex device-code auth and preserves Codex token refresh semantics

### `packages/ya-oauth-provider`

- Pydantic AI provider/model package for OAuth token sources
- owns Codex request auth/header alignment, including bearer token, ChatGPT account ID, FedRAMP, originator/version, both underscore/hyphen session and thread headers, and `x-client-request-id`
- reuses the SDK generic `WebsocketResponsesModel`; only Codex-specific headers, beta header, token refresh, and payload normalization belong here
- refreshes once on HTTP 401 through the configured token source

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
- browser workspace downloads have a configurable per-file cap through `YA_CLAW_WORKSPACE_DOWNLOAD_MAX_BYTES` (100 MiB by default); the server enforces the cap both before and during streaming
- implementation style: organize runtime code by `api/`, `controller/`, and `orm/`
- internal data objects use Pydantic `BaseModel`
- code prefers explicit typing and `isinstance` checks
- session API is the high-level surface and run API is the low-level surface
- session metadata lives in the database
- committed continuity blobs live in `run-store/{run_id}/state.json` and `run-store/{run_id}/message.json`
- `message.json` stores the compacted replay list of AGUI-aligned events as a top-level JSON array
- session GET exposes paginated runs with optional raw `input_parts` and compacted message replay lists, returns optional top-level committed state/message from `head_success_run_id` (skippable with `include_head_payload=false`), and derives session status from the latest run
- the Web session index uses `GET /api/v1/sessions/page`, a lightweight `(updated_at, id)` keyset page with total count; it omits latest-run output and live Docker reconciliation by default, while the backwards-compatible `GET /api/v1/sessions` list remains available
- session turns API returns successful completed turns with raw `input_parts` and `output_text`
- run GET returns `session + run + optional state + optional message`; run trace API returns compact tool-call/tool-response projections from `message.json`
- built-in `session` toolset lets agents inspect only their current session via internal HTTP client tools `list_session_turns` and `get_run_trace`; session ID and bearer token stay inside the client resource
- runtime instance heartbeat lives in `runtime_instances`; run records carry claim ownership through `claimed_by` and `claimed_at`
- rerun can explicitly target failed or interrupted runs through `restore_from_run_id`
- input payloads use `input_parts` rather than a single `input_text`; run records preserve `input_parts` as original JSON-compatible payloads for replay/UI reconstruction
- successful run records store final `output_text` directly in the database for replay and UI rendering
- foundational execution modules live under `ya_claw/execution/`
- workspace provider modules live under `ya_claw/workspace/`
- `LocalWorkspaceProvider` uses `LocalFileOperator` plus policy-driven `LocalShell` over the real workspace path; Claw passes resolved `ShellSandboxRuntimePolicy` for sandboxed execution, while SDK and YAACLI default local environments keep raw subprocess semantics unless a sandbox policy is provided
- Shell sandbox architecture target: reusable sandbox shell primitives live in `ya-agent-sdk` under `ya_agent_sdk.environment.shell_sandbox` split into `policy`, `backend`, and `shell` modules; shared subprocess lifecycle helpers live in `ya_agent_sdk.environment.process`; Claw keeps workspace-specific conversion in `ya_claw.workspace.shell_sandbox` and resolves profile/settings/workspace bindings into `ShellSandboxRuntimePolicy` before environment construction; default profile is `workspace_write`; Linux target backend is `linux_bwrap_seccomp` using bubblewrap plus seccomp with optional Landlock; macOS target backend is `macos_seatbelt`; Windows target backend is `windows_restricted_token` using restricted tokens, AppContainer, Job Objects, private desktop, and ACL grants; raw host shell is a privileged audited escalation path
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
- Agency heartbeat fires follow a fixed interval from `agency_timer_interval_seconds`; `submit_to_session` requires an explicit handoff kind (`context`, `exchange`, `reminder`, `task`, `risk`, `async_result`, `decision`, `conflict`), wraps prompts in a fixed `<system-reminder>` reference block, and uses kind-specific hints so target sessions can apply context or stay silent when useful
- `memory-context` is registered in `injected_context_tags` so SDK trim-mode handoff strips historical memory context from user prompt history
- memory orchestration state lives in `session_memory_states`; memory content lives in workspace files; session list/detail responses expose `memory_state`; file browsing uses workspace filetree APIs and agent filesystem tools; manual endpoints are `memory:extract` and `memory:summarize`

### `packages/ya-agent-platform`

- WIP stateless agent service with TBD scope

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
- `packages/ya-agent-environment/pyproject.toml`
- `packages/ya-agent-sdk/pyproject.toml`
- `packages/ya-agent-stream-protocol/pyproject.toml`
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
