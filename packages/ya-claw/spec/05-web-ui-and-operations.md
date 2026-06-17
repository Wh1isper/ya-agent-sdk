# 05 - Web UI and Operations

YA Claw ships with a bundled web shell and a simple single-node operations model.

## Web Shell Goal

The web shell is the first-party runtime console.

It should let a user:

- create and continue sessions
- manage workflows
- manage schedules
- inspect heartbeat configuration and fire history
- watch live run output
- read compacted conversation history for completed rounds
- inspect bridge endpoints and ingress activity
- inspect run summaries
- inspect workspace runtime health and session sandbox state
- manage execution profiles

The web shell acts as an application on top of YA Claw. It uses the single configured workspace exposed by the runtime.

## Web Shell Sections

```mermaid
flowchart LR
    HOME[Overview] --> SS[Sessions]
    HOME --> WF[Workflows]
    HOME --> SC[Schedules]
    HOME --> HB[Heartbeat]
    HOME --> PF[Profiles]
    HOME --> BR[Bridges]
    SS --> RV[Run View]
    RV --> RS[Run Summary]
```

### Overview

Shows runtime health, workspace backend health, active sessions, active workflows, ready sandbox count, active schedules, next schedule fire, heartbeat status, bridge activity, and recent runs. Workspace details include backend, runtime status, execution location, service path, virtual path, Docker image, and idle TTL when available.

### Sessions

Shows session lineage, latest state, workspace/sandbox state, continuation entry points, and compacted conversation history loaded from `message.json` in the run store. Docker-backed sessions can expose sandbox prepare and stop controls when the runtime advertises those capabilities.

### Workflows

Shows workflow definitions, workflow-specific schedule bindings, run history, live DAG state, node-linked sessions/runs, output previews, agent preset selection, and manual trigger controls. Workflows is the primary product column for durable orchestration.

The workflow console should use a three-pane shape:

- workflow column: searchable definitions with status, scope, tags, latest run, and linked schedule state
- detail pane: workflow description, input schema, definition document, metadata, schedule bindings, and manual trigger form
- activity pane: workflow runs, live node state, events, result projection, linked sessions, and linked run traces

The workflow detail view should support create, edit, archive, trigger, cancel, steer active node, retry node, open linked session, open linked run trace, and manage workflow schedules. Workflow schedule management creates and updates schedule records whose `execution_mode="workflow"`, while keeping their configuration visible from the selected workflow.

Workflow runs started by an agent should appear in the supervising session detail with compact progress and result links.

### Schedules

Shows schedule definitions and fire history for timed agent prompts. The default schedule list hides `deleted` schedules. The schedule console can expose a hidden view that calls `/api/v1/schedules?include_deleted=true` for auto-hidden expired one-time schedules and operator-deleted schedules.

Important fields:

- status
- trigger kind and next fire time
- execution mode: `continue_session`, `fork_session`, or `isolate_session`
- target session or source session when present
- active-session policy for `continue_session`
- last fire, last created session, and last run
- owner kind and owner session

The schedule detail view should support create, edit, pause, resume, delete, and manual trigger actions for timed agent prompts. Workflow-backed schedule records remain part of the schedule API and fire history model, and the first-party console presents their management inside the Workflows column.

### Heartbeat

Shows runtime-owned heartbeat configuration and fire history.

Important fields:

- enabled state
- cron expression and timezone
- profile
- prompt
- `HEARTBEAT.md` path and existence
- next fire time
- last fire status
- last session and run

The heartbeat view can expose an admin manual trigger action. Heartbeat management belongs to runtime and admin surfaces.

### Profiles

Shows profile CRUD, seed status, enabled built-in toolsets, model configuration, MCP filters, approval policy, and subagent configuration.

### Bridges

Shows bridge adapters, bridge dispatch mode, recent inbound events, conversation mapping, run dispatches, and channel health.

### Run View

Shows live event output, final summary, AGUI-aligned event flow, trigger type, source metadata, and error state when needed.

### Run Summary

Shows the final run result, commit metadata, source metadata, schedule or heartbeat fire links, and continuation readiness.

## Console API Contract

The web shell uses these API layers:

- `/api/v1/claw/info` for startup handshake, capability flags, storage model, workspace backend, and auth mode
- `/api/v1/claw/notifications` for global SSE notifications that refresh overview lists and selected session metadata
- `/api/v1/workspace/runtime` for workspace backend checks, execution location, Docker status, and sandbox capabilities
- `/api/v1/sessions/{session_id}/workspace` and sandbox lifecycle routes for selected-session workspace display and Docker sandbox control
- `/api/v1/sessions` and nested run routes for chat creation, continuation, lineage, turns, and committed replay
- `/api/v1/runs/{run_id}/events` and `/api/v1/sessions/{session_id}/events` for detailed AGUI-aligned live output
- `/api/v1/profiles` for AgentProfile management and workflow node preset selection
- `/api/v1/workflows` and `/api/v1/workflow-runs` for workflow definition management, triggering, live events, node steering, and run history
- `/api/v1/schedules` for schedule CRUD, manual trigger, and fire history
- `/api/v1/heartbeat/*` for effective heartbeat config, status, and fire history

The web shell should implement SSE through `fetch` and `ReadableStream` parsing so bearer authorization headers are sent consistently. The global notification stream updates collection state, while nested run and session streams render active AGUI output.

```mermaid
flowchart LR
    INFO[/claw/info/] --> SHELL[Console Shell]
    NOTIFY[/claw/notifications/] --> OVERVIEW[Overview and Lists]
    WORKSPACE[/workspace/runtime/] --> OVERVIEW
    WORKSPACE --> CHAT
    SESSIONS[/sessions/] --> CHAT[Chat Console]
    RUN_EVENTS[/runs/:id/events/] --> CHAT
    PROFILES[/profiles/] --> ADMIN[Profile Admin]
    WORKFLOWS[/workflows/] --> WF[Workflow Console]
    WF_EVENTS[/workflow-runs/:id/events/] --> WF
    SCHEDULES[/schedules/] --> SCHED[Schedule Console]
    HEARTBEAT[/heartbeat/] --> HB[Heartbeat Console]
```

## Startup Flow

The default startup path is:

01. load environment configuration
02. initialize the relational store and in-process runtime state manager
03. run migrations when auto-migrate is enabled
04. initialize execution supervisor
05. initialize workflow executor
06. initialize schedule dispatcher
07. initialize heartbeat dispatcher
08. initialize bridge subsystem
09. mount API routes
10. mount bundled web assets when present

## Health Model

`/healthz` should report:

- service status
- relational storage connectivity
- in-process runtime state manager health
- execution supervisor health
- workflow executor health
- schedule dispatcher health
- heartbeat dispatcher health
- bridge subsystem health
- optional web bundle availability

## Logging

The runtime should emit structured logs for:

- startup configuration summary
- workspace resolution failures
- run lifecycle transitions
- workflow trigger, node dispatch, supervision, and terminal lifecycle
- schedule trigger and dispatch lifecycle
- heartbeat trigger and dispatch lifecycle
- bridge ingress lifecycle
- event delivery failures
- shutdown and cleanup

## Shutdown Flow

On process shutdown, YA Claw stops ingress sources first, then waits for already active run tasks to finish before runtime state and database resources close:

1. stop heartbeat dispatcher
2. stop schedule dispatcher
3. stop workflow executor ingress
4. stop embedded bridge adapters
5. stop accepting new execution supervisor submissions
6. wait for active execution supervisor run tasks to complete
7. mark the runtime instance stopped
8. close runtime state, notification hub, and database engine

`YA_CLAW_SHUTDOWN_TIMEOUT_SECONDS` maps to Uvicorn graceful shutdown timeout. Leave it unset for an unlimited application shutdown wait, and configure orchestrator stop windows such as Docker Compose `stop_grace_period` or systemd `TimeoutStopSec` to cover the longest expected run.

## Local Deployment Baseline

Recommended local deployment shapes:

- one supervised process
- one Docker deployment
- one systemd-managed service on a host

Each shape should keep the same core baseline:

- one YA Claw web service
- one SQLite database by default
- optional PostgreSQL for external relational storage
- one persistent local data directory
- one configured workspace directory
- in-process active state, schedule dispatch, heartbeat dispatch, and bridge coordination

## Bridge Operations

The bridge subsystem lives inside the `ya-claw` package as both:

- a `ya_claw.bridge` subpackage for adapter implementations
- a `ya-claw bridge` CLI group for operational commands

Bridge deployment dispatch and run execution dispatch are separate runtime concepts:

- `embedded`: enabled adapters start inside the YA Claw HTTP server lifespan under `BridgeSupervisor`.
- `manual`: operators start bridge adapters outside the YA Claw HTTP server.

With the default `embedded` dispatch, one YA Claw service process supervises `ExecutionSupervisor`, `ScheduleDispatcher`, `HeartbeatDispatcher`, and `BridgeSupervisor`. Each enabled adapter runs as a long-lived async task under `BridgeSupervisor`. Bridge adapters submit inbound events through the same session/run controller path used by HTTP requests, so bridge ingress behaves as a self-request inside the service process before execution dispatch.

The built-in Lark adapter receives the comma-separated event allowlist from `YA_CLAW_BRIDGE_LARK_EVENT_TYPES`. The default allowlist covers `im.chat.member.bot.added_v1`, `im.chat.member.user.added_v1`, `im.message.receive_v1`, and `drive.notice.comment_add_v1`. Message receive events map `(adapter, tenant_key, chat_id)` to one session. Other accepted events use `chat_id` when present and fall back to a stable event or Drive conversation key. YA Claw stores inbound event records for idempotency and creates one queued bridge-triggered run per accepted event. The agent replies or acts from the workspace with `lark-cli`; workspace environments receive built-in `LARK_APP_ID` and `LARK_APP_SECRET` aliases from process variables or the configured Lark bridge app settings, plus variables explicitly listed in `YA_CLAW_WORKSPACE_ENV_VARS`.

Bridge adapter types are enumerated so future adapters can be added with the same controller and supervisor foundation. A bridge adapter may target platforms such as:

- Lark
- Slack
- Discord
- Telegram

## Docker Alignment

Three image definitions exist in the repository:

- `Dockerfile.ya-claw` for the active runtime
- `Dockerfile.ya-claw-workspace` for the default Docker workspace provider image
- `Dockerfile.ya-agent-platform` for the WIP stateless agent service image

### Docker Startup

The `ya-claw` image uses `tini` as PID 1 and runs `ya-claw start` as the default command.
The `start` command handles:

1. database migration when `YA_CLAW_AUTO_MIGRATE` is enabled
2. profile seeding when `YA_CLAW_AUTO_SEED_PROFILES` is enabled
3. HTTP server startup

This keeps startup logic inside the Python CLI for consistent error handling and signal propagation.

## AGUI Web UI Model

The Web UI should follow an AGUI-aligned split:

- live session interaction comes from streamed events in process memory
- committed conversation history comes from `message.json` in the run store
- state restore views read `state.json` from the run store
- schedule and heartbeat history comes from relational fire records linked to sessions and runs

## Operational Principle

Single-node operations should stay clear enough that one developer can inspect runtime health, storage, active runs, schedules, heartbeat, bridge activity, and committed conversation history through one service.
