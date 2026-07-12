# 04 - API

YA Claw exposes one local HTTP API under `/api/v1`.

The API has these layers:

- **session API** for the common high-level workflow
- **run API** for explicit low-level orchestration
- **workflow API** for Claw-managed agent-supervised workflow orchestration
- **schedule API** for timer-managed work
- **heartbeat API** for runtime-owned operational timer visibility
- **workspace binding fields** on session and run creation for multi-folder execution

## API Principle

The API accepts durable execution intent first.
Run creation writes a queued run record before active execution begins.

## Resource Groups

```mermaid
flowchart TB
    ROOT[/api/v1]
    ROOT --> SESSIONS[/sessions]
    ROOT --> RUNS[/runs]
    ROOT --> EVENTS[/events via nested session and run routes]
    ROOT --> PROFILES[/profiles and seed operations]
    ROOT --> WORKFLOWS[/workflows and workflow-runs]
    ROOT --> SCHEDULES[/schedules]
    ROOT --> HEARTBEAT[/heartbeat]
    ROOT --> WORKSPACE[/workspace runtime and sandbox state]
    ROOT --> CLAW[/claw info and notifications]
```

## Top-level Endpoints

| Method | Path                         | Purpose                                      |
| ------ | ---------------------------- | -------------------------------------------- |
| `GET`  | `/healthz`                   | service, storage, and runtime health         |
| `GET`  | `/api/v1/claw/info`          | web console startup handshake and capability |
| `GET`  | `/api/v1/claw/notifications` | global console notification SSE stream       |

## Workflows

| Method  | Path                                                            | Purpose                                                   |
| ------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| `GET`   | `/api/v1/workflows`                                             | list workflow definitions                                 |
| `POST`  | `/api/v1/workflows`                                             | create workflow definition                                |
| `GET`   | `/api/v1/workflows/{workflow_id}`                               | inspect workflow definition                               |
| `PATCH` | `/api/v1/workflows/{workflow_id}`                               | update workflow definition metadata or body               |
| `POST`  | `/api/v1/workflows/{workflow_id}:archive`                       | archive workflow definition                               |
| `POST`  | `/api/v1/workflows/{workflow_id}:trigger`                       | create a workflow run                                     |
| `GET`   | `/api/v1/workflow-runs`                                         | list workflow runs                                        |
| `GET`   | `/api/v1/workflow-runs/{workflow_run_id}`                       | inspect workflow run, node state, result, and linked runs |
| `GET`   | `/api/v1/workflow-runs/{workflow_run_id}/events`                | replay and tail workflow events                           |
| `POST`  | `/api/v1/workflow-runs/{workflow_run_id}/cancel`                | cancel workflow run                                       |
| `POST`  | `/api/v1/workflow-runs/{workflow_run_id}/nodes/{node_id}/steer` | steer an active node run                                  |

Workflow semantics live in [13-workflows.md](13-workflows.md). Workflow trigger creates a durable workflow run first, then the workflow executor starts node work through the regular queued-run execution model.

## Workspace Runtime

| Method | Path                                            | Purpose                                                 |
| ------ | ----------------------------------------------- | ------------------------------------------------------- |
| `GET`  | `/api/v1/workspace/runtime`                     | inspect configured workspace backend and runtime checks |
| `POST` | `/api/v1/workspace:resolve`                     | resolve provider binding for supplied metadata          |
| `GET`  | `/api/v1/sessions/{session_id}/workspace`       | inspect a session's resolved workspace and sandbox      |
| `GET`  | `/api/v1/sessions/{session_id}/sandbox`         | inspect a session's sandbox state                       |
| `POST` | `/api/v1/sessions/{session_id}/sandbox:prepare` | prepare a Docker-backed session sandbox                 |
| `POST` | `/api/v1/sessions/{session_id}/sandbox:stop`    | stop a Docker-backed session sandbox                    |

`workspace.runtime` exposes backend, execution location, workspace service path, virtual path, checks, Docker daemon/image/cache status, sandbox lifecycle capabilities, and update time. Session summaries and session detail responses include `workspace_state` with the persisted sandbox state when sandbox metadata exists.

Sandbox state uses `status` for lifecycle (`created`, `preparing`, `ready`, `failed`, `stopped`) and `ready_state` for client display (`not_started`, `starting`, `ready`, `failed`). Docker session sandboxes expose container reference, container id, verified container id, image, work dir, retention policy, idle TTL, computed expiry, and TTL seconds remaining.

## Sessions

| Method | Path                                      | Purpose                                         |
| ------ | ----------------------------------------- | ----------------------------------------------- |
| `POST` | `/api/v1/sessions`                        | create a session with optional first queued run |
| `GET`  | `/api/v1/sessions`                        | list sessions                                   |
| `GET`  | `/api/v1/sessions/{session_id}`           | inspect session and committed state             |
| `GET`  | `/api/v1/sessions/{session_id}/turns`     | list completed conversational turns             |
| `POST` | `/api/v1/sessions/{session_id}/runs`      | create a new queued run under the session       |
| `POST` | `/api/v1/sessions/{session_id}/steer`     | steer the active run through the session        |
| `POST` | `/api/v1/sessions/{session_id}/interrupt` | interrupt the active run through the session    |
| `POST` | `/api/v1/sessions/{session_id}/cancel`    | cancel the active run through the session       |
| `POST` | `/api/v1/sessions/{session_id}/fork`      | fork a new session lineage                      |
| `GET`  | `/api/v1/sessions/{session_id}/events`    | replay and tail session events                  |

## Runs

| Method | Path                              | Purpose                         |
| ------ | --------------------------------- | ------------------------------- |
| `POST` | `/api/v1/runs`                    | create a queued run directly    |
| `GET`  | `/api/v1/runs/{run_id}`           | inspect run and committed state |
| `GET`  | `/api/v1/runs/{run_id}/trace`     | inspect compact tool trace      |
| `POST` | `/api/v1/runs/{run_id}/steer`     | steer a specific active run     |
| `POST` | `/api/v1/runs/{run_id}/interrupt` | interrupt a specific active run |
| `POST` | `/api/v1/runs/{run_id}/cancel`    | cancel a specific active run    |
| `GET`  | `/api/v1/runs/{run_id}/events`    | replay and tail run events      |

## Request Model

Run creation and steering use structured input parts.

### Shared Input Field

```json
{
  "input_parts": [{ "type": "text", "text": "hello" }]
}
```

Supported part types:

- `text`
- `url`
- `file`
- `binary`
- `mode`
- `command`

### Session Create Request

Suggested fields:

- `profile_name`
- `metadata`
- `workspace`
- `input_parts`
- `dispatch_mode`
- `trigger_type`

### Session Continue Request

Suggested fields:

- `restore_from_run_id`
- `input_parts`
- `metadata`
- `workspace`
- `dispatch_mode`
- `trigger_type`

### Run Create Request

Suggested fields:

- `session_id`
- `restore_from_run_id`
- `profile_name`
- `input_parts`
- `metadata`
- `workspace`
- `dispatch_mode`
- `trigger_type`

### Workspace Binding Request Field

`workspace` is an optional first-class field for session and run creation. It lets clients bind a session or run to one or more host folders.

```json
{
  "workspace": {
    "mounts": [
      {
        "id": "main",
        "name": "ya-mono",
        "host_path": "/Users/jizhongsheng/code/yet-another-agents/ya-mono",
        "virtual_path": "/workspace/main",
        "mode": "rw"
      },
      {
        "id": "docs",
        "name": "product-docs",
        "host_path": "/Users/jizhongsheng/docs/product",
        "virtual_path": "/workspace/docs",
        "mode": "ro"
      }
    ],
    "default_mount_id": "main",
    "cwd": "/workspace/main"
  }
}
```

Session create stores `workspace` in session metadata. Session continuation inherits the session workspace. Run create and session-run create can provide a workspace override for that execution. Workspace mount-set semantics live in [10-workspace-mount-sets.md](10-workspace-mount-sets.md).

## Creation Semantics

JSON run-creating endpoints should:

1. write the durable run record with `status=queued`
2. update session pointers such as `head_run_id`
3. notify the in-process supervisor when execution is available
4. return the queued run record immediately

Foreground streaming creation uses dedicated SSE endpoints:

- `POST /api/v1/sessions:stream`
- `POST /api/v1/sessions/{session_id}/runs:stream`
- `POST /api/v1/runs:stream`

## Run Summary Shape

Suggested run summary fields:

- `id`
- `session_id`
- `sequence_no`
- `restore_from_run_id`
- `status`
- `trigger_type`
- `profile_name`
- `input_preview`
- `input_parts` when `include_input_parts=true`
- `output_text`
- `error_message`
- `termination_reason`
- `created_at`
- `started_at`
- `finished_at`
- `committed_at`

### Status Semantics in API

- `queued` means accepted and durable, waiting to be claimed
- `running` means claimed by the supervisor and currently executing
- `completed`, `failed`, and `cancelled` are terminal states

## GET Response Shape

Session and run GET endpoints should return the structured record plus committed blobs.

### Session List Page

`GET /api/v1/sessions/page?limit=50`

The Web UI uses this lightweight keyset endpoint instead of polling the backwards-compatible unpaginated session list. Results are ordered by descending `(updated_at, id)`. Continue with both anchors returned by the previous page:

```http
GET /api/v1/sessions/page?limit=50&before_updated_at=2026-07-12T07:00:00Z&before_id=session_123
```

```json
{
  "sessions": [],
  "total": 3000,
  "limit": 50,
  "has_more": true,
  "next_before_updated_at": "2026-07-12T07:00:00Z",
  "next_before_id": "session_123"
}
```

The page endpoint omits `latest_run.output_text` and returns the last persisted workspace state by default. It does not synchronously inspect every Docker sandbox. Selected-session workspace endpoints remain the source for live sandbox reconciliation.

### Session GET

`GET /api/v1/sessions/{session_id}?include_message=true&include_input_parts=true`

```json
{
  "session": {
    "id": "session_123",
    "head_run_id": "run_3",
    "head_success_run_id": "run_2",
    "active_run_id": "run_3",
    "workspace_state": {
      "sandbox_state": {
        "backend": "docker",
        "ready_state": "ready",
        "container_id": "container_123",
        "ttl_seconds_remaining": 1800
      }
    },
    "recent_runs": []
  },
  "state": {},
  "message": []
}
```

`include_input_parts=true` includes each listed run's original `input_parts` for UI replay. `include_head_payload=false` skips the top-level committed `state` and `message` reads while preserving paginated run messages, which avoids returning the same head payload on every history page.

### Session Turns

`GET /api/v1/sessions/{session_id}/turns?limit=20&cursor=run_2`

Returns completed runs only. Each turn includes the original `input_parts`, `output_text`,.
The endpoint paginates newest-first by descending `(sequence_no, run_id)`. Use `next_cursor` as the next request's `cursor` to load older turns for chatbox history. `before_sequence_no` remains available for sequence-number based callers.

```json
{
  "session_id": "session_123",
  "limit": 20,
  "has_more": true,
  "next_cursor": "run_2",
  "next_before_sequence_no": 2,
  "turns": []
}
```

### Run GET

`GET /api/v1/runs/{run_id}?include_message=true`

```json
{
  "run": {
    "id": "run_2",
    "session_id": "session_123",
    "restore_from_run_id": "run_1",
    "input_parts": [],
    "has_state": true,
    "has_message": true
  },
  "state": {},
  "message": []
}
```

### Run Trace

`GET /api/v1/runs/{run_id}/trace?max_item_chars=4000&max_total_chars=12000`

Returns a compact projection of committed `message.json` tool events:

- `TOOL_CALL_CHUNK` as `tool_call`
- `TOOL_CALL_RESULT` as `tool_response`

The response trims each item and the total trace payload according to query parameters.

## Agent Self-Session Tools

The built-in `session` toolset exposes read-only tools for the running agent:

- `list_session_turns` reads completed turns for the current session through an internal HTTP client.
- `get_run_trace` reads tool-call and tool-response trace for a run in the current session.

The client carries the current `session_id` and bearer token internally. Tool calls use the current session scope for session selection and trace access.

## Agent Schedule Tools

The built-in `schedule` toolset lets the running agent manage its own timer resources through an internal client.

Agent-facing tools:

- `list_schedules`
- `create_schedule`
- `update_schedule`
- `delete_schedule`
- `trigger_schedule`

The internal client carries current `session_id`, `run_id`, current profile, and bearer token. Agent-created schedules are stamped with `owner_kind="agent"`, `owner_session_id=current session id`, and `owner_run_id=current run id`. Schedule prompts are plain text and are converted into text input parts by the runtime. Agent schedules inherit the current run profile. Mutations are limited to schedules owned by the current session. Heartbeat operations are excluded from this toolset.

Schedule tool facade semantics and argument shapes are defined in [08-schedules-and-heartbeat.md](08-schedules-and-heartbeat.md).

## Control Endpoints

Control endpoints stay flat and explicit.

Recommended shape:

- `POST /sessions/{session_id}/steer`
- `POST /sessions/{session_id}/interrupt`
- `POST /sessions/{session_id}/cancel`
- `POST /runs/{run_id}/steer`
- `POST /runs/{run_id}/interrupt`
- `POST /runs/{run_id}/cancel`

Session control routes to `active_run_id`.
Run control routes to the addressed run.

## Event Streaming

Event streaming uses SSE.

### Replay Contract

- each event has a monotonic SSE ID
- reconnect uses `Last-Event-ID`
- the server replays buffered events after that cursor
- the server then tails live events

### Transport Principle

The single-node baseline keeps the event buffer in memory.
Queued-run creation and active execution are separate concerns.
SSE reflects active or recently buffered execution. Durable creation returns through JSON creation responses.

## Console Info and Notifications

The web console reads `/api/v1/claw/info` during startup to discover environment, auth mode, runtime surfaces, and feature flags.

Suggested response shape:

```json
{
  "name": "YA Claw",
  "environment": "development",
  "version": "0.1.0",
  "public_base_url": "http://127.0.0.1:9042",
  "instance_id": "host-123-abcdef",
  "auth": "bearer",
  "surfaces": ["profiles", "sessions", "runs", "schedules", "heartbeat", "notifications"],
  "workspace_provider_backend": "docker",
  "storage_model": "sqlite",
  "features": {
    "session_events": true,
    "run_events": true,
    "notifications": true,
    "notification_replay": true,
    "session_status_reasons": true,
    "hitl_status_reason": true,
    "profiles": true,
    "schedules": true,
    "heartbeat": true,
    "session_workspace_binding": true,
    "run_workspace_override": true,
    "multi_mount_workspaces": true,
    "session_docker_sandbox": true,
    "run_scoped_auto_task_sandbox": true,
    "sandbox_idle_ttl": true
  },
  "workspace_mount_modes": ["rw", "ro"],
  "sandbox_retention_policies": ["stop_on_idle", "keep_warm"],
  "limits": {
    "max_workspace_mounts_per_session": 8,
    "default_sandbox_idle_ttl_seconds": 3600
  }
}
```

The notification stream at `/api/v1/claw/notifications` is a global SSE stream for list and overview refreshes. Each event payload is JSON:

```json
{
  "id": "1",
  "type": "session.updated",
  "created_at": "2026-04-25T13:00:00Z",
  "payload": {
    "session_id": "session_123",
    "status": "running",
    "status_reason": "hitl_pending",
    "status_detail": {
      "run_id": "run_456",
      "sequence_no": 4,
      "active_interaction_count": 1
    },
    "active_run_id": "run_456"
  }
}
```

Session summaries expose stable `status` values plus `status_reason` and `status_detail` for UI-specific transition context. HITL uses `status="running"` with `status_reason="hitl_pending"` so clients can show approval prompts while the run remains active.

Run notifications also include `session_status`, `session_status_reason`, and `session_status_detail` for clients that process run events directly.

Notification events are buffered in process memory and support `Last-Event-ID` replay. The web console should still use session and run event streams for detailed AGUI output.

Initial notification types:

- `session.created`
- `session.updated`
- `run.created`
- `run.updated`
- `profile.created`
- `profile.updated`
- `profile.deleted`
- `profiles.seeded`
- `schedule.created`
- `schedule.updated`
- `schedule.deleted`
- `schedule.fire.created`
- `schedule.fire.updated`
- `heartbeat.fire.created`
- `heartbeat.fire.updated`

## Schedules

Schedules are available through `/api/v1/schedules` CRUD routes plus manual fire and fire history routes.

| Method   | Path                                      | Purpose              |
| -------- | ----------------------------------------- | -------------------- |
| `GET`    | `/api/v1/schedules`                       | list schedules       |
| `POST`   | `/api/v1/schedules`                       | create schedule      |
| `GET`    | `/api/v1/schedules/{schedule_id}`         | inspect schedule     |
| `PATCH`  | `/api/v1/schedules/{schedule_id}`         | update schedule      |
| `DELETE` | `/api/v1/schedules/{schedule_id}`         | soft-delete schedule |
| `POST`   | `/api/v1/schedules/{schedule_id}:pause`   | pause schedule       |
| `POST`   | `/api/v1/schedules/{schedule_id}:resume`  | resume schedule      |
| `POST`   | `/api/v1/schedules/{schedule_id}:trigger` | create manual fire   |
| `GET`    | `/api/v1/schedules/{schedule_id}/fires`   | list schedule fires  |

Schedule execution modes are `continue_session`, `fork_session`, and `isolate_session`. Schedule-triggered Docker runs use run-scoped sandboxes and close them at terminal state.

## Heartbeat

Heartbeat has read-oriented console routes and an admin manual trigger route.

| Method | Path                        | Purpose                                    |
| ------ | --------------------------- | ------------------------------------------ |
| `GET`  | `/api/v1/heartbeat/config`  | read effective heartbeat config            |
| `GET`  | `/api/v1/heartbeat/status`  | read heartbeat runtime status              |
| `GET`  | `/api/v1/heartbeat/fires`   | list heartbeat fires                       |
| `POST` | `/api/v1/heartbeat:trigger` | create manual heartbeat fire for admin use |

Heartbeat is runtime-owned and available through heartbeat console/admin routes. Heartbeat-triggered Docker runs use run-scoped sandboxes and close them at terminal state.

## Profiles

Profile management is available through `/api/v1/profiles` CRUD routes and `/api/v1/profiles/seed`.
Profile records remain durable database state even when seeded from YAML. Seed operations create missing YAML profiles and refresh matching database profiles, including subagent definitions. Profiles absent from YAML remain in the database unless the seed request sets `prune_missing=true`.

## Authentication

The single-node baseline uses one shared bearer token configured through `YA_CLAW_API_TOKEN`.
Every HTTP route except `/healthz` sends `Authorization: Bearer <token>`.
