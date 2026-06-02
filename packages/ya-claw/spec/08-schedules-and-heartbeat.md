# 08 - Schedules and Heartbeat

YA Claw has two time-based ingress surfaces:

- **Schedules** are user or agent managed timer resources that create or steer agent work or trigger workflows.
- **Heartbeat** is a runtime-owned internal timer that runs operational workspace checks from `HEARTBEAT.md`.

Both surfaces submit work through the same queued-run execution model. Their ownership, API surface, and guidance loading rules stay separate.

## Design Goals

- let agents manage their own schedule resources through a built-in toolset
- support conversation-like timer messages through session continuation and steering
- support timer work that starts from an existing session state and writes results into a new session
- support isolated timer work that creates a clean session each fire
- keep heartbeat configuration runtime-owned and visible to the console
- load `HEARTBEAT.md` only for heartbeat-triggered runs
- inherit workspace binding from the owning or configured session when a timed run needs workspace files
- use run-scoped Docker sandboxes for schedule and heartbeat runs and close them at terminal state

## Time-based Work Types

```mermaid
flowchart TB
    T[Time-based ingress]
    T --> S[Schedules]
    T --> H[Heartbeat]

    S --> CS[continue_session]
    S --> FS[fork_session]
    S --> IS[isolate_session]
    S --> WF[workflow]

    H --> HI[heartbeat isolate run]
```

### `schedule + continue_session`

A schedule fire targets an existing session and behaves like a timed user message.

Required execution fields:

- `execution_mode = "continue_session"`
- `target_session_id`
- `on_active = "steer" | "skip" | "queue"`

Delivery behavior:

- when the target session is idle, the runtime creates a queued run under `target_session_id`
- when the target session has an active run and `on_active="steer"`, the runtime sends the rendered input to the active run steering queue
- when the target session has an active run and `on_active="skip"`, the fire is recorded as skipped
- when the target session has an active run and `on_active="queue"`, the fire remains pending until the session becomes idle

This mode is the default for conversation reminders, recurring check-ins, and timed follow-up prompts.

### `schedule + fork_session`

A schedule fire starts a new session from the latest successful state of a source session.

Required execution fields:

- `execution_mode = "fork_session"`
- `source_session_id`

Delivery behavior:

- the runtime reads `source_session.head_success_run_id`
- the runtime creates a child session with `parent_session_id = source_session_id`
- the first run in the child session restores from the source session's latest successful run
- the child session proceeds independently after creation

This mode is useful for periodic reports, audits, or branches that should begin from conversational continuity and then produce their own history.

### `schedule + isolate_session`

A schedule fire creates a clean session and runs from the schedule prompt.

Required execution fields:

- `execution_mode = "isolate_session"`

Delivery behavior:

- the runtime creates a new session for each fire
- the first run has `trigger_type="schedule"`
- the run starts with a clean agent state
- the run uses the configured workspace binding and regular workspace guidance
- the run uses a run-scoped Docker sandbox when the workspace backend is Docker
- the run closes its Docker sandbox at terminal state
- the run loads schedule input and regular workspace guidance only

This mode is useful for standalone recurring tasks that need workspace access, clean agent state, durable run history, and a fresh container lifecycle per fire.

### `schedule + workflow`

A schedule fire creates a workflow run from the stored workflow definition.

Required execution fields:

- `execution_mode = "workflow"`
- `workflow_id`
- `workflow_inputs_template`

Delivery behavior:

- the runtime renders workflow inputs from schedule metadata and fire context
- the runtime creates a queued workflow run with `trigger_kind="schedule"`
- the workflow executor dispatches node work through Claw sessions and queued runs
- the schedule fire stores `last_workflow_run_id`
- terminal workflow state updates the schedule fire status

This mode is useful for periodic multi-step work where an agent or user wants durable orchestration state, node-level tracking, and Web UI visibility.

### `heartbeat`

Heartbeat is a runtime-owned timer.

Behavior:

- the runtime uses configured heartbeat settings
- each fire creates an isolated queued run
- the run has `trigger_type="heartbeat"`
- the runtime loads `HEARTBEAT.md` from the workspace root and injects it as heartbeat guidance
- the run uses a run-scoped Docker sandbox when the workspace backend is Docker
- the run closes its Docker sandbox at terminal state
- heartbeat fires are queryable by the console
- heartbeat management stays outside the agent schedule toolset

## Trigger Model

Schedules support cron and one-time triggers.

| Field          | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `trigger_kind` | trigger type: `cron` or `once`                                |
| `cron_expr`    | cron expression used to compute recurring fire times          |
| `run_at`       | absolute timestamp used by one-time schedules                 |
| `timezone`     | timezone used to interpret cron expressions and display times |
| `next_fire_at` | absolute timestamp used by dispatcher scans                   |

Manual fire is an action created through `trigger_schedule` or `/api/v1/schedules/{schedule_id}:trigger`. It records a fire immediately and uses the stored schedule prompt or a prompt override.

The runtime stores `next_fire_at` as an absolute timestamp. Cron and one-time parsing happens in the schedule controller. Dispatcher scans use `next_fire_at` only. A one-time schedule moves to `completed` after terminal delivery and preserves its fire history.

Completed one-time schedules older than `YA_CLAW_SESSION_PRUNE_ONCE_SCHEDULES_HIDE_AFTER_DAYS` are automatically marked `deleted` by the session prune dispatcher. The default threshold is 7 days, and this metadata-only hiding runs independently from run-store pruning. The schedule metadata records `auto_hidden=true`, `auto_hidden_reason="expired_once_schedule"`, and `auto_hidden_at`. Default schedule list APIs and the web console hide deleted schedules, while `include_deleted=true` exposes them for audit and debugging. The automatic hiding step keeps schedule definitions and fire history in the database until explicit fire-record retention applies.

## Schedule Data Model

### `schedules`

Suggested fields:

- `id`
- `name`
- `description`
- `status`: `active | paused | completed | deleted`
- `owner_kind`: `user | agent | api`
- `owner_session_id`
- `owner_run_id`
- `profile_name`
- `trigger_kind`: `cron | once`
- `cron_expr`
- `run_at`
- `timezone`
- `next_fire_at`
- `execution_mode`: `continue_session | fork_session | isolate_session | workflow`
- `workflow_id`
- `workflow_inputs_template`
- `target_session_id`
- `source_session_id`
- `on_active`: `steer | skip | queue`
- `input_parts_template`
- `metadata`
- `last_fire_at`
- `last_fire_id`
- `last_session_id`
- `last_run_id`
- `last_workflow_run_id`
- `fire_count`
- `failure_count`
- `created_at`
- `updated_at`

### Schedule validation rules

- `continue_session` requires `target_session_id`
- `fork_session` requires `source_session_id`
- `isolate_session` creates a fresh session for each fire and leaves target/source session fields empty
- `workflow` requires `workflow_id`
- `workflow_inputs_template` applies to `workflow`
- `on_active` applies to `continue_session`
- `cron` schedules require `cron_expr`
- `once` schedules require `run_at`
- `once` schedules complete after a terminal fire status and keep `pending` while queued for an active target session
- `timezone` defaults to runtime timezone or `UTC`
- `input_parts_template` uses the same JSON-compatible input part model as run creation
- schedule-owned metadata must be queryable and safe to return to the console

### `schedule_fires`

Suggested fields:

- `id`
- `schedule_id`
- `scheduled_at`
- `fired_at`
- `status`: `pending | submitted | steered | skipped | failed`
- `dedupe_key`
- `target_session_id`
- `source_session_id`
- `created_session_id`
- `run_id`
- `active_run_id`
- `workflow_run_id`
- `input_parts`
- `error_message`
- `metadata`
- `created_at`
- `updated_at`

`dedupe_key` should use this stable shape:

```text
{schedule_id}:{scheduled_at.isoformat()}
```

The dedupe key makes process restart recovery and repeated dispatcher scans idempotent.

## Heartbeat Data Model

Heartbeat configuration is runtime-owned and comes from settings. The console reads the effective configuration through the heartbeat API.

Suggested effective config fields:

- `enabled`
- `cron_expr`
- `timezone`
- `profile_name`
- `prompt`
- `guidance_path`
- `guidance_exists`
- `on_active`
- `next_fire_at`
- `last_fire_at`
- `last_session_id`
- `last_run_id`

### `heartbeat_fires`

Suggested fields:

- `id`
- `scheduled_at`
- `fired_at`
- `status`: `pending | submitted | skipped | failed`
- `dedupe_key`
- `session_id`
- `run_id`
- `error_message`
- `metadata`
- `created_at`
- `updated_at`

Heartbeat can share schedule fire implementation internally when convenient, while API and toolset ownership remain separate.

## Dispatcher Semantics

### Schedule dispatcher

`ScheduleDispatcher` is a supervised in-process component.

Responsibilities:

- scan active schedules with `next_fire_at <= now`
- create idempotent `schedule_fires`
- render schedule input parts
- dispatch each fire according to `execution_mode`
- apply `on_active` policy for `continue_session`
- update schedule counters and next fire time
- emit console notifications for schedule and fire state changes

```mermaid
sequenceDiagram
    participant DISP as ScheduleDispatcher
    participant DB as Database
    participant CTRL as ScheduleController
    participant SESS as SessionController
    participant RUN as RunController
    participant SUP as ExecutionSupervisor

    DISP->>DB: scan due schedules
    DISP->>DB: insert schedule_fire with dedupe_key
    DISP->>CTRL: dispatch fire
    alt continue_session idle
        CTRL->>SESS: create queued run under target session
        SESS->>RUN: create run trigger_type=schedule
        RUN->>SUP: submit run
    else continue_session active + steer
        CTRL->>RUN: steer active run
    else fork_session
        CTRL->>SESS: create child session from source latest success
        CTRL->>RUN: create first run trigger_type=schedule
        RUN->>SUP: submit run
    else isolate_session
        CTRL->>SESS: create new session
        CTRL->>RUN: create first run trigger_type=schedule
        RUN->>SUP: submit run
    end
    CTRL->>DB: update fire status and schedule next_fire_at
```

### Heartbeat dispatcher

`HeartbeatDispatcher` is a supervised in-process component.

Responsibilities:

- compute heartbeat due time from runtime settings
- create idempotent `heartbeat_fires`
- create isolated queued runs with `trigger_type="heartbeat"`
- make the effective heartbeat status visible to the console
- emit console notifications for heartbeat state changes

```mermaid
sequenceDiagram
    participant DISP as HeartbeatDispatcher
    participant DB as Database
    participant SESS as SessionController
    participant RUN as RunController
    participant SUP as ExecutionSupervisor
    participant WS as Workspace Guidance

    DISP->>DB: create heartbeat_fire
    DISP->>SESS: create isolated session
    DISP->>RUN: create run trigger_type=heartbeat
    RUN->>SUP: submit run
    SUP->>WS: load HEARTBEAT.md during runtime assembly
    DISP->>DB: update heartbeat_fire
```

## Runtime Assembly Rules

### Schedule runs

Schedule runs use regular runtime assembly:

- `source_kind = "schedule"`
- `source_metadata.schedule_id`
- `source_metadata.schedule_fire_id`
- configured profile or default profile
- configured workspace binding
- regular workspace guidance loading

### Heartbeat runs

Heartbeat runs use heartbeat-specific guidance loading:

- `source_kind = "heartbeat"`
- `source_metadata.heartbeat_fire_id`
- configured heartbeat profile or default profile
- configured workspace binding
- regular workspace guidance loading
- heartbeat guidance loading from `HEARTBEAT.md`

`HEARTBEAT.md` is loaded from the configured workspace root and exposed to the agent as a tagged guidance block:

```xml
<heartbeat-guidance path="/workspace/HEARTBEAT.md">
...
</heartbeat-guidance>
```

## Agent Schedule Toolset

YA Claw includes a built-in `schedule` toolset for running agents.

The agent-facing toolset is a simplified facade over the full schedule model. Agents express timer intent with plain text and booleans. The runtime maps that facade into the internal schedule record.

The toolset follows the same internal-client pattern as the existing `session` toolset:

- the client resource carries the current `session_id`, `run_id`, current profile, and bearer token
- tool calls route through internal HTTP/controller APIs
- bearer tokens stay inside the client resource
- ownership fields are assigned by the runtime
- heartbeat operations are excluded from the toolset

### Facade rules

Agent-facing schedule tools use these rules:

- schedule input is `prompt: string`
- the runtime converts `prompt` into one text input part
- schedule profile inherits the current run profile
- pause and resume use `enabled: boolean`
- session behavior uses booleans: `continue_current_session`, `start_from_current_session`, `steer_when_running`
- schedule cadence uses `cron: string`
- advanced fields stay on the HTTP/admin schedule API

### Facade mapping

| Agent field                                                          | Internal field                                                           |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `prompt`                                                             | `input_parts_template = [{"type":"text","text": prompt}]`                |
| inherited current profile                                            | `profile_name`                                                           |
| `enabled=true`                                                       | `status="active"`                                                        |
| `enabled=false`                                                      | `status="paused"`                                                        |
| `continue_current_session=true`                                      | `execution_mode="continue_session"`, `target_session_id=current session` |
| `continue_current_session=false`, `start_from_current_session=true`  | `execution_mode="fork_session"`, `source_session_id=current session`     |
| `continue_current_session=false`, `start_from_current_session=false` | `execution_mode="isolate_session"`                                       |
| `steer_when_running=true`                                            | `on_active="steer"`                                                      |
| `steer_when_running=false`                                           | `on_active="queue"`                                                      |
| `cron`                                                               | `cron_expr=cron`                                                         |

`continue_current_session` takes priority over `start_from_current_session` in the mapping. A continuing schedule naturally uses the target session continuation pointer.

### Tool instructions

The schedule toolset should inject concise guidance:

```text
Use schedule tools to create, list, update, delete, and manually trigger schedules that you own. Provide a plain text prompt and a cron expression. Use continue_current_session for timed messages in this conversation. Use start_from_current_session for recurring branches from this conversation's latest committed state. Use enabled to pause or resume. Heartbeat is runtime-owned and is visible through console APIs.
```

### Ownership and scope

When an agent creates a schedule:

- `owner_kind = "agent"`
- `owner_session_id = current session id`
- `owner_run_id = current run id`
- `profile_name = current run profile`

Agent-visible schedule scope:

- schedules owned by the current session
- schedules targeting the current session through `continue_session`
- schedules using the current session as source through `fork_session`

Agent mutation scope:

- update, delete, and manual trigger require `owner_kind="agent"` and `owner_session_id=current session id`
- `continue_current_session=true` targets the current session
- `start_from_current_session=true` sources from the current session
- `isolate_session` remains owned by the current session even though each fire creates a new session

### Tools

| Tool                   | Purpose                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `list_schedules`       | list agent-visible schedules, optionally with recent fire summaries |
| `create_schedule`      | create a recurring cron schedule owned by the current agent session |
| `create_once_schedule` | create a one-time schedule owned by the current agent session       |
| `update_schedule`      | update mutable facade fields, including `enabled`                   |
| `delete_schedule`      | soft-delete a schedule                                              |
| `trigger_schedule`     | create a manual fire for an owned schedule                          |

### Tool argument shapes

`list_schedules` suggested arguments:

```json
{
  "schedule_id": null,
  "include_disabled": true,
  "include_recent_runs": true,
  "limit": 20
}
```

`create_schedule` suggested arguments:

```json
{
  "name": "Daily follow-up",
  "prompt": "Review current progress and suggest the next concrete action.",
  "cron": "0 9 * * *",
  "timezone": "Asia/Shanghai",
  "enabled": true,
  "continue_current_session": true,
  "start_from_current_session": true,
  "steer_when_running": true
}
```

`create_once_schedule` suggested arguments:

```json
{
  "name": "One-time follow-up",
  "prompt": "Review current progress and suggest the next concrete action.",
  "run_at": "2026-05-02T09:00:00+08:00",
  "timezone": "Asia/Shanghai",
  "enabled": true,
  "continue_current_session": true,
  "start_from_current_session": false,
  "steer_when_running": true
}
```

`update_schedule` accepts a patch over:

- `name`
- `prompt`
- `trigger_kind`
- `cron`
- `run_at`
- `timezone`
- `enabled`
- `continue_current_session`
- `start_from_current_session`
- `steer_when_running`

`trigger_schedule` suggested arguments:

```json
{
  "schedule_id": "schedule_123",
  "prompt_override": "Run this schedule now and focus on blockers."
}
```

`prompt_override` is plain text. Empty `prompt_override` uses the stored schedule prompt.

### Tool return shapes

Schedule tools return compact facade records:

```json
{
  "schedule_id": "schedule_123",
  "name": "Daily follow-up",
  "enabled": true,
  "prompt": "Review current progress and suggest the next concrete action.",
  "cron": {
    "expr": "0 9 * * *",
    "timezone": "Asia/Shanghai",
    "next_fire_at": "2026-04-26T09:00:00+08:00"
  },
  "mode": {
    "continue_current_session": true,
    "start_from_current_session": true,
    "steer_when_running": true
  },
  "last_fire": {
    "status": "submitted",
    "session_id": "session_123",
    "run_id": "run_456",
    "created_at": "2026-04-25T09:00:00+08:00"
  }
}
```

Manual trigger returns the created fire summary with `status`, `created_session_id`, `run_id`, `active_run_id`, and `error_message`.

## HTTP API Surface

Schedules are first-class API resources:

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

Heartbeat has a read-oriented console API:

| Method | Path                        | Purpose                                    |
| ------ | --------------------------- | ------------------------------------------ |
| `GET`  | `/api/v1/heartbeat/config`  | read effective heartbeat config            |
| `GET`  | `/api/v1/heartbeat/status`  | read heartbeat runtime status              |
| `GET`  | `/api/v1/heartbeat/fires`   | list heartbeat fires                       |
| `POST` | `/api/v1/heartbeat:trigger` | create manual heartbeat fire for admin use |

## Console Surface

The web console should include:

- a Schedules section with list, detail, fire history, pause/resume, manual trigger, and edit actions
- a Heartbeat section or settings panel with effective config, `HEARTBEAT.md` existence, next fire time, last fire, and recent fire history
- Overview cards for active schedule count, next schedule fire, heartbeat enabled state, and last heartbeat result

## Notifications

Suggested notification event types:

- `schedule.created`
- `schedule.updated`
- `schedule.deleted`
- `schedule.fire.created`
- `schedule.fire.updated`
- `heartbeat.fire.created`
- `heartbeat.fire.updated`

## Configuration

Suggested runtime settings:

| Variable                             | Purpose                                |
| ------------------------------------ | -------------------------------------- |
| `YA_CLAW_SCHEDULE_DISPATCH_ENABLED`  | enable schedule dispatcher             |
| `YA_CLAW_SCHEDULE_TICK_SECONDS`      | schedule dispatcher scan interval      |
| `YA_CLAW_SCHEDULE_MAX_DUE_PER_TICK`  | maximum due schedules handled per scan |
| `YA_CLAW_HEARTBEAT_ENABLED`          | enable heartbeat dispatcher            |
| `YA_CLAW_HEARTBEAT_INTERVAL_SECONDS` | heartbeat interval                     |
| `YA_CLAW_HEARTBEAT_PROFILE`          | heartbeat profile name                 |
| `YA_CLAW_HEARTBEAT_PROMPT`           | heartbeat input prompt                 |
| `YA_CLAW_HEARTBEAT_ON_ACTIVE`        | heartbeat active-run policy            |

## Design Principle

Schedules are agent-manageable timer resources. Heartbeat is a runtime-owned operational timer. Both become durable queued runs before execution begins.
