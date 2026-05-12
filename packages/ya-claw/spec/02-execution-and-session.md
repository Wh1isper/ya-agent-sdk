# 02 - Execution and Session

YA Claw uses queued runs as the durable execution intent.

The execution stack has two runtime layers:

- `ExecutionSupervisor` manages queued claims and active coordinators
- `RunCoordinator` executes one claimed run from start to terminal state

## Core Model

YA Claw uses two durable layers:

- a **run** is the atomic execution and commit unit
- a **session** is the high-level view over a run lineage and its continuation pointers

```mermaid
flowchart LR
    S[Session]
    R1[Queued Run]
    R2[Running Run]
    R3[Completed Run]
    S --> R1
    S --> R2
    S --> R3
```

## Session Model

A session carries durable metadata plus continuation pointers stored in the relational database.

Suggested session fields:

- `id`
- `parent_session_id`
- `profile_name`
- `metadata` including optional `workspace` binding
- `head_run_id`
- `head_success_run_id`
- `active_run_id`
- `created_at`
- `updated_at`

### Session Meaning

- `head_run_id` points to the latest run created in the session
- `head_success_run_id` points to the latest successfully committed run
- `active_run_id` points to the currently claimed and executing run when one exists

### Session Continuation Rule

A high-level session continuation follows this rule:

1. use explicit `restore_from_run_id` when provided
2. otherwise use `head_success_run_id`

### Session Workspace Rule

A session can carry a durable workspace binding in `metadata["workspace"]`. The binding describes mounted folders, the default mount, and the default cwd for every continuation in the session.

Session creation can accept `workspace` as a first-class request field. The controller stores that value in session metadata. Session continuation inherits the session workspace and can accept a run-level override for one execution.

Workspace mount-set details live in [10-workspace-mount-sets.md](10-workspace-mount-sets.md).

## Run Model

A run is one execution attempt inside a session.

Suggested run fields:

- `id`
- `session_id`
- `sequence_no`
- `restore_from_run_id`
- `status`
- `trigger_type`
- `profile_name`
- `input_parts`
- `metadata` including optional run-level `workspace` override
- `output_text`
- `output_summary`
- `error_message`
- `termination_reason`
- `created_at`
- `started_at`
- `finished_at`
- `committed_at`

### Run States

| Status      | Meaning                                            |
| ----------- | -------------------------------------------------- |
| `queued`    | durable run intent accepted, waiting to be claimed |
| `running`   | claimed by supervisor and currently executing      |
| `completed` | committed successfully                             |
| `failed`    | ended with error                                   |
| `cancelled` | stopped before completion                          |

### Run State Flow

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> running
    queued --> cancelled
    running --> completed
    running --> failed
    running --> cancelled
```

### Why `queued` Exists

`queued` is the durable handoff point between ingress and execution.

It enables:

- API acceptance before execution begins
- unified execution for API, schedules, heartbeat, and bridges
- trigger-specific Docker sandbox scope for conversation and automatic runs
- concurrency control and future worker scaling
- cancellation before claim
- restart recovery and claim scans after process restart

## ExecutionSupervisor

`ExecutionSupervisor` is the long-lived in-process execution manager.

Responsibilities:

- accept new queued run notifications
- scan and claim queued runs on startup or recovery
- atomically transition `queued -> running`
- create and track one `RunCoordinator` task per active run
- route stop signals to active runs
- enforce concurrency limits and execution policy

### Claim Rule

Claim should be explicit and atomic.

Recommended shape:

1. load candidate queued run
2. update `status` from `queued` to `running` with a conditional write
3. set run claim ownership through `claimed_by` and `claimed_at`
4. update `started_at` and session `active_run_id`
5. record runtime instance liveness in the `runtime_instances` heartbeat table
6. only the successful claimant starts the coordinator

## RunCoordinator

`RunCoordinator` is the short-lived per-run executor.

Responsibilities:

- load session and run state
- load restore point from prior committed run when available
- resolve profile and workspace binding
- assemble the SDK runtime through `ClawRuntimeBuilder`
- stream execution events into runtime state buffers
- persist best-effort checkpoints
- commit final continuity blobs
- advance run and session pointers through the execution state machine

Workspace resolution uses session metadata plus run metadata. A run inherits the session workspace binding by default. A run-level `metadata["workspace"]` value replaces the session binding for that run.

Docker sandbox scope follows trigger type:

- API, bridge, and memory runs use the session-scoped Docker sandbox generation
- schedule and heartbeat runs use run-scoped Docker sandboxes and close them at terminal state
- local backend runs use resolved path bindings and skip Docker sandbox lifecycle

## Execution Registry and Runtime State

The single-node runtime keeps active execution state in memory.

Recommended split:

- `ExecutionRegistry` for active coordinator tasks and control handles
- runtime event buffer store for replayable events, steering queues, and termination signals

This split keeps execution ownership separate from event delivery.

## Input Model

Run creation, session continuation, and steering share one structured input protocol.

The request payload should use:

- `input_parts: Sequence[InputPart]`

`InputPart` supports:

- `text`
- `url`
- `file`
- `binary`
- `mode`
- `command`

The execution input mapper converts `input_parts` into the SDK `UserPrompt` shape.
The run record keeps the original JSON-compatible `input_parts` payload for replay and UI reconstruction.

## Event Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant SUP as ExecutionSupervisor
    participant COORD as RunCoordinator
    participant SDK as ya-agent-sdk
    participant ADAPTER as AGUI Protocol Adapter
    participant STORE as Session Message Store

    Client->>API: create run
    API->>API: write queued run
    API->>SUP: submit(run_id)
    SUP->>SUP: claim queued run
    SUP->>COORD: start coordinator
    COORD->>SDK: stream_agent(...)
    loop runtime events
        SDK-->>COORD: stream event
        COORD->>ADAPTER: transform SDK event -> AGUI event
        ADAPTER->>STORE: append + compact replay buffer
        STORE-->>Client: SSE replay and live tail
    end
```

## Commit Semantics

Successful completion commits run output into the durable run store.

A successful commit should:

1. persist `state.json` for the run
2. persist compacted AGUI replay events to `message.json`
3. persist final `output_text` and `output_summary` on the run record
4. set `committed_at`
5. set session `head_success_run_id`
6. clear session `active_run_id`

## Failure and Interrupt Semantics

A failed or interrupted run keeps value as an explicit restore source.

The coordinator should:

1. keep `head_success_run_id` unchanged when no new successful commit exists
2. best-effort persist the latest compacted AGUI replay list for the interrupted or failed run
3. make that run addressable through `restore_from_run_id`

## Trigger Types

Runs record their ingress source through `trigger_type`.

Recommended trigger types:

| Trigger type | Meaning                               |
| ------------ | ------------------------------------- |
| `api`        | direct HTTP or web console request    |
| `bridge`     | inbound bridge adapter event          |
| `schedule`   | schedule fire created or steered work |
| `heartbeat`  | runtime-owned heartbeat fire          |

Schedule and heartbeat trigger metadata should carry fire identifiers so operators can navigate from run history back to timer history.

## Cross-session Restore

A run normally restores from a committed run in the same session.

Forked sessions also need to restore from an ancestor session's latest successful run. The restore validation rule should accept a restore source when:

- the restore run belongs to the current session
- the restore run belongs to an ancestor session reachable through `parent_session_id`

This rule lets `schedule + fork_session` create a child session from the source session's latest successful state while preserving independent child history.

## Schedule Execution Modes

Schedules can dispatch work in three execution modes:

| Mode               | Session relationship            | Execution behavior                                                            |
| ------------------ | ------------------------------- | ----------------------------------------------------------------------------- |
| `continue_session` | targets an existing session     | creates a new run under the target session or steers the active run           |
| `fork_session`     | uses an existing source session | creates a child session and restores from the source session's latest success |
| `isolate_session`  | creates a clean session         | creates a new session for each fire and runs from schedule input only         |

The full schedule and heartbeat model is defined in [08-schedules-and-heartbeat.md](08-schedules-and-heartbeat.md).

## Concurrency Rule

One active run per session is the clean default for the single-node runtime.

Parallelism should come from:

- independent sessions
- supervisor-managed background execution
- schedules that create independent sessions
- bridges that map to independent sessions
- SDK subagents

For `continue_session` schedules, `on_active` selects the active-session behavior: `steer`, `skip`, or `queue`.
