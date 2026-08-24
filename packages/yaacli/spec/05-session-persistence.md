# Session Persistence and Local Execution Coordination

## Scope

YAACLI uses one hybrid session store:

- SQLite owns transactional product metadata and small coordination records.
- Deterministic JSON files own revision, checkpoint, and subagent state that can grow
  with model history or tool output.
- `LocalExecutionCoordinator` composes this store with the SDK's host-neutral
  `AgentExecutionHarness`.

There is no workflow-engine database, artifact catalog, content-addressed payload
layer, dual-read compatibility path, or replay engine.

The SDK execution harness remains stateless across native segments. YAACLI, as the
host, persists both Pydantic AI message history and the SDK `ResumableState`; neither
is reconstructed from the other.

## Storage Layout

The default layout is:

```text
~/.yaacli/sessions/
  sessions-v2.sqlite3
  <session-id>/
    revisions/<revision-id>/state.json
    checkpoints/<execution-id>/state.json
    subagents/<execution-id>/state.json
```

`[session] session_dir` changes the default directory. An explicit
`[session] database_path` remains authoritative, and state directories are placed next
to that database. The database filename remains `sessions-v2.sqlite3`; this file-state
cutover does not introduce a v3 database.

Every `state.json` is UTF-8, pretty-printed, and written with `ensure_ascii=False`, so
agents and users can inspect history directly with ordinary tools such as `rg`, `jq`,
and `less`.

### SQLite ownership

SQLite retains:

- schema metadata;
- sessions;
- logical runs and process-owned executions;
- revision metadata, counts, and bounded input/output previews;
- checkpoint identity and segment metadata;
- ordered run inputs;
- HITL action batches and decisions;
- ordered session events.

`run_inputs.content_json`, action payloads, and event payloads remain in SQLite because
they are bounded transaction-critical coordination data.

SQLite does not contain revision message/context payloads, checkpoint payloads,
subagent descriptors, subagent records, child history, child usage, or child inbox
payloads.

### File ownership

A revision file is a versioned, self-contained `RevisionRecord`. It includes canonical
Pydantic AI messages, `ResumableState`, the logical-run input ledger, bounded display
projection, usage, terminal metadata, revision identity, parent identity, and creation
time.

A checkpoint file is a versioned, self-contained `ExecutionCheckpointRecord`, including
the complete revision payload and deferred requests when suspended.

A subagent file is a versioned, self-contained document containing:

- the exact `SubagentPlanDescriptor`;
- the complete `SubagentExecutionRecord`;
- persisted child inputs and their transitions;
- input-open and cancellation fences;
- the owning process ID plus a unique process-generation token backed by a process-held
  filesystem lock, used to distinguish dead-process orphans from children owned by
  another live terminal; and
- creation and update timestamps.

Each child owns a separate file. Concurrent children never read-modify-write one shared
session document.

## Data Model

```mermaid
flowchart TB
    Session[SQLite session metadata]
    Run[SQLite logical run]
    Execution[SQLite execution]
    Input[SQLite run input]
    Action[SQLite action batch and items]
    Event[SQLite session event]
    RevisionMeta[SQLite revision metadata]
    CheckpointMeta[SQLite checkpoint metadata]

    RevisionFile[Revision state.json]
    CheckpointFile[Checkpoint state.json]
    ChildFile[Subagent state.json]

    Session -->|head_revision_id| RevisionMeta
    Run --> Session
    Run --> Execution
    Run --> Input
    Run --> Action
    Session --> Event
    Run --> RevisionMeta
    Execution --> CheckpointMeta
    RevisionMeta -. deterministic identity .-> RevisionFile
    CheckpointMeta -. deterministic identity .-> CheckpointFile
    Session -. owner scope .-> ChildFile
```

Paths are derived from typed identities; SQLite stores no file path columns.

## Publication and Durability

Large state uses file-first publication:

1. serialize the complete typed state;
2. create a temporary file in the destination directory;
3. flush and `fsync` the temporary file;
4. atomically replace `state.json` with `os.replace`;
5. `fsync` the parent directory; and
6. publish metadata, head, run status, input fences, and terminal event in SQLite.

A crash before the SQLite commit can leave an orphan file, but SQLite never publishes
a revision or checkpoint whose file was not already installed. Session directories
carry an explicit YAACLI ownership marker; startup maintenance locks and re-reads one
marked session at a time before removing its orphans. Unmarked sibling directories are
never claimed or deleted.

Retention uses the reverse order: commit metadata deletion first, then remove the now
unreferenced files. A failed file deletion therefore leaves only an orphan, which a
later maintenance pass can remove.

Revision files are immutable after publication. Checkpoint and subagent files use
atomic replacement for each stable transition. A staged checkpoint temporarily retains
the previous committed checkpoint in the same document, so a failed SQLite metadata
advance still reads the committed segment; the backup is removed after a successful
metadata commit.

## Session and Revision

A session has a stable ID, workspace reference, active/tombstoned status, one committed
head revision, and timestamps. Session list/show projections read only SQLite metadata;
they do not load message history merely to compute counts or previews.

Every successful, failed, cancelled, or interrupted logical run publishes one terminal
revision. Revision metadata insertion, session-head publication, run/execution terminal
state, unresolved input rejection, checkpoint metadata deletion, and the canonical
terminal event share one SQLite transaction. Repeated publication with the same logical
identity and payload is idempotent.

`/dump`, `/load`, TUI restore, and headless restore load the complete revision file.
Current runtime and environment policy remain authoritative; persisted context cannot
weaken current capability, filesystem, shell, or approval policy.

## Logical Runs, Inputs, Checkpoints, and HITL

Run states are `pending`, `running`, `suspended`, `cancelling`, `completed`, `failed`,
`cancelled`, and `interrupted`.

A checkpoint records only a completed or suspended native segment boundary. It does not
advance the session head. Deferred continuation restores the complete checkpoint file,
including message history and `ResumableState`, then applies audited
`DeferredToolResults`.

Additional user and feature input is stored before acknowledgement. Input states are
`accepted`, `enqueued`, `applied`, and `rejected`. `DurableInboxPumpCapability` applies
persisted rows through Pydantic AI's native enqueue mechanism. The terminal input fence
requires every accepted/enqueued row to become applied or rejected before publication.

A native `DeferredToolRequests` result creates a transactionally persisted action batch.
Each decision has stable identity, actor, timestamp, and typed payload. A partial batch
remains pending; a completed batch wakes continuation from the stable checkpoint.
Process-local events are wake optimizations, never durable truth.

## Local Execution and Restart

The coordinator creates a fresh `TUIContext` for each native segment. It restores either
the expected head revision for a new turn or the latest checkpoint for continuation,
then calls the SDK execution harness.

The coordinator persists every stable checkpoint before terminal commit or HITL wait.
Usage limits remain cumulative across continuation segments. Runtime locks cover only a
native segment and are released before waiting for actions.

YAACLI never replays an arbitrary in-flight graph segment after restart:

- pending accepted work remains dispatchable;
- cancelling runs settle as cancelled;
- process-owned running or suspended runs settle as interrupted;
- interrupted publication uses the latest stable checkpoint when available; and
- a later user continuation is a new logical run from the published head.

Controlled failure and shutdown may publish the SDK's safe recoverable subset. The host
preserves message history and resumable context as separate typed fields. Canonical model
history follows the SDK's recoverable-message rules and is never reconstructed from UI
events. The non-authoritative display projection independently retains observed tool
invocations so interrupted activity remains visible after reattach.

## Process-Local Subagents

`FileSubagentExecutionStore` implements the SDK `SubagentExecutionStore` protocol.
`FileSubagentExecutionStore.restart_durable` and `LocalSubagentDriver.restart_durable`
are both `False`: storage survives restart for inspection and delivery reconciliation,
but model/tool execution is process-owned and is not resumed automatically.

Consequences:

- frontend startup marks `pending`, `running`, or `suspended` children whose exact
  process-generation lock is no longer held `lost` and rejects unresolved child input,
  while leaving children owned by another live terminal untouched;
- terminal child records remain inspectable after completion;
- exact descriptors are embedded in each child file and restored lazily for linked
  continuation or nested authorization;
- unavailable historical capability code fails only the operation that needs it and
  does not block current runtime startup;
- spawn and steering idempotency are owner scoped;
- child steering is persisted before graph application;
- suspended children retain their inbox for the continuation segment;
- child usage is cumulative across deferred continuation;
- terminal completion enters the parent SQLite `run_inputs` path idempotently; and
- pending completion-delivery state remains in the child file for later inspection and
  reconciliation.

The TUI's `/agents` view scans only the current session's child files. Switching sessions
does not reparent work.

## Locking and Tombstone Fence

Locks are scoped by `session_id`, not global. Different terminals/sessions use different
lock files and remain concurrent. The lock covers only short local state transitions:
owner-status validation, JSON replacement, and tombstone fencing. Model calls, tool
calls, child execution, waits, and network operations never hold it.

All children in one session briefly share the session lifecycle lock, but each child
writes its own state file. This prevents a tombstone/write race without creating a
shared-document lost-update problem. A separate global lock covers only the short child
execution-ID creation claim; it never covers execution, steering, waits, or ordinary
state writes.

Tombstoning follows this order while holding the owner session lock:

1. verify no main logical run is nonterminal;
2. commit the SQLite session tombstone;
3. mark nonterminal children cancelled;
4. reject unresolved child input; and
5. persist child cancellation intent before releasing the lock.

Child writes acquire the same session lock and recheck the SQLite owner status. A write
that starts after tombstone publication is rejected, and a pre-tombstone writer cannot
escape after the fence. Different sessions are unaffected.

## Retention and Maintenance

Retention keeps the configured number of complete terminal run bundles per session and
the configured number/age of quiescent sessions. It never deletes nonterminal main or
child work.

Startup maintenance:

1. physically purges previously tombstoned, quiescent sessions;
2. prunes complete old run bundles;
3. tombstones only quiescent count/age candidates under their individual session locks;
4. removes orphan revision/checkpoint files and directories for purged sessions;
5. performs a passive WAL checkpoint; and
6. runs rate-limited `VACUUM` only when configured thresholds permit it.

`VACUUM` is never part of a product write transaction. Busy-database failures defer it
rather than blocking or failing startup.

## Schema Cutover

The internal SQLite schema marker is version 6. The default path is still
`<session_dir>/sessions-v2.sqlite3`.

Schema-v5 YAACLI 2 data is intentionally disposable for this cutover. When the known v5
marker is found, YAACLI explicitly deletes and recreates the same database path; no
payload migration, dual read, or v3 database is created. The complete normalized schema-v5 object set is fingerprinted before destructive
cutover under a sibling filesystem lock. Other incompatible, malformed, or unmarked
databases are rejected rather than silently interpreted.

The former pre-2.0 default `sessions.sqlite3` remains untouched and is not migrated.

## Verification Invariants

Tests cover:

- schema-v5 destructive cutover at the same path and strict rejection of unknown schema;
- absence of subagent tables and large revision/checkpoint columns in SQLite;
- grep-readable revision, checkpoint, and child files;
- file-first publication and orphan cleanup;
- metadata-only session summaries;
- checkpoint and terminal publication idempotency;
- input, HITL, terminal, and tombstone state transitions;
- interrupted recovery without model replay;
- process-local child deferred continuation, cumulative usage, persisted steering,
  owner fencing, exact descriptor restoration, terminal inspection, and startup
  orphan-to-lost recovery;
- independent session-lock concurrency and same-session multi-child writes;
- retention refusal for nonterminal main or child work; and
- identical revision semantics in TUI and headless frontends.
