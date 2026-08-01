# Session Persistence and Restore

## Scope

This document defines YAACLI's durable session schema, safe legacy recognition, transactional TUI restore, and headless terminal-event persistence contract.

## Durable Layout

Schema-v2 sessions live under the configured sessions directory:

```text
sessions/<session-id>/
  metadata.json
  turns/<turn-id>/
    metadata.json
    message_history.json
    context_state.json       # optional
    display_messages.json    # optional
```

The root metadata contains:

- `schema_version = 2`;
- `session_id` equal to the directory name;
- `head_turn_id` naming the committed turn;
- creation and update timestamps; and
- model, workspace, output, and save-reason metadata when available.

A turn is committed by writing its artifacts atomically and updating root metadata to point at it. Retention never follows symbolic links and never treats an escaping turn ID as valid.

## Artifact Roles

- `message_history.json` stores Pydantic AI model messages.
- `context_state.json` stores `ResumableState` conversation data.
- `display_messages.json` stores a bounded AGUI-aligned replay list.
- turn and root `metadata.json` files provide identity, ordering, and summary fields.

Context state and display replay are optional for continuation. Missing context or display artifacts load as absent. Invalid or oversized display replay is skipped while valid model history still loads; malformed model history remains fatal.

## Session Identity and Filesystem Safety

A recognized schema-v2 session must have:

- a real, non-symlink session directory;
- a safe single-segment directory name;
- regular root metadata plus regular turn `metadata.json` and `message_history.json` identity artifacts;
- root `session_id` equal to the directory name;
- a safe `head_turn_id`;
- a real head-turn directory contained by the session directory; and
- optional `context_state.json` and `display_messages.json` artifacts that, when present, are regular non-symlink files.

Session list, resolve, delete, upgrade, and retention operations reject symbolic-link and path-escape boundaries. Unrecognized non-empty directories are never overwritten by session saves.

## Legacy Recognition

Legacy sessions stored artifacts at the session root. A directory is eligible for upgrade only when it has a complete, parseable legacy identity:

1. real non-symlink directory;
2. regular `metadata.json`;
3. regular `message_history.json`;
4. metadata is an object without schema-v2 `schema_version`;
5. metadata `session_id` is a string exactly equal to the directory name;
6. message history parses with `ModelMessagesTypeAdapter`;
7. optional `context_state.json`, when present, parses as `ResumableState`; and
8. optional `display_messages.json`, when present, parses as validated display events.

A partial, malformed, missing-identity, wrong-identity, symlinked, or schema-v2-looking directory is not a legacy session and must not be modified.

Upgrade copies recognized artifacts into a generated turn, writes turn metadata, commits schema-v2 root metadata, and only then removes the known root legacy artifacts.

## Session Selector

`/session` without an ID opens a responsive selector backed only by committed session
metadata. Rows contain bounded input/output previews from metadata; listing does not
read model history, context state, or display replay artifacts and does not wait for
artifact-writer locks. Selecting a session then enters the explicit load path, where
full validation and any recognized legacy migration occur.

## TUI Restore Preparation

`/load <folder>` and `/session <id>` use the same transactional restore pipeline.

All fallible preparation happens before the active conversation changes:

1. resolve artifact paths;
2. parse required model history;
3. parse optional `ResumableState`;
4. validate or safely skip display replay;
5. derive `candidate_ctx = old_ctx.prepare_new_run()`;
6. reset all conversation-scoped fields on the candidate; and
7. restore persisted state into the candidate.

If preparation fails, YAACLI keeps the current runtime context, session ID, history, replay, output, message bus, and background monitor state unchanged.

The runtime and environment are not rebuilt. The candidate preserves current model/runtime configuration and environment resources while isolating conversation state such as provider session/thread IDs, shell environment, task and note managers, subagent history, goal state, stream queues, and message bus.

## Background Isolation Boundary

After preparation succeeds, YAACLI establishes the old-session isolation boundary before the first `await`:

1. `BackgroundMonitor.begin_session_reset()` revokes the inherited shell-session lease, tombstones and cancels old subagent tasks, drops old shell wakeups, and suppresses reset-time shell callbacks;
2. background usage is drained;
3. `reset_session_state()` asks the reusable shell backend to terminate every owned foreground and background execution and discard every background output buffer and retained terminal result, then performs the bounded subagent wait;
4. old subagents and child tasks retain the revoked lease even if they ignore cancellation or outlive the bounded wait, so later shell access is rejected;
5. late old subagent results remain discarded; and
6. the monitor, environment, and shell backend remain available for new-session work, but old session work does not.

Process creation itself is shell-owned and shielded from caller cancellation until it either fails or yields an `ExecutionHandle`; an eventual handle is then registered or terminated before cancellation propagates. Foreground `Shell.execute()` calls stay registered for their full lifetime, so reset can terminate commands already waiting in `communicate()` rather than merely cancelling the calling subagent.

Non-shell cleanup failures roll forward after the tombstone boundary because old results are already isolated. Cancellation also commits the isolated candidate and is then re-raised, but only after process termination has completed successfully. If an execution handle's kill hook fails, `ShellBackgroundResetError` retains that handle for a later retry and blocks the state switch: `/load` and `/session` keep the active context and identity, while `/new` restores its previous durable session ID. A session transition must never silently commit while process ownership is unresolved.

## No-Await Commit

The final state switch is one synchronous no-`await` block:

1. reset TUI conversation state;
2. replace `runtime.ctx` with the candidate;
3. retarget the background monitor to the candidate message bus;
4. replace model history and display replay;
5. commit the durable session ID; and
6. rebuild visible output and return to the idle phase.

Observers therefore see either the complete old session or the complete restored session, not a mixed context/history/message-bus combination.

`/load` preserves the current durable session ID. `/session` commits the resolved target session ID in the same block.

`/new` publishes a fresh durable session ID before its first asynchronous cleanup boundary. If Ctrl+C cancels background cleanup after the old conversation is tombstoned and process termination succeeds, cleanup still commits the fresh context and the new ID remains authoritative; a later turn can never publish under the previous session ID. A shell termination failure instead aborts the commit and restores the previous ID. Old shell notifications and terminal buffers are discarded rather than delivered to a fresh message bus.

## Runtime Policy

Persisted state cannot weaken the current runtime approval policy. `restore_resumable_state_safely()` snapshots current approval lists, applies resumable conversation state transactionally, and reapplies those lists after successful restore.

A fresh conversation starts from the configured `shell_env` runtime baseline rather than an empty mapping. Session restore overlays its persisted shell environment onto that baseline, so `/new` drops conversation-specific values without disabling process configuration and `/load` preserves both configured and restored values.

HITL `asyncio.Event` objects and current approval-panel cursors are process-local and are not reconstructed from model history. See `08-hitl.md`.

## TUI Terminal Events

The interactive TUI records terminal display events for completed, cancelled, and failed runs. For failed runs it appends `RUN_ERROR` before writing the error-recovery snapshot, so durable replay contains the terminal event. A response that already emitted `RUN_FINISHED` is not reclassified as cancelled if post-response persistence is interrupted. Persistence errors are shown to the user without duplicating or replacing the terminal event.

## Headless NDJSON Contract

Headless mode writes protocol events as one JSON object per stdout line and flushes each line. Non-protocol diagnostics are redirected to stderr.

Success ordering is strict:

1. build `RUN_FINISHED`;
2. append it to the bounded replay to be persisted;
3. save history, state, replay, and output; and
4. emit `RUN_FINISHED` to stdout only after the save succeeds.

If saving fails, headless mode emits `RUN_ERROR`, emits no `RUN_FINISHED`, re-raises the failure, and exits non-zero through the CLI. Exception text uses defensive formatting, so a broken exception `__str__` cannot suppress the terminal protocol event or the CLI diagnostic.

Empty `DeferredToolRequests` payloads are rejected. Auto-denied deferred continuations share one cumulative model-request budget for the full headless invocation rather than resetting the configured limit on every continuation.

Headless startup resolves one effective model profile: an explicit `--profile` for this invocation, otherwise the persisted startup profile. The same resolved profile drives runtime construction and saved session provenance (`model_profile_id`, label, and model); explicit CLI selection remains non-persistent.

For `asyncio.CancelledError`, headless mode emits custom `run_cancelled` with reason `cancelled` and re-raises cancellation. `AgentInterrupted` and `KeyboardInterrupt` emit `run_cancelled` with reason `interrupted`. Cancellation does not emit a false success or error terminal event.

## Retention

Retention keeps bounded turn and session histories according to configuration. The current session and current head turn are protected while trimming. A turn created by a failed save is removed before the exception escapes and never consumes a retention slot. Ordering uses committed metadata timestamps with filesystem time only as a fallback.

## Verification Invariants

Tests must cover:

- schema-v2 save, head resolution, and retention;
- symlink and path-escape rejection;
- valid legacy upgrade;
- partial, malformed, missing-identity, and wrong-identity legacy rejection without file movement;
- restore preparation failure preserving the old session;
- candidate context and message-bus isolation;
- non-shell cleanup failure and cancellation roll-forward;
- shell termination failure retaining a retry handle and blocking `/new`, `/load`, and `/session` commits;
- `/new` cancellation preserving a fresh identity for the next save;
- old subagents and inherited child tasks losing shell access after reset;
- old shell readiness and terminal buffers being absent after real `/new` dispatch;
- `/load` ID preservation and `/session` ID replacement;
- configured shell environment baseline plus restored session overlays;
- current approval policy preservation;
- successful headless replay ending in persisted `RUN_FINISHED`;
- persistence failure ending in `RUN_ERROR` without `RUN_FINISHED`; and
- cancellation ending in `run_cancelled` without success/error terminals.
