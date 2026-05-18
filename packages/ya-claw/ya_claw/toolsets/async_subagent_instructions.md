# Session-backed Async Subagents

Use async subagent tools for durable long-running child work that can continue after the current parent run finishes.

Available tools:

- `spawn_delegate`: create or resume a named async subagent session.
- `list_async_subagents`: list async subagents owned by the current parent session.
- `get_async_subagent`: inspect task metadata, child session, latest run, result summary, and trace references.
- `steer_async_subagent`: send new input to a currently running child subagent. Queued or idle children return status and next-step guidance.
- `cancel_async_subagent`: request cancellation for queued or running child work.

Naming rules:

- Use `name` as the stable parent-session-local handle.
- Choose descriptive names like `repo-map`, `patch-review`, or `research-pricing`.
- Reuse the same `name` with `spawn_delegate` to continue a terminal child session.

Spawn behavior:

- `spawn_delegate(subagent_name, prompt, name, context)` returns immediately with task/session/run identifiers.
- The child uses the same profile subagent configuration as blocking `delegate` for the given `subagent_name`.
- The child session has separate continuity from the parent session.
- Completion wakes the parent session with an `async_task_completed` command input.

Steering behavior:

- Use `steer_async_subagent` when the child is running and needs additional direction.
- Use `get_async_subagent` to inspect result details and trace references after completion.
- Use `list_async_subagents` when the current mapping from names to child sessions is unclear.
