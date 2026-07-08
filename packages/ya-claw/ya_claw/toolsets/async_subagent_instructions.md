Use async subagents for durable child work that can continue after the current parent run finishes.

Best practices:

- Choose stable, descriptive parent-session-local names such as `repo-map`, `patch-review`, or `research-pricing`.
- Reuse the same name to continue a terminal child session instead of starting duplicate work.
- Spawn only bounded work with clear scope, independent value, or useful parallelism.
- Use durable async children for work expected to outlive the current parent run; keep quick current-turn fan-out local or blocking.
- Let completion wake the parent; do not poll in loops.
- Steer only when a running child needs additional direction, and inspect terminal children only when their details are needed for integration.
