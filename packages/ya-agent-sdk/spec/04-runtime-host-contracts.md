# Runtime Host Contracts

This document records current cross-cutting SDK contracts that host applications must
preserve. Feature-specific behavior remains documented in the package README and the
other SDK specifications.

## Stream Recovery Budgets

`stream_agent` separates transient model transport recovery from non-transport
execution recovery.

- `stream_transport_resume_max_attempts` limits each consecutive HTTP/WebSocket
  transport-failure streak through an independent budget.
- A successful model request resets the current transport-failure streak.
- Mid-stream disconnects such as incomplete chunked reads do not consume
  `stream_resume_max_attempts`, which remains the non-transport execution recovery
  budget.
- Delegated subagents and self forks inherit the root run's effective recovery policy,
  recover their own message history independently, and accumulate usage across
  attempts.

## Skill Routing and Runtime Catalog

Skill routing has two stages:

1. Inspect plausible candidates with high recall to determine applicability.
2. Activate only skills whose scope directly governs the requested work.

Candidate inspection is non-binding. Once activated, a skill's applicable workflow is
mandatory. Compaction carries forward only activated skills that remain relevant to
unfinished work.

`SkillToolset` publishes its priority-resolved catalog through
`AgentContext.available_skills`. A host that classifies explicit skill syntax must call
`refresh_context()` immediately before classification so it uses the effective runtime
catalog. The catalog is derived runtime state and is not persisted in `ResumableState`.

## Main-Agent Interaction Boundary

`ask_user_question` is opt-in and main-agent-only. It uses Pydantic AI `CallDeferred`
and requires a host that can collect answers and return matching
`DeferredToolResults.calls`. It is not part of the default SDK tool surface. Static and
dynamic SDK toolsets enforce the child-agent boundary even when unavailable tools are
otherwise retained or wrapped by opaque proxy/search composites.

See the [package README](../README.md#structured-clarifying-questions) for host setup.

## CodeAct Boundary

CodeAct is opt-in through `create_agent(codeact=CodeActConfig(...))`, while
`pydantic-monty` is a core SDK dependency. The YA wrapper exposes only eligible direct
tools, dispatches nested calls through `ToolManager`, and keeps inline Monty state
scoped to one agent run. `run_program` loads bounded source bytes through the current
Environment `FileOperator` and starts a fresh execution session.

Monty receives neither a workspace mount nor ambient operating-system authority. Host
argument materialization is admission-bounded before dispatch. `timeout_seconds`
initiates cancellation, while ownership-safe cleanup may finish afterward, so tools
marked `codeact=True` must cooperate with cancellation. Host-managed
`NamedMCPToolset` tools carry CodeAct eligibility by default while retaining normal
approval and agent boundaries.

See [03-codeact-programs.md](03-codeact-programs.md) for the complete execution,
security, replay, and limit contracts.
