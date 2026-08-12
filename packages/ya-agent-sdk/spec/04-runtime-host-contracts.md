# Runtime Host Contracts

This document records current cross-cutting SDK contracts that host applications must
preserve. Feature-specific behavior remains documented in the package README and the
other SDK specifications.

## Model Correction Retry Budgets

SDK-created agents expose five category limits through `RetryConfig` plus one separate
run-wide boundary:

- Pydantic AI tool retries are tracked per tool name and reset after that tool succeeds.
- Output retries are cumulative within one Pydantic AI run.
- SDK `Toolset`, Tool Search, and Tool Proxy wrappers each resolve their category from
  the runtime context unless the wrapper has an explicit local `max_retries` override.
- The SDK `overall_retries` ceiling is cumulative across all retry-prompt-producing
  tool, output, and capability paths and never resets within a run.

All five `RetryConfig` categories default to 5; `overall_retries` independently defaults
to 3 and can therefore terminate a run before any category reaches 5.
`overall_retries=None` disables only the run-wide ceiling. Existing `retries`,
`output_retries`, and `toolset_max_retries` arguments remain explicit compatibility
overrides. The resolved Pydantic AI tool/output limits propagate to regular subagents
and self forks, while child contexts inherit the runtime `RetryConfig` used by SDK
wrappers.

A `BaseTool` `ModelRetry` must propagate to Pydantic AI so these budgets can account for
it; SDK tool wrappers and post-hooks must not convert it to an ordinary value. Transport
request retry and stream recovery are separate failure domains and do not consume model
correction retries.

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

## Steering Continuation Boundary

Hosts send actual steering content through `AgentContext.send_message()` and the SDK
message bus. The message-bus completion capability checks the final node boundary and
uses Pydantic AI `RunContext.enqueue(priority="asap")` only to keep an ending run alive. The next model
request still consumes and injects the authoritative bus messages, preserving target
routing, cursor idempotency, `MessageReceivedEvent`, multimodal rendering, and compact
state. User-source bus messages are appended to `AgentContext.steering_messages`; there
is no `AgentContext.inputs` field and steering is intentionally not merged into the
canonical `user_prompts`. Cache-friendly compact, legacy compact, handoff, and
`ResumableState` all preserve this dedicated text representation. Successful context
reset clears the accumulated list only after it has been replayed. Steering
continuation is not a `ModelRetry` and consumes no correction budget.

Because Pydantic AI drains end-of-run enqueued messages from node lifecycle hooks, SDK
main-agent, subagent, and self-fork stream drivers must use hook-aware graph advancement;
bare `async for node in AgentRun` is not compatible with this contract.

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
approval and agent boundaries. Their MCP adapter keeps `structuredContent` as the
nested Python value and forwards accompanying media as supplemental
`ToolReturn.content`. A completed MCP `isError` response is returned as structured data
for explicit inspection rather than raised as `ModelRetry`; transport or protocol
failures that produce no result are terminal `ToolFailed` outcomes. This prevents
completed side-effecting calls from consuming retry budgets or being implicitly
replayed.

See [03-codeact-programs.md](03-codeact-programs.md) for the complete execution,
security, replay, and limit contracts.
