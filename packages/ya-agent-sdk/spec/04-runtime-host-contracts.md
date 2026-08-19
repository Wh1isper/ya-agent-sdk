# Runtime Host Contracts

This document records current cross-cutting SDK contracts that host applications must
preserve. The capability-first architecture and product-specific durability contracts
are defined in [05-capability-first-runtime.md](05-capability-first-runtime.md).

## Runtime Entry and Composition

`create_agent()` returns an unentered `AgentRuntime`. `capabilities=` is the sole public
behavior-composition surface.

Runtime entry occurs in this order:

1. enter the `Environment`;
2. enter and restore `AgentContext`;
3. collect explicit, context, Environment, and resource capability source groups;
4. validate duplicate singleton and ordering constraints over all leaves;
5. register runtime-managed capability cleanup; and
6. construct and enter the Pydantic AI `Agent`.

`runtime.agent`, final capabilities, and provenance are unavailable before entry. Hosts
must not prebuild a live Agent and then patch in environment tools or filter a final
tool list.

Raw functions and external Pydantic AI toolsets are wrapped by native Pydantic AI
capabilities. SDK feature toolsets remain private implementation adapters owned by
feature capabilities.

## Agent Segment Harness

`AgentExecutionHarness` is the host-neutral convenience boundary for one native agent
segment. `AgentSegmentRequest` supplies prompt/history/deferred continuation and
process-local policies. The outcome is either completed or suspended and always carries
an `AgentExecutionCheckpoint` with canonical messages, `ResumableState`, segment usage,
and detached input-ledger state.

The harness is stateless. It owns no database, scheduler, queue, lease, action store,
product retry policy, or crash recovery. A persistent host must commit accepted intent
before execution, persist stable segment checkpoints, and own terminal/interrupted
publication. An incomplete active segment is not an SDK replay boundary.

See [Host-Owned Durable Sessions and Agent Segment Execution](05-capability-first-runtime/06-yaacli-durable-sessions.md).

## Model-Correction and Tool-Execution Budgets

Native Pydantic AI tool/output correction limits are configured through
`create_agent(retries=...)`.

`OverallRetryBudget` is an explicit capability that counts retry prompts cumulatively
within one native run. It defaults to three when included through
`RuntimeFoundationCapability`; hosts that compose foundation leaves directly must add
or deliberately omit it themselves.

`ToolRetryCapability` is a separate host-execution wrapper for configured transient
exceptions. `ToolTimeoutCapability` bounds one validated execution attempt. Neither
consumes the native model-correction budget. SDK adapter-local Tool Search or Tool Proxy
retry settings remain scoped to those wrappers.

`ModelRetry`, `ApprovalRequired`, and `CallDeferred` are Pydantic AI control flow and
must propagate unchanged through SDK adapters and policy wrappers.

## Stream Recovery Budgets

`stream_agent()` separates transient model transport recovery from non-transport
execution recovery:

- `stream_transport_resume_max_attempts` limits each consecutive HTTP/WebSocket
  transport-failure streak;
- a successful model request resets that transport streak;
- `stream_resume_max_attempts` independently limits non-transport execution recovery;
- child runs inherit the root run's effective recovery policy while recovering their
  own canonical history; and
- accumulated usage survives recovered attempts.

Per-call `stream_agent()` values override `ModelConfig` defaults. Transport and stream
recovery do not consume model-correction or host-tool retry budgets.

SDK main and in-process child stream drivers use hook-aware graph advancement so
Pydantic AI node lifecycle and pending-message behavior are preserved. A host must not
replace this path with a bare node loop when it depends on SDK routing, usage, recovery,
or lifecycle events.

## Native Logical-Run Input

Actual user or feature input enters an active run through
`LogicalRunInputRouter.enqueue()`, which delegates to native Pydantic AI
`AgentRun.enqueue()`.

`RunInputLedger` is the structured retention truth for inputs applied to one logical
run. It preserves applied user input across compact, handoff, retry recovery, and
durable replay without becoming a second delivery transport.

A durable host must:

1. persist accepted input before acknowledgement;
2. identify the target logical run explicitly;
3. enqueue through the active owner when one exists;
4. retain unapplied input in its durable inbox when the run is inactive or suspended;
5. reconcile native enqueue/application events idempotently; and
6. fence terminal completion so accepted input is either applied or explicitly
   rejected, never silently lost.

The SDK 2.0 runtime has no MessageBus, bus cursor, completion guard, or steering-only
replay list. UI readiness events and lifecycle projections are observations, not
canonical model input and not automatic wake authority.

## Skill Routing and Runtime Catalog

Skill routing has two stages:

1. inspect plausible candidates with high recall to determine applicability; and
2. activate only skills whose scope directly governs the requested work.

Candidate inspection is non-binding. Once activated, the relevant skill workflow is
mandatory. Compaction carries forward only activated skills that remain relevant to
unfinished work.

`SkillToolset` publishes its priority-resolved catalog through
`AgentContext.available_skills`. A host that classifies explicit skill syntax must
refresh that capability/toolset immediately before classification so it uses the
effective runtime catalog. The catalog is derived runtime state and is not persisted in
`ResumableState`.

## Deferred Host Interaction

`UserInteractionCapability` is opt-in and owns `ask_user_question`. Tool approval is
configured by `ToolApprovalCapability`. Both yield native `DeferredToolRequests` and
require a host that can collect exact decisions/results and resume with matching
`DeferredToolResults`.

`DeferredInteractionResolver` is the typed public facade for validating host decisions.
Hosts must not inspect a runtime-private SDK toolset. `AgentRuntime` deliberately does
not expose one.

Main-agent-only tool metadata is enforced at the final `ToolVisibilityCapability`
boundary and by tool-local availability checks. Child plans should omit interactive
capabilities unless the selected driver implements nested durable suspension and host
continuation.

See [Structured Clarifying Questions](../README.md#structured-clarifying-questions).

## Portable Subagent Boundary

Declarative children use native `AgentSpec` inside `SubagentSpec`. Resolution happens
against one immutable SDK `CapabilityCatalog`, explicit host requirements, and
enumerated host-policy capabilities.

Named children receive only their native capability definitions; omission is not
inheritance. Self forks rebuild an explicit `SelfForkPolicy` and receive one bounded
spawn-time canonical-history snapshot. Neither path clones live parent Agents,
capabilities, toolsets, contexts, or model clients.

Foreground/background/resume/steer/cancel/inspect operations share
`SubagentExecutionService`. Stores and drivers declare whether they are process-local or
restart-durable, and their durability declarations must agree. See
[Portable Subagent Runtime](05-capability-first-runtime/07-subagent-runtime.md).

## CodeAct Boundary

CodeAct is opt-in through `CodeActCapability(config=CodeActConfig(...))`.
`pydantic-monty` receives no workspace mount, network, subprocess, environment,
credential, or clock authority. Every effect crosses an explicitly eligible host tool
and retains normal Pydantic AI validation, policy, approval, tracing, and child
boundaries.

`run_program` loads bounded strict UTF-8 source through the current Environment
`FileOperator` and creates a fresh Monty session. Inline `run_code` state is scoped to
one agent run.

Host argument materialization and each nested result are bounded before crossing into
Monty. `timeout_seconds` initiates cancellation, while ownership-safe cleanup may finish
afterward; tools exposed to CodeAct must therefore cooperate with cancellation.

Host-managed MCP tools preserve `structuredContent` as the nested Python value and
forward media as supplemental content. A completed MCP `isError` result is returned for
explicit inspection rather than raised as `ModelRetry`; a transport/protocol failure
without a result is terminal. This prevents implicit replay of completed side effects.

See [03-codeact-programs.md](03-codeact-programs.md).
