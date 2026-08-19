# Migration and Validation

## 1. YA Agent SDK 2.0 Cutover

Capability-first composition and native steering ship as a breaking 2.0 boundary.
There is no runtime compatibility layer, legacy capability adapter, dual API, profile
fallback, or MessageBus fallback.

Implementation proceeds in substantial end-to-end blocks, but 2.0 is released only
after all repository hosts, resources, profiles, examples, and canonical skills use the
new contracts.

The cutover rules are:

1. `create_agent()` accepts only the new capability composition surface.
2. Declarative main and child agents use native Pydantic AI `AgentSpec`; portable
   children add a thin YA `SubagentSpec` envelope. No runtime accepts `SubagentConfig`,
   duplicate agent fields, tool lists, or implicit inheritance.
3. `AgentContext` exposes `get_capabilities()` and no history-specific contribution
   hook.
4. Environment/resource providers expose provenance-grouped capabilities only.
5. YAACLI and Claw declarative profiles embed native `AgentSpec` cores; their host
   envelopes contain only product policy.
6. All steering/background producers use logical-run routers and Pydantic AI
   enqueue.
7. Every logical run records initial and accepted enqueued user inputs in a resumable
   `RunInputLedger`; compact and handoff restore applied entries.
8. MessageBus types and state are deleted.
9. YAACLI product turns execute through `LocalExecutionCoordinator` and the SDK segment
   harness while `SessionStore` owns session, input, HITL, checkpoint, event, and
   retention truth.
10. SDK, YAACLI, and YA Claw bind one public subagent registry/execution contract to
    host-specific drivers and stores.
11. Old serialized configuration and session data is migrated offline or rejected with
    a 2.0 schema error; it is not interpreted by the runtime.
12. Installed and explicitly imported custom capability classes resolve only through
    the SDK `CapabilityCatalog`; hosts do not implement entry-point or custom-registry
    logic.

## 2. Breaking API Matrix

| Removed in 2.0 | Replacement |
| --- | --- |
| `create_agent(tools=[...])` | SDK feature capabilities |
| `create_agent(agent_tools=[...])` | `Capability(tools=[...])` in `capabilities=` |
| `create_agent(toolsets=[...])` | Pydantic AI Toolset capability in `capabilities=` |
| `pre_capabilities=` | one ordered `capabilities=` list |
| pre/post/global hook dictionaries | Pydantic AI `Hooks` or tool policy capabilities |
| approval arguments | `ToolApprovalCapability` |
| timeout/availability arguments | `ToolTimeoutCapability` and `ToolVisibilityCapability` |
| compact keyword group | `ContextCompactionCapability` |
| `codeact=` | `CodeActCapability` |
| `SubagentConfig`, duplicate agent schema, `Toolset.with_subagents()`, generated delegate backends, and automatic inheritance | native `AgentSpec`, thin YA `SubagentSpec`, resolved plan, public registry/execution service, and host driver/store |
| `AgentContext.get_history_capabilities()` | `AgentContext.get_capabilities()` |
| Environment/resource `get_toolsets()` | `AgentContributionGroup.capabilities` |
| SDK `ModelFeature` | `ModelFeature` |
| Claw `builtin_toolsets`, duplicate agent fields, and allowlist-driven assembly | native profile `AgentSpec.capabilities`; final visibility policy only for host enforcement |
| `AgentContext.send_message()` / `BusMessage` | logical-run router enqueue |
| `AgentContext.steering_messages` | context-owned `RunInputLedger` plus canonical Pydantic AI history |
| `runtime.core_toolset` host access | typed deferred-interaction resolver |
| YAACLI session free functions and runtime legacy upgrade | typed SQLite `SessionStore` and explicit offline import |
| YAACLI ad hoc subagent task ownership | SQLite product records plus process-local `LocalSubagentDriver` and explicit orphan-to-`lost` recovery |

## 3. Block A: Capability Foundation

### 3.1 Deliverables

1. Add the capability package, stable IDs, and narrow dependency protocols.
2. Implement `RuntimeFoundationCapability` and semantic ordering constraints.
3. Replace `create_agent()` behavior parameters with one `capabilities=` surface and
   make it return an unentered runtime plan.
4. Enter and restore Environment/resources before resolving contributions and
   constructing the Pydantic AI Agent in `AgentRuntime.__aenter__()`.
5. Add `AgentContext.get_capabilities()` and expose resolved provenance through
   `AgentRuntime.capabilities`.
6. Add resumable `RunInputLedger` state keyed by logical run ID and make compact and
   handoff restore all applied entries without clearing the ledger.
7. Remove `AgentContext.get_history_capabilities()` and implicit history processor
   assembly.
8. Add duplicate singleton detection.
9. Implement stable instructions, cached catalog instructions, canonical processors,
   and request-envelope decorators.
10. Add request-envelope and run-input retention budgets to context lifecycle policy.
11. Remove duplicate file inspection registration.
12. Add characterization tests before replacing current behavior.
13. Add SDK-owned per-type entry-point and explicit-import discovery, native/custom
    serialization-name collision checks, deterministic catalog construction, and the
    Pydantic AI custom-type bridge specified by
    [Custom Capability Type Discovery](../06-capability-plugins/README.md).

### 3.2 Exit criteria

- Every agent-run tool, instruction, model setting, and Pydantic AI hook contribution
  is represented by a capability.
- Runtime foundation and all context/Environment/host contributions, including
  resources discovered during Environment entry/restore, are inspectable in one
  resolved list.
- Request-envelope data is absent from `AgentRun.all_messages()` and exported state.
- Canonical processors do not mutate input history objects.
- Runtime foundation ordering is deterministic and tested.
- Repeated compact/handoff cycles retain every applied logical-run user input in order
  without consuming unresolved input.
- Entry-point and explicit-import classes use the same validation path; installed but
  unselected classes are not imported, and hosts consume one SDK-owned immutable type
  catalog.

## 4. Block B: Feature and History Capabilities

Each feature family is migrated bottom-up:

1. dependency protocol and state owner;
2. tool implementation/private toolset;
3. leaf capability and instructions;
4. preset and tool policy integration;
5. explicit child composition; and
6. feature, composition, and host tests.

### 4.1 Deliverables

- one read-only/writable `FilesystemCapability`;
- shell tools, logical-stream background completion, and request-only process status;
- web search/content and web preset;
- document conversion and media reading;
- task, note, optional todo, thinking, and user interaction;
- handoff and request-envelope file inspection;
- Skills, manager-backed Tool Proxy, MCP, CodeAct, and delegation;
- native Pydantic AI Tool Search adoption;
- reasoning, request-only media projection, tool argument repair, and tool ID
  normalization;
- cache-friendly compact, cold-start, request-envelope Environment/runtime context,
  and system prompt reinjection;
- approval, timeout, visibility, observation, and retry policy leaves extracted from
  the monolithic SDK Toolset;
- typed deferred-interaction facade replacing `runtime.core_toolset`; and
- catalog-resolvable main capability plans used later by portable named-child and
  self-fork policies.

### 4.2 Exit criteria

- No repository feature uses public SDK Toolset composition.
- No default processor exists outside its owning capability.
- No migrated SDK processor mutates canonical-history input objects.
- Tools and instructions share one availability decision.
- Custom context subclasses replace stores/authorities through narrow protocols.
- Child composition does not slice capability internals.
- Tool Proxy nested calls cross the active Pydantic AI `ToolManager` boundary.
- Native Tool Search covers all retained 2.0 discovery behavior.

## 5. Block C: Logical Stream and Native Steering

### 5.1 Deliverables

1. Add root `LogicalRunInputRouter` and expose `EnqueueReceipt` through
   `AgentStreamer`.
2. Bind/unbind every streamer segment and native `AgentRun` attempt around graph
   advancement.
3. Add stable logical input metadata to enqueued `ModelRequest` objects.
4. Reconcile applied input against recovery history before attempt rebind.
5. Requeue unresolved user and feature envelopes across transport/execution recovery
   and deferred continuation segments.
6. Add `DeferredTerminalCapability` so `DeferredToolRequests` reaches the host before
   native pending-message terminal redirect.
7. Add a root/shared `ActiveRunRegistry`; child `for_run()` instances own only
   registration tokens.
8. Route YAACLI steering and every SDK background result through logical-run routers.
9. Map native `EnqueuedMessagesEvent` to logical applied events.
10. Apply one count/byte budget across bound, recovery, and suspended ingress.
11. Remove `MessageBus`, `BusMessage`, bus filters, cursors, guard, and bus events.
12. Replace `steering_messages` and its text-only builders with `RunInputLedger`;
    initial and enqueued user inputs share one structured retention contract.

### 5.2 Exit criteria

- Input accepted during a bound attempt, recoverable failure, between attempts, or a
  deferred continuation gap is applied once or explicitly rejected.
- Failed/cancelled native attempts cannot return a false successful receipt.
- Applied input is present in recovery history and never rebound.
- Main and child routers are discoverable through the shared registry.
- Text and multimodal input appear exactly once in canonical history and once in the
  logical run ledger, with distinct delivery and retention responsibilities.
- Enqueue consumes no retry budget.
- Hook-aware graph advancement drains non-deferred terminal pending input, while a
  deferred-tool terminal remains host-visible with pending envelopes preserved.

## 6. Block D: YA Claw Active Input

### 6.1 Deliverables

1. Add `SteerMode` to run/session steer models and `active_mode` to unified submit.
2. Replace raw steering lists with typed logical-run envelope state and unified
   count/byte backpressure.
3. Bind Claw to the SDK logical-run router rather than a native attempt.
4. Track logical input IDs and all attempt-specific native enqueue IDs.
5. Persist user and background-completion envelopes through the durable HITL
   continuation store with correlation and delivery state.
6. Restore all HITL-gap envelopes before deferred continuation graph advancement.
7. Remove 100 ms signal polling, user BusMessage forwarding, and late-steering stream
   re-segmentation.
8. Emit accepted, enqueued, applied, and rejected events.
9. Add “Steer now” and “Queue after current work” UI actions.
10. Update Claw execution, storage/streaming, API, UI/operations, and runtime assembly
    specifications.

### 6.2 Exit criteria

- `steer` maps only to `asap`; `queue` maps only to `when_idle`.
- `DispatchMode.QUEUE` is never used for active input.
- Pre-bind, bound, recovery, and HITL ingress share one bounded policy.
- SDK recovery preserves unresolved input and mode.
- `DeferredToolRequests` reaches the host even when `asap` or `when_idle` input is
  pending.
- HITL continuation gaps preserve user mode and canonical background completions
  without initial-prompt or bus injection.
- Terminal acceptance/commit races are deterministic.
- UI state distinguishes acceptance, per-attempt enqueue, and application.
- Active queued input is never represented as a durable next run.

## 7. Block E: YAACLI Durable Sessions and Execution

### 7.1 Deliverables

1. Replace session free functions with a transactional SQLite `SessionStore` owning
   immutable revisions, head CAS, logical runs, inbox/outbox, actions, event
   projections, usage, tombstones, export, and retention.
2. Add one-way offline import for committed filesystem schema-v2 sessions; reject
   unmigrated data at runtime.
3. Add stable YAACLI agent/capability/toolset IDs, used custom-type provenance, plan
   fingerprints, durable descriptors, and worker-side runtime reconstruction.
4. Add the SDK `AgentExecutionHarness` and execute every interactive/headless product
   turn through a host-owned local coordinator selected by exact runtime descriptor.
5. Persist initial and accepted user input before acknowledgement; deliver it from the
   workflow through the logical-run router and reconcile `RunInputLedger` on replay.
6. Replace process-local HITL waiters with durable pending actions, audited decisions,
   workflow communication, and continuation inbox restoration.
7. Add committed event/display projection distinct from best-effort live durable-step
   streaming.
8. Classify mutating tools as replay-safe, idempotency-keyed, or ambiguous-effect and
   suspend rather than blind-retry ambiguous effects.
9. Reconcile workflow/session state on restart, cancellation, terminal commit, and
   tombstoned deletion.
10. Add a workflow-owned node driver that drains the durable inbox at bind, after each
    graph boundary, after recovery/deferred continuation, and through an atomic terminal
    fence before native enqueue or closure.
11. Record one exact executable identity separately from capability-plan fingerprints,
    retain exact descriptors required by pending work, and bound every persisted segment
    checkpoint with artifact references.
12. Migrate TUI and headless product paths to the same session application service.

### 7.2 Exit criteria

- No execution-engine history is queried as the product session repository.
- Process termination commits active work as `interrupted` from the latest stable
  checkpoint without replaying the incomplete segment or its effects.
- A live `TUIContext`, Environment, credential, or asyncio primitive never enters a
  durable payload; `DurableAgentDeps` resolves a session-scoped worker lease only inside
  a registered durable adapter.
- Every nondeterministic tool/hook is rejected unless it executes through a registered
  durable adapter with replay-safe, idempotency-keyed, or ambiguous-effect semantics.
- An input arriving before the terminal fence is drained into the current logical run;
  one arriving after closure is rejected or routed to the next turn exactly once.
- Compact and handoff retain all applied initial/steer/queue user inputs after durable
  replay and repeated reduction cycles.
- Pending HITL batches survive restart and consume each audited per-call decision once.
- `RUN_FINISHED` is externally visible only after the immutable session revision
  commits.
- Session tombstones reject every late workflow, input, action, and child commit.

## 8. Block F: Portable Subagent Runtime

### 8.1 Deliverables

1. Adopt native Pydantic AI `AgentSpec` as the child agent-definition core and add a
   versioned thin YA `SubagentSpec` envelope, `SelfForkPolicy`, immutable resolved plans,
   plan fingerprints, and a resolver over the SDK `CapabilityCatalog` plus
   Environment/context dependency availability.
2. Add public `SubagentRegistry`, `SubagentExecutionService`, execution records,
   lifecycle events, typed driver/store contracts, and immutable portable plan
   descriptors.
3. Make `DelegationCapability` a thin model-facing adapter over the public services;
   foreground becomes spawn-plus-wait and background becomes spawn-plus-handle.
4. Give every child an isolated context, history, logical-run router, input ledger,
   capability state, and explicit shared-store references; bind exactly its native
   feature grants plus an enumerated fingerprinted host infrastructure/policy set.
5. Keep final main-only/authorization/recursion enforcement in
   `ToolVisibilityCapability` at child execution.
6. Add the SDK in-process/in-memory driver without restart-durability claims.
7. Add the YAACLI SQLite/`LocalSubagentDriver` composition and remove ad hoc subagent
   ownership from `BackgroundMonitor`; restart orphans become `lost`.
8. Bind both YA Claw foreground and background delegation to its existing SQL child
   sessions/runs and `session_async_tasks`, while replacing copied profile derivation and
   resolver logic.
9. Add persistent completion delivery/outbox semantics to active, suspended, and idle
   parents; typed events remain notification only.
10. Define one cross-host `AgentTemplateContext` projection and compose native structured
    output with `DeferredToolRequests` in the effective plan where needed.
11. Migrate builtin/custom definitions offline to native `AgentSpec` plus the YA
    envelope and delete `SubagentConfig`, duplicate agent fields,
    `Toolset.with_subagents()`, generated private backends, MessageBus delivery, and ID
    naming inference.

### 8.2 Exit criteria

- No child plan is derived by slicing parent tools or opaque capability internals.
- Context/Environment contributions omitted from a named child's native spec remain
  unavailable; child entry does not rerun root contribution merging.
- Named children and self forks resolve only explicit catalog references.
- SDK, YAACLI, and Claw produce the same target availability and plan semantics from
  one native `AgentSpec` core and YA envelope without duplicating native fields.
- Repeated spawn, completion, wake, steer, and cancel commands have one product effect.
- Parent and child never share mutable run-local context.
- YAACLI resumes active child workflows after restart; both Claw foreground and
  background calls retain SQL session-backed durability.
- Claw recovers a child from its immutable plan descriptor after the mutable source
  profile or registry changes or is deleted.
- Native structured output and stop-the-world deferred output compose without override,
  and templates render through one cross-host serializable projection.
- No host imports private generated-tool attributes or infers background mode from IDs.

## 9. Block G: Repository-Wide 2.0 Cutover

### 9.1 Deliverables

1. Migrate YAACLI built-ins to capability presets and the durable session/subagent
   application services.
2. Migrate all YAACLI and Claw profile models to native `AgentSpec` cores with only
   host policy in their outer envelopes; make both hosts consume the SDK custom-type
   catalog without copied discovery or registry logic; migrate seeded YAML, controllers,
   workflow/schedule paths, and async-subagent paths to capabilities and portable
   subagent specs.
3. Migrate every Environment/resource provider to provenance-grouped capabilities.
4. Add Pydantic AI's `spec` extra to every package that parses native YAML profiles or
   compiles `TemplateStr`, keep YAACLI free of workflow-engine extras, update `uv.lock`, and
   fail startup with a direct dependency diagnostic rather than a late import/template
   error.
5. Migrate examples, benchmarks, tests, canonical skills, and package README files.
6. Delete old construction arguments, ToolSearch/Skill public toolset exports, filter
   exports, old context methods, MessageBus, old session runtime paths, and old Claw
   profile/event schemas.
7. Rename `ModelFeature` to `ModelFeature` across code and serialized config.
8. Update package versioning and 2.0 release notes with the breaking API matrix and
   explicit migration commands.

### 9.2 Release criteria

- Repository search finds no removed API usage in production, tests, examples, or
  canonical skills.
- Bundled profiles serialize native `AgentSpec` cores and YA host/subagent envelopes
  without duplicating native agent-definition fields.
- Every Environment/resource provider returns provenance-grouped capabilities.
- No MessageBus type, user-source bus producer, steering-only replay field, private
  subagent backend, or process-local YAACLI execution owner remains.
- Generalized run input retention and YAACLI durable session execution are present.
- No host depends on `runtime.core_toolset`.
- All current package specs describe the 2.0 boundaries consistently.

## 10. Capability and Composition Tests

- stable capability IDs and duplicate errors;
- ordering constraints and native dependency-cycle construction failure;
- every SDK/YAACLI leaf's `get_ordering().position` is `None`, while native outermost
  pending/tool-search behavior and native innermost durable execution remain intact;
- canonical, request-envelope, node-terminal, and tool-execution matrices produce the
  same semantic order from deliberately shuffled caller lists;
- direct pairwise edges preserve required order when optional intermediate leaves are
  absent, including repair/budget-plus-`ReinjectSystemPrompt`-only compositions;
- absent optional targets add no edge, a fully unconstrained list retains caller order,
  and exact final order is asserted only when the complete active graph defines it;
- `for_run()` state isolation across concurrent runs;
- tool/instruction availability coupling;
- explicit child capability subsets and main-only exclusion;
- external Pydantic AI capability pass-through, including local type relationships with
  built-in leaves and an unconstrained extension tested both alone and in a fully
  unconstrained caller-ordered list;
- custom-type metadata-only discovery, explicit selection, direct-import equivalence,
  deterministic catalog ordering, collision diagnostics, Pydantic AI `AgentSpec`/schema
  round trips, and proof that catalog/import order never changes runtime ordering;
- named-child omission remains authoritative even when its Environment advertises the
  omitted feature, while mandatory host injections are enumerated and fingerprinted;
- custom `AgentContext.get_capabilities()` contributions and provenance;
- context/store subclass replacement through narrow protocols;
- per-resource `AgentContributionGroup.source_id` diagnostics;
- Environment setup and restored resources contribute capabilities before Agent
  construction without requiring callers to pre-enter the Environment;
- deferred capability tools/instructions loading together;
- native Pydantic AI Tool Search behavior for deferred SDK features; and
- built-in, Pydantic AI toolset, and entry-point-contributed function tools all cross
  the same visibility, approval, observation, retry, and timeout policy hooks, with
  logical-call observation outside retries and one timeout per attempt; provider-native
  model tools retain their native provider contract.

## 11. History and Prompt Tests

- canonical processors preserve input object identity;
- `before_model_request` output intentionally appears in canonical history;
- `wrap_model_request` envelope decoration is absent from `all_messages()` and export;
- replacement request contexts retain `model_id`, streaming mode, and all untouched
  fields across nested wrappers;
- file reminders remain pending when the model handler fails and clear after response;
- fully decorated requests fit the reserved envelope budget;
- compact/handoff never sees Environment/runtime envelope context;
- all applied initial and enqueued user inputs survive repeated compact/handoff and
  export/restore in original order;
- unresolved or rejected ledger entries are never materialized as applied history;
- run-wide retry prompts are counted before optional handoff, compaction, or cold-start
  reduction;
- system prompt reinjection after restore/lifecycle reset;
- stable instruction prefix across requests;
- native `AgentSpec` structured output composes with approval and external deferred-call
  alternatives without override;
- every SDK/YAACLI/Claw native template validates and renders through the same bounded
  `AgentTemplateContext` projection with the native `spec` extra installed;
- skill refresh I/O only after catalog invalidation;
- media projection order and absence from canonical history;
- provider-valid tool return/retry ordering; and
- stable tool ID normalization across response, event, persistence, and restore.

## 12. SDK Logical-Run Tests

- enqueue during model streaming and foreground tools;
- enqueue after output validation but before final completion;
- enqueue immediately before a recoverable transport failure;
- unresolved input rebind to the next attempt;
- input accepted between attempts;
- applied input present in resume history and not rebound;
- failed/cancelled attempt with `result is None` cannot accept native input directly;
- FIFO within `asap` and within `when_idle`;
- `asap` before `when_idle` at an idle boundary;
- deferred-tool output with simultaneous `asap` and `when_idle` input reaches the host,
  then applies pending envelopes in the continuation segment;
- background shell/subagent completion accepted during a deferred gap applies once in
  the continuation segment;
- bound, recovery-gap, and suspended ingress all enforce the same count/byte budget;
- multimodal content preserved without text flattening in both canonical history and
  the run input ledger;
- stable logical ID with distinct per-attempt native enqueue IDs;
- main-to-child lookup across concurrent delegated runs;
- ownership-safe child cleanup;
- terminal rejection of unresolved input; and
- no tool/output/overall/transport/stream-recovery budget consumption by enqueue.

## 13. YA Claw Tests

- API default mode and explicit queue mapping;
- queued `RunRecord` merge distinct from active queue mode;
- direct steer to a queued run rejected;
- unified backpressure before bind, while bound, during recovery, and during HITL;
- logical-run router remains available across SDK attempt recovery and deferred
  continuation segments;
- accepted/enqueued/applied/rejected event correlation across attempts;
- input during HITL continuation gap persists mode and logical ID;
- HITL user input and background completions enqueue before continuation graph
  advancement;
- submit racing terminal commit enters the current logical-run router or next run
  exactly once;
- unresolved input rejected on cancellation/failure;
- no polling dependency or late-steering stream segment;
- UI labels and outstanding count;
- immutable child plan recovery after source profile/registry mutation or deletion; and
- process loss does not claim active model execution was resumed.

## 14. Risk Controls

| Risk | Control |
| --- | --- |
| capabilities become a renamed monolith | leaf ownership and `CombinedCapability` presets |
| context becomes a service locator | narrow typed protocols and explicit root services |
| ordering is incorrect | semantic ordering and request-history characterization tests |
| transient context enters canonical history | request-only `wrap_model_request` decorators and export assertions |
| request envelope or retained user input bypasses token pressure | envelope reserve, run-input retention budget, and fully rendered request tests |
| child agents receive main-only tools | explicit child sets and final visibility policy |
| Tool Proxy bypasses execution policy | nested calls use active Pydantic AI `ToolManager` |
| recovery loses accepted input | logical-run router with rebind, `RunInputLedger`, and canonical metadata reconciliation |
| deferred terminal is redirected past HITL | `DeferredTerminalCapability` suspends before native pending drain |
| registry isolation hides child runs | root shared registry; per-run token ownership only |
| HITL gap loses user or feature content | generalized durable continuation store and pre-advancement enqueue |
| active ingress grows without bound | one count/byte budget across every logical-run state |
| terminal input is acknowledged but not driven | logical gate, state recheck, and shared terminal lock |
| active `queue` is confused with durable work | distinct fields, product copy, and persistence boundary |
| workflow history becomes product session truth | independent `SessionStore`, portable revisions, and no engine-table reads |
| session/workflow dual writes diverge | transactional outbox, idempotent commands, and startup reconciliation |
| durable replay loses user steering | durable inbox plus `RunInputLedger` and canonical metadata reconciliation |
| durable step mutation disappears on replay | explicit return/store channels; no `ctx.enqueue()` or implicit state mutation in steps |
| execution starts before exact plans are registered | worker boot constructs every required plan before outbox dispatch |
| serialized deps contain live authorities | reference-only `DurableAgentDeps` plus session-scoped worker leases acquired inside durable adapters |
| effectful function tools replay in workflow code | durable-plan validation rejects unwrapped I/O and requires named step adapters and operation journals |
| persisted input has no active delivery pump | ordered workflow `after_node_run` pump, registered drain step, and atomic close-and-drain terminal fence before native pending processing |
| plan identity misses executable compatibility | full executable identity plus exact immutable descriptor validation |
| segment checkpoints grow without product limits | serialized-payload budgets and artifact references at every stable segment boundary |
| an ambiguous tool effect repeats | recovery classification and HITL reconciliation instead of blind retry |
| subagent abstraction hides host durability differences | shared service semantics with explicit SDK, YAACLI, and Claw drivers |
| child resolution leaks parent behavior | exact native `AgentSpec` grants; context/Environment only populate catalog/dependencies; enumerated host injection and final visibility policy |
| a child hash detects but cannot reconstruct an old plan | descriptor plus compatible deployed code; stored provenance does not replace executable artifacts |
| host output override drops native schema or HITL | one fingerprinted effective output combining `StructuredDict` and `DeferredToolRequests` |
| templates couple to concrete host deps | canonical serializable `AgentTemplateContext` projection validated across SDK, YAACLI, and Claw |
| child completion wakes a parent twice | durable result/delivery records and idempotent outbox correlation |
| Environment diagnostics lose provenance | source-scoped `AgentContributionGroup` |
| restored resource capabilities are resolved too early | Agent construction is deferred until Environment entry and restore |

## 15. Verification Commands

Each block uses focused package tests during implementation. The 2.0 release requires:

```bash
make lint
make check
make test
```

Documentation, examples, `.env.example` files, profiles, canonical skills, and package
exports change in the same blocks as their behavior.
