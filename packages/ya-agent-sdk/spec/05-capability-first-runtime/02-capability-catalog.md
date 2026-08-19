# Capability Catalog

## 1. Decomposition Rules

A leaf capability has one behavior owner and one authority boundary.

- Tools and their workflow guidance live in the same capability.
- A capability depends on the smallest context protocol needed by its tools.
- Read-only/writable or enabled/disabled access modes are configuration, not separate
  capabilities, unless they introduce a different authority or lifecycle.
- Cross-cutting authorization, visibility, timeout, and observation policy wraps the
  assembled toolset instead of being copied into every feature.
- Durable state is injected through a store protocol.
- Mutable derived state is isolated by `for_run()`.
- Presets combine leaves and add no hidden implementation.
- Alternative tools are selected explicitly by presets or visibility policy. Global
  tag-based arbitration is not used.
- Native Pydantic AI capabilities replace SDK implementations when their contracts
  match.
- Context/Environment contributions are automatic only for root assembly. For named
  children they populate the catalog/dependencies; only native `AgentSpec` grants plus
  enumerated host infrastructure/policy capabilities enter the resolved plan.

## 2. Filesystem and Execution

| Capability | Tools and behavior | Dependencies |
| --- | --- | --- |
| `FilesystemCapability` | inspect and mutate files; filesystem workflow guidance; read-only or writable resolution | filesystem dependency protocol |
| `ShellCapability` | execute, wait, status, input, signal, kill; shell guidance; canonical background-completion enqueue; request-only process status | shell protocol; optional filesystem for overflow output; logical-run input router |
| `DocumentConversionCapability` | PDF and Office conversion | filesystem; optional conversion libraries |
| `MediaReadCapability` | `read_media`, URL media loading, and media-reader policy | filesystem and media loader |

Filesystem reads and writes stay together because they share one authority, path-safety
policy, and instruction owner. `FilesystemCapability` resolves a read-only or writable
tool surface and matching guidance as one unit. General host allow/deny restrictions
remain in `ToolVisibilityCapability`. Shell stays separate because process lifecycle
and asynchronous completion delivery form a different authority boundary.

`ReadImageTool`, `ReadAudioTool`, and `ReadVideoTool` are removed. `MediaReadCapability`
owns the canonical media surface.

## 3. Web and External Integrations

| Capability | Tools and behavior | Dependencies |
| --- | --- | --- |
| `WebSearchCapability` | general search and optional stock/image search | typed search provider configuration |
| `WebContentCapability` | fetch, scrape, download | HTTP/content clients; filesystem for download |
| `SkillsCapability` | skill catalog, routing guidance, load tools, and cached catalog generation | filesystem/catalog sources; resumable activated-skill state where required |
| `ToolProxyCapability` | proxy protocol, proxy tools, and private proxy adapter | active Pydantic AI `ToolManager` |
| Pydantic AI `MCP` | native or local MCP without YA-specific wrapping | native Pydantic AI MCP contract |
| Pydantic AI `ToolSearch` | deferred capability/tool discovery | capabilities with `defer_loading=True` |

### 3.1 Tool Search

SDK `ToolSearchToolSet` is not a public alternative to native Pydantic AI Tool Search.
A feature intended for discovery marks its capability `defer_loading=True`, coupling
its tools, description, model settings, and instructions under one identity.

Pydantic AI owns deferred capability discovery and loading. `ToolSearchToolSet` and its
`AgentContext.tool_search_loaded_*` state are removed; there is no fallback Tool Search
implementation. Tool Proxy is a separate fixed-surface protocol and persists its typed
state under `AgentContext.tool_proxy`.

### 3.2 Tool Proxy

The private proxy adapter cannot call an underlying toolset directly if doing so
bypasses:

- Pydantic AI validation;
- prepared/filtered tools;
- approval and deferred calls;
- retry accounting;
- capability hooks; or
- the active tool execution policy.

Nested proxy calls execute through the active Pydantic AI `ToolManager`, following the
same execution-boundary principle as CodeAct. A deliberately reduced remote proxy is a
separate capability with the reduced contract stated explicitly.

### 3.3 Skills

`SkillsCapability` owns both the catalog snapshot and routing guidance. Tools and
instructions read the same snapshot. Hot reload invalidates a catalog generation;
filesystem or network scans do not run independently from multiple prompt paths.

## 4. Agent Behavior and Collaboration

| Capability | Tools and behavior | State boundary |
| --- | --- | --- |
| `TaskCapability` | create/get/list/update task tools and task workflow guidance | context-owned `TaskManager` |
| `NoteCapability` | note read/write/delete tools and note guidance | context-owned `NoteManager` |
| `ThinkingCapability` | optional explicit thinking tool | no durable state |
| `TodoCapability` | optional todo read/write workflow distinct from dependency-aware tasks | context-owned todo store when enabled |
| `UserInteractionCapability` | `ask_user_question`, deferred interaction guidance, main-agent-only enforcement | host resolves Pydantic AI deferred calls |
| `HandoffCapability` | handoff tool, history transition, applied run-input restoration, file-inspection transfer, guidance | resumable handoff state and `RunInputLedger` |
| `DelegationCapability` | model-facing spawn/resume/steer/cancel/wait/list operations and registry-backed roster guidance | public `SubagentRegistry` and `SubagentExecutionService`; shared `ActiveRunRegistry`; host driver/store |
| `CodeActCapability` | `run_code`, `run_program`, wrapper toolset, CodeAct instructions | Pydantic AI `ToolManager`; run-local Monty state |

Handoff's callable tool and history transition cannot be configured independently.
Task and note capabilities own tools and instructions but not their durable data.
Structured user interaction is absent from child capability sets.

## 5. Runtime and History Policy

| Capability | Behavior |
| --- | --- |
| `ReasoningCompatibilityCapability` | normalize reasoning across model/provider changes |
| `MediaCompatibilityCapability` | request-envelope projection that splits, resizes/compresses, limits, uploads, and filters retained media without persisting provider-specific projection |
| `ToolArgumentRepairCapability` | repair recoverable truncated tool arguments before lifecycle reduction |
| `ToolIdCompatibilityCapability` | normalize tool call/result IDs without mutating canonical input objects |
| `ContextCompactionCapability` | cache-friendly compaction, typed events, and ordered restoration of applied `RunInputLedger` entries |
| `ColdStartCapability` | cold-start trim |
| `FileInspectionCapability` | one-shot request-envelope continuation reminder; replaces the misleading auto-load name |
| `EnvironmentContextCapability` | request-only `Environment.get_context_instructions()` envelope decoration |
| `RuntimeContextCapability` | request-only elapsed time, usage, active-agent state, reminders, and context pressure |
| `DeferredTerminalCapability` | preserve `DeferredToolRequests` as a host-visible suspension boundary before native pending-message redirect |
| YAACLI `DurableInboxPumpCapability` | drain the host SQLite inbox, enqueue only from bound `after_node_run`, and fence ordinary terminal closure |
| Pydantic AI `ReinjectSystemPrompt` | restore the configured system prompt after reconstruction/reset |
| `OverallRetryBudget` | cumulative model-correction ceiling |

The historical `handle_model_switch` export is removed. The non-composable model
metadata enum is renamed from `ModelFeature` to `ModelFeature` without an alias.

## 6. Tool Execution Policy

| Capability | Behavior |
| --- | --- |
| `ToolApprovalCapability` | project approval policy onto native tool definitions and deferred-call resolution |
| `ToolTimeoutCapability` | bound each execution attempt around the final validated tool call |
| `ToolVisibilityCapability` | filter prepared definitions, then recheck final host allow/deny and main-agent-only policy before execution |
| `ToolObservationCapability` | observe one logical validated call around all non-native execution attempts |
| `ToolRetryCapability` | account for and run non-native execution attempts, separate from model and recovery budgets |

The policy applies through native capability hooks after all built-in and external tool
contributions are assembled. Visibility and approval use preparation/execution-guard
phases; they are not tool-execution wrappers. The only semantic execution nesting is
`ToolObservationCapability` outside `ToolRetryCapability` outside
`ToolTimeoutCapability`. Direct pairwise ordering keeps the same meaning when retry or
observation is omitted. CodeAct eligibility stays trusted tool metadata consumed by
`CodeActCapability`.

## 7. Existing Toolset Mapping

| Existing implementation | Capability boundary | Final internal form |
| --- | --- | --- |
| SDK `BaseTool`, `Toolset`, and `BaseToolset` | owning feature capability or native `Toolset` capability | retained tool-authoring adapters, not `create_agent()` composition inputs |
| `ToolSearchToolSet` | Pydantic AI `ToolSearch` and deferred feature capabilities | old implementation removed |
| `ToolProxyToolset` | `ToolProxyCapability` | manager-backed implementation adapter; compose through the capability |
| `SkillToolset` | `SkillsCapability` | catalog/tool implementation adapter |
| host-managed MCP adapter | Pydantic AI `MCP`, native `Toolset`, or `ToolProxyCapability` | private only when YA approval, result mapping, or CodeAct policy is required |
| `CodeActToolset` | `CodeActCapability` | private wrapper toolset |
| generated subagent tools | `DelegationCapability` | removed; one public-service-backed model tool surface over resolved plans |
| Environment/resource toolsets | provider-returned capability groups | raw provider toolset contribution removed |

`BaseTool`, SDK `Toolset`, and `BaseToolset` remain public tool-authoring adapters for
built-in and custom feature implementations. Their `call()`, approval metadata,
availability, preprocessing, and instruction aggregation are used inside an owning
capability or native Pydantic AI `Toolset` capability. They are not alternate behavior
inputs to `create_agent()`.

## 8. Existing Tool Metadata Mapping

| `BaseTool` metadata | Target |
| --- | --- |
| `tags` | descriptive tool metadata and visibility-policy input only |
| `superseded_by_tags` | removed; explicit preset/visibility selection |
| `auto_inherit` | removed; explicit child capability composition |
| `codeact` | trusted tool metadata consumed by `CodeActCapability` |
| `main_agent_only` | final tool policy capability plus explicit child composition |
| `is_context_manage_tool` | owning context lifecycle capability identity |
| `is_available()` | run resolution shared by capability tools and instructions |
| `get_instruction()` | owning capability `get_instructions()` |

A capability with no usable tools for a resolved run contributes neither stale tool
guidance nor an empty wrapper. `for_run()` or `DynamicCapability` resolves availability
once for tools and instructions together.

## 9. Existing Filter Mapping

| Existing filter/processor | Capability owner | Final behavior |
| --- | --- | --- |
| `normalize_reasoning_for_model` | `ReasoningCompatibilityCapability` | immutable replacement transform |
| `handle_model_switch` | none | removed in 2.0 |
| `split_large_images` | `MediaCompatibilityCapability` | local shaping stage |
| `compress_large_images` | `MediaCompatibilityCapability` | local shaping stage |
| `drop_extra_images` | `MediaCompatibilityCapability` | configured media limit stage |
| `drop_gif_images` | `MediaCompatibilityCapability` | provider compatibility stage |
| `drop_extra_videos` | `MediaCompatibilityCapability` | configured media limit stage |
| `filter_by_capability` | `MediaCompatibilityCapability` | model-feature filtering stage |
| `create_media_upload_filter` | configured media compatibility leaf | upload after local shaping and before unsupported-media filtering |
| `fix_truncated_tool_args` | `ToolArgumentRepairCapability` | canonical repair before lifecycle reduction |
| `ToolIdWrapper.wrap_messages` | `ToolIdCompatibilityCapability` | replacement transform and event normalization |
| `process_handoff_message` | `HandoffCapability` | handoff transition |
| cache-friendly compact filters | `ContextCompactionCapability` | lifecycle reduction |
| `cold_start_trim` | `ColdStartCapability` | lifecycle reduction |
| removed `process_auto_load_files` processor | `FileInspectionCapability` | one-shot `wrap_model_request` reminder, cleared after a successful handler response |
| `create_environment_instructions_filter` | `EnvironmentContextCapability` | request-only `wrap_model_request` decoration |
| `inject_runtime_instructions` | `RuntimeContextCapability` | request-only `wrap_model_request` decoration |
| `inject_background_results` | `ShellCapability` | canonical exactly-once completion enqueue plus transient status |
| `inject_bus_messages` | none | removed in 2.0; canonical producers use logical-run routers |
| `create_system_prompt_filter` | Pydantic AI `ReinjectSystemPrompt` | native system prompt boundary |
| `MessageBusGuardCapability` | none | removed; native pending lifecycle is authoritative, with `DeferredTerminalCapability` preserving HITL suspension |

Private compact/handoff builders move beside their owning capabilities. The
steering-only builder is replaced by shared structured rendering from
`RunInputLedger`; there is no text-only steering replay field.

## 10. YA Agent SDK 2.0 Construction Surface

| Removed surface | 2.0 replacement |
| --- | --- |
| `create_agent(tools=[BaseTool...])` | SDK feature capabilities |
| `create_agent(agent_tools=[...])` | `Capability(tools=[...])` inside `capabilities=` |
| `create_agent(toolsets=[...])` | `pydantic_ai.capabilities.Toolset(...)` inside `capabilities=` |
| `pre_capabilities=` | one ordered `capabilities=` list |
| pre/post/global hook dictionaries | Pydantic AI `Hooks` or tool policy capabilities |
| approval arguments | `ToolApprovalCapability` |
| timeout and availability arguments | `ToolTimeoutCapability` and `ToolVisibilityCapability` |
| compact keyword group | `ContextCompactionCapability` |
| `codeact=` | `CodeActCapability` |
| `SubagentConfig`, duplicate agent fields, tool lists, and implicit inheritance | native `AgentSpec` core, thin YA `SubagentSpec` envelope, resolved plan, and final visibility policy |
| `AgentContext.get_history_capabilities()` | `AgentContext.get_capabilities()` |
| Environment/resource `get_toolsets()` | source-scoped `AgentContributionGroup.capabilities` |
| Claw `builtin_toolsets` and duplicate agent fields | native profile `AgentSpec` with custom `capabilities` entries |

Only the replacement surfaces are accepted. Duplicate singleton capability IDs fail
during construction.

## 11. External Capability Type Availability

[Custom Capability Type Discovery](../06-capability-plugins/README.md) defines how one
installed entry point or explicit Python import contributes one serializable class to
an SDK-owned immutable `CapabilityCatalog`. The catalog contains classes and lightweight
provenance only; it does not add instances to an agent.

Standalone SDK, YAACLI, YA Claw, main-agent, and portable-subagent resolution consume
the same SDK catalog contract. Native `AgentSpec.capabilities` or programmatic
`capabilities=` remains the exact grant/composition surface. Environment and context
contributions remain a separate root-instance and typed-dependency source and do not
mutate the static custom-type catalog.
