# 01 - Lifecycle Extensions

`ya-agent-sdk` should expose lifecycle extensions as first-class runtime components.

The goal is to let products such as YA Claw attach cross-cutting runtime behavior around agent execution without rebuilding `stream_agent` or passing many one-off hook functions at every call site.

## Current State

The SDK already has these pieces:

- `create_agent(...)` accepts Pydantic AI capabilities through `pre_capabilities` and `capabilities`.
- SDK history behavior is implemented with `ProcessHistory` capabilities.
- `stream_agent(...)` accepts call-site hooks:
  - `on_runtime_ready`
  - `on_agent_start`
  - `on_agent_complete`
  - `pre_node_hook`
  - `post_node_hook`
  - `pre_event_hook`
  - `post_event_hook`
- compact is implemented as a history processor and emits:
  - `CompactStartEvent`
  - `CompactCompleteEvent`
  - `CompactFailedEvent`
- the cache-friendly compact filter reuses the current agent and keeps cache-sensitive model settings intact.
- the legacy compact filter pre-trims history through `_trim_history_for_compact(...)` before running the compact agent.

This provides a strong base for memory, observability, and runtime policy hooks.

## Problem

Runtime products need stable extension points that span the whole agent lifecycle.

Examples:

- inject runtime memory into the prompt after the runtime has entered
- observe compact and summarize operations with trimmed history
- schedule background extract or consolidation jobs after an execution completes
- collect usage and terminal result data through one interface
- register reusable runtime behavior once at agent construction time

Passing hook callables into `stream_agent(...)` works for local scripts. Service runtimes benefit from reusable extension objects that are configured during runtime assembly.

## Design Goals

- Keep the existing `stream_agent(...)` hook API compatible.
- Support extension objects that can be registered through `create_agent(...)` or `AgentRuntime`.
- Preserve Pydantic AI capabilities for model-history processing.
- Make compact and summarize activity observable through typed callback contexts.
- Support trimmed message-history handoff for memory extraction.
- Avoid cache coupling between main execution and memory extraction.
- Keep extension failures isolated by policy.

## Proposed API

### Lifecycle Extension Protocol

```python
from typing import Protocol, Sequence, Generic, TypeVar

class AgentLifecycleExtension(Protocol, Generic[AgentDepsT, OutputT, EnvT]):
    name: str

    async def on_runtime_ready(
        self,
        ctx: RuntimeReadyContext[AgentDepsT, OutputT, EnvT],
    ) -> None: ...

    async def on_agent_start(
        self,
        ctx: AgentStartContext[AgentDepsT, OutputT, EnvT],
    ) -> None: ...

    async def on_before_node(
        self,
        ctx: NodeHookContext[AgentDepsT, OutputT],
    ) -> None: ...

    async def on_after_node(
        self,
        ctx: NodeHookContext[AgentDepsT, OutputT],
    ) -> None: ...

    async def on_before_event(
        self,
        ctx: EventHookContext[AgentDepsT, OutputT],
    ) -> None: ...

    async def on_after_event(
        self,
        ctx: EventHookContext[AgentDepsT, OutputT],
    ) -> None: ...

    async def on_agent_complete(
        self,
        ctx: AgentCompleteContext[AgentDepsT, OutputT, EnvT],
    ) -> None: ...

    async def on_agent_error(
        self,
        ctx: AgentErrorContext[AgentDepsT, OutputT, EnvT],
    ) -> None: ...
```

Each method is optional in concrete classes. The SDK should ship a base class with no-op implementations:

```python
class BaseLifecycleExtension(Generic[AgentDepsT, OutputT, EnvT]):
    name = "base"

    async def on_runtime_ready(self, ctx): pass
    async def on_agent_start(self, ctx): pass
    async def on_before_node(self, ctx): pass
    async def on_after_node(self, ctx): pass
    async def on_before_event(self, ctx): pass
    async def on_after_event(self, ctx): pass
    async def on_agent_complete(self, ctx): pass
    async def on_agent_error(self, ctx): pass
```

### Registration

Add an optional parameter to `create_agent(...)`:

```python
def create_agent(
    ...,
    lifecycle_extensions: Sequence[AgentLifecycleExtension[AgentDepsT, OutputT, EnvT]] | None = None,
) -> AgentRuntime[AgentDepsT, OutputT, EnvT]: ...
```

Store extensions on `AgentRuntime`:

```python
@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT, EnvT]):
    env: EnvT
    ctx: AgentDepsT
    agent: Agent[AgentDepsT, OutputT]
    core_toolset: Toolset[AgentDepsT] | None = None
    lifecycle_extensions: list[AgentLifecycleExtension[AgentDepsT, OutputT, EnvT]] = field(default_factory=list)
```

`stream_agent(...)` should merge runtime extensions with call-site hooks:

```python
extensions = list(runtime.lifecycle_extensions)

await run_extensions("on_runtime_ready", RuntimeReadyContext(...))
if on_runtime_ready is not None:
    await on_runtime_ready(context)
```

Ordering rule:

1. runtime extensions in registration order
2. call-site hook

This gives runtime products deterministic baseline behavior and lets local callers add one-off logic.

### Error Policy

Each extension can declare its failure policy:

```python
class ExtensionFailurePolicy(StrEnum):
    RAISE = "raise"
    LOG_AND_CONTINUE = "log_and_continue"

class BaseLifecycleExtension:
    failure_policy = ExtensionFailurePolicy.RAISE
```

Default `RAISE` keeps behavior explicit during development. Runtime products can set `LOG_AND_CONTINUE` for optional telemetry extensions.

## Compact and Summary Hooks

Compact currently emits typed events through `AgentContext.emit_event(...)`. Runtime extensions can observe those events through `on_after_event(...)`, but compact-specific hooks provide a cleaner contract for systems that need trimmed histories or compact output.

### Compact Hook Contexts

Add typed compact callback contexts:

```python
@dataclass
class CompactStartContext(Generic[AgentDepsT]):
    event_id: str
    deps: AgentDepsT
    original_messages: list[ModelMessage]
    trigger: CompactTrigger
    mode: CompactMode

@dataclass
class CompactCompleteContext(Generic[AgentDepsT]):
    event_id: str
    deps: AgentDepsT
    original_messages: list[ModelMessage]
    trimmed_messages: list[ModelMessage]
    compacted_messages: list[ModelMessage]
    summary_markdown: str
    condense_result: CondenseResult | None
    usage: Usage | None
    trigger: CompactTrigger
    mode: CompactMode

@dataclass
class CompactFailedContext(Generic[AgentDepsT]):
    event_id: str
    deps: AgentDepsT
    original_messages: list[ModelMessage]
    trimmed_messages: list[ModelMessage] | None
    error: BaseException
    trigger: CompactTrigger
    mode: CompactMode
```

### Compact Trigger and Mode

```python
class CompactTrigger(StrEnum):
    TOKEN_THRESHOLD = "token_threshold"
    MANUAL_COMPACT = "manual_compact"
    MANUAL_SUMMARIZE = "manual_summarize"

class CompactMode(StrEnum):
    CACHE_FRIENDLY = "cache_friendly"
    LEGACY_AGENT = "legacy_agent"
    TRIM_ONLY = "trim_only"
```

`MANUAL_SUMMARIZE` represents an application-level handoff or summarize action that clears context and preserves a summary. SDK call sites can pass the trigger when invoking a manual compact/summarize helper.

### Compact Callback Registration

Compact filters should accept callbacks:

```python
CompactCallback = Callable[[CompactCompleteContext[AgentContext]], Awaitable[None]]

def create_cache_friendly_compact_filter(
    model_cfg: ModelConfig | None = None,
    callbacks: Sequence[CompactLifecycleCallback] | None = None,
) -> HistoryProcessor[AgentContext]: ...

def create_compact_filter(
    ...,
    callbacks: Sequence[CompactLifecycleCallback] | None = None,
) -> HistoryProcessor[AgentContext]: ...
```

`AgentLifecycleExtension` can also expose compact-specific methods:

```python
class AgentLifecycleExtension(...):
    async def on_compact_start(self, ctx: CompactStartContext[AgentDepsT]) -> None: ...
    async def on_compact_complete(self, ctx: CompactCompleteContext[AgentDepsT]) -> None: ...
    async def on_compact_failed(self, ctx: CompactFailedContext[AgentDepsT]) -> None: ...
```

The compact filter can load callbacks from `agent_ctx.lifecycle_extensions` when the context provides them, then run them alongside filter-local callbacks.

## Trim Mode for Memory Handoff

Memory extraction wants a compact-safe view of the conversation. It does not need provider cache consistency or a replayable history that will be sent back to the main model.

The SDK should expose the compact pre-trim operation as a public utility:

```python
@dataclass
class TrimHistoryOptions:
    preserve_last_turn: bool = False
    injected_context_tags: tuple[str, ...] = (
        RUNTIME_CONTEXT_TAG,
        ENVIRONMENT_CONTEXT_TAG,
    )
    max_tool_return_chars: int = 500
    strip_media: bool = True
    strip_injected_context: bool = True
    preserve_keep_tagged_messages: bool = True

@dataclass
class TrimHistoryResult:
    messages: list[ModelMessage]
    original_message_count: int
    trimmed_message_count: int
    removed_part_count: int
    truncated_tool_return_count: int
    stripped_media_count: int
    stripped_injected_context_count: int

async def trim_history_for_summary(
    message_history: Sequence[ModelMessage],
    options: TrimHistoryOptions | None = None,
) -> TrimHistoryResult: ...
```

The current private `_trim_history_for_compact(...)` can become the implementation base.

### Memory Handoff on Compact Complete

When compact or manual summarize completes, the SDK should let extensions receive:

- `original_messages`: full message history before compact
- `trimmed_messages`: trim-mode view for secondary summarization or extraction
- `compacted_messages`: replay messages used by the main agent continuation
- `summary_markdown`: continuation summary
- `condense_result`: structured result for legacy compact

Memory extensions should consume `trimmed_messages` and `summary_markdown`.

The main continuation keeps cache-friendly compact behavior. Memory extraction receives trim-mode input because its output goes to a separate memory store.

## Manual Summarize Contract

A runtime may provide a user-facing summarize or handoff action. The SDK should expose a shared helper that mirrors compact output:

```python
async def summarize_history(
    *,
    agent: Agent[AgentDepsT, OutputT],
    deps: AgentDepsT,
    message_history: Sequence[ModelMessage],
    prompt: str | None = None,
    mode: CompactMode = CompactMode.CACHE_FRIENDLY,
    trigger: CompactTrigger = CompactTrigger.MANUAL_SUMMARIZE,
) -> CompactCompleteContext[AgentDepsT]: ...
```

This helper should:

1. build `trimmed_messages` using trim mode
2. produce `summary_markdown`
3. build `compacted_messages` for handoff continuation
4. emit compact lifecycle events
5. call compact lifecycle extension methods

## Forked Background Agents

Memory extractors and consolidators need isolated execution with restricted tools. A lifecycle extension may start an internal agent after compact or after run completion.

Proposed helper:

```python
@dataclass
class ForkAgentOptions(Generic[AgentDepsT]):
    name: str
    model: str | Model | None = None
    model_settings: ModelSettings | None = None
    system_prompt: str | None = None
    message_history: Sequence[ModelMessage] | None = None
    user_prompt: str | Sequence[UserContent] | None = None
    tools: Sequence[type[BaseTool]] | None = None
    toolsets: Sequence[AbstractToolset[Any]] | None = None
    capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None
    inherit_model_wrapper: bool = True
    inherit_usage_sink: bool = True
    isolated_context: bool = True

async def fork_agent(
    runtime: AgentRuntime[AgentDepsT, OutputT, EnvT],
    options: ForkAgentOptions[AgentDepsT],
) -> AgentRunResult[Any]: ...
```

This is useful for SDK consumers beyond YA Claw: background summarizers, evaluators, validators, and auditors.

## Interaction with Pydantic AI Capabilities

Lifecycle extensions complement Pydantic AI capabilities.

Recommended split:

- Pydantic AI capabilities own model-history processing, tool availability, and model settings.
- SDK lifecycle extensions own runtime orchestration, event observation, secondary agent jobs, and service integrations.

`create_agent(...)` should keep `pre_capabilities` and `capabilities` as the low-level model request composition mechanism.

## Implementation Plan

1. Add `lifecycle_extensions` to `AgentRuntime` and `create_agent(...)`.
2. Add extension runner utilities with failure policy handling.
3. Route existing `stream_agent(...)` hook sites through the extension runner.
4. Promote trim-mode history processing to a public helper.
5. Add compact callback contexts and compact-specific extension methods.
6. Feed compact callbacks from both cache-friendly and legacy compact filters.
7. Add `summarize_history(...)` helper for manual handoff flows.
8. Add `fork_agent(...)` after the lifecycle API stabilizes.

## Compatibility

Existing callers can keep using the current hook parameters. Existing compact behavior remains the default. The new API mainly moves service-level integration from ad hoc call-site hooks into reusable extension objects.
