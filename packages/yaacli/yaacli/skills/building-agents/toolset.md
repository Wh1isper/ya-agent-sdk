# Capability-Owned Tools and Execution Policy

Pydantic AI capabilities are the public composition boundary. A feature should own its
tools, instructions, request/history hooks, and run-local state together. SDK `Toolset`
and `BaseTool` remain implementation adapters for the built-in tool library; they are
not a second `create_agent()` composition plane.

## Public Composition

### SDK feature capabilities

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.capabilities import (
    FilesystemCapability,
    RuntimeFoundationCapability,
    ShellCapability,
    WebContentCapability,
    WebSearchCapability,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[
        RuntimeFoundationCapability(),
        FilesystemCapability(),
        ShellCapability(),
        WebSearchCapability(),
        WebContentCapability(),
    ],
)
```

Each feature capability contributes one coherent tool family and its guidance. The
runtime preserves source provenance from explicit capabilities, context contributions,
Environment contributions, and resource contributions before Pydantic AI validates
ordering and singleton constraints.

### Raw functions

Wrap ordinary Pydantic AI tools in native `Capability`:

```python
from pydantic_ai.capabilities import Capability
from ya_agent_sdk.agents import create_agent


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

runtime = create_agent(
    "openai-chat:gpt-4o",
    capabilities=[Capability(tools=[add], id="math")],
)
```

### Existing Pydantic AI toolsets

Wrap an external `AbstractToolset` with the native `Toolset` capability:

```python
from pydantic_ai.capabilities import Toolset as ToolsetCapability

runtime = create_agent(
    "openai-chat:gpt-4o",
    capabilities=[ToolsetCapability(external_toolset, id="external")],
)
```

Do not pass `tools=` or `toolsets=` to SDK `create_agent()`.

## SDK `BaseTool` Adapter

Use `BaseTool` only when extending the SDK's existing typed tool infrastructure:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import Toolset as ToolsetCapability
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset


class ReadProjectFile(BaseTool):
    name = "read_project_file"
    description = "Read one project-relative text file."

    async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
        return await ctx.deps.file_operator.read_file(path)


adapter = Toolset(tools=[ReadProjectFile], toolset_id="project_files")
runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[ToolsetCapability(adapter, id="project_files")],
)
```

Useful `BaseTool` metadata includes:

| Attribute                | Purpose                                               |
| ------------------------ | ----------------------------------------------------- |
| `tags`                   | Generic capability/search metadata                    |
| `superseded_by_tags`     | Hide a tool when another tag is active                |
| `main_agent_only`        | Mark host-interaction tools for final child exclusion |
| `is_context_manage_tool` | Identify context compaction/handoff tools             |
| `codeact`                | Opt into CodeAct nested dispatch                      |

There is no `auto_inherit`. Child behavior is declared through its own capability list.

## Tool Instructions

An SDK `BaseTool` may return a string or grouped `Instruction` from
`get_instruction()`. The adapter deduplicates the first instruction for each group:

```python
from ya_agent_sdk.toolsets import Instruction


class TaskCreateTool(BaseTool):
    name = "task_create"

    def get_instruction(self, ctx):
        return Instruction(
            group="task-manager",
            content="Use task tracking only for meaningful multi-step work.",
        )
```

For new public features, prefer a capability's `get_instructions()` so the tool and
instruction have the same owner.

## Native Execution-Policy Capabilities

The SDK provides independent policy leaves:

| Capability                  | Contract                                               |
| --------------------------- | ------------------------------------------------------ |
| `ToolVisibilityCapability`  | Final allow/deny and main-agent-only enforcement       |
| `ToolApprovalCapability`    | Marks configured native tool definitions as unapproved |
| `ToolObservationCapability` | Observes one logical call around retries/timeouts      |
| `ToolRetryCapability`       | Retries configured transient execution failures        |
| `ToolTimeoutCapability`     | Bounds one validated execution attempt                 |

Compose them explicitly and in semantic order:

```python
from ya_agent_sdk.capabilities import (
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolRetryCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
)

policies = [
    ToolVisibilityCapability(deny=frozenset({"dangerous_tool"})),
    ToolApprovalCapability(tools=frozenset({"shell_exec", "write"})),
    ToolObservationCapability(),
    ToolRetryCapability(max_attempts=2),
    ToolTimeoutCapability(timeout=60),
]
```

Pydantic AI resolves the ordering declarations. Duplicate singleton capabilities fail
runtime entry instead of silently selecting one.

### Control-flow exceptions

`ApprovalRequired`, `CallDeferred`, and `ModelRetry` are Pydantic AI control flow. Tool
wrappers must re-raise them unchanged. Do not transform them into error strings or
retry them as ordinary transient exceptions.

## Retry Domains

Keep the failure domains separate:

- `create_agent(retries=...)` configures native Pydantic AI tool/output correction
  limits;
- `OverallRetryBudget(max_retries=...)` is the explicit cumulative run-wide
  correction ceiling and is included by `RuntimeFoundationCapability` with a default
  of three;
- `ToolRetryCapability(max_attempts=...)` retries transient host execution failures;
- `stream_agent()` recovery settings cover interrupted graph/model transport attempts.

Transport recovery and host execution retries do not consume model-correction budgets.

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.capabilities import OverallRetryBudget

runtime = create_agent(
    "openai-chat:gpt-4o",
    retries={"tools": 5, "output": 5},
    capabilities=[OverallRetryBudget(max_retries=3)],
)
```

## Deferred Approval

`ToolApprovalCapability` makes matching tool definitions produce native
`DeferredToolRequests.approvals`. A compatible host must:

1. include `DeferredToolRequests` in the output type;
2. display every exact request;
3. collect an explicit approve/deny result;
4. build matching `DeferredToolResults`; and
5. resume with the pending message history.

Use `ya_agent_sdk.interactions.DeferredInteractionResolver`. Do not inspect
`runtime.core_toolset`; that 1.x surface does not exist.

See [`user-input.md`](user-input.md) for external deferred calls such as
`ask_user_question`.

## Child Boundary

A child receives only capabilities declared by its normalized native `AgentSpec` plus
explicitly fingerprinted host-policy capabilities. The resolver injects final
`ToolVisibilityCapability` if missing. This enforcement runs at tool preparation and
execution, so stale caches or opaque wrappers cannot authorize a main-agent-only tool.

Do not filter a parent's final tools, slice a `CombinedCapability`, or infer child
availability from tool names. See [`subagent.md`](subagent.md).

## Internal Adapter Notes

The SDK `Toolset` adapter still supports pre/post/global hooks for existing `BaseTool`
implementations. Post hooks may observe ordinary exceptions and return a fallback. For
Pydantic AI control-flow exceptions their return values are ignored and the original
exception is always re-raised. New cross-cutting behavior should normally be a native
capability hook rather than another adapter hook layer.

## See Also

- [`subagent.md`](subagent.md) - exact child capability grants
- [`user-input.md`](user-input.md) - deferred host continuation
- [`tool-search.md`](tool-search.md) - dynamic tool discovery
- [`tool-proxy.md`](tool-proxy.md) - fixed proxy entry points
- [`codeact.md`](codeact.md) - restricted nested orchestration
- [`environment.md`](environment.md) - filesystem, shell, and resource authority
