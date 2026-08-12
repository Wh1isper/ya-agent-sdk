# Toolset Architecture

The toolset system for managing tools with hooks and HITL support.

## Overview

- **BaseTool**: Abstract base class for individual tools
- **Toolset**: Container that manages tools with hooks
- **Hook System**: Pre/post hooks for intercepting tool execution
- **Error Handling**: Post-hooks can intercept and handle exceptions

```mermaid
flowchart LR
    subgraph Toolset
        GlobalPre[Global Pre-Hook]
        ToolPre[Tool Pre-Hook]
        Execute[Tool Execution]
        ToolPost[Tool Post-Hook]
        GlobalPost[Global Post-Hook]
    end

    Args[Tool Args] --> GlobalPre --> ToolPre --> Execute
    Execute --> ToolPost --> GlobalPost --> Result[Result]
    Execute -- "ApprovalRequired / CallDeferred" --> ToolPost
    ToolPost -- observe only --> GlobalPost
    GlobalPost -- re-raise --> ControlFlow[Control Flow Exception]
```

## Creating Tools

Inherit from `BaseTool` and implement the `call` method:

```python
from ya_agent_sdk.toolsets.core.base import BaseTool
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "A custom tool example"

    async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
        return await ctx.deps.file_operator.read_file(path)

    # Optional class attributes:
    # tags: frozenset[str] = frozenset()           # Capability tags
    # superseded_by_tags: frozenset[str] = frozenset()  # Auto-hide when tag active
    # auto_inherit: bool = False                    # Include in subagent toolsets
    # main_agent_only: bool = False                 # Exclude from every SDK subagent toolset
    # is_context_manage_tool: bool = False          # Context management (e.g., summarize)

    # Optional overrides:
    # def is_available(self, ctx) -> bool: ...
    # def get_instruction(self, ctx) -> str | Instruction | None: ...
    # def get_approval_metadata(self) -> dict | None: ...
```

> Full interface: `ya_agent_sdk/toolsets/base.py`

## Tool Instructions

Tools can provide instructions to inject into the system prompt via `get_instruction()`.

### Basic Usage

Return a plain string (uses tool name as group):

```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "..."

    def get_instruction(self, ctx):
        return "Guidelines for using my_tool..."
```

### Grouped Instructions (Deduplication)

When multiple related tools share the same instruction, use `Instruction` with a `group`:

```python
from ya_agent_sdk.toolsets import Instruction

class TaskCreateTool(BaseTool):
    name = "task_create"

    def get_instruction(self, ctx):
        return Instruction(
            group="task-manager",  # Same group = deduplicated
            content="Task manager guidelines..."
        )

class TaskListTool(BaseTool):
    name = "task_list"

    def get_instruction(self, ctx):
        return Instruction(
            group="task-manager",  # Same group, only first one kept
            content="Task manager guidelines..."
        )
```

**Deduplication behavior**: When `Toolset.get_instructions()` collects instructions, tools with the same `group` only contribute once (first wins). This reduces prompt bloat for related tool families.

| Return Type   | Group ID  | Deduplicated |
| ------------- | --------- | ------------ |
| `str`         | tool name | No           |
| `Instruction` | `.group`  | Yes          |

## Using Toolset

Typically via `create_agent`:

```python
from ya_agent_sdk.agents import create_agent, stream_agent

runtime = create_agent(
    "openai-chat:gpt-4",
    tools=[ViewTool, EditTool, GrepTool],
    pre_hooks={"view": my_pre_hook},
    post_hooks={"edit": my_post_hook},
    global_hooks=GlobalHooks(pre=global_pre, post=global_post),
)
```

> For manual toolset creation, see `ya_agent_sdk/toolsets/core/base.py`

## Hook System

### Execution Order

```
global_pre -> tool_pre -> execute -> tool_post -> global_post
```

### Hook Signatures

All hooks receive a shared `metadata` dict that persists throughout a single `call_tool` invocation.

| Hook Type            | Signature                                      |
| -------------------- | ---------------------------------------------- |
| `PreHookFunc`        | `(ctx, args, metadata) -> args`                |
| `PostHookFunc`       | `(ctx, result, metadata) -> result`            |
| `GlobalPreHookFunc`  | `(ctx, tool_name, args, metadata) -> args`     |
| `GlobalPostHookFunc` | `(ctx, tool_name, result, metadata) -> result` |

### Error Handling in Post-Hooks

Post-hooks receive the result, which **may be an Exception instance** if tool execution failed:

- Check `isinstance(result, Exception)` to handle errors
- Return a fallback value for recovery, or pass through to re-raise

### Control Flow Exceptions

Pydantic AI uses `ModelRetry`, `ApprovalRequired`, and `CallDeferred` as control flow exceptions. These are handled specially:

- **Post-hooks are called for observation only** -- their return values are discarded
- The original exception is **always re-raised** to preserve pydantic-ai control flow and retry accounting
- This ensures tracing/cleanup hooks (e.g., closing spans) still run

This means post-hooks that receive a `ModelRetry`, `ApprovalRequired`, or `CallDeferred` instance should treat it as a notification, not an opportunity to transform the result. If an observation-only post-hook raises an ordinary exception, the SDK logs that hook failure and still propagates the original control-flow exception; cancellation and other `BaseException` signals are not suppressed.

> Examples: `ya_agent_sdk/toolsets/core/base.py` docstrings

## Extending Toolset

Override `_call_tool_func` for custom execution behavior (timeout, retry, error wrapping):

```python
class TimeoutToolset(Toolset):
    async def _call_tool_func(self, args, ctx, tool) -> Any:
        return await asyncio.wait_for(tool.call_func(args, ctx), timeout=30.0)
```

**Important**: `ModelRetry`, `ApprovalRequired`, and `CallDeferred` must NOT be converted to ordinary values in `_call_tool_func` overrides. The base implementation already re-raises them. If you override this method, ensure these exceptions propagate:

```python
class CustomToolset(Toolset):
    async def _call_tool_func(self, args, ctx, tool) -> Any:
        try:
            return await tool.call_func(args, ctx)
        except (ModelRetry, ApprovalRequired, CallDeferred):
            raise  # Must propagate
        except Exception as e:
            return e
```

> Full examples: `ya_agent_sdk/toolsets/core/base.py`

## Retry Budgets

Configure all SDK model-correction categories from `create_agent()`:

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.context import RetryConfig

runtime = create_agent(
    "openai-chat:gpt-4o",
    retry_config=RetryConfig(
        tools=5,
        output=5,
        toolset=5,
        tool_search=5,
        tool_proxy=5,
    ),
)
```

All five categories default to 5. Pydantic AI tool retries are tracked per tool name and reset after that tool succeeds; output retries are cumulative within one run. SDK Toolset, Tool Search, and Tool Proxy wrappers resolve their category from `AgentContext.retry_config` at run time. A wrapper-local `max_retries` value wins, and existing `retries`, `output_retries`, and `toolset_max_retries` arguments remain higher-priority compatibility overrides. Regular subagents and self forks inherit the resolved Pydantic AI tool/output policy, while child contexts inherit the SDK wrapper configuration.

The SDK additionally defaults `overall_retries=3`. This separate run-wide correction ceiling counts requests containing tool, output, or capability retry prompts and never resets after successful calls, so it may terminate the run before a category reaches 5. Set it to `None` only when another explicit boundary owns that risk. Transport request retries and stream recovery do not consume this budget.

## Human-in-the-Loop (HITL) Approval

Configure tools requiring user approval:

```python
ctx.need_user_approve_tools = ["shell", "edit", "write"]
```

When called, these tools raise `ApprovalRequired`. Implement `get_approval_metadata()` in your tool to provide context.

The optional `ask_user_question` tool uses `CallDeferred` instead of approval control flow. It is not part of the SDK default tool surface because the host must collect structured answers and resume the run. It sets `main_agent_only=True`, so SDK subagent builders exclude it at construction time and sanitize capability wrappers plus dynamic Toolset factory results at the final subagent execution boundary. SDK `Toolset` listing and calling enforce the policy in subagent contexts regardless of `skip_unavailable`, opaque search/proxy composites, or stale caches, while the tool's availability check provides another runtime guard. See [user-input.md](user-input.md) for registration and continuation details.

> HITL flow details: `ya_agent_sdk/toolsets/core/base.py`

## Architecture

| Component             | Purpose                                 |
| --------------------- | --------------------------------------- |
| `BaseTool`            | Abstract base for individual tools      |
| `Toolset`             | Manages tools with hooks                |
| `HookableToolsetTool` | Internal wrapper with hook support      |
| `GlobalHooks`         | Container for global pre/post hooks     |
| `_call_tool_func`     | Overridable method for custom execution |

## See Also

- [context.md](context.md) - AgentContext configuration
- [user-input.md](user-input.md) - structured deferred clarifying questions
- [subagent.md](subagent.md) - Subagent system
- [environment.md](environment.md) - Environment and resource management
