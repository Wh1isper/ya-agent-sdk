---
name: agent-builder
description: Build and configure AI agents with ya-agent-sdk and Pydantic AI. Covers capability-first create_agent(), stream_agent(), AgentSpec, AgentContext, ResumableState, portable subagents, environments, native steering, and deferred HITL. Use when implementing agent applications, composing capabilities, restoring sessions, configuring child-agent plans, adding approval flows, or working with ya-agent-sdk runtime APIs.
---

# Building Agents with ya-agent-sdk

Build agents with the 2.0 capability-first runtime. Pydantic AI capabilities are the
only public behavior-composition surface. Do not pass SDK `tools=` or `toolsets=` to
`create_agent()`, slice toolsets for children, or use removed MessageBus/generated
subagent APIs.

## Start Here

- Construct an unentered runtime with `create_agent()`.
- Put every agent behavior in the ordered `capabilities=` list.
- Enter `AgentRuntime` before accessing `runtime.agent` or resolved capabilities.
- Use `stream_agent()` for SDK lifecycle events, native steering, and recovery.
- Persist both Pydantic AI message history and `runtime.ctx.export_state()`.
- Use native `AgentSpec` for declarative agent fields.
- Use `SubagentSpec`, `SubagentPlanResolver`, `SubagentRegistry`, and
  `SubagentExecutionService` for delegation.
- Use `DeferredInteractionResolver` for approvals and external deferred calls.

Read the focused references before changing the corresponding subsystem:

- Sessions: [`./context.md`](./context.md)
- Streaming and events: [`./streaming.md`](./streaming.md), [`./events.md`](./events.md)
- Capability-owned tools and policies: [`./toolset.md`](./toolset.md)
- Structured deferred input: [`./user-input.md`](./user-input.md)
- Portable subagents: [`./subagent.md`](./subagent.md)
- Environment authority: [`./environment.md`](./environment.md),
  [`./resumable-resources.md`](./resumable-resources.md)
- Tool Search, proxying, and CodeAct: [`./tool-search.md`](./tool-search.md),
  [`./tool-proxy.md`](./tool-proxy.md), [`./codeact.md`](./codeact.md)

## Installation

```bash
pip install 'ya-agent-sdk[all]'
uv add 'ya-agent-sdk[all]'
```

Use selective extras such as `docker`, `web`, `document`, `s3`, `tool-proxy`,
`oauth`, or `rs` when a smaller installation is needed.

## Core Workflows

### Create and enter a runtime

```python
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import RuntimeFoundationCapability

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[RuntimeFoundationCapability()],
)

async with runtime:
    result = await runtime.agent.run("Summarize this project", deps=runtime.ctx)
    print(result.output)
```

`create_agent()` returns an unentered `AgentRuntime`. Runtime entry first enters the
Environment and context, collects their contribution groups, validates capability
ordering and singleton constraints, and then constructs the Pydantic AI `Agent`.

### Compose SDK features

```python
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    FilesystemCapability,
    RuntimeFoundationCapability,
    ShellCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolRetryCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[
        RuntimeFoundationCapability(),
        FilesystemCapability(),
        ShellCapability(),
        ToolVisibilityCapability(),
        ToolApprovalCapability(tools=frozenset({"shell_exec"})),
        ToolObservationCapability(),
        ToolRetryCapability(),
        ToolTimeoutCapability(),
    ],
)
```

Capabilities own tools, instructions, request/history hooks, and run-local state as one
coherent feature. `RuntimeFoundationCapability` is explicit; `create_agent()` does not
inject it.

### Stream responses

```python
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.capabilities import RuntimeFoundationCapability

runtime = create_agent(
    "openai-chat:gpt-4o",
    capabilities=[RuntimeFoundationCapability()],
)

async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
    streamer.raise_if_exception()
```

Use the SDK stream driver instead of manually advancing Pydantic AI graph nodes when
you need SDK lifecycle events, logical-run input routing, usage snapshots, or recovery.

### Persist and restore sessions

```python
from ya_agent_sdk.agents.main import create_agent

async with create_agent("openai-chat:gpt-4o") as runtime:
    result = await runtime.agent.run("Remember this", deps=runtime.ctx)
    messages = result.all_messages()
    state = runtime.ctx.export_state()

restored = create_agent("openai-chat:gpt-4o", state=state)
# Pass `messages` as message_history on the next run.
```

`ResumableState` stores SDK context state, not canonical Pydantic AI message history.
Hosts persist both.

### Add deferred host interaction

```python
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    RuntimeFoundationCapability,
    ToolApprovalCapability,
    UserInteractionCapability,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[
        RuntimeFoundationCapability(),
        UserInteractionCapability(),
        ToolApprovalCapability(tools=frozenset({"shell_exec"})),
    ],
    output_type=[str, DeferredToolRequests],
)
```

The host must present every deferred request and resume with matching
`DeferredToolResults`. Use the typed `DeferredInteractionResolver`; do not inspect a
runtime-private toolset.

### Add portable subagents

Define each child with native `AgentSpec` inside the thin YA `SubagentSpec` envelope,
resolve it against one immutable capability catalog, register the resulting plan, and
inject one `DelegationCapability` backed by a store and driver. See
[`./subagent.md`](./subagent.md) for a complete example and durability boundaries.

## Public Boundary Checklist

- `capabilities=` is the sole public composition plane.
- `AgentSpec` owns model, settings, instructions, output schema, and serialized
  capability definitions.
- `SubagentSpec` adds only delegation policy.
- Named children receive only capabilities declared in their own native spec.
- Self forks rebuild an explicit policy and bounded history snapshot; they never clone
  live parent capabilities.
- `ToolVisibilityCapability` is the final child execution-boundary defense.
- Steering uses Pydantic AI `AgentRun.enqueue()` through `LogicalRunInputRouter`.
- Durable hosts persist input before acknowledgement and keep canonical delivery in
  their inbox/store. SDK lifecycle events and UI projections are notifications only.
- There is no MessageBus, generated delegate class, implicit tool inheritance, or
  runtime compatibility layer in 2.0.

## Reference Routing

| Topic                | Local path                                             | Read when                                                  |
| -------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| Context and sessions | [`./context.md`](./context.md)                         | Persisting context, history, or custom context fields      |
| Streaming and hooks  | [`./streaming.md`](./streaming.md)                     | Streamed UX, recovery, or lifecycle extensions             |
| Events               | [`./events.md`](./events.md)                           | Consuming SDK or feature lifecycle events                  |
| Tools and policies   | [`./toolset.md`](./toolset.md)                         | Writing BaseTool adapters or execution-policy capabilities |
| Structured input     | [`./user-input.md`](./user-input.md)                   | Approval or external deferred continuation                 |
| Native Tool Search   | [`./tool-search.md`](./tool-search.md)                 | Deferred native capabilities and large tool libraries      |
| Subagents            | [`./subagent.md`](./subagent.md)                       | Child specs, resolution, services, stores, or drivers      |
| Environment          | [`./environment.md`](./environment.md)                 | Filesystem, shell, resources, and lifecycle authority      |
| Resumable resources  | [`./resumable-resources.md`](./resumable-resources.md) | Reconstructing long-lived external resources               |
| Skills               | [`./skills.md`](./skills.md)                           | SDK skill catalog loading and refresh                      |
| Model configuration  | [`./model.md`](./model.md)                             | Models, settings, wrappers, and presets                    |
| Media                | [`./media.md`](./media.md)                             | Image, audio, video, and file inputs                       |
| Tool proxy           | [`./tool-proxy.md`](./tool-proxy.md)                   | Search/proxy wrappers around external toolsets             |
| CodeAct              | [`./codeact.md`](./codeact.md)                         | Restricted Python orchestration                            |

## Runnable Examples

- `../../examples/general.py`: capability composition, streaming, typed HITL,
  persistence, named delegation, and self fork.
- `../../examples/deepresearch.py`: autonomous capability-first research agent with
  structured output.

After editing this canonical skill, run `scripts/sync-skills.sh` to update YAACLI's
bundled copy.
