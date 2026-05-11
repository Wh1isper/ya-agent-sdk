---
name: agent-builder
description: Build and configure AI agents with ya-agent-sdk and Pydantic AI. Covers create_agent(), stream_agent(), AgentContext, ResumableState session persistence, toolsets, subagents, environments, and HITL approval. Use when implementing agent applications, wiring tools into an agent, restoring multi-turn sessions, configuring subagent hierarchies, adding approval flows, or working with ya-agent-sdk APIs such as create_agent, export_state, stream_agent, and SubagentConfig.
---

# Building Agents with ya-agent-sdk

Build production-ready AI agents with ya-agent-sdk and Pydantic AI.

## Start Here

Choose the workflow that matches the task:

- Use `create_agent()` for one-shot runs and runtime construction.
- Use `stream_agent()` for interactive or event-driven flows.
- Use `runtime.ctx.export_state()` and `create_agent(..., state=state)` for multi-turn persistence.
- Use tool classes, toolsets, hooks, and approval settings for tool-enabled agents.
- Use `SubagentConfig` and delegation settings for hierarchical agents.
- Use custom environments and sandbox resources for richer runtime behavior.

Start with these local references:

- Sessions and restore flow: [`./context.md`](./context.md)
- Streaming and event handling: [`./streaming.md`](./streaming.md), [`./events.md`](./events.md)
- Tools, hooks, and toolsets: [`./toolset.md`](./toolset.md), [`./tool-search.md`](./tool-search.md)
- Subagents and delegation: [`./subagent.md`](./subagent.md)
- Environments and resumable resources: [`./environment.md`](./environment.md), [`./resumable-resources.md`](./resumable-resources.md)

## Installation

Install the core package for basic agent construction:

```bash
pip install ya-agent-sdk
uv add ya-agent-sdk
```

Install the full toolkit when you want examples, document tools, tool search, and common integrations:

```bash
pip install ya-agent-sdk[all]
uv add ya-agent-sdk[all]
```

Install selective extras when you want a smaller dependency set:

```bash
pip install ya-agent-sdk[docker]
pip install ya-agent-sdk[web]
pip install ya-agent-sdk[document]
pip install ya-agent-sdk[s3]
pip install ya-agent-sdk[tool-search]
```

## Core Workflows

### Create a basic agent

```python
from ya_agent_sdk.agents import create_agent

async with create_agent("anthropic:claude-sonnet-4") as runtime:
    result = await runtime.agent.run("Summarize this project", deps=runtime.ctx)
    print(result.output)
```

### Stream responses

```python
from ya_agent_sdk.agents import create_agent, stream_agent

runtime = create_agent("openai:gpt-4o")

async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

Read [`./streaming.md`](./streaming.md) and [`./events.md`](./events.md) when the task needs lifecycle hooks, custom event handling, or streamed UX.

### Add tools and toolsets

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from ya_agent_sdk.toolsets.core.shell import tools as shell_tools

async with create_agent(
    model="anthropic:claude-sonnet-4",
    system_prompt="You are a coding assistant.",
    tools=[*filesystem_tools, *shell_tools],
) as runtime:
    result = await runtime.agent.run("List the repository files", deps=runtime.ctx)
    print(result.output)
```

Read [`./toolset.md`](./toolset.md) when the task needs custom toolsets, hooks, retries, or timeouts. Read [`./tool-search.md`](./tool-search.md) when the available tool surface is large or dynamic.

### Persist and restore sessions

```python
from ya_agent_sdk.agents import create_agent

async with create_agent("openai:gpt-4o") as runtime:
    await runtime.agent.run("Remember that I prefer concise answers.", deps=runtime.ctx)
    state = runtime.ctx.export_state()

restored_runtime = create_agent("openai:gpt-4o", state=state)
```

Read [`./context.md`](./context.md) for `ResumableState`, message history handling, and restore semantics.

### Configure approval flows

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from ya_agent_sdk.toolsets.core.shell import tools as shell_tools

async with create_agent(
    model="anthropic:claude-sonnet-4",
    tools=[*filesystem_tools, *shell_tools],
    need_user_approve_tools=["shell", "edit"],
) as runtime:
    ...
```

Read [`./toolset.md`](./toolset.md) when approval rules interact with hooks, tool composition, or custom tool registration.

### Add subagents

```python
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.subagents import SubagentConfig

researcher = SubagentConfig(
    name="researcher",
    description="Research specialist for web search tasks",
    system_prompt="You gather relevant facts and sources.",
    tools=["search_with_tavily", "visit_webpage"],
)

async with create_agent(
    "openai:gpt-4o",
    subagent_configs=[researcher],
    unified_subagents=True,
) as runtime:
    ...
```

Read [`./subagent.md`](./subagent.md) for subagent loading, unified delegation, and builtin subagent behavior.

## Reference Routing

All paths below are local paths relative to this file.

| Topic                | Local path                                             | Read when                                                          |
| -------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| Context and sessions | [`./context.md`](./context.md)                         | You need session state, message history, or restore behavior       |
| Streaming and hooks  | [`./streaming.md`](./streaming.md)                     | You need streaming output, lifecycle hooks, or interactive runs    |
| Events               | [`./events.md`](./events.md)                           | You need the event model or sideband event handling                |
| Toolsets             | [`./toolset.md`](./toolset.md)                         | You need tools, toolsets, hooks, retries, or approval settings     |
| Tool search          | [`./tool-search.md`](./tool-search.md)                 | You need discovery across a large or dynamic tool library          |
| Subagents            | [`./subagent.md`](./subagent.md)                       | You need delegation, subagent configs, or unified subagent tools   |
| Environment          | [`./environment.md`](./environment.md)                 | You need custom environments or sandbox-backed execution           |
| Resumable resources  | [`./resumable-resources.md`](./resumable-resources.md) | You need long-lived external resource state                        |
| Skills system        | [`./skills.md`](./skills.md)                           | You need SDK skill loading, reload, or skill integration details   |
| Message bus          | [`./message-bus.md`](./message-bus.md)                 | You need agent coordination or user steering through messages      |
| Model configuration  | [`./model.md`](./model.md)                             | You need model selection, model settings, or configuration details |
| Logging              | [`./logging.md`](./logging.md)                         | You need runtime logging, diagnostics, or instrumentation setup    |
| Media upload         | [`./media.md`](./media.md)                             | You need image, audio, video, or file media handling               |
| Tool proxy           | [`./tool-proxy.md`](./tool-proxy.md)                   | You need wrapped, remote, or proxy-style tools                     |

## Example Programs

Use these examples when you need a full application flow:

| Scenario                   | Repository source                | Bundled CLI skill            |
| -------------------------- | -------------------------------- | ---------------------------- |
| General production pattern | `../../examples/general.py`      | `./examples/general.py`      |
| Autonomous research agent  | `../../examples/deepresearch.py` | `./examples/deepresearch.py` |

## Workspace Context

This directory is the canonical source for the skill at `skills/agent-builder/`.

- Repository examples live at `../../examples/`.
- The sync script lives at `../../scripts/sync-skills.sh`.
- The CLI bundle target is `../../packages/yaacli/yaacli/skills/building-agents/`.
