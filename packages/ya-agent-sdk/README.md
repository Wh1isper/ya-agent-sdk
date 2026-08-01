# Ya Agent SDK

> Yet Another Agent SDK

[![Release](https://img.shields.io/github/v/release/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/wh1isper/ya-mono/main.yml?branch=main)](https://github.com/wh1isper/ya-mono/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/wh1isper/ya-mono/branch/main/graph/badge.svg)](https://codecov.io/gh/wh1isper/ya-mono)
[![Commit activity](https://img.shields.io/github/commit-activity/m/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/commits/main)
[![License](https://img.shields.io/github/license/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/blob/main/LICENSE)

Yet Another Agent SDK for building AI agents with [Pydantic AI](https://ai.pydantic.dev/).

## Key Features

- Environment-based architecture for file operations, shell access, and resources
- Fully typed SDK validated with pyright
- Resumable sessions with state export and restore
- Hierarchical agents with subagent delegation
- Tool search for large tool libraries
- Skills system with hot reload and progressive loading
- Human-in-the-loop approval and optional structured clarification workflows
- Event system and streaming support
- Message bus for agent coordination and user steering

## Installation

```bash
pip install 'ya-agent-sdk[all,rs]'
uv add 'ya-agent-sdk[all,rs]'
```

`[rs]` adds the native Rust filesystem search binding. Selective extras:

```bash
pip install 'ya-agent-sdk[rs]'
pip install 'ya-agent-sdk[docker]'
pip install 'ya-agent-sdk[web]'
pip install 'ya-agent-sdk[document]'
pip install 'ya-agent-sdk[s3]'
pip install 'ya-agent-sdk[tool-search]'
pip install 'ya-agent-sdk[oauth]'
```

## OAuth-backed Codex

Use your ChatGPT/Codex subscription through `ya-oauth`:

```bash
uv run --package ya-oauth ya-oauth login codex
```

Then select the OAuth model string:

```python
from ya_agent_sdk.agents import create_agent

runtime = create_agent("oauth@codex:gpt-5.5")
```

The SDK passes stable session and thread headers into the OAuth provider. YA Claw sets the provider session header from the session ID and the provider thread header from the run ID.

## OpenAI Responses WebSocket

`ya-agent-sdk` includes a built-in OpenAI Responses WebSocket transport for streaming calls. Use either alias to prefer WebSocket with automatic HTTP fallback:

```python
from ya_agent_sdk.agents import create_agent

runtime = create_agent("openai-responses-ws:gpt-5.5")
# Equivalent alias:
# runtime = create_agent("openai-responses-rs:gpt-5.5")
```

Set `YA_AGENT_OPENAI_RESPONSES_WEBSOCKET_MODE` to `auto`, `websocket`, or `http` to control the transport. The OAuth Codex provider reuses this SDK transport and only adds Codex-specific headers and payload normalization.

GPT-5.6 supports independent reasoning effort and reasoning mode controls. Use `openai_responses_pro` for `pro` mode with balanced `medium` effort:

```python
runtime = create_agent(
    "openai-responses:gpt-5.6",
    model_settings="openai_responses_pro",
)
```

Choose `openai_responses_pro_low`, `openai_responses_pro_medium`, `openai_responses_pro_high`, `openai_responses_pro_xhigh`, or `openai_responses_pro_max` to pair pro mode with an explicit effort. `openai_responses_pro` is the medium-effort convenience preset. Existing OpenAI Responses effort presets remain in the default `standard` mode. GPT-5.6 Sol can use `openai_responses_max` for `max` reasoning effort. Terra and Luna convenience aliases are available as `openai_responses_terra` and `openai_responses_luna`. Use `gpt5_350k` for subscription-backed Codex access with a 350K context window; keep using the other GPT-5 `model_cfg` presets when they match the provider's documented context window.

## Quick Start

For workspace development, copy [`packages/ya-agent-sdk/.env.example`](.env.example) to `packages/ya-agent-sdk/.env`.
For the runnable example scripts, copy [`examples/.env.example`](../../examples/.env.example) to `examples/.env`.

```python
from ya_agent_sdk.agents import create_agent, stream_agent

runtime = create_agent("openai-chat:gpt-4o")

async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

When stream recovery is enabled, delegated subagents and self forks inherit the root run's effective recovery policy. Each child retries transient provider or network stream failures against its own transport budget and resumes from its own recovered history. Successful child-local recovery does not consume the root agent's execution recovery budget; only an exhausted child failure propagates to the root tool-call path.

## Structured Clarifying Questions

The optional `ask_user_question` tool uses Pydantic AI deferred-tool control flow to request one to four structured questions with suggested options, multi-select support, and free-text answers.

```python
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.agents import create_agent
from ya_agent_sdk.toolsets.core.interaction import tools as interaction_tools

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    tools=[*interaction_tools],
    output_type=[str, DeferredToolRequests],
)
```

The SDK deliberately does **not** include this tool in its default tool surface: hosts must opt in only when they can present `DeferredToolRequests`, collect answers, and resume with matching `DeferredToolResults.calls`. The tool sets `main_agent_only=True`, so regular subagents and self forks remove it from direct SDK `Toolset` instances, capability wrappers, and sync or async dynamic Toolset factory results at both per-run and per-step resolution. SDK `Toolset` also enforces this policy while listing and calling tools in a subagent context, regardless of `skip_unavailable`, so opaque search/proxy composites and stale caches cannot bypass it. Its runtime availability check additionally requires a root main-agent context as defense in depth. Nested subagent runs do not own the host's user-interaction loop. See [Structured User Input](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/user-input.md) for the question schema and a complete continuation example.

## CodeAct

CodeAct lets a model orchestrate eligible host tools from restricted Python while preserving the normal Pydantic AI validation, hooks, approval, tracing, and final-agent boundaries. Enable it explicitly:

```python
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.codeact import CodeActConfig

runtime = create_agent(
    "openai-chat:gpt-4o",
    tools=[...],
    codeact=CodeActConfig(),
)
```

SDK `BaseTool` classes opt in with `codeact = True`; external toolsets attach `ToolDefinition.metadata["codeact"] = True`. Host-managed `NamedMCPToolset` tools opt in by default. Eligible tools remain directly model-visible as well as callable through `run_code`.

When the current Environment provides a `FileOperator`, `run_program(path, inputs)` reads a strict UTF-8 `.py` file through it and executes `async def main(inputs)` in a fresh Monty session; otherwise that tool is not exposed. Monty receives no workspace mount or ambient OS access: filesystem, shell, browser, network, and computer-use operations still cross the current Environment tool boundary. `max_concurrency` admits calls before host argument materialization and covers argument serialization, nested validation, and execution. `max_output_bytes` bounds each nested argument set, result, explicit model-facing `ToolReturn.content`, cumulative nested results, and the final returned value before large supported host values are fully encoded or cross into Monty. `timeout_seconds` initiates cancellation at the execution deadline; CodeAct still drains active in-process tool ownership before returning, and `codeact=True` therefore requires cancellation-cooperative tools. See [CodeAct](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/codeact.md) and [the program specification](spec/03-codeact-programs.md).

## Environment Temporary Storage

`Environment` owns managed temporary storage. While entered, use `env.tmp_dir` to
inspect the agent-facing root and `env.resolve_tmp_path("relative/path")` to build a
contained path. Temporary files use the normal `env.file_operator` methods.
Workspace-backed environments use a hidden `.tmp/ya-agent-<id>` directory; a
`LocalEnvironment` without a workspace falls back to the system temporary directory,
and an explicit `tmp_base_dir` takes priority. Each owned instance contains a
self-ignoring `.gitignore`, so temporary contents stay out of Git status without
editing the project's root ignore file. Sandbox and YA Claw environments create the
same path below an existing shared mount so file operations and container commands use
the same path. Reusable containers therefore need no additional bind mount. Temporary
storage is removed only after resources, shell, and file operator cleanup.

`FileOperator.read_bytes_stream()` returns an async iterator directly:

```python
stream = env.file_operator.read_bytes_stream(path)
async for chunk in stream:
    ...
```

## Local Shell Sandbox Policy

`LocalShell` is the SDK's single local subprocess implementation. By default, `LocalShell` and `LocalEnvironment` preserve raw local subprocess behavior for SDK and YAACLI compatibility. Pass a resolved `ShellSandboxRuntimePolicy` to `LocalShell(sandbox_policy=...)` or `LocalEnvironment(shell_sandbox_policy=...)` to route commands through the selected local sandbox backend. `SandboxedLocalShell` is exported as a direct alias of `LocalShell` for naming convenience.

Path masks are opt-in. `ShellSandboxConfig.masked_path_aliases` provides recommended aliases such as `common_credentials`, `ssh`, `aws`, and `kube`; `masked_paths` accepts concrete paths. Linux bubblewrap applies these masks as tmpfs mounts inside the sandbox.

## Shell Command Review

Configure shell command review on `AgentContext.security.shell_review` to run a small reviewer model before shell execution:

```python
from ya_agent_sdk.agents import create_agent, stream_agent
from ya_agent_sdk.context import SecurityConfig, ShellReviewConfig

runtime = create_agent(
    "gateway@openai-responses:gpt-5.5",
    extra_context_kwargs={
        "security": SecurityConfig(
            shell_review=ShellReviewConfig(
                enabled=True,
                model="gateway@openai-responses:gpt-5.4-mini",
                model_settings="openai_responses_low",
                on_needs_approval="defer",
                risk_threshold="high",
            )
        )
    },
)

async with stream_agent(runtime, "Run the test suite") as streamer:
    async for event in streamer:
        print(event)
```

`model` is required when shell review is enabled. `model_settings` accepts SDK preset names or an inline settings dictionary. `on_needs_approval` supports `defer` for HITL-capable runtimes and `deny` for autopilot runtimes. `risk_threshold` defaults to `high` and controls when the configured action triggers.

## Model Preset Tips

For Anthropic models, `anthropic` now resolves to adaptive thinking by default.

- Use `anthropic` for the default adaptive preset.
- Use `anthropic_adaptive_xhigh` for Claude Opus 4.7 long-horizon coding and agentic workloads.
- Use `openai_responses_pro` or `openai_responses_gpt5_6_pro` for GPT-5.6 pro reasoning mode.
- Use `openai_responses_max` or `openai_responses_gpt5_6_sol` for GPT-5.6 Sol maximum reasoning effort.
- Use `openai_responses_xhigh` for GPT-5.5 hard asynchronous agentic tasks and evals.
- Use `openai_responses_terra` or `openai_responses_luna` for GPT-5.6 balanced or low-latency tiers.
- Use `anthropic_off` when you want thinking disabled.
- Use `anthropic_400k` or `claude_400k` for a 400K context window between `claude_200k` and `claude_1m`.

## Repository Context

This package lives in the [`ya-mono`](https://github.com/wh1isper/ya-mono) workspace.

- CLI package: [`packages/yaacli`](https://github.com/wh1isper/ya-mono/tree/main/packages/yaacli)
- Examples: [`examples/`](https://github.com/wh1isper/ya-mono/tree/main/examples)
- Skill source: [`skills/agent-builder/`](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder)
- agent-builder skill: [`skills/agent-builder/SKILL.md`](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/SKILL.md)

## Examples

| Example | Description |
| --- | --- |
| [`general.py`](https://github.com/wh1isper/ya-mono/tree/main/examples/general.py) | Production pattern with streaming, HITL approval, and session persistence |
| [`deepresearch.py`](https://github.com/wh1isper/ya-mono/tree/main/examples/deepresearch.py) | Autonomous research agent with web search and content extraction |

## Reference Files

- [AgentContext & Sessions](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/context.md)
- [Streaming & Hooks](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/streaming.md)
- [Events](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/events.md)
- [Toolset Architecture](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/toolset.md)
- [Structured User Input](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/user-input.md)
- [Tool Search](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/tool-search.md)
- [Subagent System](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/subagent.md)
- [Skills System](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/skills.md)
- [Message Bus](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/message-bus.md)
- [Media Upload](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/media.md)
- [Custom Environments](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/environment.md)
- [Resumable Resources](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/resumable-resources.md)
- [Model Configuration](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/model.md)
- [Logging Configuration](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/logging.md)
- [Tool Proxy](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/tool-proxy.md)

## Development

```bash
git clone git@github.com:YOUR_NAME/ya-mono.git
cd ya-mono
uv sync --all-packages
```

Workspace commands live at the repository root. See the [contributing guide](https://github.com/wh1isper/ya-mono/tree/main/CONTRIBUTING.md).
