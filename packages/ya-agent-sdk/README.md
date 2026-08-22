# Ya Agent SDK

> Yet Another Agent SDK

[![Release](https://img.shields.io/github/v/release/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/wh1isper/ya-mono/main.yml?branch=main)](https://github.com/wh1isper/ya-mono/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/wh1isper/ya-mono/branch/main/graph/badge.svg)](https://codecov.io/gh/wh1isper/ya-mono)
[![Commit activity](https://img.shields.io/github/commit-activity/m/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/commits/main)
[![License](https://img.shields.io/github/license/wh1isper/ya-mono)](https://github.com/wh1isper/ya-mono/blob/main/LICENSE)

Yet Another Agent SDK for building AI agents with [Pydantic AI](https://ai.pydantic.dev/).

## Key Features

- Capability-first composition over native Pydantic AI capabilities
- Environment-based authority for file operations, shell access, and resources
- Fully typed SDK validated with pyright
- Resumable context state plus canonical Pydantic AI message history
- Native `AgentSpec` profiles and portable subagent execution services
- Strict versioned TOML configuration for explicitly selected capability plugins
- Stateless `AgentExecutionHarness` for host-coordinated completed or suspended native segments
- Native Tool Search, Tool Proxy, skills, MCP, and CodeAct capability integrations
- Human-in-the-loop approval and structured deferred interaction
- Native logical-run steering through Pydantic AI `AgentRun.enqueue()`
- Lifecycle, usage, subagent, compact, and handoff event streaming

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
pip install 'ya-agent-sdk[tool-proxy]'
pip install 'ya-agent-sdk[oauth]'
```

## OAuth-backed Codex

Use your ChatGPT/Codex subscription through `ya-oauth`:

```bash
uv run --package ya-oauth ya-oauth login codex
```

Then select the OAuth model string:

```python
from ya_agent_sdk.agents.main import create_agent

runtime = create_agent("oauth@codex:gpt-5.5")
```

The SDK passes stable session and thread headers into the OAuth provider. YA Claw sets the provider session header from the session ID and the provider thread header from the run ID.

## OpenAI Responses WebSocket

`ya-agent-sdk` includes a built-in OpenAI Responses WebSocket transport for streaming calls. Use either alias to prefer WebSocket with automatic HTTP fallback:

```python
from ya_agent_sdk.agents.main import create_agent

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

The GPT-5 model-config presets also declare `ModelFeature.openai_prompt_cache_key`. For transports that receive the SDK's provider session headers, `create_agent()` copies the configured model settings and binds `openai_prompt_cache_key` to the exact `x-session-id` value. This includes OAuth Codex, direct Responses WebSocket aliases, and gateway-backed OpenAI Responses over HTTP or WebSocket. Gateway HTTP requests and WebSocket fallback receive the same headers as the handshake. Conflicting request-level header or body overrides are normalized on the copy; caller-owned mappings are not mutated. Models without this explicit capability are left unchanged.

## Quick Start

For workspace development, copy [`packages/ya-agent-sdk/.env.example`](.env.example) to `packages/ya-agent-sdk/.env`.
For the runnable example scripts, copy [`examples/.env.example`](../../examples/.env.example) to `examples/.env`.

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

`create_agent()` returns an unentered `AgentRuntime`. `capabilities=` is the sole public behavior-composition surface; the Pydantic AI Agent is built only after the Environment and context have entered and all contribution groups are available. `RuntimeFoundationCapability` is explicit and is not injected by `create_agent()`.

When stream recovery is enabled, delegated subagents and self forks inherit the root run's effective recovery policy. Each child retries transient provider or network stream failures against its own transport budget and resumes from its own recovered history. Successful child-local recovery does not consume the root agent's execution recovery budget; only an exhausted child failure propagates to the root tool-call path.

## Capability Plugins

Applications can select installed third-party capability types and grant configured
instances to their root agent with one SDK-owned manifest:

```toml
schema_version = 1
entry_points = ["acme.search"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }
```

Load the manifest once at trusted process bootstrap, then retain the same catalog
snapshot in every root, child, and restored runtime factory:

```python
from pydantic_ai import AgentSpec
from ya_agent_sdk.agents import validate_agent_spec_capabilities
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import load_capability_plugins
from ya_agent_sdk.context import AgentContext

plugins = load_capability_plugins("/etc/my-agent/plugins.toml")
root_spec = plugins.apply_to_root_agent_spec(
    AgentSpec.from_dict({"name": "my-agent"})
)
validate_agent_spec_capabilities(
    root_spec,
    deps_type=AgentContext,
    custom_capability_types=plugins.custom_capability_types,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4-5",
    spec=root_spec,
    custom_capability_types=plugins.custom_capability_types,
)
```

`validate_agent_spec_capabilities()` constructs a throwaway native agent with a
no-network test model, so plugin `from_spec()` values and the supplied static capability
ordering can fail before durable admission. Runtime entry still validates dynamic
Environment, context, and resource contributions.

The application still owns package installation, the trusted manifest path, missing-file
policy, and host dependencies. The SDK performs no ambient loading, installation,
upgrade, sandboxing, or hot reload. Manifest root grants do not enter named children or
self forks; a selected type is available to a child only when that child's native
`AgentSpec` explicitly grants it. Manifest arguments are durable non-secret
configuration, and secret-like keys fail validation recursively. This name-based
validation cannot recognize a secret placed under a neutral key, so every secret value
must still remain outside the manifest.

See the [file configuration specification](spec/06-capability-plugins/03-file-configuration.md),
the [application integration guide](../../skills/agent-builder/plugins.md), and the
[runnable installable plugin example](../../examples/capability_plugin/).

## Retry Boundaries

Configure native Pydantic AI tool/output correction limits with `create_agent(retries=...)`. When omitted, SDK `RetryConfig` supplies five for both tools and output. The explicit `OverallRetryBudget` capability supplies a separate cumulative run-wide ceiling and is included in `RuntimeFoundationCapability` with a default of five:

```python
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import OverallRetryBudget

runtime = create_agent(
    "openai-chat:gpt-4o",
    retries={"tools": 5, "output": 5},
    capabilities=[OverallRetryBudget(max_retries=5)],
)
```

`BaseTool` `ModelRetry` signals propagate unchanged into Pydantic AI, while HTTP/WebSocket request retries and `stream_agent()` recovery are transport/execution recovery mechanisms. SDK-created model-provider HTTP clients use `httpx2.AsyncClient`, and request retries use Pydantic AI's `AsyncHTTPX2TenacityTransport`; custom clients passed to current providers must use the same HTTPX2 boundary. Pydantic AI's deprecated, retired GitHub Models provider remains upstream-owned on legacy HTTPX until it is removed in v3. Clients created by the SDK are provider-owned, close with the model context, and are recreated when the model is entered again. None of those consumes the model-correction budget. Native steering is accepted through the active logical-run router and `AgentRun.enqueue()`; it is not a retry prompt.

`ToolTimeoutCapability()` applies a generic 600-second ceiling to one validated tool execution attempt. Set `YA_AGENT_TOOL_TIMEOUT_SECONDS` to a positive finite number to change that default when the capability is constructed, or pass `timeout=` explicitly for one capability instance. A tool definition may declare a shorter timeout; longer tool-specific values remain capped by the capability ceiling. Tools should own stricter operation-specific deadlines and return their normal bounded timeout result where appropriate. If the generic ceiling expires, the capability raises Pydantic AI's `ModelRetry` with the tool name, effective deadline, cancellation state, and partial-side-effect warning. Pydantic AI accounts for the correction against the native per-tool retry budget; the model can inspect state before deciding whether to retry, so the same potentially side-effecting call is not blindly replayed.

## Portable Subagents

`SubagentExecutionService` separates portable records and lifecycle semantics from host
coroutine ownership. The standalone SDK defaults to `InlineSubagentExecutionHost`, so
`delegate` is foreground-only and runs in the calling tool task. Applications that
intentionally support background work inject a `SubagentExecutionHost` such as
`AsyncioSubagentExecutionHost` or their own durable scheduler. Both modes return the
same bounded structured result with a short route-prefixed `execution_id` such as
`code-reviewer-f1a2`; mode is explicit data and is not encoded in the handle. Use the
separate `resume_subagent(execution_id, prompt)` model tool for linked continuation.
Inspection, wait, fan-in, and background-completion projections are paged or bounded;
internal descriptors, usage state, logical-run IDs, and correlation UUIDs remain private.

See [the portable subagent specification](spec/05-capability-first-runtime/07-subagent-runtime.md)
and [the agent-builder guide](../../skills/agent-builder/subagent.md).

## Structured Clarifying Questions

The optional `ask_user_question` tool uses Pydantic AI deferred-tool control flow to request one to four structured questions with suggested options, multi-select support, and free-text answers.

```python
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    RuntimeFoundationCapability,
    UserInteractionCapability,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[
        RuntimeFoundationCapability(),
        UserInteractionCapability(),
    ],
    output_type=[str, DeferredToolRequests],
)
```

The SDK deliberately does **not** include this capability by default: hosts must opt in only when they can present `DeferredToolRequests`, collect answers, and resume with matching `DeferredToolResults.calls`. The tool carries `main_agent_only` metadata; final `ToolVisibilityCapability` and its own availability check reject it in child execution contexts. Nested subagent runs do not own the host's user-interaction loop unless a host explicitly implements that protocol. Use `DeferredInteractionResolver` rather than runtime-private toolset access. See [Structured User Input](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/user-input.md) for the schema and complete continuation flow.

## CodeAct

CodeAct lets a model orchestrate eligible host tools from restricted Python while preserving the normal Pydantic AI validation, hooks, approval, tracing, and final-agent boundaries. Enable it explicitly:

```python
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import CodeActCapability, FilesystemCapability
from ya_agent_sdk.codeact import CodeActConfig

runtime = create_agent(
    "openai-chat:gpt-4o",
    capabilities=[
        FilesystemCapability(),
        CodeActCapability(config=CodeActConfig()),
    ],
)
```

SDK `BaseTool` classes opt in with `codeact = True`; external toolsets attach `ToolDefinition.metadata["codeact"] = True`. Actual tools produced by the private host-managed MCP adapter opt in by default. Eligible tools remain directly model-visible as well as callable through `run_code`.

Host-managed MCP results preserve `structuredContent` as the callable Python value while forwarding accompanying image, audio, or binary content through `ToolReturn.content`. Completed MCP error results are returned as structured values for explicit caller inspection rather than raised as `ModelRetry`; transport or protocol failures without a result become terminal `ToolFailed` outcomes. This avoids consuming per-tool retry budgets or implicitly replaying side effects.

When the current Environment provides a `FileOperator`, `run_program(path, inputs)` reads a strict UTF-8 `*.codeact.py` file through it and executes `async def main(inputs)` in a fresh Monty session; otherwise that tool is not exposed. The dedicated suffix distinguishes the restricted program contract from general CPython. Preflight reserves known ambient-builtin names and rejects common ambient-capability imports as an authoring diagnostic, with guidance to use injected CodeAct-eligible tools; the security boundary remains Monty's lack of ambient authority and the injected-tool dispatch boundary. Monty receives no workspace mount or ambient OS access: filesystem, shell, browser, network, and computer-use operations still cross the current Environment tool boundary. `max_concurrency` admits calls before host argument materialization and covers argument serialization, nested validation, and execution. `max_output_bytes` bounds each nested argument set, result, explicit model-facing `ToolReturn.content`, cumulative nested results, and the final returned value before large supported host values are fully encoded or cross into Monty. `timeout_seconds` initiates cancellation at the execution deadline; CodeAct still drains active in-process tool ownership before returning, so tools exposed while `CodeActCapability` is enabled must cooperate with cancellation. See [CodeAct](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/codeact.md) and [the program specification](spec/03-codeact-programs.md).

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

## Background Shell Handles

`Shell.start()` and `shell_exec(background=True)` return compact session-local handles
such as `process-1`. The same `process_id` value is used by `shell_wait`, `shell_status`,
`shell_input`, `shell_signal`, `shell_kill`, completion injection, and lifecycle events.
A successful shell-session reset restarts the sequence; foreground commands do not
consume it. The handle is not an OS PID, process-group ID, container exec token, UUID,
or durable host key. Those identities remain private to the Environment backend.

## Local Shell Sandbox Policy

`LocalShell` is the SDK's single local subprocess implementation. By default, `LocalShell` and `LocalEnvironment` preserve raw local subprocess behavior for SDK and YAACLI compatibility. Pass a resolved `ShellSandboxRuntimePolicy` to `LocalShell(sandbox_policy=...)` or `LocalEnvironment(shell_sandbox_policy=...)` to route commands through the selected local sandbox backend. `SandboxedLocalShell` is exported as a direct alias of `LocalShell` for naming convenience.

Path masks are opt-in. `ShellSandboxConfig.masked_path_aliases` provides recommended aliases such as `common_credentials`, `ssh`, `aws`, and `kube`; `masked_paths` accepts concrete paths. Linux bubblewrap applies these masks as tmpfs mounts inside the sandbox.

## Shell Command Review

Configure shell command review on `AgentContext.security.shell_review` to run a small reviewer model before shell execution:

```python
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.context import SecurityConfig, ShellReviewConfig

runtime = create_agent(
    "gateway@openai-responses:gpt-5.5",
    context_kwargs={
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
- [Native Tool Search](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/tool-search.md)
- [Portable Subagent Runtime](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/subagent.md)
- [Skills System](https://github.com/wh1isper/ya-mono/tree/main/skills/agent-builder/skills.md)
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
