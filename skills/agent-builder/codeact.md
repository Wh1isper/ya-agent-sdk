# CodeAct

Use restricted Python to compose eligible tools without weakening the current agent or Environment boundary.

## Enable CodeAct

```python
from pydantic_ai.capabilities import Capability, Toolset as ToolsetCapability
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.codeact import CodeActCapability, CodeActConfig

runtime = create_agent(
    model,
    capabilities=[
        Capability(tools=[...], id="application_tools"),
        ToolsetCapability(external_toolset, id="external"),
        CodeActCapability(config=CodeActConfig()),
    ],
)
```

CodeAct is disabled when `CodeActCapability` is absent. Monty is still an SDK core dependency so deployment shape does not change when the capability is enabled.

## Mark eligible tools

SDK tools opt in explicitly:

```python
class SearchTool(BaseTool):
    name = "search"
    description = "Search indexed records."
    codeact = True

    async def call(self, ctx, *, query: str) -> list[dict[str, object]]:
        ...
```

External Pydantic AI toolsets attach the same metadata:

```python
toolset = FunctionToolset([
    Tool(search, metadata={"codeact": True}),
])
```

Eligibility does not bypass availability, subagent filtering, validation, hooks, approval, deferred-call handling, or tracing. Host-managed `NamedMCPToolset` actual tools are marked eligible by default and still use their MCP approval hooks. Provider-native MCP is not bridged locally.

## Inline execution

The model receives `run_code(code, restart=False)`. Eligible direct tools remain visible; CodeAct augments rather than replaces the normal tool surface.

- The language is Monty's restricted Python subset.
- Tool functions accept keyword arguments only.
- Non-sequential tools are async and must be awaited.
- REPL state lasts only for the current Pydantic agent run.
- `restart=True` discards that run-local state.
- No ambient filesystem, network, process, environment, credential, or clock access is provided.
- `max_concurrency` admits calls before Monty materializes host arguments and covers argument serialization, validation hooks, and execution; a call counts as started only after validation succeeds.
- `max_output_bytes` bounds each nested argument set, result, explicit `ToolReturn.content`, cumulative nested results, and the final value before large supported host values are fully encoded or cross into Monty; unknown value types fail closed.
- Nested `ToolReturn.content` remains model-facing on successful CodeAct completion; nested metadata is not merged into outer metadata.
- Host-managed MCP calls keep `structuredContent` as the Python value and media as supplemental content. Completed MCP errors return structured data for explicit inspection instead of consuming `ModelRetry`; transport or protocol failures without a result are terminal failures.

## Reusable programs

A program is a strict UTF-8 `*.codeact.py` file with this entrypoint:

```python
import asyncio

async def main(inputs):
    results = await asyncio.gather(
        *(search(query=query) for query in inputs["queries"])
    )
    return results
```

Execute it with `run_program(path, inputs)`. The tool is exposed only when the current Environment provides a `FileOperator`. The SDK:

1. reads at most `max_source_bytes + 1` through `ctx.deps.file_operator`;
2. validates the module, safe module-scope declarations, exact `async def main(inputs)` signature, reserved ambient-builtin names, and common ambient-capability imports;
3. hashes the exact source bytes;
4. injects inputs as data rather than source interpolation;
5. checks out a fresh Monty session;
6. dispatches nested calls through the current Pydantic AI `ToolManager`;
7. closes the session before returning.

A program source file is durable; interpreter state and prior outputs are not. Running it again re-executes current tools against current external state. There is no rollback or exactly-once guarantee.

The `*.codeact.py` suffix distinguishes this restricted contract from general CPython. Names such as `open`, `eval`, and `exec` are reserved and cannot be referenced even when shadowed; imports of common ambient filesystem, process, or network modules are also rejected during preflight. Other pure builtins and supported modules such as `asyncio` remain available; host effects must use injected CodeAct-eligible tools.

These checks provide deterministic authoring diagnostics, not an exhaustive capability security policy. The security boundary remains Monty's lack of workspace mounts and ambient OS adapters plus dispatch through the current injected-tool catalog.

## Environment boundary

Environment owns workspace paths, `FileOperator`, `Shell`, browsers/computer resources, and resource toolsets. CodeAct owns only restricted orchestration and its current-agent bridge.

Do not place a CodeAct catalog, `RunContext`, `ToolManager`, or inline session in `Environment.resources`: Environment is shared across contexts, named subagents, self forks, and later runs. A future authority-free worker pool may be Environment-scoped, but run-local sessions and catalogs must remain capability-owned.

## Failure and cancellation

Before any nested call starts, syntax, missing direct-name functions, program preflight, typing, and argument problems may produce `ModelRetry`. Validation diagnostics omit raw Pydantic input values. After a nested call starts, execution failure or unresolved deferred interaction produces a terminal failed tool result with bounded, redacted trace descriptors; the SDK does not automatically repeat side effects.

Caller cancellation cancels and drains active nested-call ownership, closes the Monty session, and re-raises `CancelledError`. `timeout_seconds` is the execution deadline at which cancellation is requested; completion can occur later while in-process calls release ownership. Marking a tool `codeact=True` is a trusted declaration that it stops local work, releases locally owned resources, and propagates cancellation. An indefinitely cancellation-suppressing tool violates that contract and can block the run; hard termination requires process or container isolation.
