# Tool Proxy

`ToolProxyCapability` exposes a fixed model-facing surface over explicitly granted toolsets:

- `search_tools` discovers tool schemas;
- `call_tool` invokes a discovered tool through the active Pydantic AI `ToolManager`.

With a prefix, the names become `{prefix}_search_tool` and `{prefix}_call_tool`. The model-facing tool list remains constant, which is useful for large host-managed MCP catalogs and prompt caching.

## Basic usage

```python
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import ToolProxyCapability

async def get_weather(location: str) -> str:
    """Get current weather for a location."""
    ...

async def get_forecast(location: str, days: int = 5) -> str:
    """Get a multi-day weather forecast."""
    ...

weather = FunctionToolset(
    [get_weather, get_forecast],
    id="weather",
)

runtime = create_agent(
    "openai:gpt-5",
    capabilities=[
        ToolProxyCapability(
            toolsets=(weather,),
            namespace_descriptions={"weather": "Weather and forecast operations"},
        )
    ],
)
```

`ToolProxyCapability` is the behavior-composition boundary. The concrete proxy toolset is an implementation adapter, not a second `create_agent()` composition plane.

## Namespaces and loose tools

A wrapped toolset with an `id` is a namespace. A matching search result makes all tools in that namespace available through `call_tool`. A toolset without an `id` contributes loose tools that are discovered individually.

Descriptions are resolved from `namespace_descriptions` and toolset metadata. Prefer short descriptions that state the domain and available actions.

## Multiple proxies

Use prefixes when one agent has independent proxy catalogs:

```python
capabilities = [
    ToolProxyCapability(
        id="tool_proxy:mcp",
        toolsets=tuple(mcp_toolsets),
        prefix="mcp",
        namespace_descriptions=mcp_descriptions,
    ),
    ToolProxyCapability(
        id="tool_proxy:internal",
        toolsets=tuple(internal_toolsets),
        prefix="internal",
        namespace_descriptions=internal_descriptions,
    ),
]
```

Each proxy also needs a unique capability `id`. Prefixes must start with a letter and contain only letters, numbers, and underscores. Discovery state remains isolated by prefix.

## Optional namespaces

A namespace listed in `optional_namespaces` may fail initialization without preventing the remaining proxy from starting:

```python
ToolProxyCapability(
    toolsets=tuple(mcp_toolsets),
    optional_namespaces=frozenset({"analytics"}),
)
```

Namespace availability is reported through `NamespaceStatusEvent`. Required namespace failures still fail initialization.

## Search strategies

The default strategy is dependency-free keyword matching. Use the shared search package to select BM25 when installed:

```python
from ya_agent_sdk.toolsets.search import (
    BM25SearchStrategy,
    KeywordSearchStrategy,
    SearchStrategy,
    create_best_strategy,
)

proxy = ToolProxyCapability(
    toolsets=tuple(toolsets),
    search_strategy=create_best_strategy(),
)
```

Install BM25 support with:

```bash
pip install 'ya-agent-sdk[tool-proxy]'
```

A custom `SearchStrategy` implements `build_index()`, `search()`, and `get_search_hint()` over `ToolMetadata` values.

## Resumable state

Discovered names are stored in the typed state at:

- `AgentContext.tool_proxy.loaded_tools`;
- `AgentContext.tool_proxy.loaded_namespaces`.

`AgentContext.export_state()` and `ResumableState.restore()` copy this state across session restore. Prefix scoping is preserved in the stored names; callers should not mutate the encoding.

## Choosing Tool Proxy or native Tool Search

| Need                                                   | Use                                                        |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| Fixed search/invoke surface over host-managed toolsets | `ToolProxyCapability`                                      |
| Native tools dynamically enter the model tool list     | Pydantic AI deferred `Capability` plus native `ToolSearch` |
| Small stable tool list                                 | Ordinary capabilities                                      |

Tool Proxy invocation is manager-backed, so nested calls retain validation, policy hooks, approval, deferred-call handling, and tracing. Do not bypass it by calling wrapped toolsets directly.
