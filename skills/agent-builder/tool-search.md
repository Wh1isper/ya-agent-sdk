# Native Tool Search

Use Pydantic AI's native capability loading when an agent has a large library of tools that should remain outside the initial model request. YA Agent SDK does not provide a second `ToolSearchToolSet` implementation.

## Capability-first setup

Bundle related tools and guidance in a native `Capability`, give it a useful description, and set `defer_loading=True`:

```python
from pydantic_ai.capabilities import Capability, ToolSearch
from ya_agent_sdk.agents.main import create_agent

async def search_papers(query: str) -> str:
    """Search academic papers."""
    ...

async def read_paper(paper_id: str) -> str:
    """Read one academic paper."""
    ...

papers = Capability(
    id="papers",
    description="Search and read academic papers",
    instructions="Use paper identifiers returned by search_papers.",
    tools=[search_papers, read_paper],
    defer_loading=True,
)

runtime = create_agent(
    "openai:gpt-5",
    capabilities=[
        papers,
        ToolSearch(max_results=5),
    ],
)
```

Pydantic AI owns the search tool, deferred capability loading, schema exposure, and runtime ordering. `ToolSearch()` is optional when the native defaults are sufficient; provide it explicitly to configure search behavior or descriptions.

## Declarative specs

Native `ToolSearch` is also available in `AgentSpec`:

```yaml
model: openai:gpt-5
capabilities:
  - ToolSearch:
      max_results: 5
```

A serializable custom capability may opt into deferred loading through its native capability contract. Hosts must include its class in the validated capability catalog before loading the spec.

## Tool Proxy is a different protocol

Use `ToolProxyCapability` for host-managed toolsets such as MCP servers when the model-facing surface must stay fixed at two calls: search and invoke. Tool Proxy does not dynamically add native model tools; it invokes discovered tools through the active `ToolManager`.

Tool Proxy's reusable lexical strategies live under `ya_agent_sdk.toolsets.search`:

```python
from ya_agent_sdk.toolsets.search import create_best_strategy

strategy = create_best_strategy()
```

Install ranked BM25 retrieval only when needed:

```bash
pip install 'ya-agent-sdk[tool-proxy]'
```

Without `rank-bm25`, `create_best_strategy()` uses the dependency-free keyword strategy.

## Selection guide

| Need                                                 | Use                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| Native tools appear after model-driven discovery     | Deferred native `Capability` plus Pydantic AI `ToolSearch` |
| Constant two-tool surface over host-managed toolsets | `ToolProxyCapability`                                      |
| Small, stable tool list                              | Ordinary native capabilities without deferred loading      |

Do not combine the two mechanisms merely as a fallback. Choose the protocol that matches the host and model-facing contract.
