# Environment Management

Resource management, lifecycle hooks, and environment implementations.

> **Note**: Base abstractions (`Environment`, `FileOperator`, `Shell`, `ResourceRegistry`, etc.) are defined in the [y-agent-environment](https://github.com/wh1isper/y-agent-environment) protocol package. This SDK provides concrete implementations (`LocalEnvironment`, `SandboxEnvironment`).

## Overview

- **FileOperator**: Abstraction for file system operations
- **Shell**: Abstraction for command execution
- **ResourceRegistry**: Type-safe runtime resource management
- **Lifecycle hooks**: `_setup()` / `_teardown()` pattern for subclasses

```
Environment (ABC) - Long-lived, owns resources
  +-- _resources: ResourceRegistry
  +-- _file_operator: FileOperator
  +-- _shell: Shell
  +-- _toolsets: list[AbstractToolset]
  +-- _setup() -> None           # Subclass hook: initialization
  +-- _teardown() -> None        # Subclass hook: cleanup

AgentContext - Short-lived, references Environment resources
```

## Basic Usage

### Recommended: create_agent

```python
from ya_agent_sdk.agents import create_agent, stream_agent

# Default: uses LocalEnvironment
runtime = create_agent("openai:gpt-4")
async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

### Manual Environment Management

```python
from ya_agent_sdk.environment import LocalEnvironment
from ya_agent_sdk.context import AgentContext

async with LocalEnvironment(allowed_paths=[path]) as env:
    async with AgentContext(env=env) as ctx:
        await ctx.file_operator.read_file("test.txt")
```

## Resource Management

`ResourceRegistry` provides type-safe resource management with protocol validation:

```python
# Register resources
registry.set("database", database)  # Must implement Resource protocol (close())

# Access resources
database = registry.get_typed("database", DatabasePool)

# Collect toolsets from all resources
toolsets = await registry.get_toolsets()

# Cleanup (called automatically by Environment)
await registry.close_all()
```

> Full API: [y-agent-environment](https://github.com/wh1isper/y-agent-environment)

## Creating Custom Environments

Implement `_setup` and `_teardown` hooks instead of overriding `__aenter__`/`__aexit__`:

```python
class MyEnvironment(Environment):
    async def _setup(self) -> None:
        self._file_operator = LocalFileOperator(...)
        self._shell = LocalShell(...)
        db = await Database.connect(...)
        self._resources.set("db", db)

    async def _teardown(self) -> None:
        self._file_operator = None
        self._shell = None
        # resources.close_all() called automatically after this
```

**Why hooks instead of __aenter__/__aexit__?**

- Safe inheritance without `await super().__aenter__()` concerns
- Base class ensures `resources.close_all()` is always called

## Available Implementations

### LocalEnvironment

```python
LocalEnvironment(
    allowed_paths=[Path("/workspace")],
    default_path=Path("/workspace"),
    shell_timeout=30.0,
)
```

### SandboxEnvironment

Sandbox environment with virtual file operations and containerized shell.
Both file operations and shell commands see the same path space (e.g., `/workspace`).

```python
# Single mount with Docker
SandboxEnvironment(
    mounts=[VirtualMount(Path("/home/user/project"), Path("/workspace"))],
    image="python:3.11",
    cleanup_on_exit=True,
)

# Multiple mounts
SandboxEnvironment(
    mounts=[
        VirtualMount(Path("/home/user/project"), Path("/workspace/project")),
        VirtualMount(Path("/home/user/.config"), Path("/workspace/.config")),
    ],
    work_dir="/workspace/project",
    image="python:3.11",
)

# With custom shell backend
SandboxEnvironment(
    mounts=[VirtualMount(Path("/home/user/project"), Path("/workspace"))],
    shell=my_custom_shell,
)
```

## Environment Toolsets

Environments can provide pydantic-ai toolsets via the `get_toolsets()` method:

```python
class ContainerEnvironment(Environment):
    async def _setup(self) -> None:
        # ... setup file_operator, shell ...
        container_toolset = FunctionToolset()

        @container_toolset.tool
        def get_container_status() -> str:
            return "running"

        self._toolsets = [container_toolset]
```

`create_agent` automatically includes `env.get_toolsets()`.

## Resource-Provided Toolsets

Resources can provide their own toolsets via `get_toolsets()`. This enables better encapsulation where a resource owns both its state and the tools that operate on it:

```python
from y_agent_environment import BaseResource

class ProcessManager(BaseResource):
    async def setup(self) -> None:
        """Async initialization after factory creation."""
        self._processes = {}

    def get_toolsets(self) -> list[Any]:
        """Return tools for managing processes."""
        return [ProcessToolset(self)]

    async def close(self) -> None:
        await self._kill_all_processes()
```

Collect toolsets from all resources via `ResourceRegistry.get_toolsets()`:

```python
async with LocalEnvironment() as env:
    # Register and create resource
    env.resources.register_factory("process_manager", create_pm)
    await env.resources.get_or_create("process_manager")

    # Get toolsets from all resources
    resource_toolsets = env.resources.get_toolsets()

    # Combine with environment toolsets (includes resource toolsets)
    all_toolsets = [*core_toolsets, *env.get_toolsets()]
```

## Resumable Resources

Resources can be exported and restored across process restarts using factories.

### Using BaseResource (Recommended)

`BaseResource` is a convenience abstract class with async `close()` and default export/restore:

```python
from y_agent_environment import BaseResource

class ApiClientSession(BaseResource):
    def __init__(self, client: ApiClient):
        self._client = client

    async def setup(self) -> None:
        """Async initialization (called after factory, before restore_state)."""
        await self._client.connect()

    async def close(self) -> None:
        await self._client.close()

    async def export_state(self) -> dict[str, Any]:
        return {"cursor": self._client.cursor}

    async def restore_state(self, state: dict[str, Any]) -> None:
        self._client.cursor = state.get("cursor")

    def get_toolsets(self) -> list[Any]:
        """Provide tools backed by this API client."""
        return [ApiClientToolset(self._client)]
```

### Using Factories

Factory functions receive the `Environment` instance, allowing access to `file_operator`, `shell`, `resources`, and `tmp_dir`:

```python
from y_agent_environment import Environment

async def create_api_client(env: Environment) -> ApiClientSession:
    return ApiClientSession(
        client=ApiClient(cache_dir=env.tmp_dir),
    )

# First run: create and export
async with LocalEnvironment() as env:
    env.resources.register_factory("api_client", create_api_client)
    api_client = await env.resources.get_or_create("api_client")
    # ... use api_client ...
    state = await env.export_resource_state()
    Path("state.json").write_text(state.model_dump_json())

# Later: restore from state
state = ResourceRegistryState.model_validate_json(Path("state.json").read_text())
async with LocalEnvironment(
    resource_state=state,
    resource_factories={"api_client": create_api_client},
) as env:
    api_client = env.resources.get("api_client")  # Already restored
```

### Chaining API

```python
env = (LocalEnvironment()
    .with_resource_factory("api_client", create_api_client)
    .with_resource_state(state))
```

> Non-resumable resources (without `export_state`/`restore_state`) are silently skipped during export.

## See Also

- [context.md](context.md) - AgentContext and session management
- [toolset.md](toolset.md) - Toolset architecture
- [resumable-resources.md](resumable-resources.md) - Full resumable resources documentation
