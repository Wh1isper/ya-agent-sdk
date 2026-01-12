# Environment Management

Resource management, lifecycle hooks, and environment implementations.

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
from pai_agent_sdk.agents import create_agent, stream_agent

# Default: uses LocalEnvironment
runtime = create_agent("openai:gpt-4")
async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

### Manual Environment Management

```python
from pai_agent_sdk.environment import LocalEnvironment
from pai_agent_sdk.context import AgentContext

async with LocalEnvironment(allowed_paths=[path]) as env:
    async with AgentContext(env=env) as ctx:
        await ctx.file_operator.read_file("test.txt")
```

## Resource Management

`ResourceRegistry` provides type-safe resource management with protocol validation:

```python
# Register resources
registry.set("browser", browser)  # Must implement Resource protocol (close())

# Access resources
browser = registry.get_typed("browser", Browser)

# Cleanup (called automatically by Environment)
await registry.close_all()
```

> Full API: `pai_agent_sdk/environment/base.py`

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

### DockerEnvironment

```python
DockerEnvironment(
    mount_dir=Path("/home/user/project"),
    container_workdir="/workspace",
    image="python:3.11",
    cleanup_on_exit=True,
)
```

## Environment Toolsets

Environments can provide pydantic-ai toolsets via the `toolsets` property:

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

`create_agent` automatically includes `env.toolsets`.

## See Also

- [context.md](context.md) - AgentContext and session management
- [toolset.md](toolset.md) - Toolset architecture
