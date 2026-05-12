"""Environment abstraction for environment module.

This module provides an abstract base class for environment context managers
that manage the lifecycle of shared resources.
"""

import asyncio
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, Self

from ya_agent_environment.exceptions import EnvironmentNotEnteredError
from ya_agent_environment.file_operator import FileOperator
from ya_agent_environment.resources import ResourceFactory, ResourceRegistry, ResourceRegistryState
from ya_agent_environment.shell import Shell


class Environment(ABC):
    """Abstract base class for environment context manager.

    Environment manages the lifecycle of shared resources (file_operator, shell, resources)
    that can be reused across multiple AgentContext sessions.

    Subclasses should:
    - Call super().__init__() to initialize the resource registry
    - Implement _setup() to create file_operator, shell, and any custom resources
    - Implement _teardown() to clean up environment-specific resources
    - Optionally populate self._toolsets in _setup() to provide environment-specific tools
    - NOT override __aenter__ or __aexit__ (use _setup/_teardown instead)

    The base class handles:
    - Calling _setup() in __aenter__
    - Binding resource registry to environment (so factories can access infrastructure)
    - Calling resources.restore_all() after _setup() for resumable resources
    - Calling _teardown() then resources.close_all() in __aexit__

    Resource Factory Pattern:
        Factories receive the Environment instance, allowing access to infrastructure:

        ```python
        async def create_browser(env: Environment) -> BrowserSession:
            return BrowserSession(
                file_operator=env.file_operator,  # Access file system
                shell=env.shell,                   # Execute commands
                tmp_dir=env.tmp_dir,               # Temporary storage
            )

        async with LocalEnvironment() as env:
            env.resources.register_factory("browser", create_browser)
            browser = await env.resources.get_or_create("browser")
        ```

    Resumable Resources:
        Environment supports resource state persistence via ResourceRegistry.
        Resources implementing ResumableResource can have their state exported
        and restored across process restarts.

        Example:
            # First run
            async with LocalEnvironment() as env:
                env.resources.register_factory("browser", create_browser)
                browser = await env.resources.get_or_create("browser")
                # ... use browser ...
                state = env.export_resource_state()
                save_state(state)

            # Subsequent run
            state = load_state()
            async with LocalEnvironment(
                resource_state=state,
                resource_factories={"browser": create_browser},
            ) as env:
                # Browser automatically restored with previous state
                browser = env.resources.get("browser")

    Example:
        Using AsyncExitStack (recommended for dependent contexts):

        ```python
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            env = await stack.enter_async_context(
                LocalEnvironment(allowed_paths=[Path("/workspace")])
            )
            ctx = await stack.enter_async_context(
                AgentContext(env=env)
            )
            # Get combined toolsets from environment and resources
            toolsets = env.get_toolsets()
            agent = Agent(..., toolsets=[*core_toolsets, *toolsets])
            ...
        # Resources cleaned up when stack exits
        ```
    """

    def __init__(
        self,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            resource_state: Optional state to restore resources from.
                Resources will be restored when entering the context.
            resource_factories: Optional dictionary of resource factories.
                Required for any resources in resource_state.
        """
        self._resources = ResourceRegistry(
            state=resource_state,
            factories=resource_factories,
        )
        self._file_operator: FileOperator | None = None
        self._shell: Shell | None = None
        self._toolsets: list[Any] = []
        self._entered: bool = False
        self._enter_lock: asyncio.Lock = asyncio.Lock()

    @property
    def entered(self) -> bool:
        """Whether the environment has been entered via async context manager."""
        return self._entered

    @property
    def file_operator(self) -> FileOperator | None:
        """Return the file operator, or None if not configured.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered.

        Returns None when the subclass chooses not to provide a file operator.
        """
        if not self._entered:
            raise EnvironmentNotEnteredError("file_operator")
        return self._file_operator

    @property
    def shell(self) -> Shell | None:
        """Return the shell, or None if not configured.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered.

        Returns None when the subclass chooses not to provide a shell.
        """
        if not self._entered:
            raise EnvironmentNotEnteredError("shell")
        return self._shell

    @property
    def resources(self) -> ResourceRegistry:
        """Return the resource registry for runtime resources.

        Resources can be accessed by AgentContext and tools.
        """
        return self._resources

    def get_toolsets(self) -> list[Any]:
        """Return combined toolsets from environment and all resources.

        Collects toolsets from:
        1. Environment-level toolsets (self._toolsets, set in _setup())
        2. All registered resources via ResourceRegistry.get_toolsets()

        This is the recommended way to get all available toolsets.

        Returns:
            Combined list of toolsets from environment and resources.

        Example:
            ```python
            async with MyEnvironment() as env:
                toolsets = env.get_toolsets()
                agent = Agent(..., toolsets=[*core_toolsets, *toolsets])
            ```
        """
        toolsets = list(self._toolsets)
        toolsets.extend(self._resources.get_toolsets())
        return toolsets

    # --- Chaining API for resource factories and state ---

    def with_resource_factory(self, key: str, factory: ResourceFactory) -> "Self":
        """Register a resource factory. Can be chained.

        Args:
            key: Unique identifier for the resource.
            factory: Async callable that creates the resource.

        Returns:
            Self for method chaining.

        Example:
            env = (LocalEnvironment()
                .with_resource_factory("browser", create_browser)
                .with_resource_factory("db", create_db_pool))
        """
        self._resources.register_factory(key, factory)
        return self

    def with_resource_state(self, state: ResourceRegistryState | None) -> "Self":
        """Set resource state to restore on enter. Can be chained.

        Args:
            state: State to restore from, or None to clear pending state.

        Returns:
            Self for method chaining.

        Example:
            state = ResourceRegistryState.model_validate_json(saved_json)
            env = (LocalEnvironment()
                .with_resource_factory("browser", create_browser)
                .with_resource_state(state))
        """
        if state is not None:
            self._resources._pending_state = state
        return self

    # --- Export method ---

    async def export_resource_state(self) -> ResourceRegistryState:
        """Export resource registry state for serialization.

        Only resources implementing ResumableResource will be included.

        Returns:
            ResourceRegistryState that can be serialized to JSON.

        Example:
            state = await env.export_resource_state()
            Path("state.json").write_text(state.model_dump_json())
        """
        return await self._resources.export_state()

    # --- Subclass hooks ---

    @abstractmethod
    async def _setup(self) -> None:
        """Initialize environment resources.

        Subclasses must implement this to:
        - Optionally create and assign self._file_operator
        - Optionally create and assign self._shell
        - Optionally register custom resources via self._resources.set()

        Both file_operator and shell may be left as None if the environment
        does not need them. Tools should check availability before use.

        This is called by __aenter__.
        """
        ...

    @abstractmethod
    async def _teardown(self) -> None:
        """Clean up environment-specific resources.

        Subclasses must implement this to:
        - Clean up tmp_dir, containers, connections, etc.
        - Set self._file_operator = None
        - Set self._shell = None

        Note: self._resources.close_all() is called automatically after _teardown().
        This is called by __aexit__.
        """
        ...

    # --- Workspace isolation ---

    def fork(self) -> AbstractAsyncContextManager["Self"]:
        """Create an isolated copy of this environment.

        Returns an async context manager that yields a new Environment instance
        with its own file_operator and shell pointing to an isolated workspace
        directory. The forked environment starts with an empty resource registry.

        Subclasses that support workspace isolation should override this method.
        The mechanism for creating the isolated directory (e.g., git worktree,
        file copy) is an implementation detail of each subclass.

        Yields:
            A new Environment instance with isolated workspace.

        Raises:
            NotImplementedError: If the subclass does not support forking.

        Example::

            async with env.fork() as forked_env:
                # forked_env has its own workspace
                await forked_env.shell.execute("make test")
            # Isolated workspace is cleaned up automatically
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support fork(). "
            "Subclasses must override this method to provide workspace isolation."
        )

    # --- Fixed lifecycle management ---

    async def __aenter__(self) -> "Self":
        """Enter context and setup resources.

        This method:
        1. Calls _setup() to initialize file_operator, shell, etc.
        2. Binds the resource registry to this environment
        3. Calls resources.restore_all() to restore pending resources

        Raises:
            RuntimeError: If the environment has already been entered.
            KeyError: If pending state references a resource without factory.
        """
        async with self._enter_lock:
            if self._entered:
                raise RuntimeError(
                    f"{self.__class__.__name__} has already been entered. "
                    "Each Environment instance can only be entered once at a time."
                )
            self._entered = True
        await self._setup()

        # Bind resource registry to this environment so factories can access infrastructure
        self._resources.bind(self)

        # Restore resources from pending state (if any)
        await self._resources.restore_all()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and cleanup resources."""
        try:
            await self._teardown()
        finally:
            # Close file_operator and shell, then close all registered resources
            if self._file_operator is not None:
                await self._file_operator.close()
            if self._shell is not None:
                await self._shell.close()
            await self._resources.close_all()
            async with self._enter_lock:
                self._entered = False

    async def get_context_instructions(self) -> str:
        """Return combined context instructions from file_operator, shell, and resources.

        Subclasses can override this to provide additional environment-specific
        instructions. The default implementation combines file_operator, shell,
        and resources instructions.

        Returns:
            Combined XML-formatted instructions string wrapped in <environment-context> tags.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered yet.
        """
        if not self._entered:
            raise EnvironmentNotEnteredError("environment")

        parts: list[str] = []

        if self._file_operator is not None:
            file_instructions = await self._file_operator.get_context_instructions()
            if file_instructions:
                parts.append(file_instructions)

        if self._shell is not None:
            shell_instructions = await self._shell.get_context_instructions()
            if shell_instructions:
                parts.append(shell_instructions)

        # Collect resource instructions
        resource_instructions = await self._resources.get_context_instructions()
        if resource_instructions:
            parts.append(resource_instructions)

        if not parts:
            return ""

        content = "\n\n".join(parts)
        return f"<environment-context>\n{content}\n</environment-context>"
