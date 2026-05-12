"""Resource management for environment module.

This module provides classes for managing resources with lifecycle hooks,
state persistence, and factory-based creation.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

from ya_agent_environment.protocols import InstructableResource, Resource, ResumableResource

T = TypeVar("T")

# Forward reference for type hints - Environment is defined in environment.py
if TYPE_CHECKING:
    from ya_agent_environment.environment import Environment


ResourceFactory = Callable[["Environment"], Awaitable[Resource]]
"""Async callable that creates a Resource instance.

The factory receives the Environment instance, allowing access to:
- env.file_operator: For file system operations
- env.shell: For command execution
- env.resources: For accessing other registered resources
- env.tmp_dir: For temporary file storage (if available)

Example:
    async def create_browser(env: Environment) -> BrowserSession:
        return BrowserSession(
            file_operator=env.file_operator,
            tmp_dir=env.tmp_dir,
        )
"""


class ResourceEntry(BaseModel):
    """Serialized entry for a single resource."""

    state: dict[str, Any]


class ResourceRegistryState(BaseModel):
    """Serializable state for ResourceRegistry.

    Can be serialized to JSON and stored for session restoration.
    Only contains entries for resources that implement ResumableResource.
    """

    entries: dict[str, ResourceEntry] = Field(default_factory=dict)


class BaseResource(ABC):
    """Abstract base class for resources with default resumable support.

    Provides convenience implementation for Resource and ResumableResource protocols.
    Subclasses must implement close(), and can optionally override export_state()
    and restore_state() for resumable functionality.

    Example:
        class BrowserSession(BaseResource):
            def __init__(self, browser: Browser):
                self._browser = browser
                self._cookies: list[dict] = []

            async def close(self) -> None:
                await self._browser.close()

            async def export_state(self) -> dict[str, Any]:
                return {"cookies": await self._browser.get_cookies()}

            async def restore_state(self, state: dict[str, Any]) -> None:
                await self._browser.set_cookies(state.get("cookies", []))
    """

    @abstractmethod
    async def close(self) -> None:
        """Close the resource and release any held resources."""
        ...

    async def setup(self) -> None:  # noqa: B027
        """Initialize the resource after creation.

        Called by ResourceRegistry after factory creation, before restore_state().
        Override to perform async initialization (start browser, connect to DB).
        Default implementation does nothing.
        """
        pass  # Default: no-op

    def get_toolsets(self) -> list[Any]:
        """Return toolsets provided by this resource.

        Default implementation returns empty list.
        Override to provide actual toolsets.

        Returns:
            List of toolset instances.
        """
        return []

    async def export_state(self) -> dict[str, Any]:
        """Export resource state for serialization.

        Default implementation returns empty dict (no state to export).
        Override to export actual state.

        Returns:
            Dictionary of JSON-serializable state data.
        """
        return {}

    async def restore_state(self, state: dict[str, Any]) -> None:
        """Restore resource from serialized state.

        Default implementation does nothing.
        Override to restore actual state.

        Args:
            state: State dictionary from export_state().
        """
        _ = state  # Default: ignore state

    async def get_context_instructions(self) -> str | None:
        """Return context instructions for this resource.

        Override to provide resource-specific instructions that will be
        included in the environment context instructions.

        Returns:
            Instructions string, or None if no instructions.
        """
        return None


class ResourceRegistry:
    """Type-safe resource container with protocol validation and resumption support.

    Provides a registry for managing resources with:
    - Protocol validation on set()
    - Type-safe get operations
    - Factory-based lazy creation
    - State export/restore for resumable resources
    - Unified cleanup via close_all()

    Example (factory pattern):
        registry = ResourceRegistry()
        registry.register_factory("browser", create_browser_session)
        browser = await registry.get_or_create_typed("browser", BrowserSession)

        # Export state
        state = registry.export_state()

        # Later, restore
        new_registry = ResourceRegistry(state=state, factories={"browser": create_browser_session})
        await new_registry.restore_all()
        browser = new_registry.get_typed("browser", BrowserSession)  # Already restored
    """

    def __init__(
        self,
        state: ResourceRegistryState | None = None,
        factories: dict[str, ResourceFactory] | None = None,
    ) -> None:
        """Initialize ResourceRegistry.

        Args:
            state: Optional state to restore from. Resources will be restored
                when restore_all() is called (typically by Environment.__aenter__).
            factories: Optional dictionary of resource factories.
        """
        self._resources: dict[str, Resource] = {}
        self._factories: dict[str, ResourceFactory] = dict(factories) if factories else {}
        self._pending_state: ResourceRegistryState | None = state
        self._env: Environment | None = None

    def bind(self, env: "Environment") -> "ResourceRegistry":
        """Bind this registry to an environment.

        This is called automatically by Environment.__aenter__ after _setup().
        Once bound, factories can access the environment's infrastructure
        (file_operator, shell, resources, etc.).

        Args:
            env: The Environment instance to bind to.

        Returns:
            Self for method chaining.
        """
        self._env = env
        return self

    @property
    def env(self) -> "Environment":
        """Return the bound environment.

        Raises:
            RuntimeError: If registry is not bound to an environment.
        """
        if self._env is None:
            raise RuntimeError(
                "ResourceRegistry not bound to Environment. "
                "This usually means you're accessing resources before Environment.__aenter__ completed."
            )
        return self._env

    def register_factory(self, key: str, factory: ResourceFactory) -> None:
        """Register an async factory for a resource key.

        Factories are used by get_or_create() and restore_all() to
        create resource instances.

        Args:
            key: Unique identifier for the resource.
            factory: Async callable that creates the resource.
        """
        self._factories[key] = factory

    async def get_or_create(self, key: str) -> Resource:
        """Get existing resource or create via factory.

        If the resource exists, returns it immediately.
        If not, creates it using the registered factory.

        The factory receives the bound Environment instance, allowing access to:
        - env.file_operator: For file system operations
        - env.shell: For command execution
        - env.resources: For accessing other registered resources

        Args:
            key: Resource identifier.

        Returns:
            The resource instance.

        Raises:
            KeyError: If no resource exists and no factory is registered.
            RuntimeError: If registry is not bound to an environment.
        """
        if key in self._resources:
            return self._resources[key]

        if key not in self._factories:
            raise KeyError(f"No resource or factory registered for key: {key}")

        resource = await self._factories[key](self.env)
        await resource.setup()
        self._resources[key] = resource
        return resource

    async def get_or_create_typed(self, key: str, resource_type: type[T]) -> T:
        """Get or create resource with type casting.

        Provides better IDE support by returning the expected type.

        Args:
            key: Resource identifier.
            resource_type: Expected type of the resource.

        Returns:
            Resource cast to the expected type.

        Raises:
            KeyError: If no resource exists and no factory is registered.
            TypeError: If resource is not of the expected type.
        """
        resource = await self.get_or_create(key)
        if not isinstance(resource, resource_type):
            raise TypeError(f"Resource '{key}' is {type(resource).__name__}, expected {resource_type.__name__}")
        return resource

    async def export_state(self) -> ResourceRegistryState:
        """Export state of all resumable resources.

        Only resources implementing ResumableResource protocol will be
        included in the exported state. Other resources are skipped.

        Returns:
            ResourceRegistryState containing serialized resource states.
        """
        entries: dict[str, ResourceEntry] = {}
        for key, resource in self._resources.items():
            if isinstance(resource, ResumableResource):
                state = await resource.export_state()
                entries[key] = ResourceEntry(state=state)
        return ResourceRegistryState(entries=entries)

    async def restore_all(self) -> int:
        """Restore all resources from pending state.

        For each entry in pending state:
        1. Create resource via registered factory (factory receives Environment)
        2. Call setup() to initialize the resource
        3. Call restore_state() if resource implements ResumableResource

        This method is idempotent - calling it multiple times has no effect
        after the first successful call (pending_state is cleared).

        Returns:
            Number of resources restored.

        Raises:
            KeyError: If a pending state has no registered factory.
            ValueError: If restore_state() fails (propagated from resource).
            RuntimeError: If registry is not bound to an environment.
        """
        if self._pending_state is None:
            return 0

        count = 0
        for key, entry in self._pending_state.entries.items():
            if key not in self._factories:
                raise KeyError(f"No factory registered for pending resource: {key}")

            # Create resource via factory (pass environment)
            resource = await self._factories[key](self.env)
            await resource.setup()
            self._resources[key] = resource

            # Restore state if resumable
            if isinstance(resource, ResumableResource):
                await resource.restore_state(entry.state)

            count += 1

        self._pending_state = None
        return count

    async def restore_one(self, key: str) -> bool:
        """Restore a single resource from pending state.

        Useful for lazy restoration - restore resources only when needed.

        Args:
            key: Resource identifier to restore.

        Returns:
            True if resource was restored, False if not in pending state.

        Raises:
            KeyError: If key is in pending state but no factory is registered.
            RuntimeError: If registry is not bound to an environment.
        """
        if self._pending_state is None or key not in self._pending_state.entries:
            return False

        entry = self._pending_state.entries.pop(key)

        if key not in self._factories:
            raise KeyError(f"No factory registered for resource: {key}")

        resource = await self._factories[key](self.env)
        await resource.setup()
        self._resources[key] = resource

        if isinstance(resource, ResumableResource):
            await resource.restore_state(entry.state)

        return True

    def set(self, key: str, resource: Resource) -> None:
        """Register a resource with protocol validation.

        Args:
            key: Unique identifier for the resource.
            resource: Resource instance (must implement Resource protocol).

        Raises:
            TypeError: If resource doesn't implement Resource protocol.
        """
        if not isinstance(resource, Resource):
            raise TypeError(
                f"Resource must implement Resource protocol (have close() method), got {type(resource).__name__}"
            )
        self._resources[key] = resource

    def get(self, key: str) -> Resource | None:
        """Get a resource by key.

        Args:
            key: Resource identifier.

        Returns:
            Resource instance or None if not found.
        """
        return self._resources.get(key)

    def get_typed(self, key: str, resource_type: type[T]) -> T | None:
        """Get a resource with type casting.

        Provides better IDE support by returning the expected type.

        Args:
            key: Resource identifier.
            resource_type: Expected type of the resource.

        Returns:
            Resource cast to the expected type, or None if not found
            or type doesn't match.

        Example:
            browser = resources.get_typed("browser", Browser)
            if browser:
                await browser.screenshot(url)
        """
        resource = self._resources.get(key)
        if resource is not None and isinstance(resource, resource_type):
            return resource
        return None

    def remove(self, key: str) -> Resource | None:
        """Remove and return a resource.

        Args:
            key: Resource identifier.

        Returns:
            Removed resource or None if not found.
        """
        return self._resources.pop(key, None)

    def __contains__(self, key: str) -> bool:
        """Check if a resource exists."""
        return key in self._resources

    def __len__(self) -> int:
        """Return number of registered resources."""
        return len(self._resources)

    def keys(self) -> list[str]:
        """Return list of resource keys."""
        return list(self._resources.keys())

    async def close_all(self, *, parallel: bool = False) -> None:
        """Close all resources.

        Args:
            parallel: If True, close resources concurrently using asyncio.gather.
                If False (default), close in reverse registration order sequentially.

        Uses best-effort cleanup - continues even if individual
        resources fail to close. Handles both sync and async close().
        Also clears registered factories.
        """
        if parallel:
            # Close all resources concurrently
            async def _close_resource(resource: Resource) -> None:
                try:
                    result = resource.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: S110
                    pass  # Best effort cleanup

            await asyncio.gather(
                *[_close_resource(r) for r in self._resources.values()],
                return_exceptions=True,
            )
        else:
            # Close in reverse registration order (sequential)
            for resource in reversed(list(self._resources.values())):
                try:
                    result = resource.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: S110
                    pass  # Best effort cleanup

        self._resources.clear()
        self._factories.clear()

    def get_toolsets(self) -> list[Any]:
        """Collect toolsets from all resources.

        Iterates through all registered resources and collects their toolsets.

        Returns:
            Combined list of toolsets from all resources.
        """
        toolsets: list[Any] = []
        for resource in self._resources.values():
            toolsets.extend(resource.get_toolsets())
        return toolsets

    async def get_context_instructions(self) -> str | None:
        """Return combined context instructions from all resources in XML format.

        Collects instructions from resources that implement InstructableResource
        protocol and returns them combined in an XML structure.

        Returns:
            Combined instructions string in XML format, or None if no instructions.
        """
        root = ET.Element("resources")

        for key, resource in self._resources.items():
            if isinstance(resource, InstructableResource):
                try:
                    result = await resource.get_context_instructions()
                    if result:
                        resource_elem = ET.SubElement(root, "resource")
                        resource_elem.set("name", key)
                        resource_elem.text = "\n" + result + "\n  "
                except Exception:  # noqa: S110
                    pass  # Best effort - skip resources that fail

        # Return None if no resource elements were added
        if len(root) == 0:
            return None

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
