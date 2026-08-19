"""Protocol definitions for environment module.

This module defines runtime-checkable protocols for resources and operators.
"""

from typing import Any, Protocol, runtime_checkable

# Default chunk size for streaming operations (64KB)
DEFAULT_CHUNK_SIZE = 65536


@runtime_checkable
class Resource(Protocol):
    """Protocol for resources managed by Environment.

    Resources must implement a close() method that can be either
    synchronous or asynchronous. The Environment will call close()
    during cleanup.

    Resources can optionally provide agent capabilities via get_capabilities().
    The default implementation returns an empty tuple.

    Example:
        class DatabaseConnection:
            async def close(self) -> None:
                await self._pool.close()

            def get_capabilities(self) -> tuple[Any, ...]:
                return (self._database_capability,)

        class FileHandle:
            def close(self) -> None:
                self._handle.close()

            def get_capabilities(self) -> tuple[Any, ...]:
                return ()
    """

    def close(self) -> object:
        """Close the resource. Can be sync or async."""
        ...

    async def setup(self) -> None:
        """Initialize the resource after creation.

        Called by ResourceRegistry after factory creation, before restore_state().
        Use for async initialization like starting processes or establishing connections.
        """
        ...

    def get_capabilities(self) -> tuple[Any, ...]:
        """Return ordered opaque agent capabilities provided by this resource."""
        ...


@runtime_checkable
class ResumableResource(Resource, Protocol):
    """Protocol for resources that support state export/restore.

    Resources implementing this protocol can have their state serialized
    and restored across process restarts. The factory pattern ensures
    resources are properly initialized before state restoration.

    Example:
        class BrowserSession:
            def __init__(self, browser: Browser):
                self._browser = browser
                self._cookies: list[dict] = []

            async def export_state(self) -> dict[str, Any]:
                # May need to fetch current state from browser
                self._cookies = await self._browser.get_cookies()
                return {"cookies": self._cookies}

            async def restore_state(self, state: dict[str, Any]) -> None:
                self._cookies = state.get("cookies", [])
                await self._browser.set_cookies(self._cookies)

            def close(self) -> None:
                self._browser.close()
    """

    async def export_state(self) -> dict[str, Any]:
        """Export resource state for serialization.

        Returns:
            Dictionary of JSON-serializable state data.
            Should NOT include sensitive data (passwords, tokens, API keys).
        """
        ...

    async def restore_state(self, state: dict[str, Any]) -> None:
        """Restore resource from serialized state.

        Called after the resource is created via factory.
        Should restore the resource to the state it was in when
        export_state() was called.

        Args:
            state: State dictionary from export_state().

        Raises:
            ValueError: If state is invalid or incompatible.
        """
        ...


@runtime_checkable
class InstructableResource(Resource, Protocol):
    """Protocol for resources that provide context instructions.

    Resources implementing this protocol can contribute instructions
    to the environment context, which will be included in the agent's
    system prompt.

    Example:
        class BrowserSession:
            async def get_context_instructions(self) -> str | None:
                return "Browser session is active. Use browser tools for web tasks."

            def close(self) -> None:
                self._browser.close()
    """

    async def get_context_instructions(self) -> str | None:
        """Return context instructions for this resource.

        Returns:
            Instructions string to include in environment context,
            or None if no instructions.
        """
        ...
