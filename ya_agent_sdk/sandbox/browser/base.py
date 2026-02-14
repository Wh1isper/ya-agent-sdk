"""Browser sandbox implementations for isolated browser execution."""

from abc import ABC, abstractmethod
from types import TracebackType


class BrowserSandbox(ABC):
    """Abstract base class for browser sandbox implementations.

    Supports async context manager protocol for automatic resource management.

    Example:
        async with DockerBrowserSandbox() as browser_url:
            # Use browser_url for operations
            pass
        # Browser automatically stopped when exiting context
    """

    @abstractmethod
    async def start_browser(self) -> str:
        """Start the browser in the sandbox environment.

        Returns:
            str: The browser URL or connection endpoint.
        """

    @abstractmethod
    async def stop_browser(self) -> None:
        """Stop the browser in the sandbox environment."""

    async def __aenter__(self) -> str:
        """Enter async context manager.

        Returns:
            str: The browser URL returned from start_browser().
        """
        return await self.start_browser()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager.

        Ensures browser is stopped even if an exception occurs.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        await self.stop_browser()
