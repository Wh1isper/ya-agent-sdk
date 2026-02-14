"""Browser session state management."""

from __future__ import annotations

from typing import Any

from cdp_use import CDPClient
from pydantic import BaseModel, Field


class BrowserSession(BaseModel):
    """Browser session state shared across tools.

    Tools should use session.cdp_client.send.{Domain}.{method}() directly
    to get proper type hints from cdp-use library.
    """

    cdp_client: CDPClient
    page: str  # CDP session_id for the target

    # Current page state
    current_url: str = ""
    current_title: str = ""
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1280, "height": 720})

    # Navigation history
    navigation_history: list[str] = Field(default_factory=list)

    # Runtime cache
    last_screenshot_timestamp: float = 0.0
    cached_elements: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def dispose(self) -> None:
        """Clean up session resources."""
        self.cached_elements.clear()
