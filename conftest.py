"""Workspace-wide test isolation for optional background services."""

from unittest.mock import MagicMock

import pytest
from genai_prices import UpdatePrices
from pydantic_ai import prices


@pytest.fixture(autouse=True)
def mock_price_updates(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Keep application tests offline; focused tests may supply a stubbed real updater."""
    updater = MagicMock(spec=UpdatePrices)
    updater.__enter__.return_value = updater
    updater.__exit__.return_value = False
    start = MagicMock(return_value=updater)
    monkeypatch.setattr(prices, "update_in_background", start)
    return start
