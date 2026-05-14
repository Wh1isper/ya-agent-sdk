"""Capability helpers for agent construction."""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability, ProcessHistory


def is_process_history_for(capability: AbstractCapability[Any], processor: object) -> bool:
    """Return whether a capability wraps the given history processor callable."""
    return isinstance(capability, ProcessHistory) and capability.processor is processor
