"""Capability helpers for agent construction."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory

HISTORY_PROCESSORS_DEPRECATION_MESSAGE = (
    "`history_processors=` is deprecated; use `capabilities=[ProcessHistory(...)]` instead."
)


def is_process_history_for(capability: AbstractCapability[Any], processor: object) -> bool:
    """Return whether a capability wraps the given history processor callable."""
    return isinstance(capability, ProcessHistory) and capability.processor is processor


def history_processors_to_capabilities(
    history_processors: Sequence[HistoryProcessor[Any]] | None,
    *,
    stacklevel: int = 3,
) -> list[AbstractCapability[Any]]:
    """Convert deprecated history processors to ProcessHistory capabilities."""
    if not history_processors:
        return []
    warnings.warn(HISTORY_PROCESSORS_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=stacklevel)
    return [ProcessHistory(processor) for processor in history_processors]
