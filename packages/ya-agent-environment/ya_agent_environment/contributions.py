"""Agent-facing contribution values exposed by an entered Environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentContributionGroup:
    """An ordered capability group with authoritative source provenance.

    Environment remains independent from Pydantic AI, so capability values are opaque at
    this boundary. The SDK validates them after the Environment has entered and restored
    its resources.
    """

    source_id: str
    capabilities: tuple[Any, ...]
