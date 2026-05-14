"""Usage tracking models for agent token consumption.

This module provides the unified per-run usage ledger and realtime usage
snapshot models used by agents, CLI clients, and runtime services.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.usage import RunUsage

_RUN_USAGE_FIELD_NAMES = frozenset(field.name for field in fields(RunUsage))


def coerce_run_usage(usage: Any) -> RunUsage:
    """Convert Pydantic AI usage wrappers into a concrete ``RunUsage`` instance."""
    if is_dataclass(usage):
        return RunUsage(**_run_usage_data(usage))
    if callable(usage):
        usage = usage()
    return RunUsage(**_run_usage_data(usage))


def _run_usage_data(usage: Any) -> dict[str, Any]:
    if not is_dataclass(usage):
        raise TypeError(f"Expected RunUsage-compatible dataclass, got {type(usage).__name__}")

    data: dict[str, Any] = {}
    for name in _RUN_USAGE_FIELD_NAMES:
        value = getattr(usage, name)
        if name == "details":
            value = dict(value)
        data[name] = value
    return data


class UsageSnapshotEntry(BaseModel):
    """Cumulative usage for one agent/source in the current run."""

    agent_id: str
    """Agent/source instance that generated this usage (e.g., 'main', 'searcher-a1b2', 'compact')."""

    agent_name: str
    """Human-readable agent/source name (e.g., 'main', 'searcher', 'compact')."""

    model_id: str
    """Model identifier that generated this usage."""

    usage: RunUsage
    """Cumulative token usage for this agent/source instance."""

    usage_id: str | None = None
    """Stable usage record ID for idempotent updates."""

    source: str = "model_request"
    """Component that reported this usage."""


class UsageAgentTotal(BaseModel):
    """Cumulative usage grouped by agent/source."""

    agent_name: str
    model_id: str
    usage: RunUsage
    usage_id: str | None = None
    source: str = "model_request"


class UsageSnapshot(BaseModel):
    """Cumulative usage snapshot for the current run.

    Realtime consumers and billing systems treat each snapshot as a replacement
    for the previous snapshot with the same run ID.
    """

    run_id: str
    """Run identifier for the snapshot."""

    total_usage: RunUsage = Field(default_factory=RunUsage)
    """Cumulative usage across all known agents in this run."""

    entries: list[UsageSnapshotEntry] = Field(default_factory=list)
    """Per-agent/source cumulative usage entries."""

    agent_usages: dict[str, UsageAgentTotal] = Field(default_factory=dict)
    """Cumulative usage grouped by agent ID."""

    model_usages: dict[str, RunUsage] = Field(default_factory=dict)
    """Cumulative usage grouped by model identifier."""
