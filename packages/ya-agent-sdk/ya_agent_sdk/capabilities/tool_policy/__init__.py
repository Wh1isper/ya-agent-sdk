"""Native tool policy capability leaves and preset."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability

from ya_agent_sdk.context import AgentContext

from .policy import (
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolRetryCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
)


@dataclass
class ToolExecutionPolicyCapability(CombinedCapability[AgentContext]):
    """Reference policy preset over independent native hook phases."""

    capabilities: Sequence[AbstractCapability[AgentContext]] = field(
        default_factory=lambda: [
            ToolVisibilityCapability(),
            ToolApprovalCapability(),
            ToolObservationCapability(),
            ToolRetryCapability(),
            ToolTimeoutCapability(),
        ]
    )
    id: str | None = "tool_execution_policy"


__all__ = [
    "ToolApprovalCapability",
    "ToolExecutionPolicyCapability",
    "ToolObservationCapability",
    "ToolRetryCapability",
    "ToolTimeoutCapability",
    "ToolVisibilityCapability",
]
