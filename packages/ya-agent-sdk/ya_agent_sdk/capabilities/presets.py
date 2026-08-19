"""Reference capability presets with no hidden runtime injection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, ReinjectSystemPrompt

from ya_agent_sdk.context import AgentContext

from .foundation.deferred import DeferredTerminalCapability
from .foundation.history import (
    ColdStartCapability,
    ContextCompactionCapability,
    HandoffCapability,
    ReasoningCompatibilityCapability,
    ToolArgumentRepairCapability,
    ToolIdCompatibilityCapability,
)
from .foundation.request import (
    EnvironmentContextCapability,
    FileInspectionCapability,
    MediaCompatibilityCapability,
    RuntimeContextCapability,
)
from .foundation.retry import OverallRetryBudget


@dataclass
class RuntimeFoundationCapability(CombinedCapability[AgentContext]):
    """Explicit reference-host foundation for history and terminal semantics."""

    capabilities: Sequence[AbstractCapability[AgentContext]] = field(
        default_factory=lambda: [
            ReasoningCompatibilityCapability(),
            MediaCompatibilityCapability(),
            ToolArgumentRepairCapability(),
            ToolIdCompatibilityCapability(),
            OverallRetryBudget(),
            HandoffCapability(),
            ContextCompactionCapability(),
            ColdStartCapability(),
            FileInspectionCapability(),
            EnvironmentContextCapability(),
            RuntimeContextCapability(),
            DeferredTerminalCapability(),
            ReinjectSystemPrompt(),
        ]
    )
    id: str | None = "runtime_foundation"
