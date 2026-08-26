"""Default SDK capability catalog composition."""

from collections.abc import Iterable

from ya_agent_sdk.capabilities.catalog import CapabilityCatalog, CapabilityType, build_capability_catalog
from ya_agent_sdk.capabilities.features import (
    DocumentConversionCapability,
    FilesystemCapability,
    MediaReadCapability,
    NoteCapability,
    ShellCapability,
    SkillsCapability,
    TaskCapability,
    ThinkingCapability,
    UserInteractionCapability,
    WebContentCapability,
    WebSearchCapability,
)
from ya_agent_sdk.capabilities.foundation import (
    ColdStartCapability,
    ContextCompactionCapability,
    DeferredTerminalCapability,
    EnvironmentContextCapability,
    FileInspectionCapability,
    HandoffCapability,
    MediaCompatibilityCapability,
    OverallRetryBudget,
    ReasoningCompatibilityCapability,
    RuntimeContextCapability,
    ToolArgumentRepairCapability,
    ToolIdCompatibilityCapability,
)
from ya_agent_sdk.capabilities.presets import RuntimeFoundationCapability
from ya_agent_sdk.capabilities.tool_policy import (
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolSupersessionCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
)
from ya_agent_sdk.codeact import CodeActCapability

BUILTIN_CAPABILITY_TYPES = (
    CodeActCapability,
    ColdStartCapability,
    ContextCompactionCapability,
    DeferredTerminalCapability,
    DocumentConversionCapability,
    EnvironmentContextCapability,
    FileInspectionCapability,
    FilesystemCapability,
    HandoffCapability,
    MediaCompatibilityCapability,
    MediaReadCapability,
    NoteCapability,
    OverallRetryBudget,
    ReasoningCompatibilityCapability,
    RuntimeContextCapability,
    RuntimeFoundationCapability,
    ShellCapability,
    SkillsCapability,
    TaskCapability,
    ThinkingCapability,
    ToolApprovalCapability,
    ToolArgumentRepairCapability,
    ToolIdCompatibilityCapability,
    ToolObservationCapability,
    ToolSupersessionCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
    UserInteractionCapability,
    WebContentCapability,
    WebSearchCapability,
)


def build_default_capability_catalog(
    *,
    explicit_types: Iterable[CapabilityType] = (),
    selected_entry_points: Iterable[str] = (),
) -> CapabilityCatalog:
    """Build the SDK catalog with all built-in serializable capability types."""
    return build_capability_catalog(
        sdk_types=BUILTIN_CAPABILITY_TYPES,
        explicit_types=explicit_types,
        selected_entry_points=selected_entry_points,
    )
