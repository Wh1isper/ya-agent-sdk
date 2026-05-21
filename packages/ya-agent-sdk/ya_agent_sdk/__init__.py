"""ya-agent-sdk: Production-ready SDK for building AI agents with Pydantic AI."""

import importlib.metadata

from ya_agent_sdk.mcp import MCPServerSpec, NamedMCPToolset, ProcessToolCallback, create_mcp_approval_hook
from ya_agent_sdk.media import MediaUploader, S3MediaConfig, S3MediaUploader, create_s3_media_hook
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot, UsageSnapshotEntry

__all__ = [
    "MCPServerSpec",
    "MediaUploader",
    "NamedMCPToolset",
    "ProcessToolCallback",
    "S3MediaConfig",
    "S3MediaUploader",
    "UsageAgentTotal",
    "UsageSnapshot",
    "UsageSnapshotEntry",
    "__version__",
    "create_mcp_approval_hook",
    "create_s3_media_hook",
]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
