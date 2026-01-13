"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from pai_agent_sdk.environment.local import (
    LocalEnvironment,
    LocalFileOperator,
    LocalShell,
)

# Docker environment is optional (requires docker package)
try:
    from pai_agent_sdk.environment.docker import (  # noqa: F401
        DockerEnvironment,
        DockerShell,
    )

    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False

__all__ = [
    "LocalEnvironment",
    "LocalFileOperator",
    "LocalShell",
]

# Add Docker exports if available
if _DOCKER_AVAILABLE:
    __all__.extend(["DockerEnvironment", "DockerShell"])
