"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from ya_agent_environment.environment import Environment
from ya_agent_environment.exceptions import (
    EnvironmentError as EnvironmentError,
)
from ya_agent_environment.exceptions import (
    EnvironmentNotEnteredError,
    FileOperationError,
    PathNotAllowedError,
    ShellExecutionError,
    ShellTimeoutError,
)
from ya_agent_environment.file_operator import (
    DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    DEFAULT_INSTRUCTIONS_SKIP_DIRS,
    FileOperator,
    LocalTmpFileOperator,
)
from ya_agent_environment.protocols import (
    DEFAULT_CHUNK_SIZE,
    InstructableResource,
    Resource,
    ResumableResource,
    TmpFileOperator,
)
from ya_agent_environment.resources import (
    BaseResource,
    ResourceEntry,
    ResourceFactory,
    ResourceRegistry,
    ResourceRegistryState,
)
from ya_agent_environment.shell import (
    BackgroundProcess,
    CompletedProcess,
    DeferredShell,
    ExecutionHandle,
    OutputBuffer,
    ReadyState,
    Shell,
    StdinAdapter,
)
from ya_agent_environment.types import FileStat, TruncatedResult
from ya_agent_environment.utils import generate_filetree

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_INSTRUCTIONS_MAX_DEPTH",
    "DEFAULT_INSTRUCTIONS_SKIP_DIRS",
    "BackgroundProcess",
    "BaseResource",
    "CompletedProcess",
    "DeferredShell",
    "Environment",
    "EnvironmentError",
    "EnvironmentNotEnteredError",
    "ExecutionHandle",
    "FileOperationError",
    "FileOperator",
    "FileStat",
    "InstructableResource",
    "LocalTmpFileOperator",
    "OutputBuffer",
    "PathNotAllowedError",
    "ReadyState",
    "Resource",
    "ResourceEntry",
    "ResourceFactory",
    "ResourceRegistry",
    "ResourceRegistryState",
    "ResumableResource",
    "Shell",
    "ShellExecutionError",
    "ShellTimeoutError",
    "StdinAdapter",
    "TmpFileOperator",
    "TruncatedResult",
    "generate_filetree",
]
