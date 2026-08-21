"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from ya_agent_environment.contributions import AgentContributionGroup
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
)
from ya_agent_environment.protocols import (
    DEFAULT_CHUNK_SIZE,
    InstructableResource,
    Resource,
    ResumableResource,
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
    ShellBackgroundResetError,
    ShellSessionAccessError,
    StdinAdapter,
)
from ya_agent_environment.types import FileEntry, FileStat
from ya_agent_environment.utils import generate_filetree

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_INSTRUCTIONS_MAX_DEPTH",
    "DEFAULT_INSTRUCTIONS_SKIP_DIRS",
    "AgentContributionGroup",
    "BackgroundProcess",
    "BaseResource",
    "CompletedProcess",
    "DeferredShell",
    "Environment",
    "EnvironmentError",
    "EnvironmentNotEnteredError",
    "ExecutionHandle",
    "FileEntry",
    "FileOperationError",
    "FileOperator",
    "FileStat",
    "InstructableResource",
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
    "ShellBackgroundResetError",
    "ShellExecutionError",
    "ShellSessionAccessError",
    "ShellTimeoutError",
    "StdinAdapter",
    "generate_filetree",
]
