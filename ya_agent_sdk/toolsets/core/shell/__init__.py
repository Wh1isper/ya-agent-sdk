"""Shell-related tools.

Tools for executing shell commands.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.shell.shell import ShellTool

tools: list[type[BaseTool]] = [ShellTool]

__all__ = ["ShellTool", "tools"]
