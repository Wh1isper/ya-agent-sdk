"""Environment-related exceptions."""


class EnvironmentError(Exception):  # noqa: A001
    """Base exception for environment operations."""

    pass


class PathNotAllowedError(EnvironmentError):
    """Raised when a path is outside allowed directories."""

    def __init__(self, path: str, allowed_paths: list[str] | None = None):
        self.path = path
        self.allowed_paths = allowed_paths or []
        msg = f"Path '{path}' is not within allowed directories"
        if self.allowed_paths:
            msg += f": {', '.join(self.allowed_paths)}"
        super().__init__(msg)


class FileOperationError(EnvironmentError):
    """Raised when a file operation fails."""

    def __init__(self, operation: str, path: str, reason: str | None = None):
        self.operation = operation
        self.path = path
        self.reason = reason
        msg = f"Failed to {operation} '{path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ShellExecutionError(EnvironmentError):
    """Raised when shell command execution fails."""

    def __init__(
        self,
        command: list[str],
        exit_code: int | None = None,
        stderr: str | None = None,
    ):
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        cmd_str = " ".join(command)
        msg = f"Command failed: {cmd_str}"
        if exit_code is not None:
            msg += f" (exit code: {exit_code})"
        if stderr:
            msg += f"\n{stderr}"
        super().__init__(msg)


class ShellTimeoutError(ShellExecutionError):
    """Raised when shell command times out."""

    def __init__(self, command: list[str], timeout: float):
        self.timeout = timeout
        super().__init__(command)
        self.args = (f"Command timed out after {timeout}s: {' '.join(command)}",)


class EnvironmentNotEnteredError(EnvironmentError):
    """Raised when accessing resources before entering the environment context."""

    def __init__(self, resource: str):
        self.resource = resource
        super().__init__(f"Environment not entered. Use 'async with environment:' before accessing {resource}.")
