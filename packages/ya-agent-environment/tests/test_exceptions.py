"""Tests for exception classes."""

from ya_agent_environment.exceptions import (
    EnvironmentNotEnteredError,
    FileOperationError,
    PathNotAllowedError,
    ShellExecutionError,
    ShellTimeoutError,
)


def test_path_not_allowed_error() -> None:
    """PathNotAllowedError should format message correctly."""
    error = PathNotAllowedError("/forbidden/path", ["/allowed1", "/allowed2"])
    assert "/forbidden/path" in str(error)
    assert "/allowed1" in str(error)
    assert "/allowed2" in str(error)
    assert error.path == "/forbidden/path"
    assert error.allowed_paths == ["/allowed1", "/allowed2"]


def test_path_not_allowed_error_no_paths() -> None:
    """PathNotAllowedError should work without allowed_paths."""
    error = PathNotAllowedError("/forbidden/path")
    assert "/forbidden/path" in str(error)
    assert error.allowed_paths == []


def test_file_operation_error() -> None:
    """FileOperationError should format message correctly."""
    error = FileOperationError("read", "/test/file.txt", "Permission denied")
    assert "read" in str(error)
    assert "/test/file.txt" in str(error)
    assert "Permission denied" in str(error)
    assert error.operation == "read"
    assert error.path == "/test/file.txt"
    assert error.reason == "Permission denied"


def test_file_operation_error_no_reason() -> None:
    """FileOperationError should work without reason."""
    error = FileOperationError("write", "/test/file.txt")
    assert "write" in str(error)
    assert "/test/file.txt" in str(error)
    assert error.reason is None


def test_shell_execution_error() -> None:
    """ShellExecutionError should format message correctly."""
    error = ShellExecutionError("ls -la", exit_code=1, stderr="Permission denied")
    assert "ls -la" in str(error)
    assert "1" in str(error)
    assert "Permission denied" in str(error)
    assert error.command == "ls -la"
    assert error.exit_code == 1
    assert error.stderr == "Permission denied"


def test_shell_execution_error_minimal() -> None:
    """ShellExecutionError should work with minimal args."""
    error = ShellExecutionError("command")
    assert "command" in str(error)
    assert error.exit_code is None
    assert error.stderr is None


def test_shell_timeout_error() -> None:
    """ShellTimeoutError should format message correctly."""
    error = ShellTimeoutError("long_command", timeout=30.0)
    assert "long_command" in str(error)
    assert "30" in str(error)
    assert "timed out" in str(error)
    assert error.timeout == 30.0
    assert error.command == "long_command"


def test_environment_not_entered_error() -> None:
    """EnvironmentNotEnteredError should format message correctly."""
    error = EnvironmentNotEnteredError("file_operator")
    assert "file_operator" in str(error)
    assert "not entered" in str(error)
    assert error.resource == "file_operator"
