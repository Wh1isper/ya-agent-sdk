"""Shell command execution tools.

This module provides tools for executing shell commands
using the shell provided by AgentContext, including
background process management (start, wait, kill).
"""

from typing import Annotated, Any, cast

from pydantic import Field
from pydantic_ai import ApprovalRequired, RunContext
from typing_extensions import TypedDict
from ya_agent_environment import Shell

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalFileOperator, LocalShell
from ya_agent_sdk.environment.sandbox import DockerShell
from ya_agent_sdk.events import BackgroundShellKilledEvent, BackgroundShellStartEvent
from ya_agent_sdk.toolsets.core._output import (
    append_guidance,
    dump_tool_output,
    fit_text_fields_to_limit,
    output_too_large_message,
    tool_output_size,
    write_tmp_output,
)
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.shell.review import (
    ShellReviewBlockedResult,
    ShellReviewContextSnapshot,
    ShellReviewRecord,
    ShellReviewRequest,
    get_previous_shell_reviews,
    review_shell_command,
)

logger = get_logger(__name__)

OUTPUT_TRUNCATE_LIMIT = 20000
DEFAULT_TIMEOUT_SECONDS = 180
SHELL_REVIEW_HISTORY_LIMIT = 10


class ShellResult(TypedDict, total=False):
    """Result of shell command execution."""

    stdout: str
    stderr: str
    return_code: int
    process_id: str  # Present when background=True
    stdout_file_path: str  # Present when stdout exceeds limit
    stderr_file_path: str  # Present when stderr exceeds limit
    output_file_path: str  # Present when the full structured result exceeds limit
    truncated: bool
    error: str  # Present on execution error
    hint: str  # Guidance on next available actions


def _merge_shell_environment(
    ctx: AgentContext,
    environment: dict[str, str] | None,
) -> dict[str, str] | None:
    """Merge context shell env with per-call shell env."""
    shell_env = ctx.shell_env
    if shell_env or environment:
        return {**shell_env, **(environment or {})}
    return environment


def _shell_context(shell: Shell | None) -> tuple[str | None, list[str], str | None, str | None]:
    """Return shell cwd, allowed paths, platform, and executable for review context."""
    if isinstance(shell, LocalShell):
        return (
            str(shell._default_cwd) if shell._default_cwd is not None else None,
            [str(path) for path in shell._allowed_paths],
            shell._platform_name,
            shell._shell_executable,
        )
    if isinstance(shell, DockerShell):
        return (shell._container_workdir, [shell._container_workdir], "docker", None)
    if isinstance(shell, Shell):
        return (
            str(shell._default_cwd) if shell._default_cwd is not None else None,
            [str(path) for path in shell._allowed_paths],
            None,
            None,
        )
    return None, [], None, None


def _file_operator_context(file_operator: object) -> tuple[str | None, list[str]]:
    """Return file operator default path and allowed paths for review context."""
    if isinstance(file_operator, LocalFileOperator):
        return (
            str(file_operator._default_path) if file_operator._default_path is not None else None,
            [str(path) for path in file_operator._allowed_paths],
        )
    return None, []


def _build_shell_review_context(
    run_ctx: RunContext[AgentContext],
    *,
    timeout_seconds: int,
    tool_call_id: str | None,
) -> ShellReviewContextSnapshot:
    """Build compact execution context for shell review."""
    shell_default_cwd, shell_allowed_paths, shell_platform, shell_executable = _shell_context(run_ctx.deps.shell)
    file_default_path, file_allowed_paths = _file_operator_context(run_ctx.deps.file_operator)
    tool_call_approved = run_ctx.tool_call_approved if isinstance(run_ctx.tool_call_approved, bool) else False

    return ShellReviewContextSnapshot(
        timeout_seconds=timeout_seconds,
        tool_call_id=tool_call_id,
        tool_call_approved=tool_call_approved,
        default_cwd=shell_default_cwd or file_default_path,
        allowed_paths=shell_allowed_paths or file_allowed_paths,
        shell_platform=shell_platform,
        shell_executable=shell_executable,
    )


async def _review_shell_command_or_block(
    run_ctx: RunContext[AgentContext],
    *,
    command: str,
    cwd: str | None,
    background: bool,
    environment_keys: list[str],
    timeout_seconds: int,
) -> ShellResult | None:
    """Review a shell command and return a blocked result when policy denies execution."""
    ctx = run_ctx.deps
    tool_call_id = run_ctx.tool_call_id if isinstance(run_ctx.tool_call_id, str) else None
    request = ShellReviewRequest(
        command=command,
        cwd=cwd,
        background=background,
        environment_keys=environment_keys,
        context_snapshot=_build_shell_review_context(
            run_ctx,
            timeout_seconds=timeout_seconds,
            tool_call_id=tool_call_id,
        ),
    )
    tool_call_approved = run_ctx.tool_call_approved if isinstance(run_ctx.tool_call_approved, bool) else False
    if tool_call_approved:
        records = [record for record in ctx.shell_review_records if isinstance(record, ShellReviewRecord)]
        fingerprint = request.command_fingerprint()
        for record in reversed(records):
            if tool_call_id is not None and record.tool_call_id == tool_call_id:
                record.approved = True
                break
        else:
            for record in reversed(records):
                if record.request.command_fingerprint() == fingerprint:
                    record.approved = True
                    break
        logger.info("Shell review approval replay bypassed reviewer")
        return None

    request.previous_reviews = get_previous_shell_reviews(ctx, request, tool_call_id=tool_call_id)
    review = await review_shell_command(ctx, request=request, usage_uuid=tool_call_id)
    review_record = ShellReviewRecord(request=request, decision=review, tool_call_id=tool_call_id)
    ctx.shell_review_records.append(review_record)
    if not review.requires_approval(ctx):
        review_record.approved = True
        return None

    logger.info(
        "Shell review requested approval command_chars=%d risk_level=%s reason=%s",
        len(command),
        review.risk_level,
        review.reason,
    )
    metadata = request.to_approval_metadata(review)
    if review.requires_defer(ctx):
        logger.info("Shell review deferring command for approval risk_level=%s", review.risk_level)
        raise ApprovalRequired(metadata=metadata)
    if review.requires_deny(ctx):
        logger.warning("Shell review blocked command risk_level=%s reason=%s", review.risk_level, review.reason)
        blocked = ShellReviewBlockedResult(
            error=f"Shell command blocked by review: {review.reason}",
            shell_review=review,
        )
        return cast(ShellResult, blocked.model_dump(mode="json"))
    review_record.approved = True
    return None


async def _start_background_shell_command(
    ctx: AgentContext,
    shell: Shell,
    *,
    command: str,
    cwd: str | None,
    environment: dict[str, str] | None,
) -> ShellResult:
    """Start a background shell command."""
    try:
        process_id = await shell.start(command, env=environment, cwd=cwd)
        await ctx.emit_event(
            BackgroundShellStartEvent(
                event_id=f"bg-{process_id}",
                process_id=process_id,
                command=command,
            )
        )
        return ShellResult(
            stdout="",
            stderr="",
            return_code=-1,
            process_id=process_id,
            hint=(
                f"Background process started (id={process_id}). "
                "Use shell_wait to poll/wait for output, "
                "shell_input to send stdin, "
                "shell_kill to terminate."
            ),
        )
    except Exception as e:
        return ShellResult(
            stdout="",
            stderr="",
            return_code=1,
            error=f"Failed to start background command: {e}",
        )


async def _spill_large_shell_streams(
    file_op: Any,
    result: dict[str, Any],
    *,
    prefix: str,
) -> None:
    """Save oversized stdout/stderr streams individually for compatibility."""
    for field in ("stdout", "stderr"):
        value = result.get(field)
        if not isinstance(value, str) or len(value) <= OUTPUT_TRUNCATE_LIMIT:
            continue
        output_path = await write_tmp_output(
            file_op,
            prefix=f"{prefix}-{field}",
            content=value,
            extension="log",
        )
        if output_path is not None:
            result[f"{field}_file_path"] = output_path


async def _guard_shell_result(
    ctx: AgentContext,
    result: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Guarantee shell result previews stay under the shared output limit."""
    file_op = ctx.file_operator
    await _spill_large_shell_streams(file_op, result, prefix=prefix)
    if tool_output_size(result) <= OUTPUT_TRUNCATE_LIMIT:
        return result

    full_result = dump_tool_output(result)
    output_path = await write_tmp_output(
        file_op,
        prefix=prefix,
        content=full_result,
        extension="json",
    )

    preview = dict(result)
    preview["truncated"] = True
    if output_path is not None:
        preview["output_file_path"] = output_path

    guidance = output_too_large_message(
        size=len(full_result),
        output_path=output_path,
        noun="shell result",
    )
    hint = preview.get("hint")
    preview["hint"] = append_guidance(hint if isinstance(hint, str) else None, guidance)

    suffix = (
        "\n...(truncated; full output saved in `output_file_path`)"
        if output_path is not None
        else "\n...(truncated; failed to save full output)"
    )
    return fit_text_fields_to_limit(
        preview,
        text_fields=("stdout", "stderr"),
        limit=OUTPUT_TRUNCATE_LIMIT,
        suffix=suffix,
    )


async def _execute_foreground_shell_command(
    ctx: AgentContext,
    shell: Shell,
    *,
    command: str,
    timeout_seconds: int,
    cwd: str | None,
    environment: dict[str, str] | None,
) -> ShellResult:
    """Execute a foreground shell command."""
    try:
        exit_code, stdout, stderr = await shell.execute(
            command,
            timeout=float(timeout_seconds),
            env=environment,
            cwd=cwd,
        )

        result = ShellResult(
            stdout=stdout,
            stderr=stderr,
            return_code=exit_code,
        )
        return cast(ShellResult, await _guard_shell_result(ctx, cast(dict[str, Any], result), prefix="shell-exec"))

    except Exception as e:
        return ShellResult(
            stdout="",
            stderr="",
            return_code=1,
            error=f"Failed to execute command: {e}",
        )


class ShellTool(BaseTool):
    """Tool for executing shell commands."""

    name = "shell_exec"
    description = "Execute a shell command."
    tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires shell)."""
        if ctx.deps.shell is None:
            logger.debug("ShellTool unavailable: shell is not configured")
            return False
        return True

    async def call(
        self,
        ctx: RunContext[AgentContext],
        command: Annotated[str, Field(description="The shell command to execute.")],
        timeout_seconds: Annotated[
            int,
            Field(
                default=DEFAULT_TIMEOUT_SECONDS,
                description="Maximum execution time in seconds.",
            ),
        ] = DEFAULT_TIMEOUT_SECONDS,
        environment: Annotated[
            dict[str, str] | None,
            Field(description="Environment variables to set for the command."),
        ] = None,
        cwd: Annotated[
            str | None,
            Field(description="Working directory (relative or absolute path)."),
        ] = None,
        background: Annotated[
            bool,
            Field(
                default=False,
                description="Run command in background. Returns immediately with a process_id. "
                "Use shell_wait to check results, shell_kill to terminate.",
            ),
        ] = False,
    ) -> ShellResult:
        if not command or not command.strip():
            return ShellResult(
                stdout="",
                stderr="",
                return_code=1,
                error="Command cannot be empty.",
            )

        shell = cast(Shell, ctx.deps.shell)
        environment = _merge_shell_environment(ctx.deps, environment)

        blocked_result = await _review_shell_command_or_block(
            ctx,
            command=command,
            cwd=cwd,
            background=background,
            environment_keys=sorted((environment or {}).keys()),
            timeout_seconds=timeout_seconds,
        )
        if blocked_result is not None:
            return blocked_result

        # Background mode: start and return immediately
        if background:
            return await _start_background_shell_command(
                ctx.deps,
                shell,
                command=command,
                cwd=cwd,
                environment=environment,
            )

        # Foreground mode: execute and wait
        return await _execute_foreground_shell_command(
            ctx.deps,
            shell,
            command=command,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
            environment=environment,
        )


class ShellWaitResult(TypedDict, total=False):
    """Result of waiting for a background process."""

    stdout: str
    stderr: str
    return_code: int
    is_running: bool  # True when process is still running
    process_id: str
    stdout_file_path: str
    stderr_file_path: str
    output_file_path: str
    truncated: bool
    error: str
    hint: str  # Guidance on next available actions


class ShellWaitTool(BaseTool):
    """Tool for waiting on a background shell process."""

    name = "shell_wait"
    description = (
        "Wait for a background shell process. "
        "Set timeout_seconds=0 to poll (drain current output without waiting). "
        "Use shell_status to list process IDs."
    )
    tags = frozenset({"shell"})
    superseded_by_tags: frozenset[str] = frozenset()

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.shell is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID returned by shell with background=True.")],
        timeout_seconds: Annotated[
            int,
            Field(
                default=DEFAULT_TIMEOUT_SECONDS,
                description="Maximum seconds to wait. 0 means poll (drain output immediately). "
                "Process keeps running if timeout is exceeded.",
            ),
        ] = DEFAULT_TIMEOUT_SECONDS,
    ) -> ShellWaitResult:
        shell = cast(Shell, ctx.deps.shell)

        try:
            stdout, stderr, is_running, exit_code = await shell.wait_process(
                process_id,
                timeout=float(timeout_seconds),
            )
        except KeyError:
            return ShellWaitResult(
                process_id=process_id,
                error=f"No background process with id: {process_id}",
            )
        except Exception as e:
            return ShellWaitResult(
                process_id=process_id,
                error=f"Failed to wait for process: {e}",
            )

        result = ShellWaitResult(
            process_id=process_id,
            stdout=stdout,
            stderr=stderr,
            is_running=is_running,
            return_code=exit_code if exit_code is not None else -1,
        )

        if is_running:
            result["hint"] = (
                f"Process {process_id} is still running. "
                "Use shell_input to send stdin, "
                "shell_wait to poll again, "
                "shell_kill to terminate."
            )

        return cast(
            ShellWaitResult, await _guard_shell_result(ctx.deps, cast(dict[str, Any], result), prefix="shell-wait")
        )


class ShellKillResult(TypedDict, total=False):
    """Result of killing a background process."""

    process_id: str
    killed: bool
    stdout: str
    stderr: str
    stdout_file_path: str
    stderr_file_path: str
    output_file_path: str
    truncated: bool
    error: str
    hint: str


class ShellKillTool(BaseTool):
    """Tool for killing a background shell process."""

    name = "shell_kill"
    description = (
        "Kill a running background shell process. Returns final buffered output. Use shell_status to list process IDs."
    )
    tags = frozenset({"shell"})
    superseded_by_tags: frozenset[str] = frozenset()

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.shell is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID of the background process to kill.")],
    ) -> ShellKillResult:
        shell = cast(Shell, ctx.deps.shell)

        try:
            bg_proc = shell._background_processes.get(process_id)
            bg_command = bg_proc.command if bg_proc else ""
            stdout, stderr = await shell.kill_process(process_id)
            await ctx.deps.emit_event(
                BackgroundShellKilledEvent(
                    event_id=f"bg-{process_id}",
                    process_id=process_id,
                    command=bg_command,
                )
            )
            result = ShellKillResult(
                process_id=process_id,
                killed=True,
                stdout=stdout,
                stderr=stderr,
            )
            return cast(
                ShellKillResult,
                await _guard_shell_result(ctx.deps, cast(dict[str, Any], result), prefix="shell-kill"),
            )
        except KeyError:
            return ShellKillResult(
                process_id=process_id,
                killed=False,
                error=f"No background process with id: {process_id}",
            )
        except Exception as e:
            return ShellKillResult(
                process_id=process_id,
                killed=False,
                error=f"Failed to kill process: {e}",
            )


class ShellStatusTool(BaseTool):
    """Tool for querying background shell process status."""

    name = "shell_status"
    description = "List all background shell processes and their status (running, completed, failed)."
    tags = frozenset({"shell"})
    superseded_by_tags: frozenset[str] = frozenset()

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.shell is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
    ) -> str:
        shell = cast(Shell, ctx.deps.shell)
        summary = shell.background_status_summary_with_retained_results()
        if summary is None:
            return "No background processes."
        return summary


class ShellInputResult(TypedDict, total=False):
    """Result of writing to a background process's stdin."""

    process_id: str
    written: bool
    error: str


class ShellInputTool(BaseTool):
    """Tool for writing to a background process's stdin."""

    name = "shell_input"
    description = (
        "Write text to a background process's stdin for interactive input. "
        "Use for answering prompts, sending commands to REPLs, or piping data. "
        "Set close_stdin=true to send EOF after writing."
    )
    tags = frozenset({"shell"})
    superseded_by_tags: frozenset[str] = frozenset()

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.shell is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID of the background process.")],
        text: Annotated[str, Field(description="Text to write to stdin. A trailing newline is added automatically.")],
        close_stdin: Annotated[
            bool,
            Field(
                default=False,
                description="Close stdin after writing (sends EOF to the process).",
            ),
        ] = False,
    ) -> ShellInputResult:
        shell = cast(Shell, ctx.deps.shell)

        try:
            # Add trailing newline (simulates pressing Enter)
            data = text if text.endswith("\n") else text + "\n"
            await shell.write_stdin(process_id, data)
        except KeyError as e:
            return ShellInputResult(
                process_id=process_id,
                written=False,
                error=str(e),
            )
        except Exception as e:
            return ShellInputResult(
                process_id=process_id,
                written=False,
                error=f"Failed to write to stdin: {e}",
            )

        if close_stdin:
            await shell.close_stdin(process_id)

        return ShellInputResult(
            process_id=process_id,
            written=True,
        )


class ShellSignalResult(TypedDict, total=False):
    """Result of sending a signal to a background process."""

    process_id: str
    signal: int
    sent: bool
    error: str


class ShellSignalTool(BaseTool):
    """Tool for sending a Unix signal to a background process."""

    name = "shell_signal"
    description = (
        "Send a Unix signal to a background process. "
        "Common signals: 2 (SIGINT/Ctrl+C), 15 (SIGTERM). "
        "Use shell_kill to terminate and clean up instead."
    )
    tags = frozenset({"shell"})
    superseded_by_tags: frozenset[str] = frozenset()

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.shell is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID of the background process.")],
        signal: Annotated[
            int,
            Field(
                description="Signal number to send. Common values: 2 (SIGINT/Ctrl+C), 15 (SIGTERM), 9 (SIGKILL), 18 (SIGCONT).",
            ),
        ],
    ) -> ShellSignalResult:
        shell = cast(Shell, ctx.deps.shell)

        try:
            await shell.send_signal(process_id, signal)
        except KeyError as e:
            return ShellSignalResult(
                process_id=process_id,
                signal=signal,
                sent=False,
                error=str(e),
            )
        except Exception as e:
            return ShellSignalResult(
                process_id=process_id,
                signal=signal,
                sent=False,
                error=f"Failed to send signal {signal}: {e}",
            )

        return ShellSignalResult(
            process_id=process_id,
            signal=signal,
            sent=True,
        )
