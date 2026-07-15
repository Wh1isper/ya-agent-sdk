"""Background shell results injection filter.

This filter consumes completed background shell process results and
injects them into the conversation, along with a status summary of
all background processes.

Results are injected as UserPromptPart into the last ModelRequest.
Large output is truncated and full content is written to tmp files.
"""

from __future__ import annotations

from html import escape as _html_escape

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import RunContext
from ya_agent_environment import CompletedProcess, FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import BackgroundShellCompleteEvent

logger = get_logger(__name__)

# Truncation limit for injected output (per stream)
_INJECT_TRUNCATE_LIMIT = 20000


def _xml_escape(s: str, *, quote: bool = False) -> str:
    """Escape XML-special characters.

    Args:
        s: String to escape.
        quote: If True, also escape quote characters (for attributes).
    """
    return _html_escape(s, quote=quote)


def _format_stream(tag: str, content: str, *, source_capped: bool) -> str:
    """Format a stdout/stderr stream element, truncating if needed."""
    if len(content) > _INJECT_TRUNCATE_LIMIT:
        escaped = _xml_escape(content[:_INJECT_TRUNCATE_LIMIT])
        saved_label = "stored output" if source_capped else "full output"
        return f'  <{tag} truncated="true">\n{escaped}\n...(truncated, {saved_label} at `{tag}_file_path`)\n  </{tag}>'
    return f"  <{tag}>{_xml_escape(content)}</{tag}>"


def _format_completed_result(result: CompletedProcess) -> str:
    """Format a single completed process result for injection."""
    parts: list[str] = [
        f'<background-result process-id="{_xml_escape(result.process_id, quote=True)}" '
        f'command="{_xml_escape(result.command, quote=True)}" exit-code="{result.exit_code}">'
    ]

    if result.stdout:
        parts.append(_format_stream("stdout", result.stdout, source_capped=result.truncated))
    if result.stderr:
        parts.append(_format_stream("stderr", result.stderr, source_capped=result.truncated))
    if result.truncated:
        parts.append(
            "  <note>Output was capped for automatic injection; shell_wait can retrieve the retained terminal "
            "output while available.</note>"
        )

    parts.append("</background-result>")
    return "\n".join(parts)


async def _write_truncated_files(
    result: CompletedProcess,
    file_op: FileOperator,
) -> list[str]:
    """Write available output to tmp files for truncated streams. Returns path info lines."""
    path_lines: list[str] = []
    label = "Stored" if result.truncated else "Full"
    if len(result.stdout) > _INJECT_TRUNCATE_LIMIT:
        path = await file_op.write_tmp_file(f"bg-stdout-{result.process_id}.log", result.stdout)
        path_lines.append(f"  {label} stdout: {path}")
    if len(result.stderr) > _INJECT_TRUNCATE_LIMIT:
        path = await file_op.write_tmp_file(f"bg-stderr-{result.process_id}.log", result.stderr)
        path_lines.append(f"  {label} stderr: {path}")
    return path_lines


async def inject_background_results(
    ctx: RunContext[AgentContext],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Inject completed background shell results into the conversation.

    This filter:
    1. Consumes completed background process results (one-time)
    2. Formats each result with truncation for large output
    3. Writes the available output to tmp files when the prompt preview is truncated
    4. Appends a background status summary
    5. Injects everything as a UserPromptPart in the last ModelRequest

    Filter Order:
        Should run BEFORE inject_runtime_instructions and AFTER
        inject_bus_messages.

    Args:
        ctx: Run context containing AgentContext.
        messages: Current message history.

    Returns:
        Modified message history with injected background results.
    """
    if not messages or not isinstance(messages[-1], ModelRequest):
        return messages

    shell = ctx.deps.shell
    if shell is None:
        return messages

    completed = shell.consume_completed_results()
    summary = shell.background_status_summary()

    if not completed and not summary:
        return messages

    injection_parts: list[str] = []
    file_op = ctx.deps.file_operator

    for result in completed:
        formatted = _format_completed_result(result)
        if file_op is not None:
            path_lines = await _write_truncated_files(result, file_op)
            if path_lines:
                formatted += "\n" + "\n".join(path_lines)
        injection_parts.append(formatted)

        await ctx.deps.emit_event(
            BackgroundShellCompleteEvent(
                event_id=f"bg-{result.process_id}",
                process_id=result.process_id,
                command=result.command,
                exit_code=result.exit_code,
            )
        )

    if summary:
        injection_parts.append(summary)

    content = "\n\n".join(injection_parts)
    messages[-1].parts = [*messages[-1].parts, UserPromptPart(content=content)]

    logger.debug("Injected %d background result(s)", len(completed))
    return messages
