"""Background shell results injection filter.

This filter consumes completed background shell process results and
injects them into the conversation, along with a status summary of
all background processes.

Results are injected as UserPromptPart into the last ModelRequest.
Large output is truncated and full content is written to tmp files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from html import escape as _html_escape

from ya_agent_environment import CompletedProcess
from ya_agent_environment.output import truncate_text_head_tail

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import BackgroundShellCompleteEvent
from ya_agent_sdk.toolsets.core._output import DEFAULT_OUTPUT_TRUNCATE_LIMIT, write_tmp_output

logger = get_logger(__name__)

# Truncation limit for injected output (per stream)
_INJECT_TRUNCATE_LIMIT = DEFAULT_OUTPUT_TRUNCATE_LIMIT


def _xml_escape(s: str, *, quote: bool = False) -> str:
    """Escape XML-special characters.

    Args:
        s: String to escape.
        quote: If True, also escape quote characters (for attributes).
    """
    return _html_escape(s, quote=quote)


def _format_stream(tag: str, content: str, *, source_capped: bool) -> str:
    """Format a stdout/stderr stream element, retaining its head and tail when truncated."""
    if len(content) > _INJECT_TRUNCATE_LIMIT:
        saved_label = "stored output" if source_capped else "full output"
        marker = f"\n...(truncated, {saved_label} at `{tag}_file_path`)...\n"
        preview = truncate_text_head_tail(content, _INJECT_TRUNCATE_LIMIT, marker=marker)
        return f'  <{tag} truncated="true">\n{_xml_escape(preview)}\n  </{tag}>'
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
    context: AgentContext,
) -> list[str]:
    """Write available output to tmp files for truncated streams. Returns path info lines."""
    path_lines: list[str] = []
    label = "Stored" if result.truncated else "Full"
    if len(result.stdout) > _INJECT_TRUNCATE_LIMIT:
        path = await write_tmp_output(
            context,
            prefix=f"bg-stdout-{result.process_id}",
            content=result.stdout,
            extension="log",
        )
        if path is not None:
            path_lines.append(f"  {label} stdout: {path}")
    if len(result.stderr) > _INJECT_TRUNCATE_LIMIT:
        path = await write_tmp_output(
            context,
            prefix=f"bg-stderr-{result.process_id}",
            content=result.stderr,
            extension="log",
        )
        if path is not None:
            path_lines.append(f"  {label} stderr: {path}")
    return path_lines


async def _guard_injected_output(
    content: str,
    completed: list[CompletedProcess],
    summary: str | None,
    context: AgentContext,
) -> str:
    """Bound the aggregate completion injection and preserve retained results."""
    if len(content) <= _INJECT_TRUNCATE_LIMIT:
        return content

    serialized = json.dumps(
        {
            "completed": [asdict(result) for result in completed],
            "status_summary": summary,
        },
        default=str,
        ensure_ascii=False,
    )
    output_path = await write_tmp_output(
        context,
        prefix="background-shell-results",
        content=serialized,
        extension="json",
    )
    marker = (
        f"\n...(truncated; retained background results saved in `{output_path}`)...\n"
        if output_path is not None
        else "\n...(truncated; failed to save retained background results)...\n"
    )
    return truncate_text_head_tail(content, _INJECT_TRUNCATE_LIMIT, marker=marker)


async def consume_background_results(context: AgentContext) -> str | None:
    """Consume and format completed shell work for canonical feature input."""
    shell = context.shell
    if shell is None:
        return None

    completed = shell.consume_completed_results()
    if not completed:
        return None
    summary = shell.background_status_summary()

    injection_parts: list[str] = []
    for result in completed:
        formatted = _format_completed_result(result)
        path_lines = await _write_truncated_files(result, context)
        if path_lines:
            formatted += "\n" + "\n".join(path_lines)
        injection_parts.append(formatted)
        await context.emit_event(
            BackgroundShellCompleteEvent(
                event_id=f"bg-{result.process_id}",
                process_id=result.process_id,
                command=result.command,
                exit_code=result.exit_code,
            )
        )

    if summary:
        injection_parts.append(summary)
    logger.debug("Collected %d background result(s)", len(completed))
    content = "\n\n".join(injection_parts)
    return await _guard_injected_output(content, completed, summary, context)
