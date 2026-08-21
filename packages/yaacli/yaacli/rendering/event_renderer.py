"""Event rendering for agent stream events.

Provides EventRenderer for rendering agent events to display-ready output.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from ya_agent_sdk.events import (
    FileChange,
    FileChangeAction,
    FileChangeEvent,
    NamespaceStatus,
    NoteEvent,
)

from yaacli.rendering.renderer import RichRenderer
from yaacli.rendering.tool_message import ToolMessage
from yaacli.rendering.tool_panels.base import generate_unified_diff
from yaacli.rendering.tracker import ToolCallTracker


class EventRenderer:
    """Render agent stream events to display-ready output.

    Tracks text streaming and tool calls, producing Rich-rendered output.
    """

    def __init__(
        self,
        width: int | None = None,
        code_theme: str = "monokai",
        max_tool_result_lines: int = 2,
        max_arg_length: int = 50,
    ) -> None:
        self._renderer = RichRenderer(width=width)
        self._code_theme = code_theme
        self._max_tool_result_lines = max_tool_result_lines
        self._max_arg_length = max_arg_length
        self._tracker = ToolCallTracker()
        self._tool_messages: dict[str, ToolMessage] = {}

        # Current streaming text
        self._current_text: str = ""
        self._current_thinking: str = ""

    def configure_rendering(
        self,
        *,
        code_theme: str,
        max_tool_result_lines: int,
        max_arg_length: int,
    ) -> None:
        """Update rendering options without discarding in-flight event state."""
        self._code_theme = code_theme
        self._max_tool_result_lines = max_tool_result_lines
        self._max_arg_length = max_arg_length

    @property
    def tracker(self) -> ToolCallTracker:
        """Get the tool call tracker."""
        return self._tracker

    def clear(self) -> None:
        """Clear all state for new conversation turn."""
        self._tracker.clear()
        self._tool_messages.clear()
        self._current_text = ""
        self._current_thinking = ""

    def get_current_text(self) -> str:
        """Get current accumulated text content."""
        return self._current_text

    def get_current_thinking(self) -> str:
        """Get current accumulated thinking content."""
        return self._current_thinking

    def update_thinking(self, delta: str) -> None:
        """Update current thinking content with delta."""
        self._current_thinking += delta

    def start_thinking(self, content: str = "") -> None:
        """Start a new thinking block."""
        self._current_thinking = content

    def render_thinking(self, content: str | None = None, width: int | None = None) -> str:
        """Render thinking content as a styled blockquote.

        Uses dim style with '>' prefix for each line to visually distinguish
        model's internal reasoning from regular output.

        Args:
            content: Optional content to render. If None, uses current thinking.
            width: Optional render width.

        Returns:
            Rendered ANSI string.
        """
        thinking_content = content if content is not None else self._current_thinking
        if not thinking_content:
            return ""

        # Format as blockquote with dim style
        lines = thinking_content.split("\n")
        text = Text()
        for i, line in enumerate(lines):
            if i > 0:
                text.append("\n")
            text.append("> ", style="dim magenta")
            text.append(line, style="dim italic")

        return self._renderer.render(text, width=width)

    def render_tool_call_start(self, name: str, tool_call_id: str) -> str:
        """Render tool call start indicator."""
        text = Text()
        text.append("Calling: ", style="dim")
        text.append(name, style="bold cyan")
        return self._renderer.render(text)

    def render_tool_call_complete(
        self,
        tool_message: ToolMessage,
        duration: float = 0.0,
        width: int | None = None,
    ) -> str:
        """Render completed tool call.

        Special tools (edit and thinking) use Panel format.
        Normal tools use inline Text format for cleaner display.
        """
        render_width = width or 120
        if tool_message.name == "thinking":
            panel = tool_message.to_special_panel(code_theme=self._code_theme)
            return self._renderer.render(panel, width=render_width)
        else:
            # Use inline text format for normal tools
            # Calculate available width for args/output (reserve space for labels)
            max_line_len = max(50, render_width - 20)
            text = tool_message.to_inline_text(
                duration=duration,
                max_arg_length=min(self._max_arg_length, max_line_len),
                max_result_lines=self._max_tool_result_lines,
                max_line_length=max_line_len,
            )
            return self._renderer.render(text, width=render_width)

    def render_markdown(self, text: str) -> str:
        """Render markdown text."""
        return self._renderer.render_markdown(text, code_theme=self._code_theme)

    def render_text(self, text: str, style: str | None = None) -> str:
        """Render styled text."""
        return self._renderer.render_text(text, style=style)

    # =========================================================================
    # Event Panel Rendering
    # =========================================================================

    def render_mcp_unavailable(self, server_name: str, status: NamespaceStatus) -> str:
        """Render an optional MCP availability warning."""
        reason = "failed to connect" if status == NamespaceStatus.skipped else "became unavailable"
        text = Text()
        text.append("! ", style="yellow")
        text.append(f"Optional MCP server {server_name!r} {reason}; continuing without it.", style="yellow")
        return self._renderer.render(text)

    def render_compact_start(self, message_count: int) -> str:
        """Render compact start notification (single line)."""
        text = Text()
        text.append("> ", style="cyan")
        text.append(f"Context compacting {message_count} messages...", style="dim")
        return self._renderer.render(text)

    def render_compact_complete(
        self,
        original_count: int,
        compacted_count: int,
        summary: str = "",
    ) -> str:
        """Render compact complete panel."""
        reduction = int((1 - compacted_count / original_count) * 100) if original_count > 0 else 0
        content = Text()
        content.append(f"{original_count} -> {compacted_count} messages ", style="bold")
        content.append(f"({reduction}% reduction)", style="dim")
        if summary:
            content.append(summary, style="dim italic")
        panel = Panel(content, border_style="cyan", title="[cyan]Context Compacted[/cyan]", title_align="left")
        return self._renderer.render(panel)

    def render_compact_failed(self, error: str) -> str:
        """Render compact failed notification (single line)."""
        text = Text()
        text.append("x ", style="red")
        text.append("Compact failed: ", style="bold red")
        text.append(error[:100], style="dim")
        return self._renderer.render(text)

    def render_handoff_start(self, message_count: int) -> str:
        """Render handoff start notification (single line)."""
        text = Text()
        text.append("> ", style="magenta")
        text.append(f"Summarizing progress ({message_count} messages)...", style="dim")
        return self._renderer.render(text)

    def render_handoff_complete(self, content: str) -> str:
        """Render handoff complete panel."""
        panel_content = Text()
        panel_content.append("Progress summarized, continuing with fresh context\n", style="bold green")
        if content:
            panel_content.append(content, style="dim")
        panel = Panel(
            panel_content, border_style="magenta", title="[magenta]Summary Complete[/magenta]", title_align="left"
        )
        return self._renderer.render(panel)

    def render_handoff_failed(self, error: str) -> str:
        """Render handoff failed notification (single line)."""
        text = Text()
        text.append("x ", style="red")
        text.append("Summary failed: ", style="bold red")
        text.append(error[:100], style="dim")
        return self._renderer.render(text)

    def render_steering_injected(self, messages: list[str], max_line_len: int = 100) -> str:
        """Render steering message injected panel."""
        panel_content = Text()
        panel_content.append(f"Guidance injected ({len(messages)} message(s))\n", style="bold")
        for msg in messages:
            line = msg.replace("\n", " ")
            preview = line[:max_line_len] + "..." if len(line) > max_line_len else line
            panel_content.append(f'"{preview}"\n', style="dim italic")
        panel = Panel(panel_content, border_style="yellow", title="[yellow]Steering[/yellow]", title_align="left")
        return self._renderer.render(panel)

    # =========================================================================
    # Memory Event Rendering
    # =========================================================================

    # =========================================================================
    # File Change Event Rendering
    # =========================================================================

    def render_file_change_event(self, event: FileChangeEvent, width: int | None = None) -> str:
        """Render file change event as diff panels or notification lines.

        For edit/multi_edit: shows file path + diff using TextReplacement data.
        For write: shows file creation notification.
        For move/copy: shows source -> destination.
        """
        if not event.changes:
            return ""

        parts: list[RenderableType] = []

        for change in event.changes:
            if change.replacements:
                # Edit operations: render diffs
                parts.extend(self._render_edit_change(change, event.tool_name))
            elif change.action == FileChangeAction.created:
                # Write/new file
                line = Text()
                line.append("+ ", style="bold green")
                line.append("Created: ", style="dim")
                line.append(change.path, style="bold")
                parts.append(line)
            elif change.action == FileChangeAction.moved:
                line = Text()
                line.append("> ", style="bold yellow")
                line.append("Moved: ", style="dim")
                line.append(change.path, style="bold")
                line.append(" -> ", style="dim")
                line.append(change.destination or "?", style="bold")
                parts.append(line)
            elif change.action == FileChangeAction.copied:
                line = Text()
                line.append("= ", style="bold cyan")
                line.append("Copied: ", style="dim")
                line.append(change.path, style="bold")
                line.append(" -> ", style="dim")
                line.append(change.destination or "?", style="bold")
                parts.append(line)
            elif change.action == FileChangeAction.modified:
                # Modified without replacements (e.g., write to existing file)
                line = Text()
                line.append("~ ", style="bold yellow")
                line.append("Modified: ", style="dim")
                line.append(change.path, style="bold")
                parts.append(line)

        if not parts:
            return ""

        render_width = width or 120
        if any(c.replacements for c in event.changes):
            # Has diffs: use panel
            file_path = event.changes[0].path
            title_label = "Edit" if event.tool_name in ("edit", "multi_edit") else "File Change"
            panel = Panel(
                Group(*parts),
                border_style="green",
                title=f"[green]{title_label}: {file_path}[/green]",
                title_align="left",
            )
            return self._renderer.render(panel, width=render_width)
        else:
            # No diffs: render as simple lines
            rendered_parts = []
            for part in parts:
                rendered_parts.append(self._renderer.render(part, width=render_width).rstrip())
            return "\n".join(rendered_parts)

    def _render_edit_change(self, change: FileChange, tool_name: str) -> list[RenderableType]:
        """Render a single file change with text replacements as diffs."""
        parts: list[RenderableType] = []
        code_theme = self._code_theme

        is_new_file = (
            len(change.replacements) == 1
            and not change.replacements[0].old_string
            and change.action == FileChangeAction.created
        )

        if is_new_file:
            # New file creation: show content preview
            new_content = change.replacements[0].new_string
            lines = new_content.split("\n")
            preview = "\n".join(lines[:20])
            if len(lines) > 20:
                preview += f"\n... ({len(lines) - 20} more lines)"
            syntax = Syntax(
                preview,
                lexer="text",
                theme=code_theme,
                line_numbers=False,
                background_color="default",
            )
            header = Text()
            header.append("+ ", style="bold green")
            header.append(f"New file ({len(lines)} lines)", style="dim")
            parts.append(header)
            parts.append(syntax)
        else:
            # Modifications: show diffs
            for i, replacement in enumerate(change.replacements):
                if len(change.replacements) > 1:
                    edit_header = Text(f"Edit #{i + 1}", style="bold blue")
                    parts.append(edit_header)

                diff_content = generate_unified_diff(replacement.old_string, replacement.new_string)
                if diff_content.strip() and diff_content != "No changes detected":
                    syntax_diff = Syntax(
                        diff_content,
                        lexer="diff",
                        theme=code_theme,
                        line_numbers=False,
                        background_color="default",
                    )
                    parts.append(syntax_diff)
                else:
                    parts.append(Text("No changes detected", style="dim"))

                # Spacing between edits
                if i < len(change.replacements) - 1:
                    parts.append(Text(""))

        return parts

    def render_note_event(self, event: NoteEvent) -> str:
        """Render note event as a note panel with full snapshot."""
        if not event.entries:
            content = Text("No entries", style="dim")
            panel = Panel(content, border_style="magenta", title="[magenta]Notes[/magenta]", title_align="left")
            return self._renderer.render(panel)

        parts: list[RenderableType] = []
        for key in sorted(event.entries):
            value = event.entries[key]
            line = Text()
            line.append(f"{key}: ", style="bold")
            # Truncate long values for display
            display_value = value if len(value) <= 120 else value[:120] + "..."
            line.append(display_value, style="dim")
            parts.append(line)

        panel = Panel(
            Group(*parts),
            border_style="magenta",
            title=f"[magenta]Notes ({len(event.entries)} entries)[/magenta]",
            title_align="left",
        )
        return self._renderer.render(panel)
