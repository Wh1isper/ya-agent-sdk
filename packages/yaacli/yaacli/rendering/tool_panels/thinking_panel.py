"""Thinking tool panel rendering."""

from __future__ import annotations

import json

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel


def create_thinking_panel(args: str | dict[str, object] | None, code_theme: str = "monokai") -> Panel:
    """Create a special panel for thinking tools.

    The thought content is in args['thought'], not in the result content.
    """
    panel_content: RenderableType = ""
    try:
        # Extract thought from args (where thinking tool stores the content)
        thought: str | None = None
        if isinstance(args, dict):
            raw_thought = args.get("thought")
            if isinstance(raw_thought, str):
                thought = raw_thought
        elif isinstance(args, str):
            # Try parsing args as JSON string
            try:
                args_data = json.loads(args)
                if isinstance(args_data, dict):
                    raw_thought = args_data.get("thought")
                    if isinstance(raw_thought, str):
                        thought = raw_thought
            except json.JSONDecodeError:
                pass

        if thought:
            panel_content = Markdown(thought, code_theme=code_theme)
        else:
            panel_content = "No thinking content available"
    except Exception:
        panel_content = "Error processing thinking data"

    return Panel(
        panel_content,
        title="[TOOL] thinking",
        title_align="left",
        border_style="magenta",
    )
