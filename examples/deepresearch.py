"""
Deep Research Example for pai-agent-sdk

This example demonstrates a deep research agent that autonomously conducts
comprehensive research on a given objective without human-in-the-loop interaction.

The agent will:
1. Search the web for relevant information
2. Scrape and download important resources
3. Organize findings in notes/
4. Produce a final report in report.md

Usage:
    cd examples
    cp .env.example .env  # Configure API keys

    # Run with a research objective:
    uv run python deepresearch.py "Research the latest developments in AI agents"

    # Or run interactively (will prompt for objective):
    uv run python deepresearch.py

Required API Keys (at least one search API is required):
    - GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CX (preferred)
    - TAVILY_API_KEY (alternative)

Optional API Keys:
    - FIRECRAWL_API_KEY (better web scraping, falls back to MarkItDown if not set)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

import json

from pydantic_ai import ModelSettings, PartEndEvent, PartStartEvent, TextPart, ToolOutput
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    TextPartDelta,
)

from pai_agent_sdk.agents.main import create_agent, stream_agent
from pai_agent_sdk.context import ModelCapability, ModelConfig, ResumableState, StreamEvent, ToolConfig
from pai_agent_sdk.presets import GEMINI_THINKING_LEVEL_HIGH
from pai_agent_sdk.toolsets.core.content import tools as content_tools
from pai_agent_sdk.toolsets.core.context import tools as context_tools
from pai_agent_sdk.toolsets.core.document import tools as document_tools
from pai_agent_sdk.toolsets.core.enhance import tools as enhance_tools
from pai_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from pai_agent_sdk.toolsets.core.multimodal import tools as multimodal_tools
from pai_agent_sdk.toolsets.core.shell import tools as shell_tools
from pai_agent_sdk.toolsets.core.web import tools as web_tools

# =============================================================================
# Configuration
# =============================================================================

PROMPT_FILE = Path(__file__).parent / "prompts" / "deepresearch.md"
OUTPUT_DIR = Path(__file__).parent / "research"
SESSION_DIR = OUTPUT_DIR / ".session" / "deepresearch"
MESSAGE_HISTORY_FILE = SESSION_DIR / "message_history.json"
STATE_FILE = SESSION_DIR / "context_state.json"

# Output formatting
MAX_TOOL_CONTENT_LENGTH = 200


def load_system_prompt(objective: str) -> str:
    """Load and format system prompt with the research objective."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found: {PROMPT_FILE}")
    template = PROMPT_FILE.read_text(encoding="utf-8")
    return template.replace("{objective}", objective)


def validate_api_keys() -> tuple[bool, list[str]]:
    """Validate required API keys are configured.

    Returns:
        (is_valid, messages) - is_valid is True if search is available
    """
    # Load tool config to check API keys
    tool_config = ToolConfig()

    messages: list[str] = []

    # Check search APIs (at least one required)
    has_google = bool(tool_config.google_search_api_key and tool_config.google_search_cx)
    has_tavily = bool(tool_config.tavily_api_key)

    if not has_google and not has_tavily:
        return False, [
            "ERROR: No search API configured.",
            "Please set one of the following in your .env file:",
            "  - GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX (preferred)",
            "  - TAVILY_API_KEY",
        ]

    # Info about which search API is active
    if has_google:
        messages.append("Search API: Google Custom Search (active)")
    elif has_tavily:
        messages.append("Search API: Tavily (active)")

    # Check scraping API (optional but recommended)
    if not tool_config.firecrawl_api_key:
        messages.append("Note: FIRECRAWL_API_KEY not set - web scraping will use MarkItDown (limited quality)")
    else:
        messages.append("Scraping API: Firecrawl (active)")

    return True, messages


# =============================================================================
# Session Persistence
# =============================================================================


def ensure_output_dirs() -> None:
    """Ensure output directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "notes").mkdir(parents=True, exist_ok=True)


def load_message_history() -> list[ModelMessage] | None:
    """Load message history from JSON file."""
    if not MESSAGE_HISTORY_FILE.exists():
        return None
    try:
        with open(MESSAGE_HISTORY_FILE) as f:
            data = json.load(f)
        return ModelMessagesTypeAdapter.validate_python(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to load message history: {e}")
        return None


def save_message_history(messages_json: bytes) -> None:
    """Save message history to JSON file."""
    ensure_output_dirs()
    with open(MESSAGE_HISTORY_FILE, "wb") as f:
        f.write(messages_json)


def load_state() -> ResumableState | None:
    """Load context state from JSON file."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return ResumableState.model_validate_json(f.read())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to load context state: {e}")
        return None


def save_state(state: ResumableState) -> None:
    """Save context state to JSON file."""
    ensure_output_dirs()
    with open(STATE_FILE, "w") as f:
        f.write(state.model_dump_json(indent=2))


# =============================================================================
# Stream Event Formatting
# =============================================================================


def truncate(text: str, max_length: int = MAX_TOOL_CONTENT_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_tool_call(event: FunctionToolCallEvent) -> str:
    """Format a tool call event for display."""
    tool_name = event.part.tool_name
    args_str = json.dumps(event.part.args, ensure_ascii=False) if event.part.args else "{}"
    return f"[ToolCall] {tool_name}({truncate(args_str)})"


def format_tool_result(event: FunctionToolResultEvent) -> str:
    """Format a tool result event for display."""
    result = event.result
    tool_name = getattr(result, "tool_name", "unknown")
    content = event.content
    if content is None:
        content = getattr(result, "content", "")
    content_str = str(content) if content else ""
    return f"[ToolResult] {tool_name}: {truncate(content_str)}"


def print_stream_event(event: StreamEvent) -> None:
    """Print stream event with appropriate formatting."""
    message_event = event.event
    if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
        print(message_event.part.content, end="", flush=True)
    if isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, TextPartDelta):
        print(message_event.delta.content_delta, end="", flush=True)
    if isinstance(message_event, PartEndEvent) and isinstance(message_event.part, TextPart):
        print()
    elif isinstance(message_event, FunctionToolCallEvent):
        print(format_tool_call(message_event))
        print()
    elif isinstance(message_event, FunctionToolResultEvent):
        print(format_tool_result(message_event))
        print()


# =============================================================================
# Main
# =============================================================================


def get_objective() -> str:
    """Get research objective from command line args or user input."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    print("Deep Research Agent")
    print("=" * 50)
    print("Enter your research objective:")
    try:
        objective = input("> ").strip()
        if not objective:
            print("No objective provided, exiting.")
            sys.exit(0)
        return objective
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)


class TaskComplete(BaseModel):
    """Research task completion result."""

    summary: str
    """Brief summary of the research findings and key insights."""

    report_path: str
    """Path to the final research report file (e.g., 'report.md')."""


async def main() -> None:
    # Validate API keys first
    is_valid, messages = validate_api_keys()
    for msg in messages:
        print(msg)
    if not is_valid:
        sys.exit(1)

    print()

    # Get research objective
    objective = get_objective()
    print(f"\nResearch Objective: {objective}")
    print(f"Output Directory: {OUTPUT_DIR.absolute()}")
    print("=" * 50)
    print()

    # Ensure output directories exist
    ensure_output_dirs()

    # Load previous session state (for potential resume)
    message_history: list[ModelMessage] | None = load_message_history()
    state: ResumableState | None = load_state()

    if message_history:
        print(f"Resuming from {len(message_history)} previous messages")
    else:
        print("Starting new research session")

    # Load system prompt with objective
    system_prompt = load_system_prompt(objective)

    # Create and run the agent
    async with create_agent(
        model="gemini@google-vertex:gemini-3-pro-preview",
        model_settings=cast(ModelSettings, GEMINI_THINKING_LEVEL_HIGH),
        system_prompt=system_prompt,
        tools=[
            *content_tools,
            *context_tools,
            *document_tools,
            *enhance_tools,
            *filesystem_tools,
            *multimodal_tools,
            *shell_tools,
            *web_tools,
        ],
        # No need_user_approve_tools - fully autonomous
        model_cfg=ModelConfig(context_window=200_000, capabilities={ModelCapability.vision}),
        tool_config=ToolConfig(),
        state=state,
        output_type=ToolOutput(
            type_=TaskComplete,
            name="task_complete",
            description=(
                "Call this tool to finalize the research when ALL of the following are complete: "
                "(1) sufficient information gathered to address the objective, "
                "(2) notes organized in notes/ directory, "
                "(3) comprehensive report written to report.md with citations. "
                "Provide a brief summary of key findings and the report file path."
            ),
            max_retries=3,
        ),
        include_builtin_subagents=True,
    ) as agent_runtime:
        agent = agent_runtime.agent

        async with stream_agent(
            agent,
            user_prompt=f"""<objective>
{objective}
</objective>

<system-reminder>
Please conduct deep research on the following objective and produce a comprehensive report in {OUTPUT_DIR} directory.
</system-reminder>
""",
            ctx=agent_runtime.ctx,
            message_history=message_history,
        ) as stream:
            async for event in stream:
                print_stream_event(event)
            print()
            stream.raise_if_exception()
            run = stream.run

        if run and run.result:
            print(run.result.output.summary)
            print(f"\nUsage: {run.usage()}")
            print(f"Total messages: {len(run.all_messages())}")

            # Save session state
            save_message_history(run.all_messages_json())
            new_state = agent_runtime.ctx.export_state()
            save_state(new_state)

            print(f"\nResearch complete. Check {OUTPUT_DIR.absolute()} for results.")
            print(f"  - Report: {OUTPUT_DIR / 'report.md'}")
            print(f"  - Notes: {OUTPUT_DIR / 'notes/'}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
