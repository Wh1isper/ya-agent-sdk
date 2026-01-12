"""
Browser Use Example for pai-agent-sdk

This example demonstrates browser automation capabilities using the pai-agent-sdk.
It uses DockerBrowserSandbox to provide a headless Chrome browser and BrowserUseToolset
for browser automation tools.

Usage:
    cd examples
    cp .env.example .env

    # To run the example:
    uv run python browser_use.py

    # With debug logging enabled:
    PAI_AGENT_LOG_LEVEL=DEBUG uv run python browser_use.py

Key features demonstrated:
- Docker-based headless Chrome browser sandbox
- Browser automation via CDP (Chrome DevTools Protocol)
- Session state persistence
- Streaming agent responses with formatted output
- Context management with handoff tool
"""

from __future__ import annotations

from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

import json
from pathlib import Path

from pydantic_ai import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelSettings,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ToolOutput,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
)

from pai_agent_sdk.agents.main import create_agent, stream_agent
from pai_agent_sdk.context import (
    ModelCapability,
    ModelConfig,
    ResumableState,
    RunContextMetadata,
    StreamEvent,
    ToolConfig,
)
from pai_agent_sdk.presets import GEMINI_THINKING_LEVEL_HIGH
from pai_agent_sdk.sandbox.browser.docker_ import DockerBrowserSandbox
from pai_agent_sdk.toolsets.browser_use import BrowserUseToolset
from pai_agent_sdk.toolsets.core.context import tools as context_tools

# =============================================================================
# Prompt Configuration
# =============================================================================

PROMPT_FILE = Path(__file__).parent / "prompts" / "browser_use.md"


class TaskComplete(BaseModel):
    """Browser task completion result."""

    summary: str
    """Brief summary of the task results and key findings."""


def load_system_prompt() -> str:
    """Load system prompt from markdown file."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


# =============================================================================
# Session Management
# =============================================================================

SESSION_DIR = Path(__file__).parent / ".session" / "browser_use"
MESSAGE_HISTORY_FILE = SESSION_DIR / "message_history.json"
STATE_FILE = SESSION_DIR / "context_state.json"


def ensure_session_dir() -> None:
    """Ensure session directory exists."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


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
    ensure_session_dir()
    with open(MESSAGE_HISTORY_FILE, "wb") as f:
        f.write(messages_json)
    print(f"Message history saved to {MESSAGE_HISTORY_FILE}")


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
    ensure_session_dir()
    with open(STATE_FILE, "w") as f:
        f.write(state.model_dump_json(indent=2))
    print(f"Context state saved to {STATE_FILE}")


def get_user_input(prompt: str = "You: ") -> str:
    """Get user input from console with graceful interrupt handling."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# =============================================================================
# Stream Event Formatting
# =============================================================================

MAX_TOOL_CONTENT_LENGTH = 200


def truncate(text: str, max_length: int = MAX_TOOL_CONTENT_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


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
# Main Entry
# =============================================================================


async def main():
    # Load previous session state
    message_history: list[ModelMessage] | None = load_message_history()
    state: ResumableState | None = load_state()

    if message_history:
        print(f"Loaded {len(message_history)} messages from history")
    if state:
        print("Loaded previous context state")

    # Get user input
    user_prompt = get_user_input()

    if not user_prompt:
        print("No input provided, exiting.")
        return

    # Load system prompt from file
    system_prompt = load_system_prompt()

    # Start browser sandbox and create agent
    async with DockerBrowserSandbox() as cdp_url:
        print(f"Browser started at: {cdp_url}")

        async with BrowserUseToolset(
            cdp_url=cdp_url,
            always_use_new_page=True,
            auto_cleanup_page=True,
        ) as browser_toolset:
            runtime = create_agent(
                model="gemini@google-vertex:gemini-3-pro-preview",
                model_settings=cast(ModelSettings, GEMINI_THINKING_LEVEL_HIGH),
                system_prompt=system_prompt,
                tools=[
                    *context_tools,  # handoff tool for context management
                ],
                toolsets=[browser_toolset],
                model_cfg=ModelConfig(
                    context_window=200_000,
                    capabilities={ModelCapability.vision},
                ),
                tool_config=ToolConfig(),
                state=state,
                output_type=ToolOutput(
                    type_=TaskComplete,
                    name="task_complete",
                    description=(
                        "Call this tool when the browser task is complete. "
                        "Provide a brief summary of what was accomplished, "
                        "key findings, or results of the browser automation."
                    ),
                    max_retries=3,
                ),
                metadata=RunContextMetadata(context_manage_tool="handoff"),
            )

            async with stream_agent(
                runtime,
                user_prompt=user_prompt,
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
                print(f"Messages so far: {len(run.all_messages())}")
                save_message_history(run.all_messages_json())
                new_state = runtime.ctx.export_state()
                save_state(new_state)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
