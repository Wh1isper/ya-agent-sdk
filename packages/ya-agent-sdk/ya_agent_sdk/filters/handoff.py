"""Handoff message history processor.

This module provides a history processor that injects handoff summaries
into the message history when a context reset occurs.

Handoff summaries are injected as restored context with a system reminder. This
avoids fabricating assistant/tool-call history and keeps reasoning-model message
history compatible with OpenAI-compatible APIs.

Note:
    This processor must be used together with `ya_agent_sdk.toolsets.context.handoff.HandoffTool`.
    See HandoffTool for usage example.
"""

from collections.abc import Sequence
from uuid import uuid4

from pydantic_ai import UserContent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.lifecycle import ContextHandoffCompleteContext, ContextHandoffSource, run_extension_method
from ya_agent_sdk.agents.trim import TrimHistoryOptions, trim_history_for_summary
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import (
    HandoffCompleteEvent,
    HandoffFailedEvent,
    HandoffStartEvent,
)
from ya_agent_sdk.filters._builders import (
    KEEP_HANDOFF,
    KEEP_TAG_KEY,
    build_context_restored_part,
    build_original_request_parts,
    build_steering_parts,
)

logger = get_logger(__name__)


def _build_handoff_messages(
    summary: str,
    original_prompt: str | Sequence[UserContent] | None = None,
    steering_messages: list[str] | None = None,
    tool_call_id: str = "handoff-ack",
) -> list[ModelMessage]:
    """Build restored message history after handoff.

    The restored history uses a single user request containing the handoff
    summary, original prompt, steering messages, and a system reminder that the
    handoff has already completed. It intentionally avoids mock assistant tool
    calls so reasoning providers do not receive fabricated assistant messages
    without provider-required reasoning fields.

    Messages are tagged with ``keep:handoff`` metadata so that subsequent
    compaction cycles preserve them instead of trimming away the summary.

    Args:
        summary: The handoff summary content.
        original_prompt: The initial user prompt from the session.
        steering_messages: Additional steering messages from user during execution.
        tool_call_id: Deprecated compatibility parameter; ignored.

    Returns:
        List of ModelMessage representing the restored history.
    """
    _ = tool_call_id
    keep_metadata = {KEEP_TAG_KEY: KEEP_HANDOFF}

    # Placeholder will be replaced by create_system_prompt_filter downstream.
    request_parts: list[SystemPromptPart | UserPromptPart] = [
        SystemPromptPart(content="Placeholder system prompt"),
        *build_original_request_parts(original_prompt),
        UserPromptPart(content=summary),
        *build_steering_parts(steering_messages),
        build_context_restored_part(),
        UserPromptPart(
            content=(
                "<system-reminder>"
                "<item>The summarize tool has already completed this handoff. "
                "Continue work directly from the restored context summary, original request, "
                "and any user steering messages.</item>"
                "</system-reminder>"
            )
        ),
    ]

    return [ModelRequest(parts=request_parts, metadata=keep_metadata)]


async def process_handoff_message(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Inject handoff summary into message history after context reset.

    This is a pydantic-ai history processor that can be wrapped in a
    ProcessHistory capability. When a handoff occurs, the previous context is
    cleared but a summary message is preserved in ctx.deps.handoff_message.

    The restored history is a single request containing the context summary,
    original prompt, user steering, and a system reminder that handoff already
    completed. Downstream filters like auto_load_files can append to that request.

    Note: Subagents created via enter_subagent() have handoff_message cleared,
    so they won't be affected by the main agent's handoff state.

    Args:
        ctx: Runtime context containing AgentContext with handoff_message.
        message_history: Current message history to process.

    Returns:
        Processed message history with handoff summary injected, or unchanged
        history if no handoff is pending.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            capabilities=[ProcessHistory(process_handoff_message)],
        )
    """
    agent_ctx = ctx.deps

    # Reset force flag at the start of each filter pipeline pass
    agent_ctx.force_inject_instructions = False

    if not agent_ctx.handoff_message:
        return message_history

    # Generate event_id to correlate start/complete events
    event_id = uuid4().hex[:8]
    original_message_count = len(message_history)

    # Emit start event
    await agent_ctx.emit_event(HandoffStartEvent(event_id=event_id, message_count=original_message_count))

    handoff_content = agent_ctx.handoff_message

    try:
        trimmed_result = trim_history_for_summary(
            message_history,
            TrimHistoryOptions(injected_context_tags=agent_ctx.injected_context_tags),
        )

        # Build restored messages without fabricating assistant/tool-call history.
        result = _build_handoff_messages(
            handoff_content,
            agent_ctx.user_prompts,
            agent_ctx.steering_messages or None,
        )

        complete_ctx = ContextHandoffCompleteContext(
            event_id=event_id,
            deps=agent_ctx,
            source=ContextHandoffSource.SUMMARIZE_TOOL,
            original_messages=list(message_history),
            trimmed_messages=trimmed_result.messages,
            handoff_messages=result,
            summary_markdown=handoff_content,
            metadata={"trim": trimmed_result},
        )
    except Exception as e:
        # Emit failed event so consumers know handoff did not succeed
        await agent_ctx.emit_event(
            HandoffFailedEvent(event_id=event_id, error=str(e), message_count=original_message_count)
        )
        # On error, return original history
        return message_history

    await run_extension_method(
        agent_ctx.lifecycle_extensions,
        "on_context_handoff_complete",
        complete_ctx,
        logger=logger,
    )

    if agent_ctx.steering_messages:
        logger.debug("Including %d steering messages in handoff", len(agent_ctx.steering_messages))

    # Clear handoff state
    agent_ctx.handoff_message = None
    # Clear steering_messages after successful handoff (content is now in summary)
    agent_ctx.steering_messages.clear()

    # Force downstream filters to inject instructions after history restoration.
    agent_ctx.force_inject_instructions = True

    # Emit complete event with the actual handoff content
    await agent_ctx.emit_event(
        HandoffCompleteEvent(
            event_id=event_id,
            handoff_content=handoff_content,
            original_message_count=original_message_count,
        )
    )

    return result
