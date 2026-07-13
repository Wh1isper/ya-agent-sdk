"""Compact agent for conversation summarization.

This module provides a compact agent that can summarize conversation history
and return structured results including analysis and context for continuing
the conversation.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelSettings,
    ToolOutput,
    UsageLimits,
    UserContent,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext

from ya_agent_sdk._config import AgentSettings
from ya_agent_sdk._logger import logger
from ya_agent_sdk.agents import trim as _trim_helpers
from ya_agent_sdk.agents.lifecycle import (
    CompactCompleteContext,
    CompactFailedContext,
    CompactLifecycleCallback,
    CompactStartContext,
    ContextHandoffSource,
    run_compact_complete_callbacks,
    run_extension_method,
)
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.agents.trim import (
    TrimHistoryOptions,
    trim_history_for_summary,
)
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.context.agent import ENVIRONMENT_CONTEXT_TAG, RUNTIME_CONTEXT_TAG
from ya_agent_sdk.events import (
    CompactCompleteEvent,
    CompactFailedEvent,
    CompactStartEvent,
)
from ya_agent_sdk.filters import (
    create_system_prompt_filter,
    fix_truncated_tool_args,
)
from ya_agent_sdk.filters._builders import (
    KEEP_COMPACT,
    KEEP_TAG_KEY,
    build_context_restored_part,
    build_original_request_parts,
    build_previous_assistant_reference_parts,
    build_steering_parts,
)
from ya_agent_sdk.utils import get_latest_request_usage

# =============================================================================
# Constants
# =============================================================================

AGENT_NAME = "compact"

DEFAULT_COMPACT_INSTRUCTION = """Use `condense` to generate a summary and context of the conversation so far.
This summary covers important details of the historical conversation with the user which has been truncated.
It's crucial that you respond by ONLY asking the user what you should work on next.
You should NOT take any initiative or make any assumptions about continuing with work.
Keep this response CONCISE and wrap your analysis in `analysis` and `context` fields to organize your thoughts and ensure you've covered all necessary points.

IMPORTANT: If the message history shows that a Skill was activated and remains relevant to unfinished work, you MUST include a reminder in the context to re-read that Skill's documentation when resuming. A candidate Skill that was merely inspected or rejected was not activated and must not create a re-read requirement. Do not carry a merely inspected or rejected candidate's workflow, mandatory requirements, referenced-resource instructions, or proposed next steps into any continuation section; if historically relevant, record only that it was inspected and not activated."""

CACHE_FRIENDLY_COMPACT_INSTRUCTION = """Generate a compact continuation summary for the conversation history.
Return only the summary text. Do not call tools.
Do not carry a merely inspected or rejected candidate's workflow, mandatory requirements, referenced-resource instructions, or proposed next steps into any continuation section; if historically relevant, record only that it was inspected and not activated.
Use this exact Markdown structure:

## Condensed conversation summary

### Analysis

[Brief analysis of the conversation and what matters for continuation.]

### Context

1. Primary Request and Intent:
   [User's explicit requests and intent]

2. Key Technical Concepts:
   - [Concepts, technologies, APIs, and architecture points]

3. Files and Code Sections:
   - [Files examined, edited, or created, with important details]

4. Problem Solving:
   [Problems solved and ongoing troubleshooting]

5. Pending Tasks:
   - [Explicit pending tasks]

6. Current Work:
   [Precise current work immediately before compaction. If the current user request references numbered items, "above", "that", or similar phrases, resolve those references using the previous assistant response and spell out what they refer to.]

7. Optional Next Step:
   [Direct next step aligned with the current work]

8. Past Interactions:
   - [Key interactions already completed, including actions and outcomes]

9. Activated Skills:
   [List only Skills that were activated and remain relevant to unfinished work, and remind the next agent to re-read them. Do not include Skills that were merely inspected or rejected as candidates.]

10. Files to Inspect on Resume:
   [List only file paths that may need to be inspected when resuming. Do not include file contents.]
"""

CACHE_FRIENDLY_COMPACT_PROMPT = """Compact the conversation history into the requested continuation summary format.
Focus on details needed to continue the user's work accurately after older messages are removed.
Return only the summary text."""

# Settings keys that should NOT be inherited by compact agent.
# Cache: compact has different prompts/tools/history; inheriting cache settings
# would create separate entries that waste cache write tokens.
# Thinking: compact uses structured output; inherited thinking settings can be
# incompatible across providers and are not needed for summarization.
_COMPACT_STRIP_KEYS = frozenset({
    # Cache keys
    "anthropic_cache_tool_definitions",
    "anthropic_cache_instructions",
    "anthropic_cache_messages",
    "anthropic_cache",
    # Thinking keys (incompatible with ToolOutput)
    "thinking",
    "anthropic_thinking",
    "anthropic_effort",
})

# Anthropic beta headers that are incompatible with compact agent output mode.
_INCOMPATIBLE_BETAS = frozenset({
    "interleaved-thinking-2025-05-14",
})

# Default injected context tags used when no AgentContext is available.
_DEFAULT_INJECTED_TAGS = (RUNTIME_CONTEXT_TAG, ENVIRONMENT_CONTEXT_TAG)

# Maximum characters to keep in a single tool return content for compact
_MAX_TOOL_RETURN_CHARS = 500
# Characters to keep from the beginning and end when truncating
_TOOL_RETURN_KEEP_HEAD = 200
_TOOL_RETURN_KEEP_TAIL = 200


def _strip_beta_headers(result: dict[str, Any]) -> None:
    """Strip incompatible beta headers from extra_headers in-place."""
    extra_headers = result.get("extra_headers")
    if not extra_headers or not isinstance(extra_headers, dict):
        return
    beta_str = extra_headers.get("anthropic-beta", "")
    if not beta_str:
        return
    filtered = [b.strip() for b in beta_str.split(",") if b.strip() not in _INCOMPATIBLE_BETAS]
    if filtered:
        result["extra_headers"] = {
            **extra_headers,
            "anthropic-beta": ",".join(filtered),
        }
    else:
        result["extra_headers"] = {k: v for k, v in extra_headers.items() if k != "anthropic-beta"}
        if not result["extra_headers"]:
            del result["extra_headers"]


def _strip_clear_thinking_edits(result: dict[str, Any]) -> None:
    """Strip clear_thinking edits from context_management in extra_body in-place."""
    extra_body = result.get("extra_body")
    if not extra_body or not isinstance(extra_body, dict):
        return
    cm = extra_body.get("context_management")
    if not cm or not isinstance(cm, dict):
        return
    edits = cm.get("edits")
    if not edits or not isinstance(edits, list):
        return
    filtered_edits = [e for e in edits if not (isinstance(e, dict) and "clear_thinking" in e.get("type", ""))]
    if filtered_edits == edits:
        return
    if filtered_edits:
        result["extra_body"] = {
            **extra_body,
            "context_management": {**cm, "edits": filtered_edits},
        }
    else:
        new_body = {k: v for k, v in extra_body.items() if k != "context_management"}
        if new_body:
            result["extra_body"] = new_body
        else:
            del result["extra_body"]


def _strip_incompatible_settings(settings: ModelSettings) -> ModelSettings:
    """Strip settings incompatible with the compact agent.

    Removes:
    - Anthropic cache settings (compact has different prompts/tools/history)
    - Thinking settings (incompatible across providers for compact structured output)
    - Incompatible beta headers from extra_headers
    - clear_thinking edits from context_management (requires thinking enabled)

    Args:
        settings: Model settings potentially containing incompatible keys.

    Returns:
        A copy with incompatible settings removed.
    """
    result = {k: v for k, v in settings.items() if k not in _COMPACT_STRIP_KEYS}
    _strip_beta_headers(result)
    _strip_clear_thinking_edits(result)
    return cast(ModelSettings, result)


# =============================================================================
# Pre-trimming for compact
# =============================================================================


def _truncate_str(content: str, max_chars: int = _MAX_TOOL_RETURN_CHARS) -> str:
    options = TrimHistoryOptions(
        max_tool_return_chars=max_chars,
        tool_return_keep_head=_TOOL_RETURN_KEEP_HEAD,
        tool_return_keep_tail=_TOOL_RETURN_KEEP_TAIL,
    )
    from ya_agent_sdk.agents.trim import _truncate_str as _public_truncate_str

    return _public_truncate_str(content, options)


def _strip_injected_context(
    part: UserPromptPart,
    tags: tuple[str, ...] = _DEFAULT_INJECTED_TAGS,
) -> UserPromptPart | None:
    from ya_agent_sdk.agents.trim import _strip_injected_context as _public_strip_injected_context

    stripped, _ = _public_strip_injected_context(part, tags)
    return stripped


def _trim_history_for_compact(
    message_history: list[ModelMessage],
    *,
    preserve_last_turn: bool = False,
    injected_context_tags: tuple[str, ...] = _DEFAULT_INJECTED_TAGS,
) -> list[ModelMessage]:
    """Pre-trim message history before sending to compact agent."""
    return trim_history_for_summary(
        message_history,
        TrimHistoryOptions(
            preserve_last_turn=preserve_last_turn,
            injected_context_tags=injected_context_tags,
            max_tool_return_chars=_MAX_TOOL_RETURN_CHARS,
            tool_return_keep_head=_TOOL_RETURN_KEEP_HEAD,
            tool_return_keep_tail=_TOOL_RETURN_KEEP_TAIL,
        ),
    ).messages


_build_tag_regex = _trim_helpers._build_tag_regex
_find_last_user_turn_index = _trim_helpers._find_last_user_turn_index
_is_media_content = _trim_helpers._is_media_content
_media_to_placeholder = _trim_helpers._media_to_placeholder
_strip_media = _trim_helpers._strip_media


# =============================================================================
# Utilities
# =============================================================================


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "compact.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


# =============================================================================
# Models
# =============================================================================


class CondenseResult(BaseModel):
    analysis: str = Field(
        ...,
        description="""A summary of the conversation so far, capturing technical details, code patterns, and architectural decisions.""",
    )
    context: str = Field(
        ...,
        description="""The context to continue the conversation with. Do not carry a merely inspected or rejected candidate's workflow, mandatory requirements, referenced-resource instructions, or proposed next steps into any continuation section; if historically relevant, record only that it was inspected and not activated. If applicable based on the current task, this should include:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
6. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable. If the current user request references numbered items, "above", "that", or similar phrases, resolve those references using the previous assistant response and spell out what they refer to.
7. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
8. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
9. Past Interactions: A concise bullet list of key interactions (both sides) that already occurred, to prevent repetition. Include your actions/proposals and user's responses, approaches tried and outcomes, explanations already given.
10. Activated Skills: List only Skills that were activated and remain relevant to unfinished work, with a reminder to re-read them when resuming. Do not include Skills that were merely inspected or rejected as candidates.
11. Files to Inspect on Resume: List file paths that may need to be inspected when resuming. Do not include file contents and do not assume they will be automatically loaded.
""",
    )
    original_prompt: str = Field(
        ...,
        description="The original prompt and key information from the user. "
        "Used as fallback when agent_ctx.user_prompts is not set.",
    )


# =============================================================================
# Agent Factory
# =============================================================================


def get_compact_agent(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    main_model: str | Model | None = None,
    main_model_settings: ModelSettings | None = None,
) -> Agent[AgentContext, CondenseResult]:
    """Create a compact agent.

    Args:
        model: Model string or Model instance. Highest priority.
        model_settings: Optional model settings dict.
        main_model: Fallback model inherited from main agent. Lowest priority.
        main_model_settings: Fallback model settings inherited from main agent.

    Model resolution priority:
        1. model parameter (explicit configuration)
        2. YA_AGENT_COMPACT_MODEL environment variable
        3. main_model parameter (inherited from main agent)

    Returns:
        Agent configured for compact with AgentContext as deps type.

    Raises:
        ValueError: If no model is available from any source.
    """
    effective_model: str | Model | None = model
    effective_settings: ModelSettings | None = model_settings

    # Priority: model > env var > main_model
    if effective_model is None:
        settings = AgentSettings()
        if settings.compact_model:
            effective_model = settings.compact_model
        elif main_model is not None:
            effective_model = main_model
        else:
            raise ValueError(
                "No model specified. Provide model parameter, set YA_AGENT_COMPACT_MODEL, "
                "or pass main_model for inheritance."
            )

    # model_settings: model_settings > main_model_settings
    if effective_settings is None and main_model_settings is not None:
        effective_settings = _strip_incompatible_settings(main_model_settings)

    system_prompt = _load_system_prompt()
    return Agent[AgentContext, CondenseResult](
        model=infer_model(effective_model),
        model_settings=effective_settings,
        output_type=ToolOutput(CondenseResult),
        deps_type=AgentContext,
        system_prompt=system_prompt,
        capabilities=[
            ProcessHistory(create_system_prompt_filter(system_prompt)),  # Ensure system prompt is consistent
            ProcessHistory(fix_truncated_tool_args),
        ],
    )


# =============================================================================
# Utilities
# =============================================================================


def condense_result_to_markdown(result: CondenseResult) -> str:
    """Convert CondenseResult to markdown format.

    Args:
        result: The CondenseResult to convert.

    Returns:
        Markdown formatted string with analysis and context.
    """
    return f"""## Condensed conversation summary

### Analysis

{result.analysis}

### Context

{result.context}
"""


async def _run_compact_iter(
    agent: Agent[AgentContext, CondenseResult],
    *,
    prompt: str,
    message_history: list[ModelMessage],
    deps: AgentContext,
) -> AgentRunResult[CondenseResult]:
    """Run compact using the agent iterator path.

    Args:
        agent: Compact agent instance.
        prompt: Prompt sent to the compact agent.
        message_history: Trimmed history to summarize.
        deps: Fresh context for the compact run.

    Returns:
        AgentRunResult from the compact execution.

    Raises:
        RuntimeError: If the iterator completes without a final result.
    """
    async with agent.iter(
        prompt,
        message_history=message_history,
        deps=deps,
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue
            elif Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass

    if run.result is None:
        raise RuntimeError("Compact iteration completed without a result")

    return run.result


def _need_compact(ctx: AgentContext, message_history: list[ModelMessage]) -> bool:
    """Check if compaction is needed based on token usage threshold.

    Args:
        ctx: Agent context with model configuration.
        message_history: Current message history.

    Returns:
        True if compaction should be triggered.
    """
    if not message_history:
        return False

    model_cfg = ctx.model_cfg
    if model_cfg.context_window is None:
        logger.debug("Unknown context window, skipping compact check.")
        return False

    # Get current token usage from message history
    request_usage = get_latest_request_usage(message_history)
    if request_usage is None or request_usage.total_tokens is None:
        return False

    threshold_tokens = int(model_cfg.context_window * model_cfg.compact_threshold)
    current_tokens = request_usage.total_tokens

    logger.debug(f"Compact check: {current_tokens} tokens vs {threshold_tokens} threshold")
    return current_tokens >= threshold_tokens


def _build_compacted_messages(
    summary: str,
    original_prompt: str | Sequence[UserContent],
    steering_messages: list[str] | None = None,
    previous_assistant_reference: str | None = None,
) -> list[ModelMessage]:
    """Build compacted message history.

    Messages are tagged with ``keep:compact`` metadata so that subsequent
    compaction cycles preserve them instead of trimming away the summary.

    Args:
        summary: The compacted summary content.
        original_prompt: The initial user prompt.
        steering_messages: Additional steering messages from user during execution.
        previous_assistant_reference: Visible assistant response immediately before
            the current user request. Used only to resolve references in the request.

    Returns:
        List of ModelMessage representing the compacted history.
    """
    keep_metadata = {KEEP_TAG_KEY: KEEP_COMPACT}

    request_parts: list[SystemPromptPart | UserPromptPart] = [
        SystemPromptPart(content="Placeholder system prompt"),
        UserPromptPart(
            content="You have exceeded the maximum token limit for this conversation. "
            "Please provide a summary of the conversation so far and what you should work on next "
            "and I'll resume the conversation."
        ),
    ]

    # Build final request parts in interpretation order: restore instructions,
    # reference anchor, original request, then later user steering.
    final_parts: list[UserPromptPart] = [
        build_context_restored_part(),
        *build_previous_assistant_reference_parts(previous_assistant_reference),
        *build_original_request_parts(original_prompt),
        *build_steering_parts(steering_messages),
    ]

    return [
        ModelRequest(parts=request_parts),
        ModelResponse(parts=[TextPart(content=summary)], metadata=keep_metadata),
        ModelRequest(parts=final_parts, metadata=keep_metadata),
    ]


def create_cache_friendly_compact_filter(
    model_cfg: ModelConfig | None = None,
    callbacks: Sequence[CompactLifecycleCallback] | None = None,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a cache-friendly compact filter that reuses the current agent.

    This filter keeps the current agent's system prompt, tool definitions, and
    model settings intact. It asks the same agent for a plain text compact
    summary via run-level instructions and then replaces the history with the
    compacted replay messages.
    """

    async def compact_filter(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        agent_ctx = ctx.deps

        if agent_ctx._compact_depth > 0:
            return message_history

        compact_check_ctx = agent_ctx if model_cfg is None else agent_ctx.model_copy(update={"model_cfg": model_cfg})
        if not _need_compact(compact_check_ctx, message_history):
            logger.debug("No need to compact history.")
            return message_history

        if ctx.agent is None:
            logger.debug("No current agent in RunContext, skipping cache-friendly compact.")
            return message_history

        logger.info("Compacting conversation history with cache-friendly compact filter...")
        event_id = uuid4().hex[:8]

        object.__setattr__(agent_ctx, "_compact_depth", agent_ctx._compact_depth + 1)
        try:
            start_ctx = CompactStartContext(
                event_id=event_id,
                deps=agent_ctx,
                original_messages=list(message_history),
            )
            await run_extension_method(agent_ctx.lifecycle_extensions, "on_compact_start", start_ctx, logger=logger)
            await agent_ctx.emit_event(CompactStartEvent(event_id=event_id, message_count=len(message_history)))

            trimmed_result = None
            try:
                trimmed_result = trim_history_for_summary(
                    message_history,
                    TrimHistoryOptions(injected_context_tags=agent_ctx.injected_context_tags),
                )
                compact_agent = copy.copy(ctx.agent)
                if hasattr(compact_agent, "_output_validators"):
                    compact_agent._output_validators = []  # pyright: ignore[reportAttributeAccessIssue]
                async with compact_agent.run_stream(
                    f"{CACHE_FRIENDLY_COMPACT_INSTRUCTION}\n\n{CACHE_FRIENDLY_COMPACT_PROMPT}",
                    message_history=message_history,
                    deps=agent_ctx,
                    output_type=str,
                    usage_limits=UsageLimits(request_limit=1),
                ) as result:
                    summary_markdown = str(await result.get_output())
                    usage = result.usage

                model = ctx.model
                model_id = model.model_name if model is not None else "unknown"
                usage_id = uuid4().hex
                agent_ctx.update_usage_snapshot_entry(
                    agent_id=AGENT_NAME,
                    agent_name=AGENT_NAME,
                    model_id=model_id,
                    usage=usage,
                    source="compact",
                    usage_id=usage_id,
                    ledger_key=usage_id,
                )
                logger.info("Recorded cache-friendly compact usage: model_id=%s usage=%r", model_id, usage)

                compacted = _build_compacted_messages(
                    summary_markdown,
                    agent_ctx.user_prompts or CACHE_FRIENDLY_COMPACT_PROMPT,
                    agent_ctx.steering_messages or None,
                    agent_ctx.previous_assistant_response_reference,
                )

                complete_ctx = CompactCompleteContext(
                    event_id=event_id,
                    deps=agent_ctx,
                    source=ContextHandoffSource.COMPACT,
                    original_messages=list(message_history),
                    trimmed_messages=trimmed_result.messages,
                    handoff_messages=compacted,
                    summary_markdown=summary_markdown,
                    usage=usage,
                    metadata={"trim": trimmed_result},
                    compacted_messages=compacted,
                    condense_result=None,
                )
            except Exception as e:
                logger.error(f"Failed to compact history with cache-friendly filter: {e}")
                failed_ctx = CompactFailedContext(
                    event_id=event_id,
                    deps=agent_ctx,
                    original_messages=list(message_history),
                    trimmed_messages=trimmed_result.messages if trimmed_result is not None else None,
                    error=e,
                )
                await run_extension_method(
                    agent_ctx.lifecycle_extensions, "on_compact_failed", failed_ctx, logger=logger
                )
                await agent_ctx.emit_event(
                    CompactFailedEvent(event_id=event_id, error=str(e), message_count=len(message_history))
                )
                return message_history

            await run_extension_method(
                agent_ctx.lifecycle_extensions, "on_context_handoff_complete", complete_ctx, logger=logger
            )
            await run_extension_method(
                agent_ctx.lifecycle_extensions, "on_compact_complete", complete_ctx, logger=logger
            )
            await run_compact_complete_callbacks(callbacks, complete_ctx)
            await agent_ctx.emit_event(
                CompactCompleteEvent(
                    event_id=event_id,
                    summary_markdown=summary_markdown,
                    original_message_count=len(message_history),
                    compacted_message_count=len(compacted),
                    condense_result=None,
                )
            )

            agent_ctx.steering_messages.clear()
            agent_ctx.force_inject_instructions = True

            logger.info(f"Compacted history from {len(message_history)} messages to {len(compacted)} messages")
            return compacted

        finally:
            object.__setattr__(agent_ctx, "_compact_depth", max(0, agent_ctx._compact_depth - 1))

    return compact_filter


def create_compact_filter(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    model_cfg: ModelConfig | None = None,
    main_model: str | Model | None = None,
    main_model_settings: ModelSettings | None = None,
    callbacks: Sequence[CompactLifecycleCallback] | None = None,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a compact filter for automatic context compaction.

    The returned filter checks token usage and compacts the conversation history
    when usage exceeds the configured threshold (ModelConfig.compact_threshold).

    Args:
        model: Model string or Model instance for the compact agent. Highest priority.
        model_settings: Optional model settings for the compact agent.
        model_cfg: Optional compact-specific ModelConfig override. When None,
            threshold checks and compact-agent deps use the runtime context
            model_cfg.
        main_model: Fallback model inherited from main agent. Lowest priority.
        main_model_settings: Fallback model settings inherited from main agent.

    Model resolution priority:
        1. model parameter (explicit configuration)
        2. YA_AGENT_COMPACT_MODEL environment variable
        3. main_model parameter (inherited from main agent)

    Returns:
        An async filter function compatible with ProcessHistory capabilities.

    Example::

        compact_filter = await create_compact_filter(model="openai-chat:gpt-4o-mini")
        agent = Agent(
            'openai-chat:gpt-4',
            deps_type=AgentContext,
            capabilities=[ProcessHistory(compact_filter)],
        )
    """
    agent = get_compact_agent(
        model=model,
        model_settings=model_settings,
        main_model=main_model,
        main_model_settings=main_model_settings,
    )

    async def compact_filter(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Filter that compacts message history when threshold is exceeded.

        Args:
            ctx: Runtime context containing AgentContext.
            message_history: Current message history to potentially compact.

        Returns:
            Original or compacted message history.
        """
        agent_ctx = ctx.deps

        if not _need_compact(agent_ctx, message_history):
            logger.debug("No need to compact history.")
            return message_history

        logger.info("Compacting conversation history...")

        # Generate event_id to correlate start/complete events
        event_id = uuid4().hex[:8]

        # Apply model wrapper if configured
        original_model = agent.model
        if agent_ctx.model_wrapper is not None:
            wrapper_metadata = agent_ctx.get_wrapper_metadata()
            wrapped = agent_ctx.model_wrapper(cast(Model, original_model), AGENT_NAME, wrapper_metadata)
            agent.model = await wrapped if isawaitable(wrapped) else wrapped

        try:
            # Emit start event
            start_ctx = CompactStartContext(
                event_id=event_id,
                deps=agent_ctx,
                original_messages=list(message_history),
            )
            await run_extension_method(agent_ctx.lifecycle_extensions, "on_compact_start", start_ctx, logger=logger)
            await agent_ctx.emit_event(CompactStartEvent(event_id=event_id, message_count=len(message_history)))

            trimmed_result = None
            try:
                # Pre-trim history to reduce token count for compact agent
                trimmed_result = trim_history_for_summary(
                    message_history,
                    TrimHistoryOptions(injected_context_tags=agent_ctx.injected_context_tags),
                )
                trimmed_history = trimmed_result.messages

                # Run compact agent on trimmed message history with AgentContext as deps
                result = await _run_compact_iter(
                    agent,
                    prompt=DEFAULT_COMPACT_INSTRUCTION,
                    message_history=trimmed_history,
                    deps=AgentContext(
                        env=agent_ctx.env,
                        model_cfg=model_cfg or agent_ctx.model_cfg,
                    ),
                )

                model_id = cast(Model, agent.model).model_name
                usage_id = uuid4().hex
                agent_ctx.update_usage_snapshot_entry(
                    agent_id=AGENT_NAME,
                    agent_name=AGENT_NAME,
                    model_id=model_id,
                    usage=result.usage,
                    source="compact",
                    usage_id=usage_id,
                    ledger_key=usage_id,
                )

                condense_result: CondenseResult = result.output

                # Build summary with condense result and user prompts
                condense_markdown = condense_result_to_markdown(condense_result)

                # Build compacted messages
                # Priority: agent_ctx.user_prompts > condense_result.original_prompt
                # user_prompts is set by main agent from actual user input, while original_prompt
                # is extracted by LLM from conversation history and may be less accurate
                compacted = _build_compacted_messages(
                    condense_markdown,
                    agent_ctx.user_prompts or condense_result.original_prompt,
                    agent_ctx.steering_messages or None,
                    agent_ctx.previous_assistant_response_reference,
                )

                # Emit complete event with summary
                complete_ctx = CompactCompleteContext(
                    event_id=event_id,
                    deps=agent_ctx,
                    source=ContextHandoffSource.COMPACT,
                    original_messages=list(message_history),
                    trimmed_messages=trimmed_result.messages,
                    handoff_messages=compacted,
                    summary_markdown=condense_markdown,
                    usage=result.usage,
                    metadata={"trim": trimmed_result},
                    compacted_messages=compacted,
                    condense_result=condense_result,
                )
            except Exception as e:
                logger.error(f"Failed to compact history: {e}")
                # Emit failed event so consumers know compact did not succeed
                failed_ctx = CompactFailedContext(
                    event_id=event_id,
                    deps=agent_ctx,
                    original_messages=list(message_history),
                    trimmed_messages=trimmed_result.messages if trimmed_result is not None else None,
                    error=e,
                )
                await run_extension_method(
                    agent_ctx.lifecycle_extensions, "on_compact_failed", failed_ctx, logger=logger
                )
                await agent_ctx.emit_event(
                    CompactFailedEvent(event_id=event_id, error=str(e), message_count=len(message_history))
                )
                # On error, return original history
                return message_history

            await run_extension_method(
                agent_ctx.lifecycle_extensions, "on_context_handoff_complete", complete_ctx, logger=logger
            )
            await run_extension_method(
                agent_ctx.lifecycle_extensions, "on_compact_complete", complete_ctx, logger=logger
            )
            await run_compact_complete_callbacks(callbacks, complete_ctx)
            await agent_ctx.emit_event(
                CompactCompleteEvent(
                    event_id=event_id,
                    summary_markdown=condense_markdown,
                    original_message_count=len(message_history),
                    compacted_message_count=len(compacted),
                    condense_result=condense_result,
                )
            )

            # Clear steering_messages after successful compact (content is now in summary)
            agent_ctx.steering_messages.clear()

            # Force downstream filters to inject instructions after context reset
            agent_ctx.force_inject_instructions = True

            logger.info(f"Compacted history from {len(message_history)} messages to {len(compacted)} messages")
            return compacted

        finally:
            # Restore original model to avoid side effects on shared agent
            agent.model = original_model

    return compact_filter
