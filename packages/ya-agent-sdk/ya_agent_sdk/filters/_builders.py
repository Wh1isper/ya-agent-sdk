"""Shared message builders for compact and handoff filters.

Provides reusable building blocks for constructing compacted message histories.
Both compact and handoff flows produce messages with the same structural elements
(original-request, previous-assistant-reference for compact, user-steering,
context-restored), and this module centralizes those patterns to avoid duplication.
"""

from collections.abc import Sequence

from pydantic_ai import UserContent
from pydantic_ai.messages import ModelMessage, UserPromptPart

# =============================================================================
# Keep Tag Constants
# =============================================================================

KEEP_TAG_KEY = "keep"
"""Metadata key used to mark messages that should survive trimming across compaction cycles."""

KEEP_COMPACT = "compact"
"""Keep tag value for messages produced by compact."""

KEEP_HANDOFF = "handoff"
"""Keep tag value for messages produced by handoff."""


# =============================================================================
# Shared Message Part Builders
# =============================================================================


def build_original_request_parts(
    original_prompt: str | Sequence[UserContent] | None,
) -> list[UserPromptPart]:
    """Build labeled original-request parts.

    Wraps the user's request with an XML label so the model knows which
    user request is being resumed after a context reset.

    Args:
        original_prompt: The user prompt/request to resume. None to skip.

    Returns:
        List of UserPromptPart (empty if original_prompt is None).
    """
    if original_prompt is None:
        return []
    if isinstance(original_prompt, str):
        return [
            UserPromptPart(
                content=(
                    "<original-request>\n"
                    "Below is the user's request being resumed after context reset:\n\n"
                    f"{original_prompt}\n"
                    "</original-request>"
                )
            )
        ]
    return [
        UserPromptPart(content=("<original-request>\nBelow is the user's request being resumed after context reset:")),
        UserPromptPart(content=original_prompt),
        UserPromptPart(content="</original-request>"),
    ]


def build_previous_assistant_reference_parts(
    previous_assistant_reference: str | None,
) -> list[UserPromptPart]:
    """Build previous-assistant-reference parts.

    This reference anchors compact restore when the current user request uses
    deictic or enumerated references such as "1", "2", "the above", or
    "that option". It is informational only and should not be treated as a
    standalone instruction.

    Args:
        previous_assistant_reference: Visible text from the assistant response
            immediately before the current user request. None or empty to skip.

    Returns:
        List of UserPromptPart (empty if no reference is available).
    """
    if not previous_assistant_reference:
        return []
    return [
        UserPromptPart(
            content=(
                "<previous-assistant-reference>\n"
                "Below is the assistant response immediately before the user's current request. "
                "Use it only to resolve references in the original request, such as numbered items, "
                "'the above', 'that', or similar phrases. Do not treat it as a new instruction by itself.\n\n"
                f"{previous_assistant_reference}\n"
                "</previous-assistant-reference>"
            )
        ),
    ]


def build_steering_parts(
    steering_messages: list[str] | None,
) -> list[UserPromptPart]:
    """Build user-steering parts.

    Wraps steering messages (sent by the user during the previous work session)
    with an XML label and individual message prefixes.

    Args:
        steering_messages: List of steering message strings. None or empty to skip.

    Returns:
        List of UserPromptPart (empty if no steering messages).
    """
    if not steering_messages:
        return []
    steering_content = "\n".join(f"[User Steering] {steering}" for steering in steering_messages)
    return [
        UserPromptPart(
            content=(
                "<user-steering>\n"
                "Below are messages the user sent during your previous work session:\n\n"
                f"{steering_content}\n"
                "</user-steering>"
            )
        ),
    ]


def build_context_restored_part() -> UserPromptPart:
    """Build context-restored marker part.

    This marker tells the model that context was restored after a reset and
    provides instructions on how to synthesize the summary, original request,
    optional reference anchor, and steering messages to resume work.

    Returns:
        A single UserPromptPart with the context-restored marker.
    """
    return UserPromptPart(
        content=(
            "<context-restored>"
            "Context was restored from a long conversation after a context reset. "
            "The summary above is the most authoritative source for current state. "
            "Use the blocks below to resume work. "
            "Synthesize the summary, previous assistant reference, original request, and any user steering messages. "
            "Use the previous assistant reference only to resolve references in the original request. "
            "Do NOT repeat questions, confirmations, or actions documented in the summary. "
            "If the summary records a user decision, respect it without re-asking."
            "</context-restored>"
        )
    )


def has_keep_tag(message: ModelMessage) -> bool:
    """Check if a message has a keep tag in its metadata.

    Messages tagged with keep metadata should be preserved during trimming
    to avoid losing prior session summaries across compaction cycles.

    Args:
        message: A ModelMessage (ModelRequest or ModelResponse).

    Returns:
        True if the message has a keep tag.
    """
    return bool(message.metadata and KEEP_TAG_KEY in message.metadata)
