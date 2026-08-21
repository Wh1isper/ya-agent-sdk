"""Transactional restoration for durable YAACLI revisions."""

from __future__ import annotations

from ya_agent_sdk.context import AgentContext, ResumableState


def restore_resumable_state_safely(state: ResumableState, context: AgentContext) -> None:
    """Restore conversation state without weakening current approval policy.

    Revision payloads are conversation state, while approval configuration belongs to
    the currently running host. A failed custom state restore leaves the context
    exactly as it was before the attempt.
    """
    previous_state = context.__dict__.copy()
    approval_policy = (
        list(context.need_user_approve_tools),
        list(context.need_user_approve_mcps),
    )
    try:
        state.restore(context)
    except BaseException:
        object.__setattr__(context, "__dict__", previous_state)
        raise
    context.need_user_approve_tools = approval_policy[0]
    context.need_user_approve_mcps = approval_policy[1]
