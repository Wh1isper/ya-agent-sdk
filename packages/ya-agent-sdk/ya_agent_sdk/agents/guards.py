"""Completion guards for SDK-created agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_graph import End

from ya_agent_sdk.context import AgentContext

_PENDING_MESSAGE_REMINDER = (
    "<system-reminder>There are pending messages in your message bus. "
    "Please address them before completing.</system-reminder>"
)


@dataclass(kw_only=True)
class MessageBusGuardCapability(AbstractCapability[AgentContext]):
    """Redirect an ending run when its SDK message bus has unread messages.

    The authoritative message stays in the SDK bus. This capability only
    enqueues one lightweight wakeup at the final node boundary. Pydantic AI's
    outermost pending-message capability then redirects the run, and the normal
    history filter consumes and injects the actual bus messages on the new
    model request.
    """

    id: str | None = "message_bus_guard"

    async def after_node_run(
        self,
        ctx: RunContext[AgentContext],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        """Enqueue a continuation immediately before an otherwise terminal result."""
        if isinstance(result, End) and ctx.deps.message_bus.has_pending(ctx.deps.agent_id):
            ctx.enqueue(_PENDING_MESSAGE_REMINDER, priority="asap")
        return result
