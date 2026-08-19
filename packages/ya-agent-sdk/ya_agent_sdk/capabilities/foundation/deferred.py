"""Deferred-interaction terminal boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_graph import End


@dataclass(kw_only=True)
class DeferredTerminalCapability(AbstractCapability[Any]):
    """Keep deferred tool requests terminal despite concurrently enqueued input.

    Logical input remains retained by ``RunInputLedger`` and is rebound to the
    continuation attempt. Clearing only the native attempt queue prevents the
    auto-injected pending-message drain from replacing a host-visible deferred
    suspension with another model request.
    """

    id: str | None = "deferred_terminal"

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        if not isinstance(result, End):
            return result
        output = result.data.output
        if not isinstance(output, DeferredToolRequests):
            return result
        if ctx.pending_messages:
            ctx.pending_messages.clear()
        return result
