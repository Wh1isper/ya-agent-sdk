"""Hook-aware driver for agent runs whose nodes are streamed manually."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import _agent_graph
from pydantic_ai.run import AgentRun
from pydantic_graph import End


async def drive_streamed_run(
    run: AgentRun[Any, Any],
    process_node: Callable[[Any, AgentRun[Any, Any]], Awaitable[None]],
) -> None:
    """Stream every node while preserving Pydantic AI's node hook lifecycle.

    Bare iteration skips node hooks, including the pending-message capability's
    end-of-run redirect. Pydantic AI does not currently expose a public method
    that combines custom node streaming with hook-aware graph advancement, so
    this mirrors its own ``run_stream`` driver through two private methods kept
    behind this single compatibility boundary.
    """
    node = run.next_node
    while not isinstance(node, End):
        graph_ctx = run.ctx
        run_ctx = _agent_graph.build_run_context(graph_ctx)
        capability = graph_ctx.deps.root_capability

        # Match Pydantic AI's streamed-run boundary: replacement happens before
        # streaming, while wrap_node_run covers graph advancement only.
        node = await capability.before_node_run(run_ctx, node=node)
        await process_node(node, run)

        # Rebuild after streaming so hooks observe post-stream state such as the
        # current run step, exactly as Pydantic AI's run_stream driver does.
        run_ctx = _agent_graph.build_run_context(graph_ctx)
        node = await run._wrap_and_advance(  # pyright: ignore[reportPrivateUsage]
            run_ctx,
            node,
            run._advance_graph,  # pyright: ignore[reportPrivateUsage]
        )
