"""Source-session tools for YA Claw agency runs."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.toolsets.session import ClawSelfClient, _get_self_client


class ListSourceSessionTurnsTool(BaseTool):
    """List completed turns from the source conversation session."""

    name = "list_source_session_turns"
    description = "List completed turns from the source conversation session for an agency run."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        limit: Annotated[int, Field(description="Maximum completed turns to fetch, clamped to 1..50")] = 10,
        before_sequence_no: Annotated[
            int | None, Field(description="Fetch turns with sequence_no lower than this value")
        ] = None,
        cursor: Annotated[str | None, Field(description="Fetch turns older than this run ID cursor")] = None,
    ) -> str:
        client = _get_self_client(ctx)
        if client is None:
            return "Error: YA Claw self-session client is unavailable."
        if isinstance(client, ClawSelfClient):
            payload = await client.list_source_session_turns(
                limit=min(max(limit, 1), 50), before_sequence_no=before_sequence_no, cursor=cursor
            )
        else:
            payload = await client.list_session_turns(
                limit=min(max(limit, 1), 50), before_sequence_no=before_sequence_no, cursor=cursor
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class GetSourceRunTraceTool(BaseTool):
    """Get a source session run trace."""

    name = "get_source_run_trace"
    description = "Get tool-call and tool-response trace for a run in the source conversation session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        run_id: Annotated[str, Field(description="Source session run ID")],
        max_item_chars: Annotated[int, Field(description="Maximum characters per trace item")] = 2000,
        max_total_chars: Annotated[int, Field(description="Maximum total characters across trace items")] = 8000,
    ) -> str:
        client = _get_self_client(ctx)
        if client is None:
            return "Error: YA Claw self-session client is unavailable."
        if isinstance(client, ClawSelfClient):
            payload = await client.get_source_run_trace(
                run_id=run_id,
                max_item_chars=min(max(max_item_chars, 256), 20000),
                max_total_chars=min(max(max_total_chars, max_item_chars), 100000),
            )
        else:
            payload = await client.get_run_trace(
                run_id=run_id,
                max_item_chars=min(max(max_item_chars, 256), 20000),
                max_total_chars=min(max(max_total_chars, max_item_chars), 100000),
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ListAgencyRunsTool(BaseTool):
    """List recent agency runs for the source conversation session."""

    name = "list_agency_runs"
    description = "List recent agency runs attached to the source conversation session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        limit: Annotated[int, Field(description="Maximum agency runs to fetch, clamped to 1..50")] = 10,
    ) -> str:
        client = _get_self_client(ctx)
        if client is None:
            return "Error: YA Claw self-session client is unavailable."
        if not isinstance(client, ClawSelfClient):
            return json.dumps(
                {"source_session_id": client.session_id, "agency_session_id": None, "runs": []},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        payload = await client.list_agency_runs(limit=min(max(limit, 1), 50))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
