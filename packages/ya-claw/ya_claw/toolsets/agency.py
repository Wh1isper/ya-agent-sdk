"""Global agency tools for YA Claw agency runs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.toolsets.session import ClawSelfClient, _get_self_client


def _is_agency_run(ctx: RunContext[AgentContext]) -> bool:
    return getattr(ctx.deps, "source_kind", None) == "agency"


def _get_agency_self_client(ctx: RunContext[AgentContext]) -> ClawSelfClient | None:
    client = _get_self_client(ctx)
    if isinstance(client, ClawSelfClient) and _is_agency_run(ctx):
        return client
    return None


class ListSourceSessionTurnsTool(BaseTool):
    """List completed turns from a source conversation session."""

    name = "list_source_session_turns"
    description = "List completed turns from a source conversation session referenced by a global agency signal."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_agency_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        source_session_id: Annotated[
            str | None,
            Field(description="Source conversation session ID. Defaults to the primary source session for the run."),
        ] = None,
        limit: Annotated[int, Field(description="Maximum completed turns to fetch, clamped to 1..50")] = 10,
        before_sequence_no: Annotated[
            int | None, Field(description="Fetch turns with sequence_no lower than this value")
        ] = None,
        cursor: Annotated[str | None, Field(description="Fetch turns older than this run ID cursor")] = None,
    ) -> str:
        client = _get_agency_self_client(ctx)
        if client is None:
            return "Error: YA Claw agency source-session client is unavailable."
        payload = await client.list_source_session_turns(
            source_session_id=source_session_id,
            limit=min(max(limit, 1), 50),
            before_sequence_no=before_sequence_no,
            cursor=cursor,
        )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class GetSourceRunTraceTool(BaseTool):
    """Get a source session run trace."""

    name = "get_source_run_trace"
    description = "Get tool-call and tool-response trace for a run referenced by a global agency signal."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_agency_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        run_id: Annotated[str, Field(description="Source session run ID")],
        max_item_chars: Annotated[int, Field(description="Maximum characters per trace item")] = 2000,
        max_total_chars: Annotated[int, Field(description="Maximum total characters across trace items")] = 8000,
    ) -> str:
        client = _get_agency_self_client(ctx)
        if client is None:
            return "Error: YA Claw agency source-run trace client is unavailable."
        payload = await client.get_source_run_trace(
            run_id=run_id,
            max_item_chars=min(max(max_item_chars, 256), 20000),
            max_total_chars=min(max(max_total_chars, max_item_chars), 100000),
        )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SubmitToSessionTool(BaseTool):
    """Submit a proactive Agency nudge to a conversation session."""

    name = "submit_to_session"
    agency_only = True
    description = (
        "Submit Agency proactive context or a nudge to a chosen conversation session. "
        "Use it to exchange useful cross-session context, remind the session agent, suggest a person to ask, "
        "or prompt a lightweight next action. The target session will submit, merge, or steer automatically."
    )

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_agency_self_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if not self.is_available(ctx):
            return None
        return _submit_to_session_instructions()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        session_id: Annotated[
            str,
            Field(description="Target conversation session ID that should receive the proactive nudge"),
        ],
        prompt: Annotated[
            str,
            Field(
                description="Natural-language proactive context or guidance for the session agent to interpret freely"
            ),
        ],
        handoff_kind: Annotated[
            str,
            Field(
                description="Required engineering tag for this handoff: context, exchange, reminder, task, risk, async_result, decision, or conflict"
            ),
        ],
        metadata: Annotated[
            dict[str, Any] | None,
            Field(
                description="Optional compact provenance such as fire_ids, source_run_ids, async_task_ids, people, groups, and artifact_paths"
            ),
        ] = None,
        handoff_tags: Annotated[
            list[str] | None,
            Field(description="Optional engineering tags. agency-reminder is added automatically."),
        ] = None,
    ) -> str:
        client = _get_agency_self_client(ctx)
        if client is None:
            return "Error: YA Claw submit-to-session client is unavailable."
        try:
            payload = await client.submit_to_session(
                session_id=session_id,
                prompt=prompt,
                metadata=metadata,
                handoff_kind=handoff_kind,
                handoff_tags=handoff_tags,
            )
        except Exception as exc:
            return f"Error: {exc}"
        return _format_session_submit_response(payload)


class ListAgencyRunsTool(BaseTool):
    """List recent global agency runs for the current agency session."""

    name = "list_agency_runs"
    description = "List recent runs from the global agency session that is coordinating proactive responses."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_agency_self_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        limit: Annotated[int, Field(description="Maximum agency runs to fetch, clamped to 1..50")] = 10,
    ) -> str:
        client = _get_agency_self_client(ctx)
        if client is None:
            return "Error: YA Claw agency run list client is unavailable."
        payload = await client.list_agency_runs(limit=min(max(limit, 1), 50))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def _submit_to_session_instructions() -> str:
    path = Path(__file__).with_name("submit_to_session_instructions.md")
    return path.read_text(encoding="utf-8").strip()


def _format_session_submit_response(payload: dict[str, Any]) -> str:
    session_id = payload.get("source_session_id")
    delivery = payload.get("delivery")
    run_id = payload.get("run_id")
    status = payload.get("status")
    attrs = {
        "session-id": session_id,
        "delivery": delivery,
        "run-id": run_id,
        "status": status,
    }
    attr_text = " ".join(
        f'{key}="{_xml_escape(value)}"' for key, value in attrs.items() if isinstance(value, str) and value
    )
    return f"<submit-to-session {attr_text}>\nSubmitted Agency proactive nudge to session.\n</submit-to-session>"


def _xml_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
