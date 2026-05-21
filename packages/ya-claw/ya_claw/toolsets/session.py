"""Self-session tools for YA Claw runtime."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.json_types import JsonArray, JsonValue

CLAW_SELF_CLIENT_KEY = "claw_self_client"
_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0


@runtime_checkable
class SelfSessionClient(Protocol):
    session_id: str

    async def list_session_turns(
        self,
        *,
        limit: int,
        before_sequence_no: int | None,
        cursor: str | None,
    ) -> dict[str, Any]: ...

    async def get_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]: ...


class ClawSelfClient:
    """HTTP client scoped to the current YA Claw session."""

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        session_id: str,
        run_id: str | None = None,
        profile_name: str | None = None,
        timeout_seconds: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self.session_id = session_id
        self.run_id = run_id or ""
        self.profile_name = profile_name
        self._timeout_seconds = timeout_seconds

    def close(self) -> None:
        return None

    async def setup(self) -> None:
        return None

    def get_toolsets(self) -> list[Any]:
        return []

    async def list_source_session_turns(
        self,
        *,
        source_session_id: str | None = None,
        limit: int,
        before_sequence_no: int | None,
        cursor: str | None,
    ) -> dict[str, Any]:
        session_id = source_session_id.strip() if isinstance(source_session_id, str) else ""
        if session_id == "":
            session_id = self.session_id
        params: dict[str, str] = {"limit": str(limit)}
        if isinstance(cursor, str) and cursor.strip() != "":
            params["cursor"] = cursor.strip()
        elif isinstance(before_sequence_no, int):
            params["before_sequence_no"] = str(before_sequence_no)
        path = f"/api/v1/sessions/{urllib.parse.quote(session_id)}/turns"
        return await self._get_json(path, params=params)

    async def get_source_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]:
        params = {
            "max_item_chars": str(max_item_chars),
            "max_total_chars": str(max_total_chars),
        }
        path = f"/api/v1/runs/{urllib.parse.quote(run_id)}/trace"
        return await self._get_json(path, params=params)

    async def list_agency_runs(self, *, limit: int) -> dict[str, Any]:
        agency_session_id = self.session_id
        detail = await self._get_json(
            f"/api/v1/sessions/{urllib.parse.quote(agency_session_id)}",
            params={"runs_limit": str(limit), "include_input_parts": "true", "include_message": "false"},
        )
        session = detail.get("session") if isinstance(detail, dict) else None
        runs = session.get("runs") if isinstance(session, dict) else []
        return {
            "agency_session_id": agency_session_id,
            "runs": runs if isinstance(runs, list) else [],
        }

    async def submit_to_session(
        self,
        *,
        session_id: str,
        prompt: str,
        metadata: dict[str, Any] | None,
        handoff_kind: str,
        handoff_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._send_json(
            "/api/v1/agency/source-session:submit",
            method="POST",
            payload={
                "session_id": session_id,
                "prompt": prompt,
                "metadata": dict(metadata or {}),
                "handoff_kind": handoff_kind,
                "handoff_tags": list(handoff_tags or ["agency-reminder"]),
                "agency_session_id": self.session_id,
                "agency_run_id": self.run_id or None,
            },
        )

    async def list_session_turns(
        self,
        *,
        limit: int,
        before_sequence_no: int | None,
        cursor: str | None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {"limit": str(limit)}
        if isinstance(cursor, str) and cursor.strip() != "":
            params["cursor"] = cursor.strip()
        elif isinstance(before_sequence_no, int):
            params["before_sequence_no"] = str(before_sequence_no)
        path = f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/turns"
        return await self._get_json(path, params=params)

    async def get_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]:
        params = {
            "max_item_chars": str(max_item_chars),
            "max_total_chars": str(max_total_chars),
        }
        path = f"/api/v1/runs/{urllib.parse.quote(run_id)}/trace"
        payload = await self._get_json(path, params=params)
        response_session_id = payload.get("session_id")
        if response_session_id != self.session_id:
            raise RuntimeError("Requested run does not belong to the current session.")
        return payload

    async def spawn_delegate(
        self,
        *,
        subagent_name: str,
        prompt: str,
        name: str | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        path = f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/async-tasks:spawn"
        payload = {
            "subagent_name": subagent_name,
            "prompt": prompt,
            "name": name,
            "context": dict(context or {}),
            "parent_run_id": self.run_id or None,
            "parent_agent_id": "main",
        }
        return await self._send_json(path, method="POST", payload=payload)

    async def list_async_subagents(self, *, include_terminal: bool) -> dict[str, Any]:
        path = f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/async-tasks"
        return await self._get_json(
            path,
            params={"include_terminal": "true" if include_terminal else "false"},
        )

    async def get_async_subagent(self, *, name_or_task_id: str) -> dict[str, Any]:
        path = (
            f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/async-tasks/{urllib.parse.quote(name_or_task_id)}"
        )
        return await self._get_json(path, params={})

    async def steer_async_subagent(
        self,
        *,
        name_or_task_id: str,
        prompt: str | None,
        input_parts: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        path = f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/async-tasks/{urllib.parse.quote(name_or_task_id)}:steer"
        return await self._send_json(
            path,
            method="POST",
            payload={"prompt": prompt, "input_parts": list(input_parts or [])},
        )

    async def cancel_async_subagent(self, *, name_or_task_id: str, reason: str | None) -> dict[str, Any]:
        path = f"/api/v1/sessions/{urllib.parse.quote(self.session_id)}/async-tasks/{urllib.parse.quote(name_or_task_id)}:cancel"
        return await self._send_json(path, method="POST", payload={"reason": reason})

    async def list_schedules(
        self,
        *,
        schedule_id: str | None,
        include_disabled: bool,
        include_recent_runs: bool,
        limit: int,
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "owner_session_id": self.session_id,
            "include_deleted": "false",
            "include_recent_runs": "true" if include_recent_runs else "false",
            "limit": str(limit),
        }
        if isinstance(schedule_id, str) and schedule_id.strip() != "":
            params["schedule_id"] = schedule_id.strip()
        payload = await self._get_json("/api/v1/schedules", params=params)
        if include_disabled:
            return payload
        schedules = payload.get("schedules") if isinstance(payload, dict) else None
        if isinstance(schedules, list):
            payload["schedules"] = [
                item for item in schedules if isinstance(item, dict) and item.get("enabled") is True
            ]
        return payload

    async def create_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._send_json("/api/v1/schedules", method="POST", payload=payload)

    async def update_schedule(self, *, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = f"/api/v1/schedules/{urllib.parse.quote(schedule_id)}"
        return await self._send_json(path, method="PATCH", payload=payload)

    async def delete_schedule(self, *, schedule_id: str) -> dict[str, Any]:
        path = f"/api/v1/schedules/{urllib.parse.quote(schedule_id)}"
        return await self._send_json(path, method="DELETE", payload={})

    async def trigger_schedule(self, *, schedule_id: str, prompt_override: str | None) -> dict[str, Any]:
        path = f"/api/v1/schedules/{urllib.parse.quote(schedule_id)}:trigger"
        return await self._send_json(path, method="POST", payload={"prompt_override": prompt_override})

    async def _get_json(self, path: str, *, params: dict[str, str]) -> dict[str, Any]:
        payload = await asyncio.to_thread(self._get_json_sync, path, params)
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("YA Claw self API returned a non-object JSON payload.")

    async def _send_json(self, path: str, *, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        response_payload = await asyncio.to_thread(self._send_json_sync, path, method, payload)
        if isinstance(response_payload, dict):
            return response_payload
        raise RuntimeError("YA Claw self API returned a non-object JSON payload.")

    def _get_json_sync(self, path: str, params: dict[str, str]) -> JsonValue:
        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            raise RuntimeError("YA Claw self API URL must use http or https.")
        request = urllib.request.Request(  # noqa: S310
            url,
            headers={"Authorization": f"Bearer {self._api_token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _decode_error_detail(exc)
            raise RuntimeError(f"YA Claw self API request failed with status {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"YA Claw self API request failed: {exc.reason}") from exc

    def _send_json_sync(self, path: str, method: str, payload: dict[str, Any]) -> JsonValue:
        url = f"{self._base_url}{path}"
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            raise RuntimeError("YA Claw self API URL must use http or https.")
        request = urllib.request.Request(  # noqa: S310
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                raw_body = response.read().decode("utf-8")
                return json.loads(raw_body) if raw_body else {}
        except urllib.error.HTTPError as exc:
            detail = _decode_error_detail(exc)
            raise RuntimeError(f"YA Claw self API request failed with status {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"YA Claw self API request failed: {exc.reason}") from exc


class ListSessionTurnsTool(BaseTool):
    """List completed turns from the current YA Claw session."""

    name = "list_session_turns"
    description = "List completed turns from the current YA Claw session. Returns trimmed input and output JSON."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_self_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_self_client(ctx) is None:
            return None
        return (
            "List completed turns from the current YA Claw session. "
            "Use this to inspect earlier successful work in this conversation. "
            "The tool is scoped to the current session and returns trimmed JSON."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        limit: Annotated[int, Field(description="Maximum completed turns to fetch, clamped to 1..50")] = 10,
        before_sequence_no: Annotated[
            int | None,
            Field(description="Fetch completed turns with sequence_no lower than this value"),
        ] = None,
        cursor: Annotated[
            str | None,
            Field(description="Fetch completed turns older than this run ID cursor"),
        ] = None,
        max_input_chars: Annotated[int, Field(description="Maximum characters per turn input payload")] = 2000,
        max_output_chars: Annotated[int, Field(description="Maximum characters per turn output_text")] = 4000,
        include_binary_data: Annotated[bool, Field(description="Include trimmed binary base64 data when true")] = False,
    ) -> str:
        client = _get_self_client(ctx)
        if client is None:
            return "Error: YA Claw self-session client is unavailable."
        normalized_limit = min(max(limit, 1), 50)
        normalized_max_input_chars = min(max(max_input_chars, 200), 20000)
        normalized_max_output_chars = min(max(max_output_chars, 200), 50000)
        try:
            payload = await client.list_session_turns(
                limit=normalized_limit,
                before_sequence_no=before_sequence_no,
                cursor=cursor,
            )
            if payload.get("session_id") != client.session_id:
                return "Error: YA Claw self API returned a different session."
            return _dump_json(
                _compact_turns_payload(
                    payload,
                    max_input_chars=normalized_max_input_chars,
                    max_output_chars=normalized_max_output_chars,
                    include_binary_data=include_binary_data,
                )
            )
        except Exception as exc:
            return f"Error: {exc}"


class GetRunTraceTool(BaseTool):
    """Get tool trace for a run in the current YA Claw session."""

    name = "get_run_trace"
    description = "Get tool-call and tool-response trace for a run in the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_self_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_self_client(ctx) is None:
            return None
        return (
            "Get tool-call and tool-response trace for a run in the current YA Claw session. "
            "Use run IDs returned by list_session_turns. The tool rejects runs outside the current session."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        run_id: Annotated[str, Field(description="Run ID from the current YA Claw session")],
        max_item_chars: Annotated[int, Field(description="Maximum characters per trace item")] = 2000,
        max_total_chars: Annotated[int, Field(description="Maximum total characters across trace items")] = 8000,
    ) -> str:
        client = _get_self_client(ctx)
        if client is None:
            return "Error: YA Claw self-session client is unavailable."
        normalized_item_chars = min(max(max_item_chars, 256), 20000)
        normalized_total_chars = min(max(max_total_chars, normalized_item_chars), 100000)
        try:
            payload = await client.get_run_trace(
                run_id=run_id,
                max_item_chars=normalized_item_chars,
                max_total_chars=normalized_total_chars,
            )
            if payload.get("session_id") != client.session_id:
                return "Error: YA Claw self API returned a different session."
            return _dump_json(payload)
        except Exception as exc:
            return f"Error: {exc}"


def _get_self_client(ctx: RunContext[AgentContext]) -> SelfSessionClient | None:
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(CLAW_SELF_CLIENT_KEY)
    if isinstance(resource, SelfSessionClient):
        return resource
    return None


def _compact_turns_payload(
    payload: dict[str, Any],
    *,
    max_input_chars: int,
    max_output_chars: int,
    include_binary_data: bool,
) -> dict[str, Any]:
    turns = payload.get("turns")
    compact_turns: list[dict[str, Any]] = []
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                compact_turns.append(
                    _compact_turn(
                        turn,
                        max_input_chars=max_input_chars,
                        max_output_chars=max_output_chars,
                        include_binary_data=include_binary_data,
                    )
                )
    return {
        "session_id": payload.get("session_id"),
        "limit": payload.get("limit"),
        "has_more": payload.get("has_more"),
        "next_cursor": payload.get("next_cursor"),
        "next_before_sequence_no": payload.get("next_before_sequence_no"),
        "turns": compact_turns,
    }


def _compact_turn(
    turn: dict[str, Any],
    *,
    max_input_chars: int,
    max_output_chars: int,
    include_binary_data: bool,
) -> dict[str, Any]:
    sanitized_input_parts: JsonArray = _sanitize_input_parts(
        turn.get("input_parts"), include_binary_data=include_binary_data
    )
    input_parts, input_truncated = _trim_json_value(
        sanitized_input_parts,
        max_chars=max_input_chars,
    )
    output_text, output_truncated = _trim_text(_string_or_none(turn.get("output_text")), max_output_chars)
    return {
        "run_id": turn.get("run_id"),
        "sequence_no": turn.get("sequence_no"),
        "restore_from_run_id": turn.get("restore_from_run_id"),
        "profile_name": turn.get("profile_name"),
        "input_preview": turn.get("input_preview"),
        "input_parts": input_parts,
        "input_truncated": input_truncated,
        "output_text": output_text,
        "output_truncated": output_truncated,
        "created_at": turn.get("created_at"),
        "committed_at": turn.get("committed_at"),
    }


def _sanitize_input_parts(value: JsonValue, *, include_binary_data: bool) -> JsonArray:
    if not isinstance(value, list):
        return []
    return [
        _sanitize_input_part(part, include_binary_data=include_binary_data) for part in value if isinstance(part, dict)
    ]


def _sanitize_input_part(part: dict[str, Any], *, include_binary_data: bool) -> dict[str, Any]:
    sanitized = dict(part)
    if sanitized.get("type") == "binary":
        data = sanitized.get("data")
        if isinstance(data, str):
            sanitized["data_length"] = len(data)
            if include_binary_data:
                sanitized["data"], sanitized["data_truncated"] = _trim_text(data, 2000)
            else:
                sanitized.pop("data", None)
                sanitized["data_omitted"] = True
    return sanitized


def _trim_json_value(value: JsonValue, *, max_chars: int) -> tuple[JsonValue, bool]:
    dumped = _dump_json(value)
    if len(dumped) <= max_chars:
        return value, False
    return json.loads(json.dumps(_trim_text(dumped, max_chars)[0], ensure_ascii=False)), True


def _trim_text(value: str | None, max_chars: int) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _dump_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decode_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw_body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason
    if raw_body == "":
        return exc.reason
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body[:500]
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail[:500]
    return raw_body[:500]
