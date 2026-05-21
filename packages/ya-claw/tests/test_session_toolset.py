from __future__ import annotations

import json
import urllib.request
from typing import Any

from ya_agent_environment import Environment
from ya_agent_sdk.context import AgentContext
from ya_claw.toolsets.session import (
    CLAW_SELF_CLIENT_KEY,
    ClawSelfClient,
    GetRunTraceTool,
    ListSessionTurnsTool,
    SelfSessionClient,
)


class EmptyEnvironment(Environment):
    async def _setup(self) -> None:
        return None

    async def _teardown(self) -> None:
        return None


class FakeRunContext:
    def __init__(self, deps: AgentContext) -> None:
        self.deps = deps


class FakeSelfClient:
    def __init__(self, *, session_id: str = "session-1") -> None:
        self.session_id = session_id
        self.turn_calls: list[dict[str, Any]] = []
        self.trace_calls: list[dict[str, Any]] = []

    def close(self) -> None:
        return None

    async def setup(self) -> None:
        return None

    def get_toolsets(self) -> list[Any]:
        return []

    async def list_session_turns(
        self,
        *,
        limit: int,
        before_sequence_no: int | None,
        cursor: str | None,
    ) -> dict[str, Any]:
        self.turn_calls.append({"limit": limit, "before_sequence_no": before_sequence_no, "cursor": cursor})
        return {
            "session_id": self.session_id,
            "limit": limit,
            "has_more": True,
            "next_before_sequence_no": 2,
            "turns": [
                {
                    "run_id": "run-1",
                    "sequence_no": 2,
                    "restore_from_run_id": "run-0",
                    "profile_name": "default",
                    "input_preview": "hello",
                    "input_parts": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "binary",
                            "data": "a" * 5000,
                            "mime_type": "image/png",
                            "kind": "image",
                            "filename": "image.png",
                        },
                    ],
                    "output_text": "o" * 5000,
                    "created_at": "2026-04-25T00:00:00Z",
                    "committed_at": "2026-04-25T00:00:01Z",
                }
            ],
        }

    async def get_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]:
        self.trace_calls.append({
            "run_id": run_id,
            "max_item_chars": max_item_chars,
            "max_total_chars": max_total_chars,
        })
        return {
            "run_id": run_id,
            "session_id": self.session_id,
            "item_count": 1,
            "max_item_chars": max_item_chars,
            "max_total_chars": max_total_chars,
            "truncated": False,
            "trace": [
                {
                    "sequence_no": 1,
                    "type": "tool_call",
                    "tool_call_id": "tool-1",
                    "tool_name": "shell_exec",
                    "content": '{"command":"echo ok"}',
                    "truncated": False,
                }
            ],
        }


def _context_with_client(client: SelfSessionClient) -> FakeRunContext:
    env = EmptyEnvironment()
    ctx = AgentContext(agent_id="main", env=env)
    assert ctx.resources is not None
    ctx.resources.set(CLAW_SELF_CLIENT_KEY, client)
    return FakeRunContext(ctx)


def test_session_tools_are_available_with_self_client() -> None:
    ctx = _context_with_client(FakeSelfClient())

    assert ListSessionTurnsTool().is_available(ctx) is True  # type: ignore[arg-type]
    assert GetRunTraceTool().is_available(ctx) is True  # type: ignore[arg-type]


async def test_list_session_turns_trims_output_and_omits_binary_data() -> None:
    client = FakeSelfClient()
    ctx = _context_with_client(client)

    result = await ListSessionTurnsTool().call(
        ctx,  # type: ignore[arg-type]
        limit=100,
        before_sequence_no=4,
        max_input_chars=80,
        max_output_chars=300,
    )

    payload = json.loads(result)
    assert client.turn_calls == [{"limit": 50, "before_sequence_no": 4, "cursor": None}]
    assert payload["session_id"] == "session-1"
    turn = payload["turns"][0]
    assert turn["run_id"] == "run-1"
    assert turn["output_text"] == "o" * 300
    assert turn["output_truncated"] is True
    assert turn["input_truncated"] is False
    assert isinstance(turn["input_parts"], list)
    binary_part = turn["input_parts"][1]
    assert "data" not in binary_part
    assert binary_part["data_omitted"] is True
    assert binary_part["data_length"] == 5000


async def test_list_session_turns_can_include_trimmed_binary_data() -> None:
    client = FakeSelfClient()
    ctx = _context_with_client(client)

    result = await ListSessionTurnsTool().call(
        ctx,  # type: ignore[arg-type]
        max_input_chars=4000,
        include_binary_data=True,
    )

    payload = json.loads(result)
    input_parts = payload["turns"][0]["input_parts"]
    assert isinstance(input_parts, list)
    binary_part = input_parts[1]
    assert binary_part["data"] == "a" * 2000
    assert binary_part["data_truncated"] is True


async def test_get_run_trace_uses_current_session_client() -> None:
    client = FakeSelfClient()
    ctx = _context_with_client(client)

    result = await GetRunTraceTool().call(
        ctx,  # type: ignore[arg-type]
        run_id="run-1",
        max_item_chars=10,
        max_total_chars=20,
    )

    payload = json.loads(result)
    assert client.trace_calls == [{"run_id": "run-1", "max_item_chars": 256, "max_total_chars": 256}]
    assert payload["session_id"] == "session-1"
    assert payload["trace"][0]["tool_name"] == "shell_exec"


async def test_get_run_trace_returns_error_for_cross_session_run() -> None:
    class CrossSessionClient(FakeSelfClient):
        async def get_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]:
            return {"run_id": run_id, "session_id": "session-2", "trace": []}

    ctx = _context_with_client(CrossSessionClient())

    result = await GetRunTraceTool().call(ctx, run_id="run-2")  # type: ignore[arg-type]

    assert result == "Error: YA Claw self API returned a different session."


def test_claw_self_client_builds_session_scoped_authorized_requests(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"session_id": "session-1", "turns": []}).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["authorization"] = request.headers.get("Authorization")
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ClawSelfClient(
        base_url="http://127.0.0.1:9042/",
        api_token="secret-token",  # noqa: S106
        session_id="session-1",
        timeout_seconds=3.0,
    )

    result = client._get_json_sync("/api/v1/sessions/session-1/turns", {"limit": "2"})

    assert result == {"session_id": "session-1", "turns": []}
    assert seen["url"] == "http://127.0.0.1:9042/api/v1/sessions/session-1/turns?limit=2"
    assert seen["authorization"] == "Bearer secret-token"
    assert seen["timeout"] == 3.0
