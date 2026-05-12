from __future__ import annotations

import asyncio

import pytest
from ya_claw.bridge.lark.adapter import LarkBridgeAdapter
from ya_claw.bridge.lark.card import build_hitl_card, build_recovery_card
from ya_claw.bridge.lark.normalizer import normalize_lark_action
from ya_claw.config import ClawSettings
from ya_claw.controller.models import ActiveInteraction


def test_build_hitl_card_special_cases_shell_review() -> None:
    interaction = ActiveInteraction(
        interaction_id="hitl-1",
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-1",
        tool_name="shell_exec",
        kind="shell_review",
        title="Shell command approval required",
        description="risky command",
        metadata={
            "reviewer": "shell_command_reviewer",
            "risk_level": "extra_high",
            "command": "rm -rf dist",
            "cwd": "/workspace",
        },
        sequence_no=1,
        total_count=2,
    )

    card = build_hitl_card(interaction)

    assert card["header"]["template"] == "red"
    assert card["header"]["title"]["content"] == "YA Claw approval required"
    content = "\n".join(str(element) for element in card["elements"])
    assert "rm -rf dist" in content
    assert "extra_high" in content
    assert "Approve" in content
    assert "Deny" in content


def test_build_hitl_card_supports_generic_tool_approval() -> None:
    interaction = ActiveInteraction(
        interaction_id="hitl-2",
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-2",
        tool_name="file_write",
        kind="tool_approval",
        title="Tool approval required: file_write",
        arguments_preview={"path": "README.md"},
    )

    card = build_hitl_card(interaction)

    assert card["header"]["template"] == "blue"
    content = "\n".join(str(element) for element in card["elements"])
    assert "file_write" in content
    assert "README.md" in content


def test_build_recovery_card_supports_retry_actions() -> None:
    card = build_recovery_card({
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence_no": 3,
        "error_message": "broken message history",
    })

    assert card["header"]["template"] == "red"
    assert card["header"]["title"]["content"] == "YA Claw run failed"
    content = "\n".join(str(element) for element in card["elements"])
    assert "broken message history" in content
    assert "Retry" in content
    assert "Reset and retry" in content
    assert "recovery:session-1:run-1" in content


def test_normalize_lark_hitl_action() -> None:
    action = normalize_lark_action({
        "header": {"event_id": "event-1", "tenant_key": "tenant-1"},
        "event": {
            "action": {
                "value": {
                    "action": "approve",
                    "interaction_token": "session-1:run-1:interaction-1:1",
                }
            }
        },
    })

    assert action is not None
    assert action.tenant_key == "tenant-1"
    assert action.event_id == "event-1"
    assert action.action_type == "hitl_respond"
    assert action.approved is True
    expected = "session-1:run-1:interaction-1:1"
    assert action.token == expected


def test_normalize_lark_recovery_action() -> None:
    action = normalize_lark_action({
        "header": {"event_id": "event-2", "tenant_key": "tenant-1"},
        "event": {
            "action": {
                "value": {
                    "action": "reset_and_retry",
                    "recovery_token": "recovery:session-1:run-1",
                }
            }
        },
    })

    assert action is not None
    assert action.tenant_key == "tenant-1"
    assert action.event_id == "event-2"
    assert action.action_type == "session_recovery"
    assert action.approved is False
    expected_value = "recovery:session-1:run-1"
    assert action.token == expected_value
    assert action.metadata["action"] == "reset_and_retry"


@pytest.mark.asyncio
async def test_lark_adapter_installs_websocket_card_action_handler() -> None:
    submitted_actions: list[object] = []
    submitted_coroutines: list[object] = []

    class Handler:
        async def handle_action(self, action: object) -> object:
            submitted_actions.append(action)
            return object()

        async def handle_message(self, message: object) -> object:
            return object()

    writes: list[bytes] = []

    class FakeClient:
        async def _write_message(self, data: bytes) -> None:
            writes.append(data)

        async def _handle_message(self, message: bytes) -> None:
            raise AssertionError("card frames should be handled before the SDK event path")

    import lark_oapi as lark
    from lark_oapi.ws.enum import MessageType
    from lark_oapi.ws.pb.pbbp2_pb2 import Frame

    adapter = LarkBridgeAdapter(
        settings=ClawSettings(api_token="test-token", _env_file=None),  # noqa: S106
        handler=Handler(),  # type: ignore[arg-type]
    )
    adapter._loop = asyncio.get_running_loop()

    def submit_from_sdk_thread(coroutine: object) -> None:
        submitted_coroutines.append(coroutine)
        if hasattr(coroutine, "close"):
            coroutine.close()

    adapter._submit_from_sdk_thread = submit_from_sdk_thread  # type: ignore[method-assign]
    client = FakeClient()

    adapter._install_card_action_handler(client, lark)

    frame = Frame()
    frame.SeqID = 1
    frame.LogID = 1
    frame.service = 1
    frame.method = 1
    header = frame.headers.add()
    header.key = "type"
    header.value = MessageType.CARD.value
    frame.payload = lark.JSON.marshal({
        "tenant_key": "tenant-1",
        "action": {
            "value": {
                "action": "approve",
                "interaction_token": "session-1:run-1:interaction-1:1",
            }
        },
    }).encode("utf-8")

    await client._handle_message(frame.SerializeToString())

    assert submitted_coroutines
    assert writes
