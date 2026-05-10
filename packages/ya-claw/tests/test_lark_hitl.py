from __future__ import annotations

from ya_claw.bridge.lark.card import build_hitl_card
from ya_claw.bridge.lark.normalizer import normalize_lark_action
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
    assert action.approved is True
    expected = "session-1:run-1:interaction-1:1"
    assert action.token == expected
