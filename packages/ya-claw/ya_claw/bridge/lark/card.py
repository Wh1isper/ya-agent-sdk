from __future__ import annotations

import json
from typing import Any

from ya_claw.controller.models import ActiveInteraction


def build_hitl_card(interaction: ActiveInteraction | None, *, completed: bool = False) -> dict[str, Any]:
    if completed or interaction is None:
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "green",
                "title": {"tag": "plain_text", "content": "YA Claw approval completed"},
            },
            "elements": [
                {"tag": "div", "text": {"tag": "plain_text", "content": "All pending approvals were handled."}}
            ],
        }

    metadata = interaction.metadata
    template = _template_for_interaction(interaction)
    detail_lines = [
        f"**Progress:** {interaction.sequence_no} / {interaction.total_count}",
        f"**Tool:** {interaction.tool_name or 'unknown'}",
    ]
    risk_level = metadata.get("risk_level")
    if isinstance(risk_level, str) and risk_level.strip():
        detail_lines.append(f"**Risk:** {risk_level.strip()}")
    reason = interaction.description or metadata.get("reason")
    if isinstance(reason, str) and reason.strip():
        detail_lines.append(f"**Reason:** {reason.strip()}")
    cwd = metadata.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        detail_lines.append(f"**cwd:** {cwd.strip()}")

    elements: list[dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(detail_lines)}},
    ]
    command = metadata.get("command")
    if isinstance(command, str) and command.strip():
        elements.extend([
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": "**Command**"}},
            {"tag": "div", "text": {"tag": "plain_text", "content": _truncate(command.strip(), 1800)}},
        ])
    elif interaction.arguments_preview is not None:
        elements.extend([
            {"tag": "hr"},
            {"tag": "div", "text": {"tag": "lark_md", "content": "**Arguments**"}},
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": _truncate(_format_arguments(interaction.arguments_preview), 1800),
                },
            },
        ])

    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "Approve"},
                "type": "primary",
                "value": {"action": "approve", "interaction_token": _interaction_token(interaction)},
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "Deny"},
                "type": "danger",
                "value": {"action": "deny", "interaction_token": _interaction_token(interaction)},
            },
        ],
    })
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": "YA Claw approval required"},
            "subtitle": {"tag": "plain_text", "content": interaction.title},
        },
        "elements": elements,
    }


def build_recovery_card(payload: dict[str, Any], *, submitted_action: str | None = None) -> dict[str, Any]:
    session_id = _string_value(payload.get("session_id"))
    run_id = _string_value(payload.get("run_id"))
    sequence_no = payload.get("sequence_no")
    error_message = _string_value(payload.get("error_message"))
    detail = payload.get("session_status_detail")
    if error_message is None and isinstance(detail, dict):
        error_message = _string_value(detail.get("error_message"))
    if submitted_action is not None:
        title = "YA Claw recovery submitted"
        template = "blue"
        action_label = "Retry" if submitted_action == "retry" else "Reset and retry"
        elements: list[dict[str, Any]] = [
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": f"{action_label} was submitted for run {run_id or 'unknown'}.",
                },
            }
        ]
    else:
        title = "YA Claw run failed"
        template = "red"
        detail_lines = [
            f"**Session:** {session_id or 'unknown'}",
            f"**Run:** {run_id or 'unknown'}",
        ]
        if isinstance(sequence_no, int):
            detail_lines.append(f"**Sequence:** {sequence_no}")
        elements = [{"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(detail_lines)}}]
        if error_message is not None:
            elements.extend([
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": "**Error**"}},
                {"tag": "div", "text": {"tag": "plain_text", "content": _truncate(error_message, 1800)}},
            ])
        if session_id is not None and run_id is not None:
            token = _recovery_token(session_id, run_id)
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Retry"},
                        "type": "primary",
                        "value": {"action": "retry", "recovery_token": token},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Reset and retry"},
                        "type": "danger",
                        "value": {"action": "reset_and_retry", "recovery_token": token},
                    },
                ],
            })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


def _interaction_token(interaction: ActiveInteraction) -> str:
    return f"{interaction.session_id}:{interaction.run_id}:{interaction.interaction_id}:{interaction.sequence_no}"


def _recovery_token(session_id: str, run_id: str) -> str:
    return f"recovery:{session_id}:{run_id}"


def _template_for_interaction(interaction: ActiveInteraction) -> str:
    risk_level = interaction.metadata.get("risk_level")
    if risk_level in {"extra_high", "high"}:
        return "red" if risk_level == "extra_high" else "orange"
    if interaction.kind == "shell_review":
        return "orange"
    return "blue"


def _format_arguments(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _string_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
