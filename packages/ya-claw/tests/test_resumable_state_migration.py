from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from ya_claw.execution.resumable_state import load_resumable_state


def test_load_resumable_state_upgrades_unversioned_claw_state() -> None:
    state = load_resumable_state({
        "subagent_history": {"code-reviewer-1": [{"kind": "request", "parts": []}]},
        "steering_messages": ["Focus on the restore failure"],
        "agent_registry": {
            "code-reviewer-1": {
                "agent_id": "code-reviewer-1",
                "agent_name": "code-reviewer",
                "parent_agent_id": "main",
            }
        },
        "auto_load_files": ["packages/ya-claw/README.md"],
        "tool_search_loaded_tools": ["view"],
        "tool_search_loaded_namespaces": ["builtin:filesystem"],
        "notes": {"decision": "keep the main conversation"},
        "tasks": {},
    })

    assert state.schema_version == 2
    assert state.files_to_inspect == ["packages/ya-claw/README.md"]
    assert state.tool_proxy.loaded_tools == ["view"]
    assert state.tool_proxy.loaded_namespaces == ["builtin:filesystem"]
    assert state.notes == {"decision": "keep the main conversation"}

    applied_messages = state.run_input_ledger.applied_user_messages()
    assert len(applied_messages) == 1
    request = applied_messages[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content == "Focus on the restore failure"


@pytest.mark.parametrize(
    ("raw_state", "expected_tools", "expected_namespaces"),
    [
        ({"tool_search_loaded_tools": ["view"]}, ["view"], []),
        ({"tool_search_loaded_namespaces": ["builtin:filesystem"]}, [], ["builtin:filesystem"]),
    ],
)
def test_load_resumable_state_preserves_independent_tool_proxy_fields(
    raw_state: dict[str, object],
    expected_tools: list[str],
    expected_namespaces: list[str],
) -> None:
    state = load_resumable_state(raw_state)

    assert state.tool_proxy.loaded_tools == expected_tools
    assert state.tool_proxy.loaded_namespaces == expected_namespaces


def test_load_resumable_state_keeps_versioned_state_strict() -> None:
    with pytest.raises(ValidationError, match="subagent_history"):
        load_resumable_state({
            "schema_version": 2,
            "subagent_history": {},
        })


def test_load_resumable_state_rejects_malformed_legacy_fields() -> None:
    with pytest.raises(ValidationError, match="steering_messages"):
        load_resumable_state({
            "steering_messages": [1],
        })
