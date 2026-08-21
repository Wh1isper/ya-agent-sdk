"""YA Claw migration boundary for persisted SDK resumable state."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelRequest, UserPromptPart
from ya_agent_sdk.context import ResumableState
from ya_agent_sdk.inputs import InputOrigin, RunInputLedger


def load_resumable_state(raw_state: dict[str, Any]) -> ResumableState:
    """Load persisted SDK state, upgrading the pre-2.0 shape when necessary.

    SDK 2.0 state is versioned and strict. YA Claw persisted unversioned 1.x state in
    run artifacts, so the host upgrades that exact boundary before strict validation.
    Versioned state never receives compatibility normalization.
    """
    state = raw_state
    if "schema_version" not in state:
        state = _upgrade_unversioned_state(state)
    return ResumableState.model_validate(state)


def _upgrade_unversioned_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(raw_state)

    _drop_legacy_mapping(state, "subagent_history")
    _drop_legacy_mapping(state, "agent_registry")

    steering_messages = _pop_legacy_string_list(state, "steering_messages")
    if steering_messages is not None and "run_input_ledger" not in state:
        ledger = RunInputLedger()
        for message in steering_messages:
            ledger.record_initial(
                [ModelRequest(parts=[UserPromptPart(content=message)])],
                origin=InputOrigin.user,
            )
        state["run_input_ledger"] = ledger.model_dump(mode="json")

    auto_load_files = _pop_legacy_string_list(state, "auto_load_files")
    if auto_load_files is not None and "files_to_inspect" not in state:
        state["files_to_inspect"] = auto_load_files

    loaded_tools = _pop_legacy_string_list(state, "tool_search_loaded_tools")
    loaded_namespaces = _pop_legacy_string_list(state, "tool_search_loaded_namespaces")
    if (loaded_tools is not None or loaded_namespaces is not None) and "tool_proxy" not in state:
        state["tool_proxy"] = {
            "loaded_tools": loaded_tools or [],
            "loaded_namespaces": loaded_namespaces or [],
        }

    state["schema_version"] = 2
    return state


def _drop_legacy_mapping(state: dict[str, Any], key: str) -> None:
    if isinstance(state.get(key), dict):
        state.pop(key)


def _pop_legacy_string_list(state: dict[str, Any], key: str) -> list[str] | None:
    value = state.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    state.pop(key)
    return value
