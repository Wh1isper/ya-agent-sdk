"""Tests for YAACLI typed events and context inheritance."""

from __future__ import annotations

from ya_agent_sdk.context import AgentContext
from yaacli.session import TUIContext


def test_tui_context_inherits_from_agent_context() -> None:
    assert isinstance(TUIContext(), AgentContext)


def test_subagent_context_shares_host_authority_but_owns_resumable_state() -> None:
    parent = TUIContext()

    child = parent.create_subagent_context("search")

    assert isinstance(child, AgentContext)
    assert child.task_manager is not parent.task_manager
    assert child.note_manager is not parent.note_manager
    assert child.active_run_registry is parent.active_run_registry
    assert child.run_input_ledger is not parent.run_input_ledger
