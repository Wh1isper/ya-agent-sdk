from __future__ import annotations

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolApproved, ToolDenied
from ya_agent_sdk.interactions import (
    DeferredApprovalResolution,
    DeferredCallResolution,
    DeferredInteractionResolver,
)


def test_deferred_interaction_resolver_separates_calls_and_approvals() -> None:
    requests = DeferredToolRequests(
        calls=[ToolCallPart("external", {}, tool_call_id="call")],
        approvals=[ToolCallPart("shell", {}, tool_call_id="approval")],
    )

    results = DeferredInteractionResolver().resolve(
        requests,
        [
            DeferredCallResolution(tool_call_id="call", result={"answer": 42}),
            DeferredApprovalResolution(
                tool_call_id="approval",
                approved=True,
                override_args={"command": "pwd"},
            ),
        ],
    )

    assert results.calls == {"call": {"answer": 42}}
    approval = results.approvals["approval"]
    assert isinstance(approval, ToolApproved)
    assert approval.override_args == {"command": "pwd"}


def test_deferred_interaction_resolver_rejects_wrong_kind_and_missing_result() -> None:
    requests = DeferredToolRequests(approvals=[ToolCallPart("shell", {}, tool_call_id="approval")])
    resolver = DeferredInteractionResolver()

    with pytest.raises(ValueError, match="not an external call"):
        resolver.resolve(
            requests,
            [DeferredCallResolution(tool_call_id="approval", result="wrong")],
        )
    with pytest.raises(ValueError, match="missing"):
        resolver.resolve(requests, [])


def test_deferred_interaction_resolver_preserves_denial_reason() -> None:
    requests = DeferredToolRequests(approvals=[ToolCallPart("shell", {}, tool_call_id="approval")])

    results = DeferredInteractionResolver().resolve(
        requests,
        [
            DeferredApprovalResolution(
                tool_call_id="approval",
                approved=False,
                reason="not now",
            )
        ],
    )

    denial = results.approvals["approval"]
    assert isinstance(denial, ToolDenied)
    assert denial.message == "not now"
