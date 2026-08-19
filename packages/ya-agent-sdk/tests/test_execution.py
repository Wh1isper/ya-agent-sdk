from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import DeferredToolRequests, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.context import AgentContext, StreamEvent
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.execution import (
    AgentExecutionHarness,
    AgentSegmentRequest,
    AgentSegmentStatus,
)


def _environment(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )


async def test_execution_harness_returns_typed_completed_checkpoint(tmp_path: Path) -> None:
    runtime = create_agent(
        TestModel(custom_output_text="finished"),
        env=_environment(tmp_path),
    )
    events: list[StreamEvent] = []

    async def collect(event: StreamEvent) -> None:
        events.append(event)

    outcome = await AgentExecutionHarness().execute_segment(
        runtime,
        AgentSegmentRequest(user_prompt="do the work"),
        event_sink=collect,
    )

    assert outcome.status is AgentSegmentStatus.completed
    assert outcome.output == "finished"
    assert outcome.deferred_requests is None
    assert outcome.checkpoint.messages
    assert outcome.checkpoint.state.schema_version == 2
    assert outcome.checkpoint.usage.total_usage.requests >= 1
    assert outcome.checkpoint.input_ledger.logical_run_id == (outcome.checkpoint.state.run_input_ledger.logical_run_id)
    assert outcome.checkpoint.input_ledger is not outcome.checkpoint.state.run_input_ledger
    assert events


@dataclass
class _ApprovalCapability(AbstractCapability[AgentContext]):
    def get_toolset(self) -> FunctionToolset[AgentContext]:
        async def guarded_effect(value: str) -> str:
            return value

        return FunctionToolset(
            [Tool(guarded_effect, requires_approval=True)],
            id="approval",
        )


async def test_execution_harness_exposes_suspended_segment_and_stream_controls(
    tmp_path: Path,
) -> None:
    runtime = create_agent(
        TestModel(call_tools=["guarded_effect"]),
        env=_environment(tmp_path),
        capabilities=[_ApprovalCapability()],
        output_type=[str, DeferredToolRequests],
    )
    harness = AgentExecutionHarness()

    async with harness.stream_segment(
        runtime,
        AgentSegmentRequest(user_prompt="run the guarded effect"),
    ) as segment:
        async for _event in segment:
            pass
        segment.raise_if_exception()
        outcome = segment.outcome()

    assert outcome.status is AgentSegmentStatus.suspended
    assert isinstance(outcome.output, DeferredToolRequests)
    assert outcome.deferred_requests is outcome.output
    assert outcome.checkpoint.messages
    assert outcome.checkpoint.state.run_input_ledger.records
