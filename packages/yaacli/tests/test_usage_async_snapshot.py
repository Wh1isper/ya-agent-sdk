from __future__ import annotations

from pydantic_ai.usage import RunUsage
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot
from yaacli.usage import SessionUsage


def _snapshot(*, run_id: str, include_subagent: bool) -> UsageSnapshot:
    agent_usages = {
        "main": UsageAgentTotal(
            agent_name="main",
            model_id="model-main",
            usage=RunUsage(requests=1, input_tokens=10, output_tokens=5),
        )
    }
    model_usages = {"model-main": RunUsage(requests=1, input_tokens=10, output_tokens=5)}
    if include_subagent:
        agent_usages["executor"] = UsageAgentTotal(
            agent_name="executor",
            model_id="model-sub",
            usage=RunUsage(requests=1, input_tokens=20, output_tokens=7),
        )
        model_usages["model-sub"] = RunUsage(
            requests=1,
            input_tokens=20,
            output_tokens=7,
        )
    return UsageSnapshot(
        run_id=run_id,
        total_usage=RunUsage(
            requests=2 if include_subagent else 1,
            input_tokens=30 if include_subagent else 10,
            output_tokens=12 if include_subagent else 5,
        ),
        agent_usages=agent_usages,
        model_usages=model_usages,
    )


def test_late_durable_subagent_snapshot_replaces_same_run_totals() -> None:
    usage = SessionUsage()
    usage.set_run_snapshot(_snapshot(run_id="run-1", include_subagent=False))
    usage.commit_run_snapshot("run-1")

    usage.set_run_snapshot(_snapshot(run_id="run-1", include_subagent=True))
    usage.commit_run_snapshot("run-1")

    assert usage.total_input_tokens == 30
    assert usage.total_output_tokens == 12
    assert usage.total_requests == 2
    assert usage.agent_usages["executor"].input_tokens == 20
    assert usage.model_usages["model-sub"].output_tokens == 7
