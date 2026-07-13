from __future__ import annotations

from pydantic_ai.usage import RunUsage
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot
from yaacli.usage import SessionUsage


def _snapshot(run_id: str, output_audio_tokens: int) -> UsageSnapshot:
    usage = RunUsage(output_audio_tokens=output_audio_tokens)
    return UsageSnapshot(
        run_id=run_id,
        total_usage=usage,
        agent_usages={"main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=usage)},
        model_usages={"model-a": usage},
    )


def test_late_usage_replacement_subtracts_output_audio_tokens() -> None:
    session = SessionUsage()
    session.set_run_snapshot(_snapshot("run-1", 10))
    session.commit_run_snapshot()
    session.set_run_snapshot(_snapshot("run-2", 10))
    session.commit_run_snapshot()

    session.set_run_snapshot(_snapshot("run-1", 20))
    session.commit_run_snapshot()

    assert session.model_usages["model-a"].output_audio_tokens == 30
