"""Tests for usage tracking module."""

from __future__ import annotations

from decimal import Decimal

from pydantic_ai.usage import RunUsage
from yaacli.usage import SessionUsage


class TestSessionUsage:
    """Tests for SessionUsage dataclass."""

    def test_is_empty(self) -> None:
        """Test is_empty on fresh instance."""
        session = SessionUsage()
        assert session.is_empty()

    def test_add_creates_entries(self) -> None:
        """Test adding usage creates agent and model entries."""
        session = SessionUsage()
        run_usage = RunUsage(input_tokens=100, output_tokens=50, requests=1)

        session.add("main", "openai-chat:gpt-4o", run_usage)

        assert not session.is_empty()
        assert "main" in session.agent_usages
        assert "openai-chat:gpt-4o" in session.model_usages
        assert session.agent_usages["main"].input_tokens == 100
        assert session.model_usages["openai-chat:gpt-4o"].input_tokens == 100

    def test_add_multiple_agents_same_model(self) -> None:
        """Test multiple agents using the same model."""
        session = SessionUsage()

        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("explorer", "openai-chat:gpt-4o", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        # Agent usages are separate
        assert len(session.agent_usages) == 2
        assert session.agent_usages["main"].input_tokens == 100
        assert session.agent_usages["explorer"].input_tokens == 200

        # Model usage is accumulated
        assert len(session.model_usages) == 1
        assert session.model_usages["openai-chat:gpt-4o"].input_tokens == 300

    def test_add_same_agent_different_models(self) -> None:
        """Test same agent using different models (e.g., image_understanding)."""
        session = SessionUsage()

        session.add(
            "image_understanding", "openai-chat:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1)
        )
        session.add(
            "image_understanding",
            "anthropic:claude-sonnet-4",
            RunUsage(input_tokens=200, output_tokens=100, requests=1),
        )

        # Agent usage is accumulated
        assert len(session.agent_usages) == 1
        assert session.agent_usages["image_understanding"].input_tokens == 300

        # Model usages are separate
        assert len(session.model_usages) == 2
        assert session.model_usages["openai-chat:gpt-4o"].input_tokens == 100
        assert session.model_usages["anthropic:claude-sonnet-4"].input_tokens == 200

    def test_add_same_agent_accumulates(self) -> None:
        """Test adding to same agent accumulates."""
        session = SessionUsage()

        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        assert session.agent_usages["main"].input_tokens == 300
        assert session.agent_usages["main"].requests == 2
        assert session.model_usages["openai-chat:gpt-4o"].input_tokens == 300

    def test_totals(self) -> None:
        """Test total calculations across all models."""
        session = SessionUsage()

        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("explorer", "anthropic:claude-sonnet-4", RunUsage(input_tokens=200, output_tokens=100, requests=2))

        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150
        assert session.total_tokens == 450
        assert session.total_requests == 3

    def test_clear(self) -> None:
        """Test clearing session usage."""
        session = SessionUsage()
        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))

        session.clear()

        assert session.is_empty()
        assert session.total_tokens == 0
        assert len(session.agent_usages) == 0
        assert len(session.model_usages) == 0

    def test_format_summary_empty(self) -> None:
        """Test format_summary on empty session."""
        session = SessionUsage()
        summary = session.format_summary()
        assert "No usage data" in summary

    def test_format_summary_with_data(self) -> None:
        """Test format_summary with data."""
        session = SessionUsage()
        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=1000, output_tokens=500, requests=2))
        session.add("explorer", "anthropic:claude-sonnet-4", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        summary = session.format_summary()

        assert "Token Usage Summary" in summary
        # By Model section
        assert "By Model:" in summary
        assert "openai-chat:gpt-4o:" in summary
        assert "anthropic:claude-sonnet-4:" in summary
        # By Agent section
        assert "By Agent:" in summary
        assert "main:" in summary
        assert "explorer:" in summary
        # Formatting
        assert "1,000" in summary  # Comma formatting
        assert "Total:" in summary

    def test_grok_4_5_cost_estimate(self) -> None:
        """Test Grok 4.5 local cost estimate in summary."""
        session = SessionUsage()
        session.add(
            "main",
            "xai:grok-4.5",
            RunUsage(input_tokens=1_000_000, cache_read_tokens=200_000, output_tokens=100_000, requests=1),
        )

        summary = session.format_summary()

        assert "Estimated Cost: $2.30" in summary
        assert session.estimated_total_cost is not None
        assert session.estimated_total_cost.total_cost == Decimal("2.30")

    def test_preserves_details(self) -> None:
        """Test that details field is accumulated."""
        session = SessionUsage()
        # Details values must be numeric for accumulation
        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, details={"cached_tokens": 50}))
        session.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=100, details={"cached_tokens": 30}))

        assert session.agent_usages["main"].details == {"cached_tokens": 80}
        assert session.model_usages["openai-chat:gpt-4o"].details == {"cached_tokens": 80}

    def test_cache_tokens(self) -> None:
        """Test cache token tracking."""
        session = SessionUsage()
        session.add(
            "main",
            "openai-chat:gpt-4o",
            RunUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=20,
                cache_write_tokens=10,
            ),
        )

        usage = session.agent_usages["main"]
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 10

        model_usage = session.model_usages["openai-chat:gpt-4o"]
        assert model_usage.cache_read_tokens == 20
        assert model_usage.cache_write_tokens == 10


class _CallableRunUsage:
    def __init__(self, usage: RunUsage) -> None:
        self._usage = usage
        self.details = {"provider_cached_tokens": 999}

    def __call__(self) -> RunUsage:
        return self._usage


def test_session_usage_normalizes_callable_usage_wrapper() -> None:
    session = SessionUsage()
    session.add("main", "openai-chat:gpt-4o", _CallableRunUsage(RunUsage(input_tokens=7, details={"cached": 2})))  # type: ignore[arg-type]

    assert type(session.agent_usages["main"]) is RunUsage
    assert session.agent_usages["main"].input_tokens == 7
    assert session.agent_usages["main"].details == {"cached": 2}


def test_session_usage_replaces_uncommitted_run_snapshot() -> None:
    """Realtime usage snapshots should replace the current run totals."""
    from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

    session = SessionUsage()
    session.add("previous", "model-a", RunUsage(requests=1, input_tokens=1, output_tokens=2))

    first = UsageSnapshot(
        run_id="run-1",
        total_usage=RunUsage(requests=1, input_tokens=10, output_tokens=20),
        agent_usages={
            "main": UsageAgentTotal(
                agent_name="main",
                model_id="model-b",
                usage=RunUsage(requests=1, input_tokens=10, output_tokens=20),
            )
        },
        model_usages={"model-b": RunUsage(requests=1, input_tokens=10, output_tokens=20)},
    )
    session.set_run_snapshot(first)

    assert session.total_requests == 2
    assert session.model_usages["model-b"].input_tokens == 10

    second = UsageSnapshot(
        run_id="run-1",
        total_usage=RunUsage(requests=2, input_tokens=30, output_tokens=40),
        agent_usages={
            "main": UsageAgentTotal(
                agent_name="main",
                model_id="model-b",
                usage=RunUsage(requests=2, input_tokens=30, output_tokens=40),
            )
        },
        model_usages={"model-b": RunUsage(requests=2, input_tokens=30, output_tokens=40)},
    )
    session.set_run_snapshot(second)

    assert session.total_requests == 3
    assert session.total_input_tokens == 31
    assert session.total_output_tokens == 42

    session.commit_run_snapshot()
    assert not session.has_run_snapshot
    assert session.total_requests == 3


def test_session_usage_finalization_releases_replacement_metadata() -> None:
    from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

    usage = SessionUsage()
    for index in range(140):
        snapshot = UsageSnapshot(
            run_id=f"run-{index}",
            total_usage=RunUsage(input_tokens=1, output_tokens=2, requests=1),
            agent_usages={
                "main": UsageAgentTotal(
                    agent_name="main",
                    model_id="model-a",
                    usage=RunUsage(input_tokens=1, output_tokens=2, requests=1),
                )
            },
            model_usages={"model-a": RunUsage(input_tokens=1, output_tokens=2, requests=1)},
        )
        usage.set_run_snapshot(snapshot)
        usage.commit_run_snapshot("run-" + str(index))

    assert usage._run_snapshots == {}
    assert len(usage._committed_run_contributions) == 140
    assert usage.total_input_tokens == 140
    assert usage.total_output_tokens == 280
    assert usage.total_requests == 140

    usage.finalize_run_snapshots()

    assert usage._committed_run_contributions == {}
    assert usage.total_input_tokens == 140
    assert usage.total_output_tokens == 280
    assert usage.total_requests == 140


def test_session_usage_late_snapshot_replaces_contribution_after_more_than_128_commits() -> None:
    from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

    def snapshot(run_id: str, tokens: int) -> UsageSnapshot:
        run_usage = RunUsage(input_tokens=tokens)
        return UsageSnapshot(
            run_id=run_id,
            total_usage=run_usage,
            agent_usages={"main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=run_usage)},
            model_usages={"model-a": run_usage},
        )

    usage = SessionUsage()
    usage.set_run_snapshot(snapshot("target", 10))
    usage.commit_run_snapshot("target")
    for index in range(129):
        run_id = f"other-{index}"
        usage.set_run_snapshot(snapshot(run_id, 1))
        usage.commit_run_snapshot(run_id)
        usage.finalize_run_snapshots(run_id)

    usage.set_run_snapshot(snapshot("target", 25))
    usage.commit_run_snapshot("target")

    assert usage.total_input_tokens == 25 + 129


def test_session_usage_late_snapshot_replaces_recent_committed_contribution() -> None:
    from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

    usage = SessionUsage()
    first = UsageSnapshot(
        run_id="run-1",
        total_usage=RunUsage(input_tokens=10),
        agent_usages={"main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=RunUsage(input_tokens=10))},
        model_usages={"model-a": RunUsage(input_tokens=10)},
    )
    updated = UsageSnapshot(
        run_id="run-1",
        total_usage=RunUsage(input_tokens=25),
        agent_usages={"main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=RunUsage(input_tokens=25))},
        model_usages={"model-a": RunUsage(input_tokens=25)},
    )
    usage.set_run_snapshot(first)
    usage.commit_run_snapshot()
    usage.set_run_snapshot(updated)
    usage.commit_run_snapshot()

    assert usage._run_snapshots == {}
    assert usage.total_input_tokens == 25


def test_session_usage_clear_late_replacement_restores_committed_total() -> None:
    from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

    def snapshot(tokens: int) -> UsageSnapshot:
        return UsageSnapshot(
            run_id="run-1",
            total_usage=RunUsage(input_tokens=tokens),
            agent_usages={
                "main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=RunUsage(input_tokens=tokens))
            },
            model_usages={"model-a": RunUsage(input_tokens=tokens)},
        )

    usage = SessionUsage()
    usage.set_run_snapshot(snapshot(10))
    usage.commit_run_snapshot()
    usage.set_run_snapshot(snapshot(20))
    assert usage.total_input_tokens == 20

    usage.clear_run_snapshot()
    assert usage.total_input_tokens == 10
    assert usage._run_snapshots == {}
