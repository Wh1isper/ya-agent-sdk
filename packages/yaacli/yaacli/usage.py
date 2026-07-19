"""Session-level usage tracking for yaacli.

This module provides usage tracking across multiple agent runs in a CLI session.
It consumes the SDK's realtime UsageSnapshotEvent as the primary usage surface.

Uses pydantic-ai's RunUsage directly for accurate tracking including details field.

Example:
    session_usage = SessionUsage()

    # During streaming
    session_usage.set_run_snapshot(usage_snapshot)

    # After run completion
    session_usage.commit_run_snapshot()

    # Show summary
    print(session_usage.format_summary())
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic_ai.usage import RunUsage
from ya_agent_sdk.usage import (
    CostEstimate,
    UsageAgentTotal,
    UsageSnapshot,
    coerce_run_usage,
    combine_cost_estimates,
    unavailable_cost_estimate,
)


@dataclass(frozen=True)
class TokenUsageBreakdown:
    """Aggregated token usage values used for compact delta displays."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total billable prompt/completion tokens, excluding cache detail counters."""
        return self.input_tokens + self.output_tokens

    def delta_since(self, start: TokenUsageBreakdown) -> TokenUsageBreakdown:
        """Return a non-negative field-by-field delta from a previous snapshot."""
        return TokenUsageBreakdown(
            input_tokens=max(0, self.input_tokens - start.input_tokens),
            cache_read_tokens=max(0, self.cache_read_tokens - start.cache_read_tokens),
            cache_write_tokens=max(0, self.cache_write_tokens - start.cache_write_tokens),
            output_tokens=max(0, self.output_tokens - start.output_tokens),
        )


def _format_usd(value: Decimal) -> str:
    """Format a USD Decimal compactly for CLI output."""
    if value == 0:
        return "$0.00"
    if value < Decimal("0.0001"):
        return "<$0.0001"
    if value < Decimal("0.01"):
        return f"${value:.4f}"
    return f"${value:.2f}"


@dataclass
class _UsageContribution:
    agent_usages: dict[str, RunUsage]
    model_usages: dict[str, RunUsage]
    agent_cost_estimates: dict[str, CostEstimate]
    model_cost_estimates: dict[str, CostEstimate]


@dataclass
class SessionUsage:
    """Session-level usage tracking, aggregated by both agent and model.

    Tracks token usage across all agent runs in a CLI session.
    Usage is grouped by:
    - Agent name (main, subagent names, image_understanding, etc.)
    - Model ID (openai-chat:gpt-4o, anthropic:claude-sonnet-4, etc.)

    Uses pydantic-ai's RunUsage for accurate tracking including details field.

    Attributes:
        agent_usages: Dict mapping agent name to its RunUsage.
        model_usages: Dict mapping model_id to its RunUsage.
    """

    agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    model_usages: dict[str, RunUsage] = field(default_factory=dict)
    agent_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    model_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    _manual_agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    _manual_model_usages: dict[str, RunUsage] = field(default_factory=dict)
    _manual_agent_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    _manual_model_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    _committed_agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    _committed_model_usages: dict[str, RunUsage] = field(default_factory=dict)
    _committed_agent_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    _committed_model_cost_estimates: dict[str, CostEstimate] = field(default_factory=dict)
    # Only in-flight snapshots are retained. commit_run_snapshot() folds them
    # into aggregate counters and removes them immediately.
    _run_snapshots: dict[str, UsageSnapshot] = field(default_factory=dict)
    _uncommitted_run_ids: set[str] = field(default_factory=set)
    # Committed contributions remain replaceable while background tasks can
    # still publish cumulative snapshots. finalize_run_snapshots() is the
    # explicit lifecycle boundary that releases this metadata.
    _committed_run_contributions: OrderedDict[str, _UsageContribution] = field(default_factory=OrderedDict)
    # A late replacement temporarily removes the prior contribution. Retain it
    # only until that replacement is committed or explicitly cleared.
    _superseded_committed_contributions: dict[str, _UsageContribution] = field(default_factory=dict)

    def add(
        self,
        agent: str,
        model_id: str,
        usage: RunUsage,
        cost_estimate: CostEstimate | None = None,
    ) -> None:
        """Add usage for a specific agent and model.

        Args:
            agent: Agent name (e.g., "main", "explorer", "image_understanding").
            model_id: Model identifier (e.g., "openai-chat:gpt-4o", "anthropic:claude-sonnet-4").
            usage: The RunUsage to accumulate.
        """
        usage = coerce_run_usage(usage)

        # Accumulate by agent
        if agent not in self._manual_agent_usages:
            self._manual_agent_usages[agent] = RunUsage()
        self._manual_agent_usages[agent].incr(usage)

        # Accumulate by model
        if model_id not in self._manual_model_usages:
            self._manual_model_usages[model_id] = RunUsage()
        self._manual_model_usages[model_id].incr(usage)

        estimate = cost_estimate or unavailable_cost_estimate(usage.requests)
        self._merge_cost_map(self._manual_agent_cost_estimates, {agent: estimate})
        self._merge_cost_map(self._manual_model_cost_estimates, {model_id: estimate})
        self._rebuild_totals()

    def _rebuild_totals(self) -> None:
        """Rebuild from aggregates plus only the currently in-flight snapshots.

        Committed runs are already folded into the aggregate dictionaries, so
        this cost cannot grow with the number of completed runs.
        """
        self.agent_usages = self._copy_usage_map(self._manual_agent_usages)
        self.model_usages = self._copy_usage_map(self._manual_model_usages)
        self.agent_cost_estimates = dict(self._manual_agent_cost_estimates)
        self.model_cost_estimates = dict(self._manual_model_cost_estimates)
        self._merge_usage_map(self.agent_usages, self._committed_agent_usages)
        self._merge_usage_map(self.model_usages, self._committed_model_usages)
        self._merge_cost_map(self.agent_cost_estimates, self._committed_agent_cost_estimates)
        self._merge_cost_map(self.model_cost_estimates, self._committed_model_cost_estimates)
        for snapshot in self._run_snapshots.values():
            self._merge_usage_map(
                self.agent_usages,
                {agent: coerce_run_usage(entry.usage) for agent, entry in snapshot.agent_usages.items()},
            )
            self._merge_usage_map(self.model_usages, snapshot.model_usages)
            self._merge_cost_map(self.agent_cost_estimates, self._snapshot_agent_costs(snapshot))
            self._merge_cost_map(self.model_cost_estimates, self._snapshot_model_costs(snapshot))

    @staticmethod
    def _copy_usage_map(usages: dict[str, RunUsage]) -> dict[str, RunUsage]:
        return {key: RunUsage() + coerce_run_usage(usage) for key, usage in usages.items()}

    @staticmethod
    def _merge_usage_map(target: dict[str, RunUsage], source: dict[str, RunUsage]) -> None:
        for key, usage in source.items():
            target.setdefault(key, RunUsage()).incr(coerce_run_usage(usage))

    @staticmethod
    def _merge_cost_map(target: dict[str, CostEstimate], source: dict[str, CostEstimate]) -> None:
        for key, estimate in source.items():
            target[key] = combine_cost_estimates(target.get(key), estimate) or CostEstimate()

    @staticmethod
    def _snapshot_agent_costs(snapshot: UsageSnapshot) -> dict[str, CostEstimate]:
        return {
            agent: entry.cost_estimate or unavailable_cost_estimate(entry.usage.requests)
            for agent, entry in snapshot.agent_usages.items()
        }

    @staticmethod
    def _snapshot_model_costs(snapshot: UsageSnapshot) -> dict[str, CostEstimate]:
        return {
            model_id: snapshot.model_cost_estimates.get(model_id) or unavailable_cost_estimate(usage.requests)
            for model_id, usage in snapshot.model_usages.items()
        }

    def _normalized_snapshot_contribution(self, snapshot: UsageSnapshot) -> _UsageContribution:
        return _UsageContribution(
            agent_usages={
                agent: RunUsage() + coerce_run_usage(entry.usage) for agent, entry in snapshot.agent_usages.items()
            },
            model_usages={
                model_id: RunUsage() + coerce_run_usage(usage) for model_id, usage in snapshot.model_usages.items()
            },
            agent_cost_estimates=self._snapshot_agent_costs(snapshot),
            model_cost_estimates=self._snapshot_model_costs(snapshot),
        )

    def _remove_committed_contribution(self, run_id: str) -> None:
        contribution = self._committed_run_contributions.pop(run_id, None)
        if contribution is None:
            return
        self._subtract_usage_map(self._committed_agent_usages, contribution.agent_usages)
        self._subtract_usage_map(self._committed_model_usages, contribution.model_usages)
        self._subtract_cost_map(self._committed_agent_cost_estimates, contribution.agent_cost_estimates)
        self._subtract_cost_map(self._committed_model_cost_estimates, contribution.model_cost_estimates)
        self._superseded_committed_contributions[run_id] = contribution

    @staticmethod
    def _subtract_usage_map(target: dict[str, RunUsage], removed: dict[str, RunUsage]) -> None:
        fields = (
            "requests",
            "tool_calls",
            "input_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
            "input_audio_tokens",
            "cache_audio_read_tokens",
            "output_tokens",
            "output_audio_tokens",
        )
        for key, usage in removed.items():
            total = target.get(key)
            if total is None:
                continue
            for field_name in fields:
                setattr(total, field_name, getattr(total, field_name) - getattr(usage, field_name))
            for detail, amount in usage.details.items():
                remaining = total.details.get(detail, 0) - amount
                if remaining:
                    total.details[detail] = remaining
                else:
                    total.details.pop(detail, None)
            if not any(getattr(total, field_name) for field_name in fields) and not total.details:
                target.pop(key, None)

    @staticmethod
    def _subtract_cost_map(target: dict[str, CostEstimate], removed: dict[str, CostEstimate]) -> None:
        for key, estimate in removed.items():
            total = target.get(key)
            if total is None:
                continue
            remaining = CostEstimate(
                input_amount=max(Decimal(0), total.input_amount - estimate.input_amount),
                output_amount=max(Decimal(0), total.output_amount - estimate.output_amount),
                total_amount=max(Decimal(0), total.total_amount - estimate.total_amount),
                priced_requests=max(0, total.priced_requests - estimate.priced_requests),
                unpriced_requests=max(0, total.unpriced_requests - estimate.unpriced_requests),
            )
            if remaining.total_requests or remaining.total_amount:
                target[key] = remaining
            else:
                target.pop(key, None)

    def set_run_snapshot(self, snapshot: UsageSnapshot) -> None:
        """Replace usage for one run with a realtime SDK snapshot."""
        for entry in snapshot.entries:
            entry.usage = coerce_run_usage(entry.usage)
        for entry in snapshot.agent_usages.values():
            entry.usage = coerce_run_usage(entry.usage)
        snapshot.model_usages = {model_id: coerce_run_usage(usage) for model_id, usage in snapshot.model_usages.items()}
        snapshot.total_usage = coerce_run_usage(snapshot.total_usage)
        # A final async update for an open committed run replaces its compact
        # contribution instead of being counted as a second run.
        self._remove_committed_contribution(snapshot.run_id)
        self._run_snapshots[snapshot.run_id] = snapshot
        self._uncommitted_run_ids.add(snapshot.run_id)
        self._rebuild_totals()

    @property
    def has_run_snapshot(self) -> bool:
        """Whether current session totals include an uncommitted run snapshot."""
        return bool(self._uncommitted_run_ids)

    def commit_run_snapshot(self, run_id: str | None = None) -> list[str]:
        """Fold realtime snapshots into aggregates and return committed run IDs."""
        run_ids = list(self._uncommitted_run_ids) if run_id is None else [run_id]
        committed_run_ids: list[str] = []
        for committed_run_id in run_ids:
            snapshot = self._run_snapshots.pop(committed_run_id, None)
            self._uncommitted_run_ids.discard(committed_run_id)
            if snapshot is None:
                continue
            self._superseded_committed_contributions.pop(committed_run_id, None)
            contribution = self._normalized_snapshot_contribution(snapshot)
            self._merge_usage_map(self._committed_agent_usages, contribution.agent_usages)
            self._merge_usage_map(self._committed_model_usages, contribution.model_usages)
            self._merge_cost_map(self._committed_agent_cost_estimates, contribution.agent_cost_estimates)
            self._merge_cost_map(self._committed_model_cost_estimates, contribution.model_cost_estimates)
            self._committed_run_contributions[committed_run_id] = contribution
            self._committed_run_contributions.move_to_end(committed_run_id)
            committed_run_ids.append(committed_run_id)
        self._rebuild_totals()
        return committed_run_ids

    def finalize_run_snapshots(self, run_id: str | None = None) -> None:
        """Release replacement metadata once no late snapshots can arrive."""
        if run_id is None:
            self._committed_run_contributions.clear()
            return
        self._committed_run_contributions.pop(run_id, None)

    def clear_run_snapshot(self) -> None:
        """Remove uncommitted realtime run snapshots from session totals."""
        for run_id in list(self._uncommitted_run_ids):
            self._run_snapshots.pop(run_id, None)
            superseded = self._superseded_committed_contributions.pop(run_id, None)
            if superseded is not None:
                self._merge_usage_map(self._committed_agent_usages, superseded.agent_usages)
                self._merge_usage_map(self._committed_model_usages, superseded.model_usages)
                self._merge_cost_map(self._committed_agent_cost_estimates, superseded.agent_cost_estimates)
                self._merge_cost_map(self._committed_model_cost_estimates, superseded.model_cost_estimates)
                self._committed_run_contributions[run_id] = superseded
                self._committed_run_contributions.move_to_end(run_id)
        self._uncommitted_run_ids.clear()
        self._rebuild_totals()

    def clear(self) -> None:
        """Clear all accumulated usage."""
        self.agent_usages.clear()
        self.model_usages.clear()
        self.agent_cost_estimates.clear()
        self.model_cost_estimates.clear()
        self._manual_agent_usages.clear()
        self._manual_model_usages.clear()
        self._manual_agent_cost_estimates.clear()
        self._manual_model_cost_estimates.clear()
        self._committed_agent_usages.clear()
        self._committed_model_usages.clear()
        self._committed_agent_cost_estimates.clear()
        self._committed_model_cost_estimates.clear()
        self._run_snapshots.clear()
        self._uncommitted_run_ids.clear()
        self._committed_run_contributions.clear()
        self._superseded_committed_contributions.clear()

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all models."""
        return sum(u.input_tokens or 0 for u in self.model_usages.values())

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all models."""
        return sum(u.output_tokens or 0 for u in self.model_usages.values())

    @property
    def total_cache_read_tokens(self) -> int:
        """Total cache-read tokens across all models."""
        return sum(u.cache_read_tokens or 0 for u in self.model_usages.values())

    @property
    def total_cache_write_tokens(self) -> int:
        """Total cache-write tokens across all models."""
        return sum(u.cache_write_tokens or 0 for u in self.model_usages.values())

    @property
    def token_breakdown(self) -> TokenUsageBreakdown:
        """Current aggregated token usage values."""
        return TokenUsageBreakdown(
            input_tokens=self.total_input_tokens,
            cache_read_tokens=self.total_cache_read_tokens,
            cache_write_tokens=self.total_cache_write_tokens,
            output_tokens=self.total_output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models."""
        return self.token_breakdown.total_tokens

    @property
    def total_requests(self) -> int:
        """Total LLM requests across all models."""
        return sum(u.requests or 0 for u in self.model_usages.values())

    @property
    def estimated_total_cost(self) -> CostEstimate | None:
        """Cumulative API list-price estimate supplied by SDK snapshots."""
        return combine_cost_estimates(*self.model_cost_estimates.values())

    def export_snapshot(self, *, run_id: str) -> UsageSnapshot:
        """Build a compact aggregate snapshot suitable for session persistence."""
        total_usage = RunUsage()
        for usage in self.model_usages.values():
            total_usage.incr(coerce_run_usage(usage))
        return UsageSnapshot(
            run_id=run_id,
            total_usage=total_usage,
            total_cost_estimate=self.estimated_total_cost,
            agent_usages={
                agent: UsageAgentTotal(
                    agent_name=agent,
                    model_id="multiple",
                    usage=RunUsage() + coerce_run_usage(usage),
                    cost_estimate=self.agent_cost_estimates.get(agent) or unavailable_cost_estimate(usage.requests),
                    source="session_restore",
                )
                for agent, usage in self.agent_usages.items()
            },
            model_usages={
                model_id: RunUsage() + coerce_run_usage(usage) for model_id, usage in self.model_usages.items()
            },
            model_cost_estimates=dict(self.model_cost_estimates),
        )

    def is_empty(self) -> bool:
        """Check if no usage has been recorded."""
        return len(self.model_usages) == 0

    def format_status_cost(self) -> str:
        """Format a compact status-bar API cost estimate."""
        estimate = self.estimated_total_cost
        if estimate is None or estimate.priced_requests == 0:
            return "--"
        suffix = " partial" if not estimate.is_complete else ""
        return f"~{_format_usd(estimate.total_amount)}{suffix}"

    @staticmethod
    def _format_cost_lines(estimate: CostEstimate | None, *, indent: str) -> list[str]:
        if estimate is None or estimate.total_requests == 0:
            return []
        coverage = f"{estimate.priced_requests}/{estimate.total_requests} requests priced"
        if estimate.priced_requests == 0:
            return [f"{indent}Estimated Cost: unavailable ({coverage})"]
        label = "Estimated Cost" if estimate.is_complete else "Partial Estimated Cost"
        return [f"{indent}{label}: {_format_usd(estimate.total_amount)} ({coverage})"]

    def _format_usage_entry(
        self,
        name: str,
        usage: RunUsage,
        *,
        cost_estimate: CostEstimate | None = None,
    ) -> list[str]:
        """Format a single usage entry."""
        lines = [f"  {name}:"]
        lines.append(f"    Input:  {usage.input_tokens or 0:,} tokens")
        lines.append(f"    Output: {usage.output_tokens or 0:,} tokens")
        if usage.cache_read_tokens:
            lines.append(f"    Cache Read:  {usage.cache_read_tokens:,} tokens")
        if usage.cache_write_tokens:
            lines.append(f"    Cache Write: {usage.cache_write_tokens:,} tokens")
        if usage.requests:
            lines.append(f"    Requests: {usage.requests}")
        lines.extend(self._format_cost_lines(cost_estimate, indent="    "))
        return lines

    def format_summary(self) -> str:
        """Format usage summary as a string.

        Returns:
            Formatted string with usage breakdown by model and agent.
        """
        if self.is_empty():
            return "No usage data available."

        lines = ["Token Usage Summary:", ""]

        # By Model breakdown
        lines.append("By Model:")
        for model_id, usage in sorted(self.model_usages.items()):
            lines.extend(
                self._format_usage_entry(
                    model_id,
                    usage,
                    cost_estimate=self.model_cost_estimates.get(model_id),
                )
            )
            lines.append("")

        # By Agent breakdown
        lines.append("By Agent:")
        for agent, usage in sorted(self.agent_usages.items()):
            lines.extend(
                self._format_usage_entry(
                    agent,
                    usage,
                    cost_estimate=self.agent_cost_estimates.get(agent),
                )
            )
            lines.append("")

        # Totals
        lines.append("Total:")
        lines.append(f"  Input:  {self.total_input_tokens:,} tokens")
        lines.append(f"  Output: {self.total_output_tokens:,} tokens")
        if self.total_cache_read_tokens:
            lines.append(f"  Cache Read:  {self.total_cache_read_tokens:,} tokens")
        if self.total_cache_write_tokens:
            lines.append(f"  Cache Write: {self.total_cache_write_tokens:,} tokens")
        lines.append(f"  Total:  {self.total_tokens:,} tokens")
        lines.append(f"  Requests: {self.total_requests}")
        lines.extend(self._format_cost_lines(self.estimated_total_cost, indent="  "))

        return "\n".join(lines)
