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
from ya_agent_sdk.usage import UsageSnapshot, coerce_run_usage


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


@dataclass(frozen=True)
class TokenCostEstimate:
    """Estimated USD cost for a model usage entry."""

    input_cost: Decimal
    output_cost: Decimal

    @property
    def total_cost(self) -> Decimal:
        """Total estimated USD cost."""
        return self.input_cost + self.output_cost


@dataclass(frozen=True)
class ModelTokenPrice:
    """Simple per-million-token pricing used for local cost estimates."""

    input_mtok: Decimal
    cached_input_mtok: Decimal | None
    output_mtok: Decimal

    def estimate(self, usage: RunUsage) -> TokenCostEstimate:
        """Estimate USD cost for a RunUsage value."""
        input_tokens = usage.input_tokens or 0
        cache_read_tokens = usage.cache_read_tokens or 0
        cached_input_tokens = min(input_tokens, cache_read_tokens) if self.cached_input_mtok is not None else 0
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        output_tokens = usage.output_tokens or 0

        input_cost = (Decimal(uncached_input_tokens) * self.input_mtok) / Decimal(1_000_000)
        if self.cached_input_mtok is not None:
            input_cost += (Decimal(cached_input_tokens) * self.cached_input_mtok) / Decimal(1_000_000)
        output_cost = (Decimal(output_tokens) * self.output_mtok) / Decimal(1_000_000)
        return TokenCostEstimate(input_cost=input_cost, output_cost=output_cost)


_GROK_4_5_PRICE = ModelTokenPrice(
    input_mtok=Decimal("2.00"),
    cached_input_mtok=Decimal("0.50"),
    output_mtok=Decimal("6.00"),
)

_GROK_4_5_MODEL_REFS = frozenset({
    "grok-4.5",
    "grok-4.5-latest",
    "grok-build-latest",
    "x-ai/grok-4.5",
    "x-ai/grok-4.5-latest",
})


def _normalize_model_ref(model_id: str) -> str:
    """Normalize YA/Pydantic model IDs to provider model references for pricing."""
    normalized = model_id.strip().lower()
    if "@" in normalized:
        _gateway_or_oauth, normalized = normalized.split("@", 1)
    if ":" in normalized:
        _provider, normalized = normalized.rsplit(":", 1)
    return normalized


def estimate_model_usage_cost(model_id: str, usage: RunUsage) -> TokenCostEstimate | None:
    """Estimate model usage cost when a local price table is available."""
    model_ref = _normalize_model_ref(model_id)
    if model_ref in _GROK_4_5_MODEL_REFS:
        return _GROK_4_5_PRICE.estimate(usage)
    return None


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
    _manual_agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    _manual_model_usages: dict[str, RunUsage] = field(default_factory=dict)
    _committed_agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    _committed_model_usages: dict[str, RunUsage] = field(default_factory=dict)
    # Only in-flight snapshots are retained. commit_run_snapshot() folds them
    # into aggregate counters and removes them immediately.
    _run_snapshots: dict[str, UsageSnapshot] = field(default_factory=dict)
    _uncommitted_run_ids: set[str] = field(default_factory=set)
    # Committed contributions remain replaceable while background tasks can
    # still publish cumulative snapshots. finalize_run_snapshots() is the
    # explicit lifecycle boundary that releases this metadata.
    _committed_run_contributions: OrderedDict[str, tuple[dict[str, RunUsage], dict[str, RunUsage]]] = field(
        default_factory=OrderedDict
    )
    # A late replacement temporarily removes the prior contribution. Retain it
    # only until that replacement is committed or explicitly cleared.
    _superseded_committed_contributions: dict[str, tuple[dict[str, RunUsage], dict[str, RunUsage]]] = field(
        default_factory=dict
    )

    def add(self, agent: str, model_id: str, usage: RunUsage) -> None:
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
        self._rebuild_totals()

    def _rebuild_totals(self) -> None:
        """Rebuild from aggregates plus only the currently in-flight snapshots.

        Committed runs are already folded into the aggregate dictionaries, so
        this cost cannot grow with the number of completed runs.
        """
        self.agent_usages = self._copy_usage_map(self._manual_agent_usages)
        self.model_usages = self._copy_usage_map(self._manual_model_usages)
        self._merge_usage_map(self.agent_usages, self._committed_agent_usages)
        self._merge_usage_map(self.model_usages, self._committed_model_usages)
        for snapshot in self._run_snapshots.values():
            self._merge_usage_map(
                self.agent_usages,
                {agent: coerce_run_usage(entry.usage) for agent, entry in snapshot.agent_usages.items()},
            )
            self._merge_usage_map(self.model_usages, snapshot.model_usages)

    @staticmethod
    def _copy_usage_map(usages: dict[str, RunUsage]) -> dict[str, RunUsage]:
        return {key: RunUsage() + coerce_run_usage(usage) for key, usage in usages.items()}

    @staticmethod
    def _merge_usage_map(target: dict[str, RunUsage], source: dict[str, RunUsage]) -> None:
        for key, usage in source.items():
            target.setdefault(key, RunUsage()).incr(coerce_run_usage(usage))

    def _normalized_snapshot_contribution(
        self, snapshot: UsageSnapshot
    ) -> tuple[dict[str, RunUsage], dict[str, RunUsage]]:
        return (
            {agent: RunUsage() + coerce_run_usage(entry.usage) for agent, entry in snapshot.agent_usages.items()},
            {model_id: RunUsage() + coerce_run_usage(usage) for model_id, usage in snapshot.model_usages.items()},
        )

    def _remove_committed_contribution(self, run_id: str) -> None:
        contribution = self._committed_run_contributions.pop(run_id, None)
        if contribution is None:
            return
        agents, models = contribution
        self._subtract_usage_map(self._committed_agent_usages, agents)
        self._subtract_usage_map(self._committed_model_usages, models)
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
            agents, models = self._normalized_snapshot_contribution(snapshot)
            self._merge_usage_map(self._committed_agent_usages, agents)
            self._merge_usage_map(self._committed_model_usages, models)
            self._committed_run_contributions[committed_run_id] = (agents, models)
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
                agents, models = superseded
                self._merge_usage_map(self._committed_agent_usages, agents)
                self._merge_usage_map(self._committed_model_usages, models)
                self._committed_run_contributions[run_id] = superseded
                self._committed_run_contributions.move_to_end(run_id)
        self._uncommitted_run_ids.clear()
        self._rebuild_totals()

    def clear(self) -> None:
        """Clear all accumulated usage."""
        self.agent_usages.clear()
        self.model_usages.clear()
        self._manual_agent_usages.clear()
        self._manual_model_usages.clear()
        self._committed_agent_usages.clear()
        self._committed_model_usages.clear()
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
    def estimated_total_cost(self) -> TokenCostEstimate | None:
        """Estimated total cost across models with known local prices."""
        input_cost = Decimal(0)
        output_cost = Decimal(0)
        has_estimate = False
        for model_id, usage in self.model_usages.items():
            estimate = estimate_model_usage_cost(model_id, usage)
            if estimate is None:
                continue
            input_cost += estimate.input_cost
            output_cost += estimate.output_cost
            has_estimate = True
        if not has_estimate:
            return None
        return TokenCostEstimate(input_cost=input_cost, output_cost=output_cost)

    def is_empty(self) -> bool:
        """Check if no usage has been recorded."""
        return len(self.model_usages) == 0

    def _format_usage_entry(self, name: str, usage: RunUsage, *, model_id: str | None = None) -> list[str]:
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
        if model_id is not None and (cost := estimate_model_usage_cost(model_id, usage)) is not None:
            lines.append(f"    Estimated Cost: {_format_usd(cost.total_cost)}")
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
            lines.extend(self._format_usage_entry(model_id, usage, model_id=model_id))
            lines.append("")

        # By Agent breakdown
        lines.append("By Agent:")
        for agent, usage in sorted(self.agent_usages.items()):
            lines.extend(self._format_usage_entry(agent, usage))
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
        if (cost := self.estimated_total_cost) is not None:
            lines.append(f"  Estimated Cost: {_format_usd(cost.total_cost)}")

        return "\n".join(lines)
