"""Usage tracking models for agent token consumption.

This module provides the unified per-run usage ledger and realtime usage
snapshot models used by agents, CLI clients, and runtime services.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.usage import RunUsage

from ya_agent_sdk._logger import logger

_RUN_USAGE_FIELD_NAMES = frozenset(field.name for field in fields(RunUsage))


@runtime_checkable
class _ModelMessageResult(Protocol):
    def new_messages(self) -> Sequence[ModelMessage]: ...


class CostEstimate(BaseModel):
    """Best-effort API list-price estimate for one or more model requests."""

    currency: Literal["USD"] = "USD"
    input_amount: Decimal = Decimal(0)
    output_amount: Decimal = Decimal(0)
    total_amount: Decimal = Decimal(0)
    priced_requests: int = 0
    unpriced_requests: int = 0
    basis: Literal["api_list_price"] = "api_list_price"
    source: Literal["genai_prices"] = "genai_prices"

    @property
    def total_requests(self) -> int:
        """Number of requests represented by this estimate."""
        return self.priced_requests + self.unpriced_requests

    @property
    def is_complete(self) -> bool:
        """Whether every represented request was priced."""
        return self.unpriced_requests == 0


def combine_cost_estimates(*estimates: CostEstimate | None) -> CostEstimate | None:
    """Add cost estimates while preserving explicit unavailable coverage."""
    present = [estimate for estimate in estimates if estimate is not None]
    if not present:
        return None
    return CostEstimate(
        input_amount=sum((estimate.input_amount for estimate in present), start=Decimal(0)),
        output_amount=sum((estimate.output_amount for estimate in present), start=Decimal(0)),
        total_amount=sum((estimate.total_amount for estimate in present), start=Decimal(0)),
        priced_requests=sum(estimate.priced_requests for estimate in present),
        unpriced_requests=sum(estimate.unpriced_requests for estimate in present),
    )


def unavailable_cost_estimate(requests: int) -> CostEstimate:
    """Represent requests for which no reliable price could be calculated."""
    return CostEstimate(unpriced_requests=max(0, requests))


def estimate_model_message_cost(
    messages: Sequence[ModelMessage],
    *,
    request_count: int | None = None,
) -> CostEstimate:
    """Estimate cumulative cost by pricing each completed response independently.

    Pricing aggregated ``RunUsage`` is intentionally avoided because tiered,
    historical, and time-dependent prices are evaluated per request.
    """
    input_amount = Decimal(0)
    output_amount = Decimal(0)
    total_amount = Decimal(0)
    priced_requests = 0
    response_count = 0

    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        response_count += 1
        if message.state != "complete":
            continue
        try:
            price = message.cost()
        except Exception:
            # Cost estimation is optional accounting metadata and must never
            # change model execution behavior when pricing data is unavailable.
            logger.debug(
                "Unable to estimate model response cost model=%s provider=%s",
                message.model_name,
                message.provider_name,
                exc_info=True,
            )
            continue
        input_amount += price.input_price
        output_amount += price.output_price
        total_amount += price.total_price
        priced_requests += 1

    represented_requests = max(response_count, request_count or 0)
    return CostEstimate(
        input_amount=input_amount,
        output_amount=output_amount,
        total_amount=total_amount,
        priced_requests=priced_requests,
        unpriced_requests=max(0, represented_requests - priced_requests),
    )


@dataclass(repr=False, kw_only=True)
class PricedRunUsage(RunUsage):
    """Run usage carrying its independently calculated per-request cost."""

    cost_estimate: CostEstimate | None = None


def build_priced_run_usage(
    usage: object,
    messages: Sequence[ModelMessage],
) -> PricedRunUsage:
    """Normalize run usage and attach a per-response API price estimate."""
    normalized = coerce_run_usage(usage)
    return PricedRunUsage(
        **_run_usage_data(normalized),
        cost_estimate=estimate_model_message_cost(messages, request_count=normalized.requests),
    )


def estimate_latest_model_message_cost(messages: Sequence[ModelMessage]) -> CostEstimate:
    """Estimate the most recent response without repricing earlier requests."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return estimate_model_message_cost([message], request_count=1)
    return unavailable_cost_estimate(1)


def estimate_model_result_cost(result: object, *, request_count: int) -> CostEstimate:
    """Price a run result without making optional accounting affect execution."""
    if not isinstance(result, _ModelMessageResult):
        return unavailable_cost_estimate(request_count)
    try:
        messages = result.new_messages()
    except Exception:
        logger.debug("Unable to read model result messages for cost estimation", exc_info=True)
        return unavailable_cost_estimate(request_count)
    return estimate_model_message_cost(messages, request_count=request_count)


def coerce_run_usage(usage: object) -> RunUsage:
    """Convert Pydantic AI usage wrappers into a concrete ``RunUsage`` instance."""
    if is_dataclass(usage):
        return RunUsage(**_run_usage_data(usage))
    if callable(usage):
        usage = usage()
    return RunUsage(**_run_usage_data(usage))


def _run_usage_data(usage: object) -> dict[str, Any]:
    if not is_dataclass(usage):
        raise TypeError(f"Expected RunUsage-compatible dataclass, got {type(usage).__name__}")

    data: dict[str, Any] = {}
    for name in _RUN_USAGE_FIELD_NAMES:
        value = getattr(usage, name)
        if name == "details":
            value = dict(value)
        data[name] = value
    return data


class UsageSnapshotEntry(BaseModel):
    """Cumulative usage for one agent/source in the current run."""

    agent_id: str
    """Agent/source instance that generated this usage (e.g., 'main', 'searcher-a1b2', 'compact')."""

    agent_name: str
    """Human-readable agent/source name (e.g., 'main', 'searcher', 'compact')."""

    model_id: str
    """Model identifier that generated this usage."""

    usage: RunUsage
    """Cumulative token usage for this agent/source instance."""

    cost_estimate: CostEstimate | None = None
    """Cumulative API list-price estimate for this entry, when captured."""

    usage_id: str | None = None
    """Stable usage record ID for idempotent updates."""

    source: str = "model_request"
    """Component that reported this usage."""


class UsageAgentTotal(BaseModel):
    """Cumulative usage grouped by agent/source."""

    agent_name: str
    model_id: str
    usage: RunUsage
    cost_estimate: CostEstimate | None = None
    usage_id: str | None = None
    source: str = "model_request"


class UsageSnapshot(BaseModel):
    """Cumulative usage snapshot for the current run.

    Realtime consumers and billing systems treat each snapshot as a replacement
    for the previous snapshot with the same run ID.
    """

    run_id: str
    """Run identifier for the snapshot."""

    total_usage: RunUsage = Field(default_factory=RunUsage)
    """Cumulative usage across all known agents in this run."""

    total_cost_estimate: CostEstimate | None = None
    """Cumulative API list-price estimate across all known requests."""

    entries: list[UsageSnapshotEntry] = Field(default_factory=list)
    """Per-agent/source cumulative usage entries."""

    agent_usages: dict[str, UsageAgentTotal] = Field(default_factory=dict)
    """Cumulative usage grouped by agent ID."""

    model_usages: dict[str, RunUsage] = Field(default_factory=dict)
    """Cumulative usage grouped by model identifier."""

    model_cost_estimates: dict[str, CostEstimate] = Field(default_factory=dict)
    """Cumulative API list-price estimates grouped by model identifier."""


def combine_usage_snapshots(
    base: UsageSnapshot | None,
    current: UsageSnapshot,
    *,
    run_id: str | None = None,
) -> UsageSnapshot:
    """Add finalized prior usage to a current cumulative segment snapshot."""
    effective_run_id = run_id or current.run_id
    if base is None:
        return current.model_copy(update={"run_id": effective_run_id}, deep=True)

    total_usage = RunUsage() + coerce_run_usage(base.total_usage)
    total_usage.incr(coerce_run_usage(current.total_usage))

    model_usages = {model_id: RunUsage() + coerce_run_usage(usage) for model_id, usage in base.model_usages.items()}
    for model_id, usage in current.model_usages.items():
        model_usages.setdefault(model_id, RunUsage()).incr(coerce_run_usage(usage))

    model_cost_estimates = dict(base.model_cost_estimates)
    for model_id, estimate in current.model_cost_estimates.items():
        model_cost_estimates[model_id] = (
            combine_cost_estimates(model_cost_estimates.get(model_id), estimate) or CostEstimate()
        )

    agent_usages = {agent_id: entry.model_copy(deep=True) for agent_id, entry in base.agent_usages.items()}
    for agent_id, entry in current.agent_usages.items():
        previous = agent_usages.get(agent_id)
        if previous is None:
            agent_usages[agent_id] = entry.model_copy(deep=True)
            continue
        usage = RunUsage() + coerce_run_usage(previous.usage)
        usage.incr(coerce_run_usage(entry.usage))
        agent_usages[agent_id] = UsageAgentTotal(
            agent_name=entry.agent_name,
            model_id=previous.model_id if previous.model_id == entry.model_id else "multiple",
            usage=usage,
            cost_estimate=combine_cost_estimates(previous.cost_estimate, entry.cost_estimate),
            usage_id=entry.usage_id or previous.usage_id,
            source=entry.source,
        )

    return UsageSnapshot(
        run_id=effective_run_id,
        total_usage=total_usage,
        total_cost_estimate=combine_cost_estimates(base.total_cost_estimate, current.total_cost_estimate),
        entries=[
            *(entry.model_copy(deep=True) for entry in base.entries),
            *(entry.model_copy(deep=True) for entry in current.entries),
        ],
        agent_usages=agent_usages,
        model_usages=model_usages,
        model_cost_estimates=model_cost_estimates,
    )
