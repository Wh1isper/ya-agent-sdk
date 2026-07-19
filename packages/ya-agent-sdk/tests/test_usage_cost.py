from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage, RunUsage
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.usage import (
    CostEstimate,
    PricedRunUsage,
    UsageAgentTotal,
    UsageSnapshot,
    build_priced_run_usage,
    combine_usage_snapshots,
    estimate_model_message_cost,
    estimate_model_result_cost,
)


def _response(model_name: str, *, state: str = "complete") -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content="ok")],
        usage=RequestUsage(input_tokens=10, output_tokens=2),
        model_name=model_name,
        provider_name="test-provider",
        state=state,  # type: ignore[arg-type]
    )


def test_estimate_model_message_cost_uses_genai_prices_for_known_model() -> None:
    estimate = estimate_model_message_cost([
        ModelResponse(
            parts=[TextPart(content="ok")],
            usage=RequestUsage(input_tokens=1_000, output_tokens=100),
            model_name="gpt-4o",
            provider_name="openai",
        )
    ])

    assert estimate.total_amount > 0
    assert estimate.priced_requests == 1
    assert estimate.unpriced_requests == 0
    assert estimate.basis == "api_list_price"
    assert estimate.source == "genai_prices"


def test_estimate_model_message_cost_prices_each_response_independently(monkeypatch) -> None:
    def fake_cost(response: ModelResponse) -> SimpleNamespace:
        if response.model_name == "unknown":
            raise LookupError("missing price")
        multiplier = Decimal("1") if response.model_name == "first" else Decimal("2")
        return SimpleNamespace(
            input_price=Decimal("0.001") * multiplier,
            output_price=Decimal("0.002") * multiplier,
            total_price=Decimal("0.003") * multiplier,
        )

    monkeypatch.setattr(ModelResponse, "cost", fake_cost)
    estimate = estimate_model_message_cost(
        [
            _response("first"),
            _response("second"),
            _response("unknown"),
            _response("interrupted", state="interrupted"),
        ],
        request_count=4,
    )

    assert estimate.input_amount == Decimal("0.003")
    assert estimate.output_amount == Decimal("0.006")
    assert estimate.total_amount == Decimal("0.009")
    assert estimate.priced_requests == 2
    assert estimate.unpriced_requests == 2
    assert not estimate.is_complete


def test_build_priced_run_usage_preserves_run_usage_and_cost(monkeypatch) -> None:
    monkeypatch.setattr(
        ModelResponse,
        "cost",
        lambda _: SimpleNamespace(
            input_price=Decimal("0.01"),
            output_price=Decimal("0.02"),
            total_price=Decimal("0.03"),
        ),
    )

    usage = build_priced_run_usage(
        RunUsage(requests=1, input_tokens=10, output_tokens=2),
        [_response("priced")],
    )

    assert isinstance(usage, PricedRunUsage)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    assert usage.cost_estimate is not None
    assert usage.cost_estimate.total_amount == Decimal("0.03")


def test_usage_snapshot_aggregates_cost_only_request_entries() -> None:
    ctx = AgentContext()
    ctx.update_usage_snapshot_entry(
        ledger_key="main",
        agent_id="main",
        agent_name="main",
        model_id="model-a",
        usage=RunUsage(requests=1, input_tokens=10, output_tokens=2),
        cost_estimate=CostEstimate(),
    )
    ctx.update_usage_snapshot_entry(
        ledger_key="cost:request-1",
        agent_id="main",
        agent_name="main",
        model_id="model-a",
        usage=RunUsage(),
        cost_estimate=CostEstimate(
            input_amount=Decimal("0.01"),
            output_amount=Decimal("0.02"),
            total_amount=Decimal("0.03"),
            priced_requests=1,
        ),
    )

    snapshot = ctx.build_usage_snapshot()

    assert snapshot.total_usage.requests == 1
    assert snapshot.total_usage.total_tokens == 12
    assert snapshot.total_cost_estimate is not None
    assert snapshot.total_cost_estimate.total_amount == Decimal("0.03")
    assert snapshot.total_cost_estimate.priced_requests == 1
    assert snapshot.total_cost_estimate.unpriced_requests == 0
    assert snapshot.model_cost_estimates["model-a"].total_amount == Decimal("0.03")
    assert snapshot.agent_usages["main"].cost_estimate is not None
    assert snapshot.agent_usages["main"].cost_estimate.total_amount == Decimal("0.03")


def test_estimate_model_result_cost_marks_missing_message_surface_unpriced() -> None:
    estimate = estimate_model_result_cost(object(), request_count=2)

    assert estimate.total_amount == 0
    assert estimate.priced_requests == 0
    assert estimate.unpriced_requests == 2


def test_combine_usage_snapshots_accumulates_finalized_segments_as_outer_run() -> None:
    def snapshot(run_id: str, requests: int, amount: str) -> UsageSnapshot:
        usage = RunUsage(requests=requests, input_tokens=requests * 10, output_tokens=requests * 2)
        estimate = CostEstimate(total_amount=Decimal(amount), priced_requests=requests)
        return UsageSnapshot(
            run_id=run_id,
            total_usage=usage,
            total_cost_estimate=estimate,
            agent_usages={
                "main": UsageAgentTotal(
                    agent_name="main",
                    model_id="model-a",
                    usage=usage,
                    cost_estimate=estimate,
                )
            },
            model_usages={"model-a": usage},
            model_cost_estimates={"model-a": estimate},
        )

    first = snapshot("sdk-1", 1, "0.003")
    second = snapshot("sdk-2", 2, "0.007")

    combined = combine_usage_snapshots(first, second, run_id="claw-run")

    assert combined.run_id == "claw-run"
    assert combined.total_usage.requests == 3
    assert combined.total_usage.input_tokens == 30
    assert combined.total_cost_estimate is not None
    assert combined.total_cost_estimate.total_amount == Decimal("0.010")
    assert combined.total_cost_estimate.priced_requests == 3
    assert combined.model_usages["model-a"].requests == 3
    assert combined.agent_usages["main"].usage.requests == 3
    assert first.total_usage.requests == 1
