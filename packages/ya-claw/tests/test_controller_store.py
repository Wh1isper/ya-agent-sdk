from __future__ import annotations

from decimal import Decimal

from ya_claw.controller.store import (
    RUN_USAGE_SNAPSHOT_METADATA_KEY,
    extract_latest_usage_snapshot,
    extract_usage_snapshot_from_metadata,
    with_usage_snapshot_metadata,
)


def _usage_event(*, run_id: str, total_amount: str, transport_run_id: str | None = None) -> dict[str, object]:
    return {
        "type": "CUSTOM",
        "name": "ya_agent.usage_snapshot",
        "value": {
            "run_id": transport_run_id,
            "payload": {
                "run_id": run_id,
                "total_usage": {"requests": 1, "input_tokens": 10, "output_tokens": 2},
                "total_cost_estimate": {
                    "currency": "USD",
                    "input_amount": "0.001",
                    "output_amount": "0.002",
                    "total_amount": total_amount,
                    "priced_requests": 1,
                    "unpriced_requests": 0,
                    "basis": "api_list_price",
                    "source": "genai_prices",
                },
            },
        },
    }


def test_extract_latest_usage_snapshot_returns_latest_valid_replacement() -> None:
    events = [
        _usage_event(run_id="sdk-1", transport_run_id="run-1", total_amount="0.003"),
        _usage_event(run_id="sdk-2", transport_run_id="run-1", total_amount="0.007"),
        {
            "type": "CUSTOM",
            "name": "ya_agent.usage_snapshot",
            "value": {"payload": {"run_id": 123}},
        },
    ]

    snapshot = extract_latest_usage_snapshot(events)  # type: ignore[arg-type]

    assert snapshot is not None
    assert snapshot.run_id == "run-1"
    assert snapshot.total_cost_estimate is not None
    assert snapshot.total_cost_estimate.total_amount == Decimal("0.007")


def test_usage_snapshot_metadata_round_trip_uses_reserved_index() -> None:
    snapshot = extract_latest_usage_snapshot([
        _usage_event(run_id="run-1", transport_run_id="run-1", total_amount="0.007")
    ])
    assert snapshot is not None

    metadata = with_usage_snapshot_metadata({"source": "test"}, snapshot)
    restored = extract_usage_snapshot_from_metadata(metadata)

    assert metadata["source"] == "test"
    assert RUN_USAGE_SNAPSHOT_METADATA_KEY in metadata
    assert restored == snapshot


def test_extract_latest_usage_snapshot_ignores_unrelated_and_malformed_events() -> None:
    events = [
        {"type": "CUSTOM", "name": "ya_agent.other", "value": {}},
        {"type": "CUSTOM", "name": "ya_agent.usage_snapshot", "value": "bad"},
    ]

    assert extract_latest_usage_snapshot(events) is None  # type: ignore[arg-type]
    assert extract_latest_usage_snapshot(None) is None
