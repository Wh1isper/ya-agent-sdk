from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from ya_agent_sdk.usage import UsageSnapshot
from ya_agent_stream_protocol.agui import parse_required_message_events

from ya_claw.config import ClawSettings
from ya_claw.controller.models import RUN_USAGE_SNAPSHOT_METADATA_KEY
from ya_claw.json_types import JsonObject, JsonValue


def _parse_state_payload(payload: JsonValue) -> JsonObject:
    if isinstance(payload, dict):
        return payload
    raise HTTPException(status_code=500, detail="Run state blob must be a JSON object.")


def ensure_run_dir(settings: ClawSettings, run_id: str) -> Path:
    run_dir = settings.run_store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_blob_path(settings: ClawSettings, run_id: str, blob_name: str) -> Path:
    return settings.run_store_dir / run_id / blob_name


def read_run_state_blob_if_exists(settings: ClawSettings, run_id: str) -> JsonObject | None:
    blob_path = run_blob_path(settings, run_id, "state.json")
    if not blob_path.exists():
        return None
    return _parse_state_payload(load_json_blob(blob_path))


def read_run_message_blob_if_exists(settings: ClawSettings, run_id: str) -> list[JsonObject] | None:
    blob_path = run_blob_path(settings, run_id, "message.json")
    if not blob_path.exists():
        return None
    try:
        return parse_required_message_events(load_json_blob(blob_path), payload_name="Run message blob")
    except TypeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _validate_usage_snapshot(payload: object) -> UsageSnapshot | None:
    if not isinstance(payload, dict):
        return None
    try:
        return UsageSnapshot.model_validate(payload)
    except ValidationError:
        return None


def extract_usage_snapshot_from_state(state: JsonObject | None) -> UsageSnapshot | None:
    """Read optional cumulative usage metadata from the existing state blob."""
    if state is None:
        return None
    return _validate_usage_snapshot(state.get("usage_snapshot"))


def extract_usage_snapshot_from_metadata(metadata: dict[str, Any] | None) -> UsageSnapshot | None:
    """Read the lightweight database index for a committed state snapshot."""
    if not isinstance(metadata, dict):
        return None
    return _validate_usage_snapshot(metadata.get(RUN_USAGE_SNAPSHOT_METADATA_KEY))


def with_usage_snapshot_metadata(
    metadata: dict[str, Any] | None,
    usage_snapshot: UsageSnapshot,
) -> dict[str, Any]:
    """Return run metadata with a lightweight index of the state usage snapshot."""
    indexed_metadata = dict(metadata or {})
    indexed_metadata[RUN_USAGE_SNAPSHOT_METADATA_KEY] = usage_snapshot.model_dump(mode="json")
    return indexed_metadata


def extract_latest_usage_snapshot(events: list[JsonObject] | None) -> UsageSnapshot | None:
    """Return the latest valid cumulative SDK usage snapshot from AGUI replay."""
    if events is None:
        return None
    for event in reversed(events):
        if event.get("type") != "CUSTOM" or event.get("name") != "ya_agent.usage_snapshot":
            continue
        value = event.get("value")
        if not isinstance(value, dict):
            continue
        payload = value.get("payload")
        if not isinstance(payload, dict):
            continue
        snapshot = _validate_usage_snapshot(payload)
        if snapshot is None:
            continue
        transport_run_id = value.get("run_id")
        if isinstance(transport_run_id, str) and transport_run_id:
            return snapshot.model_copy(update={"run_id": transport_run_id})
        return snapshot
    return None


def write_run_blob(settings: ClawSettings, run_id: str, blob_name: str, payload: JsonValue) -> Path:
    run_dir = ensure_run_dir(settings, run_id)
    blob_path = run_dir / blob_name
    blob_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return blob_path


def load_json_blob(path: Path) -> JsonValue:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
