"""YAACLI-local bounded projection of complete AGUI events."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ya_agent_stream_protocol.agui import AguiReplayBuffer, AguiReplayConfig, validate_display_events
from ya_agent_stream_protocol.json_types import JsonObject, JsonValue

_DEFAULT_MAX_RUNS = 20
_DEFAULT_MAX_EVENTS = 2_000
_DEFAULT_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_CHUNK_CHARS = 64 * 1024
_DEFAULT_MAX_EVENT_PAYLOAD_CHARS = 16 * 1024
MAX_DISPLAY_REPLAY_LOAD_BYTES = 32 * 1024 * 1024
_COMPACTION_TARGET_RATIO = 0.75
_MAX_IDENTITY_CHARS = 1024
_TRUNCATION_SUFFIX = "\n... [YAACLI replay projection truncated]"
_IDENTITY_FIELDS = frozenset({
    "type",
    "runId",
    "run_id",
    "messageId",
    "message_id",
    "toolCallId",
    "tool_call_id",
    "parentMessageId",
    "parent_message_id",
    "toolCallName",
    "tool_call_name",
    "role",
    "name",
    "timestamp",
    "yaacliAgentId",
    "yaacli_agent_id",
})


@dataclass(frozen=True, slots=True)
class DisplayReplayLimits:
    """Budgets for YAACLI's local, lossy session display projection."""

    max_runs: int = _DEFAULT_MAX_RUNS
    max_events: int = _DEFAULT_MAX_EVENTS
    max_bytes: int = _DEFAULT_MAX_BYTES
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS
    max_event_payload_chars: int = _DEFAULT_MAX_EVENT_PAYLOAD_CHARS

    def normalized(self) -> DisplayReplayLimits:
        return DisplayReplayLimits(
            max_runs=max(1, self.max_runs),
            max_events=max(1, self.max_events),
            max_bytes=max(1, self.max_bytes),
            max_chunk_chars=max(1, self.max_chunk_chars),
            max_event_payload_chars=max(1, self.max_event_payload_chars),
        )


@dataclass(slots=True)
class _ProjectionBudget:
    remaining: int
    truncated: bool = False


def load_display_replay(path: Path, *, max_bytes: int = MAX_DISPLAY_REPLAY_LOAD_BYTES) -> list[dict[str, Any]] | None:
    """Load validated display events through one descriptor and a bounded read."""
    with path.open("rb") as display_file:
        if os.fstat(display_file.fileno()).st_size > max_bytes:
            return None
        payload = display_file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        return None
    return validate_display_events(json.loads(payload.decode("utf-8")))


class BoundedDisplayReplay:
    """Receive complete AGUI events and retain only a bounded UI projection.

    The shared stream protocol remains lossless. Truncation happens here, after
    receipt, and affects only YAACLI's local display/session replay window.
    """

    def __init__(
        self,
        *,
        config: AguiReplayConfig | None = None,
        limits: DisplayReplayLimits | None = None,
    ) -> None:
        self._config = config or AguiReplayConfig()
        self._limits = (limits or DisplayReplayLimits()).normalized()
        self._replay = AguiReplayBuffer(config=self._config)
        self._chunk_chars: dict[tuple[str, str, str], int] = {}
        self._truncated_chunk_keys: set[tuple[str, str, str]] = set()
        self._current_run_id = ""
        self._estimated_bytes = 0
        self._run_count = 0

    @property
    def retained_bytes(self) -> int:
        """Conservative retained serialized-byte estimate."""
        return self._estimated_bytes

    def append(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "RUN_STARTED":
            self._current_run_id = _bounded_identifier(_identifier(event.get("runId") or event.get("run_id")))
            self._run_count += 1
        chunk_key = (
            (_event_run_id(event, self._current_run_id), event_type, _chunk_identifier(event, event_type))
            if event_type in {"TEXT_MESSAGE_CHUNK", "REASONING_MESSAGE_CHUNK", "TOOL_CALL_CHUNK"}
            else None
        )
        existing_chunk = chunk_key is not None and chunk_key in self._chunk_chars
        truncated_before = len(self._truncated_chunk_keys)
        projected = self._project_event(event)
        if projected is None:
            if len(self._truncated_chunk_keys) > truncated_before:
                self._estimated_bytes += 32
                if self._estimated_bytes > self._limits.max_bytes:
                    self._compact(reserve_capacity=True)
            return
        self._replay.append(projected)
        if existing_chunk:
            # Conservatively count the entire projected event. This includes
            # worst-case JSON escaping for metadata that the shared replay may
            # fill on a later chunk, while compaction resets to the exact size.
            self._estimated_bytes += _json_size(projected)
        else:
            self._estimated_bytes += _json_size(projected)
        if (
            self._estimated_bytes > self._limits.max_bytes
            or len(self._replay.events) > self._limits.max_events
            or self._run_count > self._limits.max_runs
        ):
            self._compact(reserve_capacity=True)

    def extend_snapshot(self, events: list[dict[str, Any]]) -> None:
        self.clear()
        for event in events:
            self.append(event)
        self._compact(reserve_capacity=False)

    def snapshot(self) -> list[JsonObject]:
        self._compact(reserve_capacity=False)
        return self._replay.snapshot()

    def clear(self) -> None:
        self._replay.clear()
        self._chunk_chars.clear()
        self._truncated_chunk_keys.clear()
        self._current_run_id = ""
        self._estimated_bytes = 0
        self._run_count = 0

    def _project_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("type", ""))
        if event_type in {"TEXT_MESSAGE_CHUNK", "REASONING_MESSAGE_CHUNK", "TOOL_CALL_CHUNK"}:
            return self._project_chunk(event, event_type)

        budget = _ProjectionBudget(self._limits.max_event_payload_chars)
        projected: dict[str, Any] = {}
        for key, value in event.items():
            projected[key] = _project_identity(value) if key in _IDENTITY_FIELDS else _project_json(value, budget)
        if budget.truncated:
            projected["yaacliReplayTruncated"] = True
        return projected

    def _project_chunk(self, event: dict[str, Any], event_type: str) -> dict[str, Any] | None:
        identifier = _chunk_identifier(event, event_type)
        key = (_event_run_id(event, self._current_run_id), event_type, identifier)
        used = self._chunk_chars.get(key, 0)
        available = max(0, self._limits.max_chunk_chars - used)
        delta = event.get("delta")
        delta_text = delta if isinstance(delta, str) else str(delta or "")
        if available <= 0:
            self._truncated_chunk_keys.add(key)
            return None

        budget = _ProjectionBudget(self._limits.max_event_payload_chars)
        projected = {
            field: (
                delta_text
                if field == "delta"
                else _project_identity(value)
                if field in _IDENTITY_FIELDS
                else _project_json(value, budget)
            )
            for field, value in event.items()
        }
        if budget.truncated:
            projected["yaacliReplayTruncated"] = True
        if len(delta_text) > available:
            projected["delta"] = _truncate_text(delta_text, available)
            projected["yaacliReplayTruncated"] = True
            self._truncated_chunk_keys.add(key)
            self._chunk_chars[key] = self._limits.max_chunk_chars
        else:
            projected["delta"] = delta_text
            self._chunk_chars[key] = used + len(delta_text)
        return projected

    def _compact(self, *, reserve_capacity: bool) -> None:
        active_run_id = self._current_run_id
        events = _annotate_truncated_chunks(self._replay.snapshot(), self._truncated_chunk_keys)
        events = _retain_recent_runs(events, self._limits.max_runs)
        target_events = (
            max(1, int(self._limits.max_events * _COMPACTION_TARGET_RATIO))
            if reserve_capacity
            else self._limits.max_events
        )
        if len(events) > target_events:
            events = events[-target_events:]

        sizes = [_json_size(event) for event in events]
        total_bytes = sum(sizes)
        target_bytes = (
            max(1, int(self._limits.max_bytes * _COMPACTION_TARGET_RATIO))
            if reserve_capacity
            else self._limits.max_bytes
        )
        remove_count = 0
        while remove_count < len(events) and total_bytes > target_bytes:
            total_bytes -= sizes[remove_count]
            remove_count += 1
        if remove_count:
            events = events[remove_count:]

        self._replay.clear()
        self._chunk_chars.clear()
        self._truncated_chunk_keys.clear()
        self._current_run_id = ""
        self._run_count = 0
        for event in events:
            event_type = str(event.get("type", ""))
            if event_type == "RUN_STARTED":
                self._current_run_id = _bounded_identifier(_identifier(event.get("runId") or event.get("run_id")))
                self._run_count += 1
            self._replay.append(dict(event))
            if event_type in {"TEXT_MESSAGE_CHUNK", "REASONING_MESSAGE_CHUNK", "TOOL_CALL_CHUNK"}:
                key = (_event_run_id(event, self._current_run_id), event_type, _chunk_identifier(event, event_type))
                delta = event.get("delta")
                self._chunk_chars[key] = len(delta) if isinstance(delta, str) else 0
                if event.get("yaacliReplayTruncated") is True:
                    self._truncated_chunk_keys.add(key)
        if not self._current_run_id:
            self._current_run_id = active_run_id
        self._estimated_bytes = sum(_json_size(event) for event in events)


def _project_identity(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | float):
        return value
    return _bounded_identifier(value if isinstance(value, str) else str(value))


def _annotate_truncated_chunks(
    events: list[JsonObject],
    truncated_keys: set[tuple[str, str, str]],
) -> list[JsonObject]:
    current_run_id = ""
    annotated: list[JsonObject] = []
    chunk_types = {"TEXT_MESSAGE_CHUNK", "REASONING_MESSAGE_CHUNK", "TOOL_CALL_CHUNK"}
    for source in events:
        event = dict(source)
        event_type = str(event.get("type", ""))
        if event_type == "RUN_STARTED":
            current_run_id = _identifier(event.get("runId") or event.get("run_id"))
        if event_type in chunk_types:
            key = (_event_run_id(event, current_run_id), event_type, _chunk_identifier(event, event_type))
            if key in truncated_keys:
                delta = event.get("delta")
                if isinstance(delta, str):
                    event["delta"] = _replace_tail_with_suffix(delta)
                event["yaacliReplayTruncated"] = True
        annotated.append(event)
    return annotated


def _replace_tail_with_suffix(value: str) -> str:
    if len(value) <= len(_TRUNCATION_SUFFIX):
        return value[: max(0, len(value) - 3)] + "..." if len(value) >= 3 else value
    return value[: len(value) - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _project_json(value: Any, budget: _ProjectionBudget, *, depth: int = 0) -> JsonValue:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= budget.remaining:
            budget.remaining -= len(value)
            return value
        budget.truncated = True
        projected = _truncate_text(value, budget.remaining)
        budget.remaining = 0
        return projected
    if depth >= 6 or budget.remaining <= 0:
        budget.truncated = True
        return _truncate_text(str(value), budget.remaining)
    if isinstance(value, list | tuple):
        projected_items: list[JsonValue] = []
        for item in value[:64]:
            projected_items.append(_project_json(item, budget, depth=depth + 1))
            if budget.remaining <= 0:
                break
        if len(value) > len(projected_items):
            budget.truncated = True
        return projected_items
    if isinstance(value, dict):
        projected_mapping: dict[str, JsonValue] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 64 or budget.remaining <= 0:
                budget.truncated = True
                break
            projected_mapping[str(key)] = _project_json(item, budget, depth=depth + 1)
        return projected_mapping
    return _project_json(str(value), budget, depth=depth + 1)


def _truncate_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= len(_TRUNCATION_SUFFIX):
        return value[:limit]
    return value[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _retain_recent_runs(events: list[JsonObject], max_runs: int) -> list[JsonObject]:
    run_starts = [index for index, event in enumerate(events) if event.get("type") == "RUN_STARTED"]
    if len(run_starts) <= max_runs:
        return events
    return events[run_starts[-max_runs] :]


def _event_run_id(event: dict[str, Any], fallback: str) -> str:
    value = _identifier(event.get("runId") or event.get("run_id"))
    return _bounded_identifier(value) if value else fallback


def _chunk_identifier(event: dict[str, Any], event_type: str) -> str:
    if event_type == "TOOL_CALL_CHUNK":
        value = _identifier(event.get("toolCallId") or event.get("tool_call_id"))
    else:
        value = _identifier(event.get("messageId") or event.get("message_id"))
    return _bounded_identifier(value)


def _identifier(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bounded_identifier(value: str) -> str:
    if len(value) <= _MAX_IDENTITY_CHARS:
        return value
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()
    prefix_length = _MAX_IDENTITY_CHARS - len(digest) - 1
    return f"{value[:prefix_length]}#{digest}"


def _json_size(event: JsonObject) -> int:
    return len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
