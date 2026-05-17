from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from ya_claw.config import ClawSettings
from ya_claw.json_types import JsonArray, JsonObject, JsonValue


def _parse_message_events(payload: JsonValue) -> list[JsonObject]:
    if not isinstance(payload, list):
        raise TypeError("Expected top-level JSON array for AGUI replay events.")
    parsed_events: list[JsonObject] = [event for event in payload if isinstance(event, dict)]
    if len(parsed_events) != len(payload):
        raise ValueError("Expected AGUI replay event objects only.")
    return parsed_events


class RunStore:
    def __init__(self, settings: ClawSettings):
        self._settings = settings

    @property
    def root_dir(self) -> Path:
        return self._settings.run_store_dir

    def run_dir(self, run_id: str) -> Path:
        path = self.root_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def message_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "message.json"

    def has_state(self, run_id: str) -> bool:
        return self.state_path(run_id).exists()

    def has_message(self, run_id: str) -> bool:
        return self.message_path(run_id).exists()

    def read_state(self, run_id: str) -> JsonObject | None:
        payload = self._read_json_if_exists(self.state_path(run_id))
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Expected JSON object at {self.state_path(run_id)}.")

    def read_message(self, run_id: str) -> list[JsonObject] | None:
        payload = self._read_json_if_exists(self.message_path(run_id))
        if payload is None:
            return None
        try:
            return _parse_message_events(payload)
        except TypeError as exc:
            raise ValueError(f"Invalid AGUI replay list at {self.message_path(run_id)}.") from exc

    def write_state(self, run_id: str, payload: JsonObject) -> Path:
        return self._atomic_write_json(self.state_path(run_id), payload)

    def write_message(self, run_id: str, payload: list[JsonObject]) -> Path:
        message_payload: JsonArray = list(payload)
        return self._atomic_write_json(self.message_path(run_id), message_payload)

    def write_checkpoint_message(
        self,
        run_id: str,
        payload: list[JsonObject],
        *,
        checkpoint_kind: str,
    ) -> Path:
        _ = checkpoint_kind
        return self.write_message(run_id, payload)

    def _atomic_write_json(self, path: Path, payload: JsonValue) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp_file:
            tmp_file.write(serialized)
            tmp_file.flush()
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
        return path

    def _read_json_if_exists(self, path: Path) -> JsonValue | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
