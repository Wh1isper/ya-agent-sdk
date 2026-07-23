"""End-to-end integration test for the full chat execution pipeline.

Verifies:
1. Session create via HTTP POST /api/v1/sessions
2. Auto-dispatch through an AgentProfile
3. Supervisor claim + coordinator execution
4. AGUI events via SSE stream
5. Commit artifacts (state.json, message.json)
6. Session GET returns state + message

Requires configured gateway credentials. Set GATEWAY_API_KEY and GATEWAY_BASE_URL.
If gateway credentials are missing, the test is skipped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx_sse import connect_sse
from ya_claw.app import create_app
from ya_claw.config import get_settings


def _has_model() -> bool:
    import os

    return os.environ.get("GATEWAY_API_KEY", "").strip() != "" and os.environ.get("GATEWAY_BASE_URL", "").strip() != ""


@pytest.fixture(autouse=True)
def _clear_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    initialize_sqlite_database: Callable[[str], None],
) -> None:
    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    get_settings.cache_clear()
    settings = get_settings()
    initialize_sqlite_database(settings.resolved_database_url)
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.mark.skipif(not _has_model(), reason="Gateway credentials are not configured")
def test_full_chat_pipeline_creates_run_streams_events_and_commits_artifacts() -> None:
    """Test the complete execution pipeline end-to-end.

    Creates a session with input, lets the coordinator execute a real agent
    run, and verifies SSE events, commit artifacts, and session GET.
    """
    settings = get_settings()

    with TestClient(create_app()) as client:
        # 1. Create session with input
        create_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={
                "profile_name": "default",
                "metadata": {"source": "e2e-test"},
                "input_parts": [{"type": "text", "text": "Say exactly: e2e-test-complete"}],
                "dispatch_mode": "async",
            },
        )
        assert create_response.status_code == 201, create_response.text
        payload = create_response.json()
        session_id = payload["session"]["id"]
        run_id = payload["run"]["id"]
        assert payload["session"]["status"] == "queued"
        assert payload["run"]["status"] == "queued"

        # 2. Wait for run to complete via polling
        completed = _poll_run_status(client, run_id, "completed", timeout=120)
        assert completed, f"Run {run_id} did not complete within timeout"

        # 3. Verify run detail via GET
        run_response = client.get(
            f"/api/v1/runs/{run_id}?include_state=true&include_message=true",
            headers=_auth_headers(),
        )
        assert run_response.status_code == 200, run_response.text
        run_detail = run_response.json()

        assert run_detail["run"]["status"] == "completed"
        assert run_detail["run"]["termination_reason"] == "completed"
        assert run_detail["session"]["id"] == session_id

        # 4. Verify committed state exists
        assert run_detail["state"] is not None, "state.json should have been committed"
        assert isinstance(run_detail["state"], dict)
        assert run_detail["state"]["session_id"] == session_id
        assert run_detail["state"]["run_id"] == run_id

        # 5. Verify committed message exists
        assert run_detail["message"] is not None, "message.json should have been committed"
        assert isinstance(run_detail["message"], list)

        # 6. Verify session GET returns completed status
        session_response = client.get(
            f"/api/v1/sessions/{session_id}?include_message=true",
            headers=_auth_headers(),
        )
        assert session_response.status_code == 200, session_response.text
        session_detail = session_response.json()
        assert session_detail["session"]["status"] == "completed"
        assert session_detail["session"]["head_success_run_id"] == run_id

        # 7. Verify continuation works (no restore_from_run_id -> uses head_success_run_id)
        continue_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={
                "input_parts": [{"type": "text", "text": "Continue with: say 'e2e-continue-complete'"}],
            },
        )
        assert continue_response.status_code == 201, continue_response.text
        continue_run_id = continue_response.json()["id"]

        completed2 = _poll_run_status(client, continue_run_id, "completed", timeout=120)
        assert completed2, f"Continue run {continue_run_id} did not complete"

        continue_detail = client.get(
            f"/api/v1/runs/{continue_run_id}?include_state=true&include_message=true",
            headers=_auth_headers(),
        ).json()
        assert continue_detail["run"]["status"] == "completed"
        assert continue_detail["run"]["restore_from_run_id"] == run_id
        assert continue_detail["state"] is not None
        assert continue_detail["message"] is not None

    # 8. Verify artifacts on disk
    run_dir = settings.run_store_dir / run_id
    assert (run_dir / "state.json").exists()
    assert (run_dir / "message.json").exists()

    continue_run_dir = settings.run_store_dir / continue_run_id
    assert (continue_run_dir / "state.json").exists()
    assert (continue_run_dir / "message.json").exists()


@pytest.mark.skipif(not _has_model(), reason="Gateway credentials are not configured")
def test_sse_event_stream_during_run_execution() -> None:
    """Test SSE event streaming during a real run execution.

    Connects to the SSE endpoint after creating a run and verifies
    event flow including RUN_STARTED, TEXT_MESSAGE_CONTENT, RUN_FINISHED.
    """
    with TestClient(create_app()) as client:
        create_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={
                "profile_name": "default",
                "input_parts": [{"type": "text", "text": "Say exactly: sse-test-ok"}],
                "dispatch_mode": "async",
            },
        )
        assert create_response.status_code == 201
        run_id = create_response.json()["run"]["id"]

        # Connect to SSE stream
        events: list[dict[str, object]] = []
        base_url = "http://testserver"
        with connect_sse(
            client,
            "GET",
            f"{base_url}/api/v1/runs/{run_id}/events",
            headers=_auth_headers(),
        ) as event_source:
            for event in event_source.iter_sse():
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                events.append(data)
                if data.get("type") == "RUN_FINISHED":
                    break

        event_types = [e.get("type") for e in events]
        assert "RUN_STARTED" in event_types, f"Expected RUN_STARTED in events: {event_types}"
        assert "RUN_FINISHED" in event_types, f"Expected RUN_FINISHED in events: {event_types}"

        # Verify text content was streamed
        text_events = [e for e in events if e.get("type") == "TEXT_MESSAGE_CONTENT"]
        assert len(text_events) > 0, f"Expected TEXT_MESSAGE_CONTENT events in: {event_types}"


def _poll_run_status(client: TestClient, run_id: str, target_status: str, timeout: float) -> bool:
    import time as time_module

    start = time_module.monotonic()
    while time_module.monotonic() - start < timeout:
        response = client.get(f"/api/v1/runs/{run_id}", headers=_auth_headers())
        if response.status_code == 200:
            status = response.json()["run"]["status"]
            if status == target_status:
                return True
            if status in ("failed", "cancelled"):
                error = response.json()["run"].get("error_message", "unknown")
                pytest.fail(f"Run {run_id} entered {status} state: {error}")
        time_module.sleep(1)
    return False
