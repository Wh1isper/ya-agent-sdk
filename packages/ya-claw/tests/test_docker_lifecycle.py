from __future__ import annotations

from typing import Any

import docker
import pytest
from ya_claw.workspace.docker_lifecycle import remove_docker_container


class _FakeContainer:
    def __init__(self, labels: dict[str, str]) -> None:
        self.attrs: dict[str, Any] = {"Config": {"Labels": labels}}
        self.stop_calls = 0
        self.remove_calls = 0

    def reload(self) -> None:
        return None

    def stop(self, *, timeout: int) -> None:
        assert timeout == 10
        self.stop_calls += 1

    def remove(self, *, force: bool) -> None:
        assert force is True
        self.remove_calls += 1


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container

    def get(self, container_ref: str) -> _FakeContainer:
        assert container_ref == "ya-claw-session-session-1-g1"
        return self._container


class _FakeDockerClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)
        self.closed = False

    def close(self) -> None:
        self.closed = True


async def test_remove_docker_container_requires_matching_ownership_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer({
        "io.ya-claw.workspace.managed": "true",
        "io.ya-claw.workspace.scope": "session",
        "io.ya-claw.workspace.session-id": "session-1",
    })
    client = _FakeDockerClient(container)
    monkeypatch.setattr(docker, "from_env", lambda: client)

    removed = await remove_docker_container(
        "ya-claw-session-session-1-g1",
        expected_labels={
            "io.ya-claw.workspace.managed": "true",
            "io.ya-claw.workspace.scope": "session",
            "io.ya-claw.workspace.session-id": "session-1",
        },
    )

    assert removed is True
    assert container.stop_calls == 1
    assert container.remove_calls == 1
    assert client.closed is True


async def test_remove_docker_container_refuses_mismatched_ownership_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer({
        "io.ya-claw.workspace.managed": "true",
        "io.ya-claw.workspace.scope": "session",
        "io.ya-claw.workspace.session-id": "different-session",
    })
    client = _FakeDockerClient(container)
    monkeypatch.setattr(docker, "from_env", lambda: client)

    removed = await remove_docker_container(
        "ya-claw-session-session-1-g1",
        expected_labels={
            "io.ya-claw.workspace.managed": "true",
            "io.ya-claw.workspace.scope": "session",
            "io.ya-claw.workspace.session-id": "session-1",
        },
    )

    assert removed is False
    assert container.stop_calls == 0
    assert container.remove_calls == 0
    assert client.closed is True
