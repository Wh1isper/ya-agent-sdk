from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

DOCKER_MANAGED_LABEL = "io.ya-claw.workspace.managed"
DOCKER_SCOPE_LABEL = "io.ya-claw.workspace.scope"
DOCKER_SESSION_ID_LABEL = "io.ya-claw.workspace.session-id"
DOCKER_RUN_ID_LABEL = "io.ya-claw.workspace.run-id"
DOCKER_GENERATION_LABEL = "io.ya-claw.workspace.generation"


def build_docker_container_ref(
    *,
    scope: str,
    session_id: str,
    run_id: str | None,
    generation: int,
) -> str:
    if scope == "run":
        if not isinstance(run_id, str) or run_id.strip() == "":
            raise ValueError("run_id is required for run-scoped Docker containers")
        return f"ya-claw-run-{run_id[:12]}"
    if scope != "session":
        raise ValueError(f"Unsupported Docker sandbox scope: {scope}")
    if generation < 1:
        raise ValueError("generation must be positive for session-scoped Docker containers")
    return f"ya-claw-session-{session_id[:12]}-g{generation}"


def build_docker_container_labels(
    *,
    scope: str,
    session_id: str,
    run_id: str | None,
    generation: int,
) -> dict[str, str]:
    labels = {
        DOCKER_MANAGED_LABEL: "true",
        DOCKER_SCOPE_LABEL: scope,
        DOCKER_SESSION_ID_LABEL: session_id,
        DOCKER_GENERATION_LABEL: str(generation),
    }
    if scope == "run":
        if not isinstance(run_id, str) or run_id.strip() == "":
            raise ValueError("run_id is required for run-scoped Docker containers")
        labels[DOCKER_RUN_ID_LABEL] = run_id
    elif scope != "session":
        raise ValueError(f"Unsupported Docker sandbox scope: {scope}")
    return labels


def docker_container_labels_from_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    scope = metadata.get("scope")
    session_id = metadata.get("session_id")
    run_id = metadata.get("run_id")
    generation = metadata.get("generation")
    if (
        not isinstance(scope, str)
        or not isinstance(session_id, str)
        or session_id.strip() == ""
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
    ):
        return {DOCKER_MANAGED_LABEL: "true"}
    try:
        return build_docker_container_labels(
            scope=scope,
            session_id=session_id,
            run_id=run_id if isinstance(run_id, str) else None,
            generation=generation,
        )
    except ValueError:
        return {DOCKER_MANAGED_LABEL: "true"}


async def remove_docker_container(container_ref: str, *, expected_labels: Mapping[str, str]) -> bool:
    """Stop and remove a Docker container only when its ownership labels match."""

    def _remove() -> bool:
        try:
            import docker
            import docker.errors

            client = docker.from_env()
            try:
                try:
                    container = client.containers.get(container_ref)
                except docker.errors.NotFound:
                    return True

                container.reload()
                attrs = container.attrs
                config = attrs.get("Config") if isinstance(attrs, dict) else None
                labels_value = config.get("Labels") if isinstance(config, dict) else None
                labels = labels_value if isinstance(labels_value, dict) else {}
                mismatched = {
                    key: {"expected": value, "actual": labels.get(key)}
                    for key, value in expected_labels.items()
                    if labels.get(key) != value
                }
                if mismatched:
                    logger.warning(
                        "Refusing to remove Docker workspace container with mismatched ownership labels "
                        "ref={} mismatched_labels={}",
                        container_ref,
                        mismatched,
                    )
                    return False

                try:
                    container.stop(timeout=10)
                except docker.errors.NotFound:
                    return True
                except docker.errors.APIError as exc:
                    logger.debug(
                        "Docker container stop failed before removal ref={} error={}",
                        container_ref,
                        exc,
                    )

                try:
                    container.remove(force=True)
                    return True
                except docker.errors.NotFound:
                    return True
            finally:
                client.close()
        except Exception as exc:
            logger.warning("Failed to remove Docker workspace container ref={} error={}", container_ref, exc)
            return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _remove)


async def delete_docker_container_cache(cache_path: Path) -> bool:
    """Delete one derived Docker workspace container cache file."""

    def _delete() -> bool:
        try:
            cache_path.unlink()
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.warning("Failed to delete Docker workspace cache path={} error={}", cache_path, exc)
            return False
        with contextlib.suppress(OSError):
            cache_path.parent.rmdir()
        return True

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _delete)
