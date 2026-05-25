from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_environment import FileOperationError
from ya_agent_sdk.environment import SandboxEnvironment
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path
from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    RunCreateRequest,
    SessionCreateRequest,
    SessionForkRequest,
    SessionRunCreateRequest,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session import SessionController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.sandbox_ttl import DockerSandboxTtlDispatcher
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state
from ya_claw.workspace import (
    DockerEnvironmentFactory,
    DockerExtraMount,
    DockerWorkspaceProvider,
    LocalEnvironmentFactory,
    LocalWorkspaceProvider,
)
from ya_claw.workspace.models import WorkspaceBindingSpec, derive_mount_id


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'workspace-mounts.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(api_token="test-token", data_dir=data_dir, workspace_dir=workspace_dir, _env_file=None)  # noqa: S106


def _workspace_payload(main: Path, docs: Path) -> dict[str, object]:
    return {
        "mounts": [
            {"id": "main", "host_path": str(main), "virtual_path": "/workspace/main", "mode": "rw"},
            {"id": "docs", "host_path": str(docs), "virtual_path": "/workspace/docs", "mode": "ro"},
        ],
        "default_mount_id": "main",
        "cwd": "/workspace/main/subdir",
    }


def test_workspace_binding_spec_validates_mount_set(tmp_path: Path) -> None:
    main = tmp_path / "main"
    docs = tmp_path / "docs"
    spec = WorkspaceBindingSpec.model_validate(_workspace_payload(main, docs))

    assert spec.default_mount_id == "main"
    assert spec.cwd == "/workspace/main/subdir"
    assert derive_mount_id("/workspace/product-docs") == "product-docs"

    with pytest.raises(ValueError, match="default_mount_id"):
        WorkspaceBindingSpec.model_validate({
            "mounts": [
                {"host_path": str(main), "virtual_path": "/workspace/main"},
                {"host_path": str(docs), "virtual_path": "/workspace/docs"},
            ],
        })
    with pytest.raises(ValueError, match=r"workspace\.cwd"):
        WorkspaceBindingSpec.model_validate({
            "mounts": [{"host_path": str(main), "virtual_path": "/workspace/main"}],
            "cwd": "/other",
        })


async def test_session_run_and_fork_persist_workspace_metadata(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    main = settings.resolved_workspace_dir / "main"
    docs = settings.resolved_workspace_dir / "docs"
    workspace = _workspace_payload(main, docs)
    controller = SessionController()
    runtime_state = create_runtime_state()

    created = await controller.create(
        db_session,
        settings,
        runtime_state,
        SessionCreateRequest(workspace=WorkspaceBindingSpec.model_validate(workspace)),
    )
    session_record = await db_session.get(SessionRecord, created.session.id)
    assert isinstance(session_record, SessionRecord)
    assert session_record.session_metadata["workspace"]["default_mount_id"] == "main"

    run = await controller.create_run(
        db_session,
        settings,
        runtime_state,
        created.session.id,
        SessionRunCreateRequest(
            workspace=WorkspaceBindingSpec.model_validate({
                "mounts": [{"id": "solo", "host_path": str(main), "virtual_path": "/workspace/solo"}]
            })
        ),
    )
    assert run.metadata["workspace"]["default_mount_id"] == "solo"

    session_record.head_success_run_id = run.id
    run_record = await db_session.get(RunRecord, run.id)
    assert isinstance(run_record, RunRecord)
    run_record.status = "completed"
    await db_session.commit()

    forked = await controller.fork(
        db_session,
        created.session.id,
        SessionForkRequest(
            workspace=WorkspaceBindingSpec.model_validate({
                "mounts": [{"id": "fork", "host_path": str(docs), "virtual_path": "/workspace/fork"}]
            })
        ),
    )
    fork_record = await db_session.get(SessionRecord, forked.id)
    assert isinstance(fork_record, SessionRecord)
    assert fork_record.session_metadata["workspace"]["default_mount_id"] == "fork"


async def test_run_controller_auto_session_persists_workspace(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    workspace_dir = settings.resolved_workspace_dir / "api"
    controller = RunController()
    runtime_state = create_runtime_state()

    run = await controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            workspace=WorkspaceBindingSpec.model_validate({
                "mounts": [{"id": "api", "host_path": str(workspace_dir), "virtual_path": "/workspace/api"}]
            }),
        ),
    )
    session_record = await db_session.get(SessionRecord, run.session_id)
    assert isinstance(session_record, SessionRecord)
    assert session_record.session_metadata["workspace"]["default_mount_id"] == "api"
    assert run.metadata["workspace"]["default_mount_id"] == "api"


def test_local_provider_resolves_explicit_multi_mount_and_read_only_policy(tmp_path: Path) -> None:
    main = tmp_path / "main"
    docs = tmp_path / "docs"
    provider = LocalWorkspaceProvider(tmp_path / "fallback", virtual_workspace_path=Path("/workspace"))
    binding = provider.resolve({"workspace": _workspace_payload(main, docs)})

    assert binding.host_path == main.resolve()
    assert binding.cwd == normalize_virtual_path("/workspace/main/subdir")
    assert binding.readable_paths == [
        normalize_virtual_path("/workspace/main"),
        normalize_virtual_path("/workspace/docs"),
    ]
    assert binding.writable_paths == [normalize_virtual_path("/workspace/main")]
    assert len(binding.mounts) == 2


async def test_local_read_only_mount_denies_file_writes(tmp_path: Path) -> None:
    main = tmp_path / "main"
    docs = tmp_path / "docs"
    provider = LocalWorkspaceProvider(tmp_path / "fallback", virtual_workspace_path=Path("/workspace"))
    binding = provider.resolve({"workspace": _workspace_payload(main, docs)})
    environment = LocalEnvironmentFactory().build(binding)

    async with environment as env:
        assert env.file_operator is not None
        await env.file_operator.write_file("/workspace/main/ok.txt", "ok")
        with pytest.raises(FileOperationError):
            await env.file_operator.write_file("/workspace/docs/no.txt", "no")


def test_docker_provider_resolves_multi_mount_and_extra_mount_fingerprint(tmp_path: Path) -> None:
    main = tmp_path / "main"
    docs = tmp_path / "docs"
    cache = tmp_path / "cache"
    provider = DockerWorkspaceProvider(
        tmp_path / "fallback",
        image="python:3.11",
        extra_mounts=[DockerExtraMount(cache, Path("/cache"), "ro")],
        workspace_uid=501,
        workspace_gid=20,
    )
    binding = provider.resolve({"workspace": _workspace_payload(main, docs), "session_id": "session-1"})
    factory = DockerEnvironmentFactory(
        image="python:3.11",
        container_cache_dir=tmp_path / "containers",
        extra_mounts=[DockerExtraMount(cache, Path("/cache"), "ro")],
    )
    environment = factory.build(binding)

    assert isinstance(environment, SandboxEnvironment)
    assert binding.fingerprint.startswith("sha256:")
    assert binding.metadata["workspace_fingerprint_payload"]["extra_mounts"] == [
        {"host_path": str(cache), "container_path": "/cache", "mode": "ro"}
    ]
    assert environment.container_cache_path == tmp_path / "containers" / "sessions" / "session-1" / "workspace.json"


def test_docker_provider_rejects_extra_mount_conflicts(tmp_path: Path) -> None:
    provider = DockerWorkspaceProvider(
        tmp_path / "fallback",
        image="python:3.11",
        extra_mounts=[DockerExtraMount(tmp_path / "cache", Path("/workspace/main/cache"), "rw")],
    )
    with pytest.raises(ValueError, match="conflicts"):
        provider.resolve({"workspace": _workspace_payload(tmp_path / "main", tmp_path / "docs")})


def test_docker_environment_factory_uses_run_cache_path(tmp_path: Path) -> None:
    provider = DockerWorkspaceProvider(tmp_path / "workspace", image="python:3.11")
    binding = provider.resolve({
        "run_id": "run-1",
        "sandbox": {"scope": "run", "run_id": "run-1", "generation": 1, "workspace_fingerprint": "sha256:test"},
    })
    factory = DockerEnvironmentFactory(image="python:3.11", container_cache_dir=tmp_path / "cache")
    environment = factory.build(binding)

    assert environment.container_cache_path == tmp_path / "cache" / "runs" / "run-1" / "workspace.json"


async def test_docker_sandbox_ttl_dispatcher_stops_expired_session_sandbox(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []
    cache_path = (
        settings.resolved_workspace_provider_docker_container_cache_dir / "sessions" / "session-ttl" / "workspace.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"container_id":"container-ttl"}\n', encoding="utf-8")

    async def fake_stop(container_id: str) -> bool:
        stopped.append(container_id)
        return True

    monkeypatch.setattr("ya_claw.execution.sandbox_ttl._stop_docker_container", fake_stop)
    session_record = SessionRecord(
        id="session-ttl",
        profile_name="general",
        session_metadata={
            "sandbox": {
                "provider": "docker",
                "scope": "session",
                "retention_policy": "stop_on_idle",
                "idle_ttl_seconds": 1,
                "last_used_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
                "container_id": "container-ttl",
                "container_ref": "ya-claw-session-session-ttl-g1",
                "cache_path": str(cache_path),
            }
        },
    )
    db_session.add(session_record)
    await db_session.commit()

    dispatcher = DockerSandboxTtlDispatcher(
        settings=settings.model_copy(update={"workspace_provider_backend": "docker"}),
        session_factory=create_session_factory(db_engine),
    )
    stopped_count = await dispatcher.cleanup_once()
    await db_session.refresh(session_record)

    assert stopped_count == 1
    assert stopped == ["container-ttl"]
    assert session_record.session_metadata["sandbox"]["status"] == "stopped"
    assert session_record.session_metadata["sandbox"]["container_id"] is None
    assert not cache_path.exists()
    assert not cache_path.parent.exists()


async def test_docker_sandbox_ttl_dispatcher_keeps_metadata_when_stop_fails(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stop(container_id: str) -> bool:
        return False

    monkeypatch.setattr("ya_claw.execution.sandbox_ttl._stop_docker_container", fake_stop)
    session_record = SessionRecord(
        id="session-ttl-failed",
        profile_name="general",
        session_metadata={
            "sandbox": {
                "provider": "docker",
                "scope": "session",
                "status": "running",
                "ready_state": "ready",
                "retention_policy": "stop_on_idle",
                "idle_ttl_seconds": 1,
                "last_used_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
                "container_id": "container-ttl-failed",
                "verified_container_id": "container-ttl-failed",
                "container_ref": "ya-claw-session-session-ttl-failed-g1",
            }
        },
    )
    db_session.add(session_record)
    await db_session.commit()

    dispatcher = DockerSandboxTtlDispatcher(
        settings=settings.model_copy(update={"workspace_provider_backend": "docker"}),
        session_factory=create_session_factory(db_engine),
    )
    stopped_count = await dispatcher.cleanup_once()
    await db_session.refresh(session_record)

    sandbox = session_record.session_metadata["sandbox"]
    assert stopped_count == 0
    assert sandbox["status"] == "running"
    assert sandbox["container_id"] == "container-ttl-failed"
    assert sandbox["verified_container_id"] == "container-ttl-failed"
    assert "Failed to stop idle Docker workspace container" in sandbox["error_message"]
