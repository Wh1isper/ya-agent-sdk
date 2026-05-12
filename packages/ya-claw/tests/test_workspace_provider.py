from __future__ import annotations

import json
from pathlib import Path

from ya_agent_sdk.environment import SandboxEnvironment, VirtualLocalFileOperator, VirtualMount
from ya_agent_sdk.environment.sandbox import DockerShell
from ya_claw.workspace import (
    DockerEnvironmentFactory,
    DockerExtraMount,
    DockerWorkspaceProvider,
    LocalEnvironmentFactory,
    LocalWorkspaceProvider,
    MappedLocalEnvironment,
    ReusableSandboxEnvironment,
    WorkspaceGuidance,
    WorkspaceLocalShell,
    build_workspace_container_ref,
    build_workspace_sandbox_metadata,
    format_workspace_guidance,
    load_workspace_guidance,
)


class FakeImage:
    id = "sha256:image-current"
    short_id = "sha256:image-short"

    def __init__(self, digest: str = "python:3.11@sha256:current") -> None:
        self.attrs = {"RepoDigests": [digest]}


class FakeImages:
    def __init__(self, digest: str | None = None) -> None:
        self.digest = digest

    def get(self, image: str) -> FakeImage:
        if self.digest is None:
            raise RuntimeError(f"image unavailable: {image}")
        return FakeImage(self.digest)


class FakeDockerClientBase:
    def __init__(self, *, images: FakeImages | None = None) -> None:
        self.images = images or FakeImages()


def test_local_workspace_provider_resolves_single_workspace(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    provider = LocalWorkspaceProvider(workspace_dir)

    binding = provider.resolve(metadata={"source": "api"})

    assert binding.host_path == workspace_dir.resolve()
    assert binding.host_path.exists()
    assert binding.virtual_path == workspace_dir.resolve()
    assert binding.cwd == workspace_dir.resolve()
    assert binding.readable_paths == [workspace_dir.resolve()]
    assert binding.writable_paths == [workspace_dir.resolve()]
    assert binding.metadata["source"] == "api"
    assert binding.metadata["provider"] == "local"
    assert binding.metadata["shell_backend"] == "local"
    assert binding.backend_hint == "local"


async def test_service_local_plus_local_shell_uses_real_paths_for_file_ops_and_shell(tmp_path: Path) -> None:
    provider = LocalWorkspaceProvider(tmp_path / "workspace")
    binding = provider.resolve()
    factory = LocalEnvironmentFactory()
    environment = factory.build(binding)
    assert isinstance(environment, MappedLocalEnvironment)

    async with environment as env:
        assert isinstance(env.file_operator, VirtualLocalFileOperator)
        assert env.shell is not None
        await env.file_operator.write_file("notes.txt", "hello")
        content = await env.file_operator.read_file("notes.txt")
        exit_code, stdout, stderr = await env.shell.execute("pwd && ls")

    assert content == "hello"
    assert exit_code == 0
    assert stderr == ""
    assert str(binding.host_path) in stdout
    assert "notes.txt" in stdout


async def test_local_environment_factory_passes_workspace_environment(tmp_path: Path) -> None:
    provider = LocalWorkspaceProvider(tmp_path / "workspace")
    binding = provider.resolve()
    factory = LocalEnvironmentFactory(workspace_environment={"LARK_APP_ID": "cli_test"})
    environment = factory.build(binding)

    async with environment as env:
        assert isinstance(env.shell, WorkspaceLocalShell)
        exit_code, stdout, stderr = await env.shell.execute("printf '%s' \"$LARK_APP_ID\"")

    assert exit_code == 0
    assert stderr == ""
    assert stdout == "cli_test"


def test_docker_workspace_provider_defaults_docker_host_path_to_service_path(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    provider = DockerWorkspaceProvider(workspace_dir, image="python:3.11")

    binding = provider.resolve(metadata={"session_id": "session-1"})

    assert binding.host_path == workspace_dir.resolve()
    assert binding.docker_host_path == workspace_dir.resolve()
    assert binding.virtual_path == Path("/workspace")
    assert binding.cwd == Path("/workspace")
    assert binding.metadata["provider"] == "docker"
    assert binding.metadata["docker_image"] == "python:3.11"
    assert binding.metadata["host_mount"] == str(workspace_dir.resolve())
    assert binding.metadata["service_mount"] == str(workspace_dir.resolve())
    assert binding.metadata["sandbox"] == {
        "provider": "docker",
        "container_ref": build_workspace_container_ref(image="python:3.11", workspace_dir=workspace_dir),
        "image": "python:3.11",
    }
    assert binding.backend_hint == "docker"


def test_docker_workspace_provider_supports_separate_service_and_daemon_paths(tmp_path: Path) -> None:
    service_workspace_dir = tmp_path / "service-workspace"
    host_workspace_dir = tmp_path / "host-workspace"
    provider = DockerWorkspaceProvider(
        service_workspace_dir,
        image="python:3.11",
        docker_host_workspace_dir=host_workspace_dir,
    )

    binding = provider.resolve(metadata={"session_id": "session-1"})

    assert binding.host_path == service_workspace_dir.resolve()
    assert binding.docker_host_path == host_workspace_dir.resolve()
    assert binding.virtual_path == Path("/workspace")
    assert binding.cwd == Path("/workspace")
    assert binding.metadata["host_mount"] == str(host_workspace_dir.resolve())
    assert binding.metadata["service_mount"] == str(service_workspace_dir.resolve())
    assert binding.metadata["sandbox"] == {
        "provider": "docker",
        "container_ref": build_workspace_container_ref(image="python:3.11", workspace_dir=host_workspace_dir),
        "image": "python:3.11",
    }


def test_service_local_plus_docker_shell_uses_virtual_paths_for_file_ops_and_shell(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    provider = DockerWorkspaceProvider(workspace_dir, image="python:3.11")
    binding = provider.resolve(metadata={"session_id": "session-1"})
    factory = DockerEnvironmentFactory(image="python:3.11", workspace_uid=1234, workspace_gid=2345)

    environment = factory.build(binding)

    assert isinstance(environment, SandboxEnvironment)
    assert isinstance(environment, ReusableSandboxEnvironment)
    assert binding.host_path == workspace_dir.resolve()
    assert binding.docker_host_path == workspace_dir.resolve()
    assert binding.virtual_path == Path("/workspace")
    assert binding.cwd == Path("/workspace")
    assert environment.container_ref == build_workspace_container_ref(image="python:3.11", workspace_dir=workspace_dir)
    assert binding.metadata["workspace_uid"] == 1234
    assert binding.metadata["workspace_gid"] == 2345
    assert binding.metadata["sandbox"]["container_ref"] == environment.container_ref
    assert environment._docker_exec_user == "1234:2345"


async def test_reusable_sandbox_environment_passes_workspace_identity_to_docker(tmp_path: Path) -> None:
    captured_run_kwargs: dict[str, object] = {}

    class FakeContainer:
        id = "container-123"

    class FakeContainers:
        def run(self, **kwargs: object) -> FakeContainer:
            captured_run_kwargs.update(kwargs)
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        workspace_uid=1234,
        workspace_gid=2345,
        workspace_environment={"LARK_APP_ID": "cli_test"},
    )
    environment._client = FakeDockerClient()

    container_id = await environment._create_container()

    assert container_id == "container-123"
    assert captured_run_kwargs["name"] == "workspace-container"
    assert captured_run_kwargs["working_dir"] == "/workspace"
    assert captured_run_kwargs["environment"] == {
        "LARK_APP_ID": "cli_test",
        "YA_CLAW_WORKSPACE_STARTUP_DIR": "/workspace",
        "YA_CLAW_WORKSPACE_UID": "1234",
        "YA_CLAW_HOST_UID": "1234",
        "YA_CLAW_WORKSPACE_GID": "2345",
        "YA_CLAW_HOST_GID": "2345",
    }


async def test_reusable_sandbox_environment_creates_shell_with_exec_user_and_home(tmp_path: Path) -> None:
    class FakeContainer:
        id = "container-123"
        status = "running"

        def __init__(self) -> None:
            self.attrs = {"State": {}}

        def reload(self) -> None:
            return None

    class FakeContainers:
        def get(self, container_id: str) -> FakeContainer:
            assert container_id == "container-123"
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        preferred_container_id="container-123",
        workspace_uid=1234,
        workspace_gid=2345,
        docker_exec_default_env={"HOME": "/home/claw", "USER": "claw"},
    )
    environment._client = FakeDockerClient()

    await environment._setup()

    assert isinstance(environment._shell, DockerShell)
    assert environment._shell._exec_user == "1234:2345"
    assert environment._shell._default_env == {"HOME": "/home/claw", "USER": "claw"}


def test_workspace_sandbox_metadata_preserves_workspace_identity(tmp_path: Path) -> None:
    provider = DockerWorkspaceProvider(tmp_path / "workspace", image="python:3.11")
    binding = provider.resolve(metadata={"session_id": "session-1"})
    factory = DockerEnvironmentFactory(image="python:3.11", workspace_uid=0, workspace_gid=0)
    environment = factory.build(binding)
    environment._container_id = "container-123"

    metadata = build_workspace_sandbox_metadata(binding=binding, environment=environment)

    assert metadata is not None
    assert metadata["provider"] == "docker"
    assert metadata["container_id"] == "container-123"
    assert metadata["container_ref"] == build_workspace_container_ref(
        image="python:3.11",
        workspace_dir=tmp_path / "workspace",
    )
    assert metadata["workspace_uid"] == 0
    assert metadata["workspace_gid"] == 0
    assert metadata["host_mount"] == str((tmp_path / "workspace").resolve())
    assert metadata["container_mount"] == "/workspace"
    assert metadata["cwd"] == "/workspace"


def test_docker_environment_factory_prefers_container_id_and_keeps_stable_ref(tmp_path: Path) -> None:
    provider = DockerWorkspaceProvider(tmp_path / "workspace", image="python:3.11")
    binding = provider.resolve(
        metadata={
            "session_id": "session-1",
            "sandbox": {
                "container_id": "container-123",
                "container_ref": "workspace-container",
            },
        },
    )
    factory = DockerEnvironmentFactory(image="python:3.11")

    environment = factory.build(binding)

    assert isinstance(environment, ReusableSandboxEnvironment)
    assert environment.container_id == "container-123"
    assert environment.container_ref == "workspace-container"


def test_docker_environment_factory_uses_session_cache_path(tmp_path: Path) -> None:
    provider = DockerWorkspaceProvider(tmp_path / "workspace", image="python:3.11")
    binding = provider.resolve(metadata={"session_id": "session-1"})
    factory = DockerEnvironmentFactory(image="python:3.11", container_cache_dir=tmp_path / "cache")

    environment = factory.build(binding)

    assert isinstance(environment, ReusableSandboxEnvironment)
    assert environment.container_cache_path == tmp_path / "cache" / "sessions" / "session-1" / "workspace.json"


async def test_service_docker_plus_docker_shell_uses_host_visible_mount_for_container(tmp_path: Path) -> None:
    captured_run_kwargs: dict[str, object] = {}

    class FakeContainer:
        id = "container-123"

    class FakeContainers:
        def run(self, **kwargs: object) -> FakeContainer:
            captured_run_kwargs.update(kwargs)
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    host_workspace_dir = tmp_path / "host-workspace"
    provider = DockerWorkspaceProvider(
        tmp_path / "workspace",
        image="python:3.11",
        docker_host_workspace_dir=host_workspace_dir,
    )
    binding = provider.resolve(metadata={"session_id": "session-1"})
    factory = DockerEnvironmentFactory(image="python:3.11")
    environment = factory.build(binding)
    environment._client = FakeDockerClient()

    assert isinstance(environment, ReusableSandboxEnvironment)
    await environment._create_container()

    assert captured_run_kwargs["volumes"] == {str(host_workspace_dir.resolve()): {"bind": "/workspace", "mode": "rw"}}


async def test_docker_environment_factory_passes_extra_mounts_to_container(tmp_path: Path) -> None:
    captured_run_kwargs: dict[str, object] = {}

    class FakeContainer:
        id = "container-123"

    class FakeContainers:
        def run(self, **kwargs: object) -> FakeContainer:
            captured_run_kwargs.update(kwargs)
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    workspace_dir = tmp_path / "workspace"
    home_dir = tmp_path / "home"
    cache_dir = tmp_path / "cache"
    provider = DockerWorkspaceProvider(
        workspace_dir,
        image="python:3.11",
        extra_mounts=[
            DockerExtraMount(host_path=home_dir, container_path=Path("/home/claw"), mode="rw"),
            DockerExtraMount(host_path=cache_dir, container_path=Path("/cache"), mode="ro"),
        ],
    )
    binding = provider.resolve(metadata={"session_id": "session-1"})
    factory = DockerEnvironmentFactory(
        image="python:3.11",
        extra_mounts=[
            DockerExtraMount(host_path=home_dir, container_path=Path("/home/claw"), mode="rw"),
            DockerExtraMount(host_path=cache_dir, container_path=Path("/cache"), mode="ro"),
        ],
    )
    environment = factory.build(binding)
    environment._client = FakeDockerClient()

    assert isinstance(environment, ReusableSandboxEnvironment)
    await environment._create_container()

    assert binding.metadata["extra_mounts"] == [
        {"host_path": str(home_dir), "container_path": "/home/claw", "mode": "rw"},
        {"host_path": str(cache_dir), "container_path": "/cache", "mode": "ro"},
    ]
    assert captured_run_kwargs["volumes"] == {
        str(workspace_dir.resolve()): {"bind": "/workspace", "mode": "rw"},
        str(home_dir.resolve()): {"bind": "/home/claw", "mode": "rw"},
        str(cache_dir.resolve()): {"bind": "/cache", "mode": "ro"},
    }


async def test_reusable_sandbox_environment_reads_and_refreshes_container_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "workspace.json"

    class FakeContainer:
        id = "container-123"
        status = "running"

        def __init__(self) -> None:
            self.attrs = {"State": {}}

        def reload(self) -> None:
            return None

    class FakeContainers:
        def get(self, container_id: str) -> FakeContainer:
            assert container_id == "container-123"
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        container_cache_path=cache_path,
    )
    environment._client = FakeDockerClient()
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({
            "schema_version": 1,
            "container_ref": "workspace-container",
            "container_id": "container-123",
            "image": "python:3.11",
        }),
        encoding="utf-8",
    )

    await environment._ensure_container()

    assert environment.container_id == "container-123"
    refreshed_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed_payload["container_id"] == "container-123"
    assert refreshed_payload["work_dir"] == "/workspace"


async def test_reusable_sandbox_environment_waits_for_healthy_container(tmp_path: Path) -> None:
    health_statuses = ["starting", "healthy"]

    class FakeContainer:
        id = "container-123"
        status = "running"

        def __init__(self) -> None:
            self.attrs = {"State": {"Health": {"Status": health_statuses.pop(0)}}}

        def reload(self) -> None:
            return None

    class FakeContainers:
        def get(self, container_id: str) -> FakeContainer:
            assert container_id == "container-123"
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        preferred_container_id="container-123",
    )
    environment._client = FakeDockerClient()

    await environment._ensure_container()

    assert environment.container_id == "container-123"
    assert health_statuses == []


async def test_reusable_sandbox_environment_recreates_container_when_image_digest_changes(tmp_path: Path) -> None:  # noqa: C901
    cache_path = tmp_path / "cache" / "workspace.json"
    run_calls = 0
    removed: list[str] = []

    class FakeImageRef:
        def __init__(self, digest: str) -> None:
            self.id = digest
            self.attrs = {"RepoDigests": [digest]}

    class FakeContainer:
        status = "running"

        def __init__(self, container_id: str, digest: str) -> None:
            self.id = container_id
            self.image = FakeImageRef(digest)
            self.attrs = {"State": {}, "Image": digest}

        def reload(self) -> None:
            return None

        def stop(self, timeout: int = 10) -> None:
            return None

        def remove(self, force: bool = False) -> None:
            removed.append(self.id)

    class FakeContainers:
        def get(self, container_ref: str) -> FakeContainer:
            if container_ref in {"container-old", "workspace-container"}:
                return FakeContainer("container-old", "python:3.11@sha256:old")
            if container_ref == "container-new":
                return FakeContainer("container-new", "python:3.11@sha256:new")
            raise RuntimeError(container_ref)

        def run(self, **kwargs: object) -> FakeContainer:
            nonlocal run_calls
            run_calls += 1
            return FakeContainer("container-new", "python:3.11@sha256:new")

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images or FakeImages("python:3.11@sha256:new"))
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        container_cache_path=cache_path,
    )
    environment._client = FakeDockerClient()
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({
            "schema_version": 1,
            "container_ref": "workspace-container",
            "container_id": "container-old",
            "image": "python:3.11",
            "image_digest": "python:3.11@sha256:old",
        }),
        encoding="utf-8",
    )

    await environment._ensure_container()

    assert run_calls == 1
    assert removed == ["container-old"]
    assert environment.container_id == "container-new"
    refreshed_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed_payload["container_id"] == "container-new"
    assert refreshed_payload["image_digest"] == "python:3.11@sha256:new"


async def test_reusable_sandbox_environment_refreshes_stale_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "workspace.json"
    run_calls = 0

    class NotFound(Exception):
        pass

    class FakeContainer:
        id = "container-new"

        def __init__(self) -> None:
            self.attrs = {"State": {}}

        def reload(self) -> None:
            return None

    class FakeContainers:
        def get(self, container_ref: str) -> FakeContainer:
            if container_ref == "container-new":
                return FakeContainer()
            raise NotFound(container_ref)

        def run(self, **kwargs: object) -> FakeContainer:
            nonlocal run_calls
            run_calls += 1
            return FakeContainer()

    class FakeDockerClient(FakeDockerClientBase):
        def __init__(self, *, images: FakeImages | None = None) -> None:
            super().__init__(images=images)
            self.containers = FakeContainers()

    environment = ReusableSandboxEnvironment(
        mounts=[VirtualMount(host_path=tmp_path / "workspace", virtual_path=Path("/workspace"))],
        work_dir="/workspace",
        image="python:3.11",
        container_ref="workspace-container",
        container_cache_path=cache_path,
    )
    environment._client = FakeDockerClient()
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({
            "schema_version": 1,
            "container_ref": "workspace-container",
            "container_id": "container-stale",
            "image": "python:3.11",
        }),
        encoding="utf-8",
    )

    await environment._ensure_container()

    assert run_calls == 1
    assert environment.container_id == "container-new"
    refreshed_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed_payload["container_id"] == "container-new"


def test_load_workspace_guidance_reads_workspace_agents_file(tmp_path: Path) -> None:
    provider = LocalWorkspaceProvider(tmp_path / "workspace")
    binding = provider.resolve()
    agents_path = tmp_path / "workspace" / "AGENTS.md"
    agents_path.write_text("# Workspace\nUse pytest.\n", encoding="utf-8")

    guidance = load_workspace_guidance(binding)

    assert guidance is not None
    assert guidance.host_path == agents_path.resolve()
    assert guidance.virtual_path == (tmp_path / "workspace" / "AGENTS.md").resolve()
    assert guidance.content == "# Workspace\nUse pytest.\n"


def test_load_workspace_guidance_ignores_empty_file(tmp_path: Path) -> None:
    provider = LocalWorkspaceProvider(tmp_path / "workspace")
    binding = provider.resolve()
    (tmp_path / "workspace" / "AGENTS.md").write_text("   \n", encoding="utf-8")

    assert load_workspace_guidance(binding) is None


def test_format_workspace_guidance_uses_virtual_path(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    provider = LocalWorkspaceProvider(workspace_dir)
    binding = provider.resolve()
    guidance = load_workspace_guidance(binding)
    assert guidance is None

    formatted = format_workspace_guidance(
        WorkspaceGuidance(
            host_path=workspace_dir / "AGENTS.md",
            virtual_path=Path('/workspace/path-"quoted"/AGENTS.md'),
            content="Use <safe> rules.",
        )
    )

    assert formatted == (
        '<workspace-guidance path="/workspace/path-&quot;quoted&quot;/AGENTS.md">\n'
        "Use <safe> rules.\n"
        "</workspace-guidance>"
    )


def test_load_workspace_guidance_reads_full_large_agents_file(tmp_path: Path) -> None:
    provider = LocalWorkspaceProvider(tmp_path / "workspace")
    binding = provider.resolve()
    agents_path = tmp_path / "workspace" / "AGENTS.md"
    content = "a" * (300 * 1024)
    agents_path.write_text(content, encoding="utf-8")

    guidance = load_workspace_guidance(binding)

    assert guidance is not None
    assert guidance.content == content
