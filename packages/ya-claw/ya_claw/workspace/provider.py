from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from loguru import logger
from y_agent_environment import Environment, FileOperationError, ResourceFactory, ResourceRegistryState
from ya_agent_sdk.environment import (
    LocalShell,
    SandboxEnvironment,
    VirtualLocalFileOperator,
    VirtualMount,
)
from ya_agent_sdk.environment.sandbox import DockerShell

from ya_claw.workspace.models import (
    SANDBOX_METADATA_KEY,
    SANDBOX_SCOPE_RUN,
    SANDBOX_SCOPE_SESSION,
    WORKSPACE_METADATA_KEY,
    SandboxScopeLiteral,
    WorkspaceBinding,
    WorkspaceBindingSpec,
    WorkspaceMountBinding,
    compute_workspace_fingerprint,
    extract_workspace_metadata,
    relative_virtual_path,
    virtual_path_contains,
    workspace_fingerprint_payload,
)

_DOCKER_SANDBOX_METADATA_KEY = SANDBOX_METADATA_KEY
_DOCKER_SANDBOX_PROVIDER = "docker"
_DOCKER_WORKSPACE_NAME_PREFIX = "ya-claw-workspace"
_DOCKER_CONTAINER_CACHE_SCHEMA_VERSION = 1
_DOCKER_CONTAINER_LOCKS: dict[str, asyncio.Lock] = {}
_DEFAULT_VIRTUAL_WORKSPACE_PATH = Path("/workspace")
_DEFAULT_CONTAINER_CACHE_FILE = "workspace.json"
_DEFAULT_DOCKER_WORKSPACE_HOME = "/home/claw"
_DEFAULT_DOCKER_WORKSPACE_USER = "claw"
_AUTO_DOCKER_EXEC_USER = "auto"
_ROOT_DOCKER_EXEC_USER = "root"


@dataclass(frozen=True, slots=True)
class DockerExtraMount:
    host_path: Path
    container_path: Path
    mode: str = "rw"

    def __post_init__(self) -> None:
        if self.mode not in {"rw", "ro"}:
            raise ValueError("Docker extra mount mode must be 'rw' or 'ro'")
        if not self.container_path.is_absolute():
            raise ValueError("Docker extra mount container_path must be absolute")


class WorkspaceProvider(ABC):
    @abstractmethod
    def resolve(self, metadata: dict[str, Any] | None = None) -> WorkspaceBinding:
        raise NotImplementedError


class PolicyVirtualLocalFileOperator(VirtualLocalFileOperator):
    def __init__(self, *, read_only_virtual_paths: list[Path] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._read_only_virtual_paths = [Path(path) for path in read_only_virtual_paths or []]

    def _assert_writable(self, path: str) -> None:
        virtual = self._resolve_virtual(path)
        for read_only_path in self._read_only_virtual_paths:
            if virtual_path_contains(read_only_path, virtual):
                raise FileOperationError("write", path, "path is mounted read-only")

    async def _write_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        self._assert_writable(path)
        await super()._write_file_impl(path, content, encoding=encoding)

    async def _append_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        self._assert_writable(path)
        await super()._append_file_impl(path, content, encoding=encoding)

    async def _delete_impl(self, path: str) -> None:
        self._assert_writable(path)
        await super()._delete_impl(path)

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        self._assert_writable(path)
        await super()._mkdir_impl(path, parents=parents)

    async def _move_impl(self, src: str, dst: str) -> None:
        self._assert_writable(src)
        self._assert_writable(dst)
        await super()._move_impl(src, dst)

    async def _copy_impl(self, src: str, dst: str) -> None:
        self._assert_writable(dst)
        await super()._copy_impl(src, dst)


class MappedLocalEnvironment(Environment):
    def __init__(
        self,
        *,
        mounts: list[VirtualMount],
        host_cwd: Path,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        enable_tmp_dir: bool = True,
        read_only_virtual_paths: list[Path] | None = None,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
        include_os_env: bool = True,
        environment_overrides: dict[str, str] | None = None,
    ) -> None:
        super().__init__(resource_state=resource_state, resource_factories=resource_factories)
        self._mounts = mounts
        self._host_cwd = host_cwd
        self._shell_timeout = shell_timeout
        self._tmp_base_dir = tmp_base_dir
        self._enable_tmp_dir = enable_tmp_dir
        self._include_os_env = include_os_env
        self._environment_overrides = dict(environment_overrides or {})
        self._read_only_virtual_paths = [Path(path) for path in read_only_virtual_paths or []]
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    async def _setup(self) -> None:
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="ya_claw_workspace_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name)

        allowed_paths = [mount.host_path.resolve() for mount in self._mounts]
        logger.debug(
            "Setting up local workspace environment cwd={} allowed_paths={} tmp_dir={}",
            self._host_cwd,
            allowed_paths,
            tmp_dir_path,
        )
        self._file_operator = PolicyVirtualLocalFileOperator(
            mounts=self._mounts,
            default_virtual_path=_virtual_path_for_host_cwd(self._mounts, self._host_cwd),
            read_only_virtual_paths=self._read_only_virtual_paths,
            tmp_dir=tmp_dir_path,
        )
        if tmp_dir_path is not None:
            allowed_paths.append(tmp_dir_path.resolve())
        self._shell = WorkspaceLocalShell(
            default_cwd=self._host_cwd,
            allowed_paths=allowed_paths,
            default_timeout=self._shell_timeout,
            include_os_env=self._include_os_env,
            environment_overrides=self._environment_overrides,
        )

    async def _teardown(self) -> None:
        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None


class WorkspaceLocalShell(LocalShell):
    def __init__(
        self,
        *,
        environment_overrides: dict[str, str],
        default_cwd: Path | None = None,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        include_os_env: bool = True,
    ) -> None:
        super().__init__(
            default_cwd=default_cwd,
            allowed_paths=allowed_paths,
            default_timeout=default_timeout,
            include_os_env=include_os_env,
        )
        self._environment_overrides = dict(environment_overrides)

    def _build_effective_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        merged_env = {**self._environment_overrides, **dict(env or {})}
        if not merged_env:
            return super()._build_effective_env(env)
        return super()._build_effective_env(merged_env)


class ReusableSandboxEnvironment(SandboxEnvironment):
    def __init__(
        self,
        *,
        mounts: list[VirtualMount],
        work_dir: str,
        image: str,
        container_ref: str,
        preferred_container_id: str | None = None,
        workspace_uid: int | None = None,
        workspace_gid: int | None = None,
        workspace_environment: dict[str, str] | None = None,
        docker_exec_user: str | None = None,
        docker_exec_default_env: dict[str, str] | None = None,
        shell_timeout: float = 30.0,
        cleanup_on_exit: bool = False,
        container_cache_path: Path | None = None,
        docker_host_paths: list[Path] | None = None,
        docker_mount_modes: list[str] | None = None,
        read_only_virtual_paths: list[Path] | None = None,
        sandbox_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            mounts=mounts,
            work_dir=work_dir,
            container_id=preferred_container_id,
            image=image,
            cleanup_on_exit=cleanup_on_exit,
            shell_timeout=shell_timeout,
        )
        self._container_ref = container_ref
        self._workspace_uid = workspace_uid
        self._workspace_gid = workspace_gid
        self._workspace_environment = dict(workspace_environment or {})
        self._docker_exec_user = _resolve_docker_exec_user(
            docker_exec_user,
            workspace_uid=workspace_uid,
            workspace_gid=workspace_gid,
        )
        self._docker_exec_default_env = dict(docker_exec_default_env or {})
        self._container_cache_path = container_cache_path.expanduser() if container_cache_path is not None else None
        self._docker_host_paths = (
            [path.expanduser() for path in docker_host_paths] if docker_host_paths is not None else []
        )
        self._docker_mount_modes = list(docker_mount_modes or [])
        self._read_only_virtual_paths = [Path(path) for path in read_only_virtual_paths or []]
        self._sandbox_metadata = dict(sandbox_metadata or {})

    @property
    def container_ref(self) -> str:
        return self._container_ref

    @property
    def container_cache_path(self) -> Path | None:
        return self._container_cache_path

    @property
    def sandbox_metadata(self) -> dict[str, Any]:
        return dict(self._sandbox_metadata)

    async def _setup(self) -> None:
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="ya_agent_sandbox_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name)

        for mount in self._mounts:
            mount.host_path.resolve().mkdir(parents=True, exist_ok=True)

        logger.debug(
            "Setting up Docker workspace environment ref={} image={} work_dir={} mounts={} docker_host_paths={} cache={}",
            self._container_ref,
            self._image,
            self._work_dir,
            [(str(mount.host_path), str(mount.virtual_path)) for mount in self._mounts],
            [str(path) for path in self._docker_host_paths],
            self._container_cache_path,
        )
        if self._custom_shell is None:
            lock = get_docker_container_lock(
                cache_path=self._container_cache_path,
                container_ref=self._container_ref,
            )
            async with lock:
                self._sandbox_metadata["last_used_at"] = _utc_now_iso()
                await self._ensure_container()

        self._file_operator = PolicyVirtualLocalFileOperator(
            mounts=self._mounts,
            default_virtual_path=Path(self._work_dir),
            read_only_virtual_paths=self._read_only_virtual_paths,
            tmp_dir=tmp_dir_path,
        )
        logger.info(
            "Docker workspace environment ready ref={} container_id={}", self._container_ref, self._container_id
        )

        if self._custom_shell is not None:
            self._shell = self._custom_shell
        else:
            if self._container_id is None:
                raise RuntimeError("container_id must be set when no custom shell is provided")
            self._shell = DockerShell(
                container_id=self._container_id,
                container_workdir=self._work_dir,
                default_timeout=self._shell_timeout,
                exec_user=self._docker_exec_user,
                default_env=self._docker_exec_default_env,
            )

    async def _ensure_container(self) -> None:
        await self._refresh_current_image_digest()
        cached_container_id = self._container_id or await self._read_cached_container_id()
        if cached_container_id is not None:
            self._container_id = cached_container_id
            try:
                await self._verify_container()
                await self._wait_for_container_ready(cached_container_id)
                await self._write_cached_container_id(cached_container_id)
                logger.info(
                    "Reusing cached Docker workspace container ref={} id={}",
                    self._container_ref,
                    cached_container_id,
                )
                return
            except RuntimeError:
                await self._clear_cached_container_id(cached_container_id)
                await self._remove_container(cached_container_id)
                self._container_id = None

        discovered_container_id = await self._resolve_container_id_from_ref()
        if discovered_container_id is not None:
            self._container_id = discovered_container_id
            try:
                await self._verify_container()
                await self._wait_for_container_ready(discovered_container_id)
                await self._write_cached_container_id(discovered_container_id)
                logger.info(
                    "Reusing discovered Docker workspace container ref={} id={}",
                    self._container_ref,
                    discovered_container_id,
                )
                return
            except RuntimeError:
                await self._clear_cached_container_id(discovered_container_id)
                await self._remove_container(discovered_container_id)
                self._container_id = None

        self._container_id = await self._create_container()
        self._created_container = True
        await self._wait_for_container_ready(self._container_id)
        await self._write_cached_container_id(self._container_id)

    async def _teardown(self) -> None:
        if self._cleanup_on_exit and self._container_id is not None:
            await self._clear_cached_container_id(self._container_id)
            await self._stop_container()

        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None

    async def _create_container(self) -> str:
        if self._image is None:
            raise ValueError("Image must be provided to create a new container")

        await self._refresh_current_image_digest()
        image = self._image
        work_dir = self._work_dir
        mounts = self._mounts
        tmp_dir = self.tmp_dir
        container_ref = self._container_ref
        workspace_uid = self._workspace_uid
        workspace_gid = self._workspace_gid
        workspace_environment = dict(self._workspace_environment)

        def _run_container() -> str:
            try:
                volumes = {
                    str(_resolve_docker_mount_host_path(mounts, self._docker_host_paths, index).resolve()): {
                        "bind": str(mount.virtual_path),
                        "mode": _resolve_docker_mount_mode(self._docker_mount_modes, index),
                    }
                    for index, mount in enumerate(mounts)
                }
                if tmp_dir is not None:
                    volumes[str(tmp_dir)] = {"bind": str(tmp_dir), "mode": "rw"}
                environment = {**workspace_environment, "YA_CLAW_WORKSPACE_STARTUP_DIR": work_dir}
                logger.info(
                    "Starting reusable Docker workspace container ref={} image={} work_dir={} volumes={}",
                    container_ref,
                    image,
                    work_dir,
                    volumes,
                )
                if isinstance(workspace_uid, int):
                    environment["YA_CLAW_WORKSPACE_UID"] = str(workspace_uid)
                    environment["YA_CLAW_HOST_UID"] = str(workspace_uid)
                if isinstance(workspace_gid, int):
                    environment["YA_CLAW_WORKSPACE_GID"] = str(workspace_gid)
                    environment["YA_CLAW_HOST_GID"] = str(workspace_gid)
                container = self.client.containers.run(
                    image=image,
                    volumes=volumes,
                    working_dir=work_dir,
                    environment=environment,
                    detach=True,
                    stdin_open=True,
                    tty=True,
                    name=container_ref,
                )
                container_id = container.id
                if container_id is None:
                    raise RuntimeError("Container was created but has no ID")
                logger.info("Started Docker workspace container ref={} id={}", container_ref, container_id)
                return container_id
            except Exception as exc:
                raise RuntimeError(f"Failed to start reusable container '{container_ref}': {exc}") from exc

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_container)

    async def _verify_container(self) -> None:
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("Container ID is not set")
        expected_image_digest = _normalize_optional_str(self._sandbox_metadata.get("image_digest"))

        def _check_and_start_container() -> None:
            try:
                container = self.client.containers.get(container_id)
                container.reload()
                container_image_digest = _container_image_digest(container)
                if expected_image_digest is not None and container_image_digest != expected_image_digest:
                    raise RuntimeError(
                        f"Container {container_id} image digest changed: {container_image_digest} != {expected_image_digest}"
                    )
                status = _normalize_optional_str(getattr(container, "status", None))
                if status == "running":
                    return
                if status in ("exited", "created", "paused"):
                    container.start()
                    container.reload()
                    next_status = _normalize_optional_str(getattr(container, "status", None))
                    if next_status != "running":
                        raise RuntimeError(f"Container {container_id} failed to start (status: {next_status})")
                    return
                raise RuntimeError(f"Container {container_id} is in unrecoverable state: {status}")
            except RuntimeError:
                raise
            except Exception as exc:
                if exc.__class__.__name__ == "NotFound":
                    raise RuntimeError(f"Container not found: {container_id}") from exc
                raise RuntimeError(f"Failed to verify/start container: {exc}") from exc

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _check_and_start_container)

    async def _wait_for_container_ready(self, container_id: str) -> None:
        timeout_seconds = 60.0
        poll_interval_seconds = 0.25

        def _wait() -> None:
            deadline = time.monotonic() + timeout_seconds
            while True:
                health_status = _inspect_container_health_status(self.client, container_id)
                if health_status is None or health_status == "healthy":
                    return
                if health_status == "unhealthy":
                    raise RuntimeError(f"Container {container_id} is unhealthy")
                if health_status != "starting":
                    raise RuntimeError(f"Container {container_id} has unexpected health status: {health_status}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Container {container_id} did not become healthy within {timeout_seconds}s")
                time.sleep(poll_interval_seconds)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _wait)

    async def _resolve_container_id_from_ref(self) -> str | None:
        container_ref = self._container_ref

        def _resolve() -> str | None:
            try:
                container = self.client.containers.get(container_ref)
                return _normalize_optional_str(getattr(container, "id", None))
            except Exception:
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _resolve)

    async def _refresh_current_image_digest(self) -> str | None:
        if self._image is None:
            return None

        image = self._image

        def _inspect_image_digest() -> str | None:
            return _resolve_image_digest(self.client, image)

        loop = asyncio.get_running_loop()
        digest = await loop.run_in_executor(None, _inspect_image_digest)
        if digest is not None:
            self._sandbox_metadata["image_digest"] = digest
        return digest

    async def _read_cached_container_id(self) -> str | None:
        cache_path = self._container_cache_path
        if cache_path is None:
            return None

        def _read() -> str | None:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None
            if payload.get("schema_version") != _DOCKER_CONTAINER_CACHE_SCHEMA_VERSION:
                return None
            if payload.get("container_ref") != self._container_ref:
                return None
            if payload.get("image") != self._image:
                return None
            cached_image_digest = _normalize_optional_str(payload.get("image_digest"))
            current_image_digest = _normalize_optional_str(self._sandbox_metadata.get("image_digest"))
            if current_image_digest is not None and cached_image_digest != current_image_digest:
                return None
            return _normalize_optional_str(payload.get("container_id"))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    async def _write_cached_container_id(self, container_id: str) -> None:
        cache_path = self._container_cache_path
        if cache_path is None:
            return
        payload = {
            **self._sandbox_metadata,
            "schema_version": _DOCKER_CONTAINER_CACHE_SCHEMA_VERSION,
            "container_ref": self._container_ref,
            "container_id": container_id,
            "image": self._image,
            "work_dir": self._work_dir,
        }

        def _write() -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

    async def _clear_cached_container_id(self, container_id: str | None = None) -> None:
        cache_path = self._container_cache_path
        if cache_path is None:
            return

        def _clear() -> None:
            cached_container_id: str | None = None
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    cached_container_id = _normalize_optional_str(payload.get("container_id"))
            except Exception:
                cached_container_id = None
            if container_id is not None and cached_container_id not in (None, container_id):
                return
            with contextlib.suppress(FileNotFoundError):
                cache_path.unlink()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _clear)

    async def _remove_container(self, container_id: str) -> None:
        def _remove() -> None:
            try:
                container = self.client.containers.get(container_id)
                with contextlib.suppress(Exception):
                    container.stop(timeout=10)
                with contextlib.suppress(Exception):
                    container.remove(force=True)
            except Exception:
                return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _remove)


class EnvironmentFactory(ABC):
    @abstractmethod
    def build(self, binding: WorkspaceBinding) -> Environment:
        raise NotImplementedError


class LocalEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        *,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        workspace_environment: dict[str, str] | None = None,
    ) -> None:
        self._shell_timeout = shell_timeout
        self._tmp_base_dir = tmp_base_dir
        self._workspace_environment = dict(workspace_environment or {})

    def build(self, binding: WorkspaceBinding) -> Environment:
        logger.debug(
            "Building local environment provider={} host_path={} cwd={} readable={} writable={}",
            binding.metadata.get("provider"),
            binding.host_path,
            binding.cwd,
            binding.readable_paths,
            binding.writable_paths,
        )
        return MappedLocalEnvironment(
            mounts=_virtual_mounts_from_binding(binding),
            host_cwd=_host_cwd_from_binding(binding),
            shell_timeout=self._shell_timeout,
            tmp_base_dir=self._tmp_base_dir,
            read_only_virtual_paths=_read_only_paths_from_binding(binding),
            environment_overrides={**self._workspace_environment, **binding.environment_overrides},
        )


class DockerEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        *,
        image: str,
        workspace_uid: int | None = None,
        workspace_gid: int | None = None,
        workspace_environment: dict[str, str] | None = None,
        docker_exec_user: str | None = None,
        docker_exec_default_env: dict[str, str] | None = None,
        shell_timeout: float = 30.0,
        cleanup_on_exit: bool = False,
        container_cache_dir: Path | None = None,
        extra_mounts: list[DockerExtraMount] | None = None,
        retention_policy: str = "stop_on_idle",
        idle_ttl_seconds: int = 3600,
    ) -> None:
        self._image = image
        self._workspace_uid = workspace_uid
        self._workspace_gid = workspace_gid
        self._workspace_environment = dict(workspace_environment or {})
        self._docker_exec_user = docker_exec_user
        self._docker_exec_default_env = dict(docker_exec_default_env or build_docker_workspace_exec_default_env())
        self._shell_timeout = shell_timeout
        self._cleanup_on_exit = cleanup_on_exit
        self._container_cache_dir = container_cache_dir.expanduser() if container_cache_dir is not None else None
        self._extra_mounts = list(extra_mounts or [])
        self._retention_policy = retention_policy
        self._idle_ttl_seconds = idle_ttl_seconds

    def build(self, binding: WorkspaceBinding) -> Environment:
        sandbox_metadata = extract_workspace_sandbox_metadata(binding.metadata) or {}
        sandbox_metadata = {
            **sandbox_metadata,
            "scope": _normalize_optional_str(sandbox_metadata.get("scope")) or binding.sandbox_scope,
            "generation": _first_optional_int(sandbox_metadata.get("generation"), binding.generation),
            "workspace_fingerprint": _normalize_optional_str(sandbox_metadata.get("workspace_fingerprint"))
            or binding.fingerprint,
            "retention_policy": _normalize_optional_str(sandbox_metadata.get("retention_policy"))
            or self._retention_policy,
            "idle_ttl_seconds": _first_optional_int(sandbox_metadata.get("idle_ttl_seconds")) or self._idle_ttl_seconds,
            "session_id": _normalize_optional_str(sandbox_metadata.get("session_id"))
            or _normalize_optional_str(binding.metadata.get("session_id")),
            "run_id": _normalize_optional_str(sandbox_metadata.get("run_id"))
            or _normalize_optional_str(binding.metadata.get("run_id")),
        }
        preferred_container_id = _normalize_optional_str(sandbox_metadata.get("container_id"))
        container_ref = _normalize_optional_str(sandbox_metadata.get("container_ref")) or build_workspace_container_ref(
            image=self._image,
            workspace_dir=_resolve_binding_docker_host_path(binding),
        )
        binding_mounts = _virtual_mounts_from_binding(binding)
        mounts = [
            *binding_mounts,
            *[VirtualMount(mount.host_path, mount.container_path) for mount in self._extra_mounts],
        ]
        docker_host_paths = [
            *_resolve_binding_docker_host_paths(binding),
            *[mount.host_path for mount in self._extra_mounts],
        ]
        docker_mount_modes = [*_docker_mount_modes_from_binding(binding), *[mount.mode for mount in self._extra_mounts]]
        workspace_environment = {**self._workspace_environment, **binding.environment_overrides}
        logger.info(
            "Building Docker environment provider={} service_path={} docker_host_path={} virtual_path={} cwd={} image={} container_ref={} preferred_container_id={} extra_mounts={}",
            binding.metadata.get("provider"),
            binding.host_path,
            _resolve_binding_docker_host_path(binding),
            binding.virtual_path,
            binding.cwd,
            self._image,
            container_ref,
            preferred_container_id,
            [(str(mount.host_path), str(mount.container_path), mount.mode) for mount in self._extra_mounts],
        )
        if isinstance(self._workspace_uid, int):
            binding.metadata["workspace_uid"] = self._workspace_uid
        if isinstance(self._workspace_gid, int):
            binding.metadata["workspace_gid"] = self._workspace_gid
        cache_path = _build_container_cache_path(self._container_cache_dir, metadata=sandbox_metadata)
        binding.metadata[_DOCKER_SANDBOX_METADATA_KEY] = {
            **sandbox_metadata,
            "provider": _DOCKER_SANDBOX_PROVIDER,
            "container_ref": container_ref,
            "image": self._image,
            "cache_path": str(cache_path) if cache_path is not None else None,
        }
        return ReusableSandboxEnvironment(
            mounts=mounts,
            work_dir=str(binding.cwd),
            image=self._image,
            container_ref=container_ref,
            preferred_container_id=preferred_container_id,
            workspace_uid=self._workspace_uid,
            workspace_gid=self._workspace_gid,
            workspace_environment=workspace_environment,
            docker_exec_user=self._docker_exec_user,
            docker_exec_default_env=self._docker_exec_default_env,
            cleanup_on_exit=self._cleanup_on_exit or sandbox_metadata.get("cleanup_on_exit") is True,
            shell_timeout=self._shell_timeout,
            container_cache_path=cache_path,
            docker_host_paths=docker_host_paths,
            docker_mount_modes=docker_mount_modes,
            read_only_virtual_paths=[
                *_read_only_paths_from_binding(binding),
                *[mount.container_path for mount in self._extra_mounts if mount.mode == "ro"],
            ],
            sandbox_metadata=binding.metadata.get(_DOCKER_SANDBOX_METADATA_KEY)
            if isinstance(binding.metadata.get(_DOCKER_SANDBOX_METADATA_KEY), dict)
            else sandbox_metadata,
        )


class DefaultEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        *,
        docker_image: str,
        workspace_uid: int | None = None,
        workspace_gid: int | None = None,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        cleanup_on_exit: bool = False,
        workspace_environment: dict[str, str] | None = None,
        docker_container_cache_dir: Path | None = None,
        docker_extra_mounts: list[DockerExtraMount] | None = None,
        docker_exec_user: str | None = None,
        docker_exec_default_env: dict[str, str] | None = None,
        docker_retention_policy: str = "stop_on_idle",
        docker_idle_ttl_seconds: int = 3600,
    ) -> None:
        self._local_factory = LocalEnvironmentFactory(
            shell_timeout=shell_timeout,
            tmp_base_dir=tmp_base_dir,
            workspace_environment=workspace_environment,
        )
        self._docker_factory = DockerEnvironmentFactory(
            image=docker_image,
            workspace_uid=workspace_uid,
            workspace_gid=workspace_gid,
            workspace_environment=workspace_environment,
            docker_exec_user=docker_exec_user,
            docker_exec_default_env=docker_exec_default_env,
            shell_timeout=shell_timeout,
            cleanup_on_exit=cleanup_on_exit,
            container_cache_dir=docker_container_cache_dir,
            extra_mounts=docker_extra_mounts,
            retention_policy=docker_retention_policy,
            idle_ttl_seconds=docker_idle_ttl_seconds,
        )

    def build(self, binding: WorkspaceBinding) -> Environment:
        backend = (binding.backend_hint or "local").strip().lower()
        logger.debug(
            "Default environment factory selected backend={} provider={}",
            backend,
            binding.metadata.get("provider"),
        )
        if backend == "docker":
            return self._docker_factory.build(binding)
        return self._local_factory.build(binding)


class LocalWorkspaceProvider(WorkspaceProvider):
    def __init__(
        self,
        workspace_dir: Path,
        *,
        virtual_workspace_path: Path | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir.expanduser().resolve()
        self._virtual_workspace_path = virtual_workspace_path or self._workspace_dir

    def resolve(self, metadata: dict[str, Any] | None = None) -> WorkspaceBinding:
        logger.debug(
            "Resolving local workspace binding workspace_dir={} metadata_keys={}",
            self._workspace_dir,
            sorted((metadata or {}).keys()),
        )
        return _build_workspace_binding(
            workspace_dir=self._workspace_dir,
            virtual_workspace_path=self._virtual_workspace_path,
            metadata=dict(metadata or {}),
            provider="local",
            backend_hint="local",
            extra_metadata={
                "shell_backend": "local",
                "file_operator": "local",
                "host_cwd": str(self._workspace_dir),
            },
        )


class DockerWorkspaceProvider(WorkspaceProvider):
    def __init__(
        self,
        workspace_dir: Path,
        *,
        image: str,
        docker_host_workspace_dir: Path | None = None,
        virtual_workspace_path: Path = _DEFAULT_VIRTUAL_WORKSPACE_PATH,
        extra_mounts: list[DockerExtraMount] | None = None,
        workspace_uid: int | None = None,
        workspace_gid: int | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir.expanduser().resolve()
        self._docker_host_workspace_dir = (
            docker_host_workspace_dir.expanduser().resolve()
            if docker_host_workspace_dir is not None
            else self._workspace_dir
        )
        self._image = image
        self._virtual_workspace_path = virtual_workspace_path
        self._extra_mounts = list(extra_mounts or [])
        self._workspace_uid = workspace_uid
        self._workspace_gid = workspace_gid

    def resolve(self, metadata: dict[str, Any] | None = None) -> WorkspaceBinding:
        resolved_metadata = dict(metadata or {})
        sandbox_metadata = extract_workspace_sandbox_metadata(resolved_metadata) or {}
        container_ref = _normalize_optional_str(sandbox_metadata.get("container_ref")) or build_workspace_container_ref(
            image=self._image,
            workspace_dir=self._docker_host_workspace_dir,
        )
        sandbox_metadata = {
            **sandbox_metadata,
            "provider": _DOCKER_SANDBOX_PROVIDER,
            "container_ref": container_ref,
            "image": _normalize_optional_str(sandbox_metadata.get("image")) or self._image,
        }
        resolved_metadata[_DOCKER_SANDBOX_METADATA_KEY] = sandbox_metadata
        logger.info(
            "Resolving Docker workspace binding service_workspace_dir={} docker_host_workspace_dir={} virtual_path={} image={} container_ref={} extra_mounts={} metadata_keys={}",
            self._workspace_dir,
            self._docker_host_workspace_dir,
            self._virtual_workspace_path,
            self._image,
            container_ref,
            [(str(mount.host_path), str(mount.container_path), mount.mode) for mount in self._extra_mounts],
            sorted(resolved_metadata.keys()),
        )
        return _build_workspace_binding(
            workspace_dir=self._workspace_dir,
            docker_host_workspace_dir=self._docker_host_workspace_dir,
            virtual_workspace_path=self._virtual_workspace_path,
            metadata=resolved_metadata,
            provider="docker",
            backend_hint="docker",
            docker_image=self._image,
            workspace_uid=self._workspace_uid,
            workspace_gid=self._workspace_gid,
            extra_mounts=self._extra_mounts,
            extra_metadata={
                "shell_backend": "docker",
                "file_operator": "virtual-local",
                "docker_image": self._image,
                "host_mount": str(self._docker_host_workspace_dir),
                "service_mount": str(self._workspace_dir),
                "container_mount": str(self._virtual_workspace_path),
                "extra_mounts": [
                    {
                        "host_path": str(mount.host_path),
                        "container_path": str(mount.container_path),
                        "mode": mount.mode,
                    }
                    for mount in self._extra_mounts
                ],
            },
        )


def _build_workspace_binding(
    *,
    workspace_dir: Path,
    virtual_workspace_path: Path,
    metadata: dict[str, Any],
    docker_host_workspace_dir: Path | None = None,
    provider: str,
    backend_hint: str,
    extra_metadata: dict[str, Any],
    docker_image: str | None = None,
    workspace_uid: int | None = None,
    workspace_gid: int | None = None,
    extra_mounts: list[DockerExtraMount] | None = None,
) -> WorkspaceBinding:
    workspace_spec = _workspace_spec_from_metadata_or_default(
        metadata=metadata,
        workspace_dir=workspace_dir,
        virtual_workspace_path=virtual_workspace_path,
        docker_host_workspace_dir=docker_host_workspace_dir,
    )
    mounts = _mount_bindings_from_spec(workspace_spec)
    _validate_logical_mounts(mounts, provider=provider)
    _validate_extra_mount_conflicts(mounts, list(extra_mounts or []))
    for mount in mounts:
        mount.host_path.mkdir(parents=True, exist_ok=True)

    default_mount = next(mount for mount in mounts if mount.id == workspace_spec.default_mount_id)
    cwd = Path(workspace_spec.cwd or str(default_mount.virtual_path))
    readable_paths = [mount.virtual_path for mount in mounts]
    writable_paths = [mount.virtual_path for mount in mounts if mount.mode == "rw"]
    fingerprint_payload = workspace_fingerprint_payload(
        provider=provider,
        workspace=workspace_spec,
        docker_image=docker_image,
        workspace_uid=workspace_uid,
        workspace_gid=workspace_gid,
        extra_mounts=_extra_mount_fingerprint_payload(extra_mounts),
    )
    fingerprint = compute_workspace_fingerprint(fingerprint_payload)
    sandbox_metadata = extract_workspace_sandbox_metadata(metadata) or {}
    generation = _first_optional_int(sandbox_metadata.get("generation"))
    sandbox_scope = _normalize_sandbox_scope(sandbox_metadata.get("scope"))

    metadata_payload = {
        **metadata,
        "provider": provider,
        **extra_metadata,
        WORKSPACE_METADATA_KEY: workspace_spec.model_dump(mode="json"),
        "workspace_fingerprint": fingerprint,
        "workspace_fingerprint_payload": fingerprint_payload,
        "mounts": [
            {
                "id": mount.id,
                "name": mount.name,
                "host_path": str(mount.host_path),
                "docker_host_path": str(mount.docker_host_path) if mount.docker_host_path is not None else None,
                "virtual_path": str(mount.virtual_path),
                "mode": mount.mode,
                "metadata": dict(mount.metadata),
            }
            for mount in mounts
        ],
    }

    return WorkspaceBinding(
        host_path=default_mount.host_path,
        virtual_path=default_mount.virtual_path,
        docker_host_path=default_mount.docker_host_path,
        cwd=cwd,
        readable_paths=readable_paths,
        writable_paths=writable_paths,
        mounts=mounts,
        fingerprint=fingerprint,
        generation=generation,
        sandbox_scope=sandbox_scope,
        metadata=metadata_payload,
        backend_hint=backend_hint,
    )


def _workspace_spec_from_metadata_or_default(
    *,
    metadata: dict[str, Any],
    workspace_dir: Path,
    virtual_workspace_path: Path,
    docker_host_workspace_dir: Path | None,
) -> WorkspaceBindingSpec:
    workspace_payload = extract_workspace_metadata(metadata)
    if workspace_payload is not None:
        return WorkspaceBindingSpec.model_validate(workspace_payload)
    mount = {
        "id": "workspace",
        "host_path": str(workspace_dir),
        "virtual_path": str(virtual_workspace_path),
        "mode": "rw",
    }
    if docker_host_workspace_dir is not None:
        mount["docker_host_path"] = str(docker_host_workspace_dir)
    return WorkspaceBindingSpec.model_validate({
        "mounts": [mount],
        "default_mount_id": "workspace",
        "cwd": str(virtual_workspace_path),
    })


def _mount_bindings_from_spec(spec: WorkspaceBindingSpec) -> list[WorkspaceMountBinding]:
    mounts: list[WorkspaceMountBinding] = []
    for mount in spec.mounts:
        if mount.id is None:
            raise ValueError("workspace mount id is required after normalization")
        mounts.append(
            WorkspaceMountBinding(
                id=mount.id,
                name=mount.name,
                host_path=Path(mount.host_path).expanduser().resolve(),
                docker_host_path=Path(mount.docker_host_path).expanduser().resolve()
                if mount.docker_host_path is not None
                else None,
                virtual_path=Path(mount.virtual_path),
                mode=mount.mode,
                metadata=dict(mount.metadata),
            )
        )
    return mounts


def _validate_logical_mounts(mounts: list[WorkspaceMountBinding], *, provider: str) -> None:
    seen_virtual_paths: set[str] = set()
    for mount in mounts:
        virtual_path = str(mount.virtual_path)
        if virtual_path in seen_virtual_paths:
            raise ValueError(f"workspace virtual path '{virtual_path}' is duplicated")
        seen_virtual_paths.add(virtual_path)
        if provider == "docker" and not virtual_path_contains(_DEFAULT_VIRTUAL_WORKSPACE_PATH, mount.virtual_path):
            raise ValueError("Docker workspace virtual paths must stay under /workspace")


def _validate_extra_mount_conflicts(
    logical_mounts: list[WorkspaceMountBinding],
    extra_mounts: list[DockerExtraMount],
) -> None:
    seen_extra_paths: set[str] = set()
    for extra_mount in extra_mounts:
        container_path = extra_mount.container_path
        normalized_container_path = str(container_path)
        if normalized_container_path in seen_extra_paths:
            raise ValueError(f"Docker extra mount virtual path '{normalized_container_path}' is duplicated")
        seen_extra_paths.add(normalized_container_path)
        for logical_mount in logical_mounts:
            if virtual_path_contains(logical_mount.virtual_path, container_path) or virtual_path_contains(
                container_path,
                logical_mount.virtual_path,
            ):
                raise ValueError(
                    f"Docker extra mount virtual path '{container_path}' conflicts with logical workspace mount "
                    f"'{logical_mount.virtual_path}'"
                )


def _extra_mount_fingerprint_payload(extra_mounts: list[DockerExtraMount] | None) -> list[dict[str, Any]]:
    return [
        {
            "host_path": str(mount.host_path.expanduser()),
            "container_path": str(mount.container_path),
            "mode": mount.mode,
        }
        for mount in extra_mounts or []
    ]


def _virtual_mounts_from_binding(binding: WorkspaceBinding) -> list[VirtualMount]:
    return [VirtualMount(host_path=mount.host_path, virtual_path=mount.virtual_path) for mount in binding.mounts]


def _resolve_binding_docker_host_path(binding: WorkspaceBinding) -> Path:
    if binding.docker_host_path is not None:
        return binding.docker_host_path
    return binding.host_path


def _resolve_binding_docker_host_paths(binding: WorkspaceBinding) -> list[Path]:
    return [mount.docker_host_path or mount.host_path for mount in binding.mounts]


def _docker_mount_modes_from_binding(binding: WorkspaceBinding) -> list[str]:
    return [mount.mode for mount in binding.mounts]


def _read_only_paths_from_binding(binding: WorkspaceBinding) -> list[Path]:
    return [mount.virtual_path for mount in binding.mounts if mount.mode == "ro"]


def _host_cwd_from_binding(binding: WorkspaceBinding) -> Path:
    for mount in binding.mounts:
        if virtual_path_contains(mount.virtual_path, binding.cwd):
            return mount.host_path / relative_virtual_path(mount.virtual_path, binding.cwd)
    raise ValueError(f"workspace cwd '{binding.cwd}' is outside declared mounts")


def _virtual_path_for_host_cwd(mounts: list[VirtualMount], host_cwd: Path) -> Path:
    resolved_host_cwd = host_cwd.expanduser().resolve()
    best_mount: VirtualMount | None = None
    best_depth = -1
    for mount in mounts:
        mount_host = mount.host_path.expanduser().resolve()
        try:
            relative_path = resolved_host_cwd.relative_to(mount_host)
        except ValueError:
            continue
        depth = len(mount_host.parts)
        if depth > best_depth:
            best_mount = VirtualMount(mount.host_path, mount.virtual_path / relative_path)
            best_depth = depth
    if best_mount is None:
        raise ValueError(f"workspace cwd host path '{host_cwd}' is outside declared mounts")
    return best_mount.virtual_path


def _resolve_docker_mount_host_path(mounts: list[VirtualMount], docker_host_paths: list[Path], index: int) -> Path:
    if index < len(docker_host_paths):
        return docker_host_paths[index]
    return mounts[index].host_path


def _resolve_docker_mount_mode(docker_mount_modes: list[str], index: int) -> str:
    if index < len(docker_mount_modes):
        return docker_mount_modes[index]
    return "rw"


def build_workspace_container_ref(*, image: str, workspace_dir: Path) -> str:
    fingerprint_source = f"{workspace_dir.expanduser().resolve()}|{image}"
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    return f"{_DOCKER_WORKSPACE_NAME_PREFIX}-{fingerprint}"


def extract_workspace_sandbox_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    raw_sandbox = metadata.get(_DOCKER_SANDBOX_METADATA_KEY)
    if not isinstance(raw_sandbox, dict):
        return None
    return dict(raw_sandbox)


def build_workspace_sandbox_metadata(*, binding: WorkspaceBinding, environment: Environment) -> dict[str, Any] | None:
    if not isinstance(environment, SandboxEnvironment):
        return None
    backend = (binding.backend_hint or binding.metadata.get("provider") or "").strip().lower()
    if backend != "docker":
        return None

    existing = extract_workspace_sandbox_metadata(binding.metadata) or {}
    container_ref = _normalize_optional_str(existing.get("container_ref"))
    if container_ref is None and isinstance(environment, ReusableSandboxEnvironment):
        container_ref = environment.container_ref

    cache_path = _normalize_optional_str(existing.get("cache_path"))
    if cache_path is None and isinstance(environment, ReusableSandboxEnvironment) and environment.container_cache_path:
        cache_path = str(environment.container_cache_path)
    environment_sandbox = environment.sandbox_metadata if isinstance(environment, ReusableSandboxEnvironment) else {}
    scope = _normalize_sandbox_scope(existing.get("scope")) or binding.sandbox_scope
    now = _normalize_optional_str(existing.get("last_used_at"))
    image_digest = _normalize_optional_str(existing.get("image_digest")) or _normalize_optional_str(
        environment_sandbox.get("image_digest")
    )

    return {
        **existing,
        "provider": _DOCKER_SANDBOX_PROVIDER,
        "scope": scope,
        "generation": _first_optional_int(existing.get("generation"), binding.generation),
        "workspace_fingerprint": _normalize_optional_str(existing.get("workspace_fingerprint")) or binding.fingerprint,
        "container_ref": container_ref or environment.container_id,
        "container_id": environment.container_id,
        "image": _normalize_optional_str(existing.get("image"))
        or _normalize_optional_str(binding.metadata.get("docker_image")),
        "image_digest": image_digest,
        "status": "running" if environment.container_id is not None else existing.get("status", "created"),
        "workspace_uid": _first_optional_int(existing.get("workspace_uid"), binding.metadata.get("workspace_uid")),
        "workspace_gid": _first_optional_int(existing.get("workspace_gid"), binding.metadata.get("workspace_gid")),
        "cache_path": cache_path,
        "last_used_at": now,
        "host_mount": str(_resolve_binding_docker_host_path(binding)),
        "service_mount": str(binding.host_path),
        "container_mount": str(binding.virtual_path),
        "cwd": str(binding.cwd),
        "mounts": [
            {
                "id": mount.id,
                "name": mount.name,
                "host_path": str(mount.host_path),
                "docker_host_path": str(mount.docker_host_path) if mount.docker_host_path is not None else None,
                "container_path": str(mount.virtual_path),
                "virtual_path": str(mount.virtual_path),
                "mode": mount.mode,
            }
            for mount in binding.mounts
        ],
    }


def remove_workspace_sandbox_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(metadata or {})
    normalized.pop(_DOCKER_SANDBOX_METADATA_KEY, None)
    return normalized


def _inspect_container_health_status(client: Any, container_id: str) -> str | None:
    try:
        container = client.containers.get(container_id)
        container.reload()
    except Exception as exc:
        if exc.__class__.__name__ == "NotFound":
            raise RuntimeError(f"Container not found: {container_id}") from exc
        raise RuntimeError(f"Failed to inspect container health: {exc}") from exc

    state = getattr(container, "attrs", {}).get("State")
    if not isinstance(state, dict):
        return None
    health = state.get("Health")
    if not isinstance(health, dict):
        return None
    return _normalize_optional_str(health.get("Status"))


def _resolve_image_digest(client: Any, image: str) -> str | None:
    try:
        image_obj = client.images.get(image)
    except Exception as exc:
        logger.warning("Failed to inspect Docker workspace image image={} error={}", image, exc)
        return None
    repo_digests = getattr(image_obj, "attrs", {}).get("RepoDigests")
    if isinstance(repo_digests, list):
        for digest in repo_digests:
            normalized = _normalize_optional_str(digest)
            if normalized is not None:
                return normalized
    image_id = _normalize_optional_str(getattr(image_obj, "id", None))
    if image_id is not None:
        return image_id
    return _normalize_optional_str(getattr(image_obj, "short_id", None))


def _container_image_digest(container: Any) -> str | None:
    image = getattr(container, "image", None)
    if image is not None:
        repo_digests = getattr(image, "attrs", {}).get("RepoDigests")
        if isinstance(repo_digests, list):
            for digest in repo_digests:
                normalized = _normalize_optional_str(digest)
                if normalized is not None:
                    return normalized
        image_id = _normalize_optional_str(getattr(image, "id", None))
        if image_id is not None:
            return image_id
    attrs_image = getattr(container, "attrs", {}).get("Image")
    return _normalize_optional_str(attrs_image)


def _build_container_cache_path(cache_dir: Path | None, *, metadata: dict[str, Any] | None = None) -> Path | None:
    if cache_dir is None:
        return None
    sandbox_metadata = dict(metadata or {})
    explicit_path = _normalize_optional_str(sandbox_metadata.get("cache_path"))
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    scope = _normalize_optional_str(sandbox_metadata.get("scope")) or SANDBOX_SCOPE_SESSION
    if scope == SANDBOX_SCOPE_RUN:
        run_id = _normalize_optional_str(sandbox_metadata.get("run_id"))
        if run_id is not None:
            return cache_dir / "runs" / run_id / _DEFAULT_CONTAINER_CACHE_FILE
    session_id = _normalize_optional_str(sandbox_metadata.get("session_id"))
    if session_id is not None:
        return cache_dir / "sessions" / session_id / _DEFAULT_CONTAINER_CACHE_FILE
    return None


def _normalize_sandbox_scope(value: Any) -> SandboxScopeLiteral | None:
    normalized = _normalize_optional_str(value)
    if normalized in {SANDBOX_SCOPE_SESSION, SANDBOX_SCOPE_RUN}:
        return cast(SandboxScopeLiteral, normalized)
    return None


def _normalize_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_docker_container_lock(*, cache_path: Path | None, container_ref: str) -> asyncio.Lock:
    lock_key = str(cache_path or container_ref)
    return _DOCKER_CONTAINER_LOCKS.setdefault(lock_key, asyncio.Lock())


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolve_docker_exec_user(
    value: str | None,
    *,
    workspace_uid: int | None,
    workspace_gid: int | None,
) -> str | None:
    normalized_value = _normalize_optional_str(value) or _AUTO_DOCKER_EXEC_USER
    if normalized_value.lower() == _AUTO_DOCKER_EXEC_USER:
        if isinstance(workspace_uid, int) and isinstance(workspace_gid, int):
            return f"{workspace_uid}:{workspace_gid}"
        return None
    return normalized_value


def build_docker_workspace_exec_default_env(*, home: str | None = None, user: str | None = None) -> dict[str, str]:
    normalized_home = _normalize_optional_str(home) or _DEFAULT_DOCKER_WORKSPACE_HOME
    normalized_user = _normalize_optional_str(user) or _DEFAULT_DOCKER_WORKSPACE_USER
    return {"HOME": normalized_home, "USER": normalized_user}


def _first_optional_int(*values: Any) -> int | None:
    for value in values:
        normalized_value = _normalize_optional_int(value)
        if normalized_value is not None:
            return normalized_value
    return None


def _normalize_optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
