from __future__ import annotations

import os
from pathlib import Path

import pytest
import ya_agent_sdk.environment.shell_sandbox.backend as backend
from ya_agent_environment import ShellExecutionError
from ya_agent_sdk.environment import (
    SandboxedLocalShell,
    ShellSandboxConfig,
    ShellSandboxMountPolicy,
    ShellSandboxRuntimePolicy,
    resolve_shell_sandbox_runtime_policy,
    shell_sandbox_diagnostics,
)
from ya_agent_sdk.environment.shell_sandbox.backend import build_linux_bwrap_command, seatbelt_profile


def test_resolve_shell_sandbox_runtime_policy_uses_profile_overrides(tmp_path: Path) -> None:
    config = ShellSandboxConfig(
        profile="read_only",
        enabled=True,
        backend_preference="raw_host",
        network="blocked",
        env_allowlist="PATH, HOME, PATH, CUSTOM",
        raw_shell_approval="allowed_for_profile",
    )
    policy = resolve_shell_sandbox_runtime_policy(
        enabled=True,
        backend="auto",
        network="restricted",
        allow_raw_host=False,
        mounts=[ShellSandboxMountPolicy(id="workspace", host_path=tmp_path, mode="ro")],
        profile_config=config,
    )

    assert policy.enabled is True
    assert policy.profile == "read_only"
    assert policy.backend == "raw_host"
    assert policy.requested_backend == "raw_host"
    assert policy.network == "blocked"
    assert policy.env_allowlist == ("PATH", "HOME", "CUSTOM")
    assert policy.raw_shell_allowed is True
    assert policy.read_only_paths == [tmp_path]
    assert policy.writable_paths == []


def test_linux_bwrap_command_binds_mounts_and_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount = tmp_path / "workspace"
    mount.mkdir()
    policy = ShellSandboxRuntimePolicy(
        enabled=True,
        profile="workspace_write",
        backend="linux_bwrap_seccomp",
        requested_backend="linux_bwrap_seccomp",
        network="blocked",
        mounts=[ShellSandboxMountPolicy(id="workspace", host_path=mount, mode="rw")],
    )

    monkeypatch.setattr(backend.shutil, "which", lambda name: "/usr/bin/bwrap")

    command = build_linux_bwrap_command(
        command="pwd",
        cwd=mount,
        policy=policy,
        shell_executable="/bin/sh",
    )

    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in command
    assert ["--bind", str(mount.resolve()), str(mount.resolve())] == command[
        command.index("--bind") : command.index("--bind") + 3
    ]
    assert command[-3:] == ["/bin/sh", "-lc", "pwd"]


def test_macos_seatbelt_profile_reflects_mount_modes(tmp_path: Path) -> None:
    readonly = tmp_path / "readonly"
    writable = tmp_path / "writable"
    readonly.mkdir()
    writable.mkdir()
    policy = ShellSandboxRuntimePolicy(
        enabled=True,
        profile="workspace_write",
        backend="macos_seatbelt",
        requested_backend="macos_seatbelt",
        network="full",
        mounts=[
            ShellSandboxMountPolicy(id="docs", host_path=readonly, mode="ro"),
            ShellSandboxMountPolicy(id="workspace", host_path=writable, mode="rw"),
        ],
    )

    profile = seatbelt_profile(policy)

    assert "(allow network*)" in profile
    assert f'(allow file-read* (subpath "{readonly.resolve()}"))' in profile
    assert f'(allow file-write* (subpath "{readonly.resolve()}"))' not in profile
    assert f'(allow file-read* (subpath "{writable.resolve()}"))' in profile
    assert f'(allow file-write* (subpath "{writable.resolve()}"))' in profile


async def test_sandboxed_local_shell_blocks_unapproved_raw_host(tmp_path: Path) -> None:
    shell = SandboxedLocalShell(
        policy=ShellSandboxRuntimePolicy(
            enabled=True,
            profile="workspace_write",
            backend="raw_host",
            requested_backend="raw_host",
            network="restricted",
            mounts=[ShellSandboxMountPolicy(id="workspace", host_path=tmp_path, mode="rw")],
            raw_shell_allowed=False,
        ),
        environment_overrides={},
        default_cwd=tmp_path,
        allowed_paths=[tmp_path],
        include_os_env=False,
    )

    with pytest.raises(ShellExecutionError, match="Raw host shell backend"):
        await shell.execute("echo denied")


async def test_sandboxed_local_shell_filters_environment_for_raw_host(tmp_path: Path) -> None:
    shell = SandboxedLocalShell(
        policy=ShellSandboxRuntimePolicy(
            enabled=True,
            profile="workspace_write",
            backend="raw_host",
            requested_backend="raw_host",
            network="restricted",
            mounts=[ShellSandboxMountPolicy(id="workspace", host_path=tmp_path, mode="rw")],
            env_allowlist=("PATH", "ALLOWED"),
            raw_shell_allowed=True,
        ),
        environment_overrides={"ALLOWED": "from-default", "BLOCKED": "from-default"},
        default_cwd=tmp_path,
        allowed_paths=[tmp_path],
        include_os_env=False,
    )

    exit_code, stdout, stderr = await shell.execute(
        'printf \'%s:%s\' "$ALLOWED" "$BLOCKED"',
        env={"ALLOWED": "from-call", "BLOCKED": "from-call", "PATH": os.environ.get("PATH", "")},
    )

    assert exit_code == 0
    assert stderr == ""
    assert stdout == "from-call:"


def test_shell_sandbox_diagnostics_reports_backend_state(tmp_path: Path) -> None:
    policy = ShellSandboxRuntimePolicy(
        enabled=True,
        profile="workspace_write",
        backend="raw_host",
        requested_backend="raw_host",
        network="restricted",
        mounts=[ShellSandboxMountPolicy(id="workspace", host_path=tmp_path, mode="rw")],
        raw_shell_allowed=True,
    )

    diagnostics = shell_sandbox_diagnostics(policy)

    assert diagnostics["enabled"] is True
    assert diagnostics["backend"] == "raw_host"
    assert diagnostics["network"] == "restricted"
    assert diagnostics["raw_host_allowed"] is True
    assert diagnostics["raw_host"] == "approved_path"
