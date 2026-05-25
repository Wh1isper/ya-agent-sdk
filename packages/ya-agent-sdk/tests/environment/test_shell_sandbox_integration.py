from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from ya_agent_sdk.environment import LocalShell, ShellSandboxMountPolicy, ShellSandboxRuntimePolicy

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap is required for Linux sandbox integration"
)


def _bubblewrap_available() -> bool:
    if shutil.which("bwrap") is None:
        return False
    result = subprocess.run(
        [
            "bwrap",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--new-session",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",  # noqa: S108 - bubblewrap tmpfs mount point inside the sandbox namespace.
            "/bin/sh",
            "-lc",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


pytestmark = [
    pytestmark,
    pytest.mark.skipif(not _bubblewrap_available(), reason="bubblewrap user namespace sandbox is unavailable"),
]


async def test_linux_bwrap_sandbox_writes_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = LocalShell(
        sandbox_policy=ShellSandboxRuntimePolicy(
            enabled=True,
            profile="workspace_write",
            backend="linux_bwrap_seccomp",
            requested_backend="linux_bwrap_seccomp",
            network="full",
            mounts=[ShellSandboxMountPolicy(id="workspace", host_path=workspace, mode="rw")],
        ),
        default_cwd=workspace,
        allowed_paths=[workspace],
        include_os_env=False,
    )

    exit_code, stdout, stderr = await shell.execute("printf ok > result.txt && cat result.txt")

    assert exit_code == 0
    assert stdout == "ok"
    assert stderr == ""
    assert (workspace / "result.txt").read_text() == "ok"


async def test_linux_bwrap_sandbox_masks_explicit_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    secret_dir = tmp_path / "secret"
    workspace.mkdir()
    secret_dir.mkdir()
    (secret_dir / "token").write_text("secret-token")
    shell = LocalShell(
        sandbox_policy=ShellSandboxRuntimePolicy(
            enabled=True,
            profile="workspace_write",
            backend="linux_bwrap_seccomp",
            requested_backend="linux_bwrap_seccomp",
            network="full",
            mounts=[ShellSandboxMountPolicy(id="workspace", host_path=workspace, mode="rw")],
            masked_paths=(secret_dir,),
        ),
        default_cwd=workspace,
        allowed_paths=[workspace],
        include_os_env=False,
    )

    exit_code, stdout, stderr = await shell.execute(f"test ! -e {secret_dir / 'token'} && printf masked")

    assert exit_code == 0
    assert stdout == "masked"
    assert stderr == ""
    assert (secret_dir / "token").read_text() == "secret-token"
