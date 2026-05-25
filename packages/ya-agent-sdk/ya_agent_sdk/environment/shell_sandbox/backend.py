from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ya_agent_environment import ShellExecutionError

from ya_agent_sdk.environment.shell_sandbox.policy import ShellSandboxMountPolicy, ShellSandboxRuntimePolicy

PROTECTED_HOME_SUFFIXES = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".config/gcloud",
    ".docker",
    ".kube",
)
LINUX_BWRAP_TMP_PATH = "/tmp"  # noqa: S108 - bubblewrap tmpfs mount point inside the sandbox namespace.

SandboxCommand = tuple[list[str], Callable[[], None]]


def build_sandbox_command(
    *,
    command: str,
    cwd: Path,
    policy: ShellSandboxRuntimePolicy,
    shell_executable: str | None,
) -> SandboxCommand:
    if policy.backend == "linux_bwrap_seccomp":
        return build_linux_bwrap_command(
            command=command,
            cwd=cwd,
            policy=policy,
            shell_executable=shell_executable,
        ), noop
    if policy.backend == "macos_seatbelt":
        return build_macos_seatbelt_command(
            command=command,
            policy=policy,
            shell_executable=shell_executable,
        )
    if policy.backend == "windows_restricted_token":
        if policy.raw_shell_allowed:
            return default_shell_command(command), noop
        raise ShellExecutionError(command, stderr="Windows restricted-token sandbox is not implemented in this build")
    raise ShellExecutionError(command, stderr=f"Unsupported shell sandbox backend: {policy.backend}")


def build_linux_bwrap_command(
    *,
    command: str,
    cwd: Path,
    policy: ShellSandboxRuntimePolicy,
    shell_executable: str | None,
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        if policy.raw_shell_allowed:
            shell = shell_executable or default_platform_shell()
            return [shell, "-lc", command]
        raise ShellExecutionError(command, stderr="bubblewrap is required for linux_bwrap_seccomp shell sandbox")
    args = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--die-with-parent",
    ]
    if policy.network in {"blocked", "restricted"}:
        args.append("--unshare-net")
    args.extend(["--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", LINUX_BWRAP_TMP_PATH])
    for protected_path in protected_home_paths(policy.mounts):
        args.extend(["--tmpfs", str(protected_path)])
    for mount in policy.mounts:
        host_path = mount.host_path.resolve()
        if not host_path.exists():
            continue
        bind_arg = "--bind" if mount.mode == "rw" else "--ro-bind"
        args.extend([bind_arg, str(host_path), str(host_path)])
    args.extend(["--chdir", str(cwd)])
    shell = shell_executable or default_platform_shell()
    args.extend([shell, "-lc", command])
    return args


def build_macos_seatbelt_command(
    *,
    command: str,
    policy: ShellSandboxRuntimePolicy,
    shell_executable: str | None,
) -> SandboxCommand:
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if not sandbox_exec.exists():
        if policy.raw_shell_allowed:
            shell = shell_executable or default_platform_shell()
            return [shell, "-lc", command], noop
        raise ShellExecutionError(command, stderr="/usr/bin/sandbox-exec is required for macos_seatbelt shell sandbox")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="ya-claw-seatbelt-",
        suffix=".sbpl",
        delete=False,
    ) as profile_file:
        profile_path = Path(profile_file.name)
        profile_file.write(seatbelt_profile(policy))

    def cleanup() -> None:
        with contextlib.suppress(FileNotFoundError):
            profile_path.unlink()

    shell = shell_executable or default_platform_shell()
    return [str(sandbox_exec), "-f", str(profile_path), shell, "-lc", command], cleanup


def shell_sandbox_diagnostics(policy: ShellSandboxRuntimePolicy) -> dict[str, Any]:
    backend = policy.backend
    diagnostics: dict[str, Any] = {
        "enabled": policy.enabled,
        "backend": backend,
        "network": policy.network,
        "raw_host_allowed": policy.raw_shell_allowed,
    }
    if backend == "linux_bwrap_seccomp":
        diagnostics["bubblewrap"] = "ok" if shutil.which("bwrap") else "missing"
        diagnostics["seccomp"] = "delegated_to_bubblewrap_helper"
        diagnostics["landlock"] = "not_probed"
    elif backend == "macos_seatbelt":
        diagnostics["sandbox_exec"] = "ok" if Path("/usr/bin/sandbox-exec").exists() else "missing"
    elif backend == "windows_restricted_token":
        diagnostics["windows_backend"] = "planned"
        diagnostics["windows_job_object"] = "planned"
        diagnostics["windows_appcontainer"] = "planned"
    elif backend == "raw_host":
        diagnostics["raw_host"] = "approved_path"
    return diagnostics


def default_platform_shell() -> str:
    if os.name == "posix":
        return "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"
    return "cmd.exe"


def default_shell_command(command: str) -> list[str]:
    shell = default_platform_shell()
    if os.name == "posix":
        return [shell, "-lc", command]
    return [shell, "/d", "/s", "/c", command]


def noop() -> None:
    return None


def protected_home_paths(mounts: list[ShellSandboxMountPolicy]) -> list[Path]:
    home = Path.home()
    protected: list[Path] = []
    for suffix in PROTECTED_HOME_SUFFIXES:
        candidate = home / suffix
        if not candidate.exists():
            continue
        if any(path_contains(candidate, mount.host_path) for mount in mounts):
            continue
        protected.append(candidate)
    return protected


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def seatbelt_profile(policy: ShellSandboxRuntimePolicy) -> str:
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        '(allow file-read* (subpath "/System"))',
        '(allow file-read* (subpath "/usr"))',
        '(allow file-read* (subpath "/bin"))',
        '(allow file-read* (subpath "/sbin"))',
        '(allow file-read* (subpath "/Library/Developer"))',
        '(allow file-read* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/zero"))',
        '(allow file-read* (literal "/dev/random"))',
        '(allow file-read* (literal "/dev/urandom"))',
        '(allow file-write* (literal "/dev/null"))',
        '(allow file-read* (subpath "/private/tmp"))',
        '(allow file-write* (subpath "/private/tmp"))',
    ]
    if policy.network in {"proxy", "full"}:
        lines.append("(allow network*)")
    for mount in policy.mounts:
        escaped = sbpl_escape(str(mount.host_path.resolve()))
        lines.append(f'(allow file-read* (subpath "{escaped}"))')
        if mount.mode == "rw":
            lines.append(f'(allow file-write* (subpath "{escaped}"))')
    return "\n".join(lines) + "\n"


def sbpl_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def shell_quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)
