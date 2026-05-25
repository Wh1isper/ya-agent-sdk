from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ya_agent_environment import ExecutionHandle, ShellExecutionError, ShellTimeoutError, StdinAdapter

from ya_agent_sdk.environment.local import LocalShell
from ya_agent_sdk.environment.process import (
    kill_process_tree,
    process_group_kwargs,
    send_process_tree_signal,
    terminate_process_tree,
)
from ya_agent_sdk.environment.shell_sandbox.backend import build_sandbox_command
from ya_agent_sdk.environment.shell_sandbox.policy import ShellSandboxRuntimePolicy


class SandboxedLocalShell(LocalShell):
    def __init__(
        self,
        *,
        policy: ShellSandboxRuntimePolicy,
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
        self._policy = policy
        self._environment_overrides = dict(environment_overrides)

    def _build_effective_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        requested = {**self._environment_overrides, **dict(env or {})}
        allowlist = set(self._policy.env_allowlist)
        if "*" in allowlist:
            if self._include_os_env:
                return {**os.environ, **requested}
            return requested
        filtered = {key: value for key, value in requested.items() if key in allowlist}
        if self._include_os_env:
            for key in allowlist:
                if key in os.environ and key not in filtered:
                    filtered[key] = os.environ[key]
        if "HOME" not in filtered and "HOME" in os.environ:
            filtered["HOME"] = os.environ["HOME"]
        if "PATH" not in filtered and "PATH" in os.environ:
            filtered["PATH"] = os.environ["PATH"]
        return filtered

    async def get_context_instructions(self) -> str | None:
        instructions = await super().get_context_instructions()
        if instructions is None:
            return None
        metadata = self._policy.to_metadata()
        sandbox_lines = [
            "  <shell-sandbox>",
            f"    <enabled>{str(self._policy.enabled).lower()}</enabled>",
            f"    <profile>{self._policy.profile}</profile>",
            f"    <backend>{self._policy.backend}</backend>",
            f"    <network>{self._policy.network}</network>",
            f"    <raw-host-allowed>{str(self._policy.raw_shell_allowed).lower()}</raw-host-allowed>",
            "  </shell-sandbox>",
        ]
        insertion = "\n".join(sandbox_lines)
        return instructions.replace("\n  <note>", f"\n{insertion}\n  <note>").replace(
            "Commands will be executed with the working directory validated.",
            f"Commands run through YA Claw shell sandbox policy {metadata['profile']} on backend {metadata['backend']}.",
        )

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        if not self._policy.enabled:
            return await super().execute(command, timeout=timeout, env=env, cwd=cwd)
        if self._policy.backend == "raw_host":
            if self._policy.raw_shell_allowed:
                return await super().execute(command, timeout=timeout, env=env, cwd=cwd)
            raise ShellExecutionError(command, stderr="Raw host shell backend is disabled by shell sandbox policy")

        resolved_cwd = self._resolve_cwd(cwd)
        effective_timeout = self._default_timeout if timeout is None else timeout
        effective_env = self._build_effective_env(env)
        args, cleanup = build_sandbox_command(
            command=command,
            cwd=resolved_cwd,
            policy=self._policy,
            shell_executable=self._shell_executable,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                env=effective_env,
                **process_group_kwargs(),
            )
            try:
                if effective_timeout is not None:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=effective_timeout
                    )
                else:
                    stdout_bytes, stderr_bytes = await process.communicate()
            except TimeoutError as exc:
                await terminate_process_tree(process)
                raise ShellTimeoutError(command, effective_timeout or 0) from exc
            return (
                process.returncode or 0,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except FileNotFoundError as exc:
            raise ShellExecutionError(command, stderr=f"Shell sandbox backend is unavailable: {args[0]}") from exc
        except PermissionError as exc:
            raise ShellExecutionError(command, stderr="Shell sandbox backend permission denied") from exc
        except OSError as exc:
            raise ShellExecutionError(command, stderr=str(exc)) from exc
        finally:
            cleanup()

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        if not self._policy.enabled:
            return await super()._create_process(command, env=env, cwd=cwd)
        if self._policy.backend == "raw_host":
            if self._policy.raw_shell_allowed:
                return await super()._create_process(command, env=env, cwd=cwd)
            raise ShellExecutionError(command, stderr="Raw host shell backend is disabled by shell sandbox policy")
        resolved_cwd = self._resolve_cwd(cwd)
        effective_env = self._build_effective_env(env)
        args, cleanup = build_sandbox_command(
            command=command,
            cwd=resolved_cwd,
            policy=self._policy,
            shell_executable=self._shell_executable,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                env=effective_env,
                **process_group_kwargs(),
            )
        except Exception as exc:
            cleanup()
            raise ShellExecutionError(command, stderr=str(exc)) from exc
        if process.stdout is None or process.stderr is None:
            cleanup()
            raise ShellExecutionError(command, stderr="Failed to capture sandboxed subprocess streams")

        async def _wait() -> int:
            try:
                await process.wait()
                return process.returncode or 0
            finally:
                cleanup()

        async def _kill() -> None:
            try:
                await kill_process_tree(process)
            finally:
                cleanup()

        async def _send_signal(sig: int) -> None:
            send_process_tree_signal(process, sig)

        return ExecutionHandle(
            stdout=process.stdout,
            stderr=process.stderr,
            wait=_wait,
            kill=_kill,
            stdin=StdinAdapter(process.stdin) if process.stdin is not None else None,
            pid=process.pid,
            send_signal=_send_signal,
        )
