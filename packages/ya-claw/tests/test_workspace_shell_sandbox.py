from __future__ import annotations

from pathlib import Path

from ya_agent_sdk.environment import ShellSandboxConfig
from ya_claw.execution.profile import ResolvedProfile
from ya_claw.workspace.models import WorkspaceBinding, WorkspaceMountBinding
from ya_claw.workspace.shell_sandbox import WorkspaceShellSandboxDefaults, resolve_workspace_shell_sandbox_policy


def _binding(main: Path, docs: Path) -> WorkspaceBinding:
    return WorkspaceBinding(
        host_path=main,
        virtual_path=Path("/workspace/main"),
        cwd=main,
        readable_paths=[main, docs],
        writable_paths=[main],
        mounts=[
            WorkspaceMountBinding(id="main", host_path=main, virtual_path=Path("/workspace/main"), mode="rw"),
            WorkspaceMountBinding(id="docs", host_path=docs, virtual_path=Path("/workspace/docs"), mode="ro"),
        ],
        fingerprint="sha256:test",
        metadata={},
    )


def test_resolve_workspace_shell_sandbox_policy_combines_defaults_and_profile(tmp_path: Path) -> None:
    profile = ResolvedProfile(
        name="restricted-profile",
        model="test-model",
        model_settings=None,
        model_config=None,
        system_prompt="",
        shell_sandbox=ShellSandboxConfig(
            enabled=True,
            profile="read_only",
            backend_preference="raw_host",
            network="blocked",
            masked_path_aliases=["common_credentials"],
            masked_paths=[tmp_path / "extra-mask"],
            raw_shell_approval="allowed_for_profile",
        ),
    )

    policy = resolve_workspace_shell_sandbox_policy(
        binding=_binding(tmp_path / "main", tmp_path / "docs"),
        defaults=WorkspaceShellSandboxDefaults(
            enabled=True,
            backend="auto",
            network="restricted",
            allow_raw_host=False,
        ),
        profile=profile,
    )

    assert policy.enabled is True
    assert policy.profile == "read_only"
    assert policy.backend == "raw_host"
    assert policy.requested_backend == "raw_host"
    assert policy.network == "blocked"
    assert policy.raw_shell_allowed is True
    assert policy.masked_paths == (
        Path.home() / ".ssh",
        Path.home() / ".gnupg",
        Path.home() / ".aws",
        Path.home() / ".config/gcloud",
        Path.home() / ".docker",
        Path.home() / ".kube",
        tmp_path / "extra-mask",
    )
    assert [mount.id for mount in policy.mounts] == ["main", "docs"]
    assert [mount.mode for mount in policy.mounts] == ["rw", "ro"]
