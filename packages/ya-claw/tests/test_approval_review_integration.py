from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.security.approval import PermissionDecision
from ya_claw.config import ClawSettings
from ya_claw.controller.run import _project_run_trace
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.profile import ClawApprovalReviewConfig, ProfileResolver, ResolvedProfile
from ya_claw.execution.runtime import ClawRuntimeBuilder
from ya_claw.orm.base import Base
from ya_claw.workspace import WorkspaceBinding
from ya_claw.workspace.models import WorkspaceMountBinding


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'approval.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


def _binding(path: Path) -> WorkspaceBinding:
    path.mkdir(parents=True, exist_ok=True)
    return WorkspaceBinding(
        host_path=path,
        virtual_path=Path("/workspace"),
        cwd=Path("/workspace"),
        mounts=[
            WorkspaceMountBinding(
                id="workspace",
                host_path=path,
                virtual_path=Path("/workspace"),
                mode="rw",
            )
        ],
        readable_paths=[Path("/workspace")],
        writable_paths=[Path("/workspace")],
        fingerprint="sha256:test",
        backend_hint="local",
    )


async def test_profile_resolver_parses_approval_review_from_yaml(tmp_path: Path, db_engine: AsyncEngine) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: default
    model: gateway@openai-responses:gpt-5.5
    security:
      approval_review:
        enabled: true
        model: gateway@openai-responses:gpt-5.5-mini
        model_settings: openai_responses_low
        timeout_seconds: 17
        max_denials: 2
        include_recent_messages: 4
        truncation:
          max_text_chars: 1000
        mcp_permissions:
          github:
            default_decision: deny
            categories: [external_integration, network, write]
            scopes: [external_service]
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    resolver = ProfileResolver(settings=settings, session_factory=create_session_factory(db_engine))

    await resolver.seed_profiles()
    profile = await resolver.resolve("default")

    assert profile.approval_review is not None
    assert profile.approval_review.enabled is True
    assert profile.approval_review.model == "gateway@openai-responses:gpt-5.5-mini"
    assert profile.approval_review.timeout_seconds == 17
    assert profile.approval_review.max_denials == 2
    assert profile.approval_review.truncation["max_text_chars"] == 1000
    assert profile.approval_review.mcp_permissions["github"]["default_decision"] == "deny"


def test_runtime_builder_mounts_approval_review(tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        _env_file=None,
    )
    profile = ResolvedProfile(
        name="default",
        model="test",
        model_settings=None,
        model_config=None,
        approval_review=ClawApprovalReviewConfig.model_validate({
            "enabled": True,
            "model": "test:model",
            "mcp_permissions": {
                "github": {
                    "default_decision": "deny",
                    "categories": ["external_integration", "network", "write"],
                    "scopes": ["external_service"],
                }
            },
        }),
    )
    runtime = ClawRuntimeBuilder(settings=settings).build(
        profile=profile,
        binding=_binding(tmp_path / "workspace"),
        environment=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path),
        restore_state=None,
        session_id="session-1",
        run_id="run-1",
        restore_from_run_id=None,
        dispatch_mode="async",
        source_kind="api",
        source_metadata={},
        claw_metadata={},
    )

    assert runtime.ctx.security.approval_review is not None
    assert runtime.ctx.security.approval_review.enabled is True
    assert runtime.ctx.security.approval_review.model == "test:model"
    github = runtime.ctx.security.approval_review.mcp_permissions["github"]
    assert github.default_decision == PermissionDecision.DENY


def test_run_trace_projects_approval_review_custom_event() -> None:
    payload = {
        "request_id": "apr_1",
        "tool_call_id": "call-1",
        "tool_name": "shell_exec",
        "source": "builtin",
        "decision": "auto_review",
        "outcome": "deny",
        "risk_level": "high",
        "authorization": "missing",
        "categories": ["execute"],
        "scopes": ["workspace", "local_system"],
        "rationale": "Command is outside the user goal.",
    }
    trace, truncated = _project_run_trace(
        [{"type": "CUSTOM", "name": "agent.approval_review_denied", "value": {"payload": payload}}],
        max_item_chars=4000,
        max_total_chars=4000,
    )

    assert truncated is False
    assert len(trace) == 1
    assert trace[0].type == "approval_review"
    content = json.loads(trace[0].content or "{}")
    assert content["request_id"] == "apr_1"
    assert content["tool_name"] == "shell_exec"
    assert content["outcome"] == "deny"
    assert content["categories"] == ["execute"]
