"""Tests for SkillToolset with extra_dir_names (.agents/skills support)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.skills.toolset import SHARED_SKILLS_DIR_NAME, SkillToolset

from ._instruction_helpers import instruction_text as _instruction_text


def _write_skill(skill_dir: Path, name: str, description: str) -> None:
    """Helper to create a SKILL.md file in a skill directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nInstructions here.\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_extra_dir_names_discovers_shared_skills(tmp_path: Path):
    """Skills in .agents/skills/ directories are discovered when extra_dir_names is set."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a shared skill in .agents/skills/
    shared_skill_dir = project_dir / ".agents" / "skills" / "shared-tool"
    _write_skill(shared_skill_dir, "shared-tool", "A shared skill")

    async with LocalEnvironment(
        allowed_paths=[project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(extra_dir_names=[SHARED_SKILLS_DIR_NAME])
            instructions = await toolset.get_instructions(mock_ctx)
            instruction_text = _instruction_text(instructions)

            assert instructions is not None
            assert "shared-tool" in instruction_text
            assert "A shared skill" in instruction_text


@pytest.mark.asyncio
async def test_no_extra_dir_names_ignores_shared_skills(tmp_path: Path):
    """Without extra_dir_names, .agents/skills/ is not scanned (backward compat)."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a shared skill in .agents/skills/
    shared_skill_dir = project_dir / ".agents" / "skills" / "shared-tool"
    _write_skill(shared_skill_dir, "shared-tool", "A shared skill")

    async with LocalEnvironment(
        allowed_paths=[project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset()  # No extra_dir_names
            instructions = await toolset.get_instructions(mock_ctx)
            assert instructions is None  # No skills found


@pytest.mark.asyncio
async def test_tool_specific_overrides_shared_within_same_path(tmp_path: Path):
    """Within the same allowed_path, skills/ overrides .agents/skills/ for same name."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create shared skill
    shared_skill_dir = project_dir / ".agents" / "skills" / "my-skill"
    _write_skill(shared_skill_dir, "my-skill", "Shared version")

    # Create tool-specific skill with same name (higher priority)
    specific_skill_dir = project_dir / "skills" / "my-skill"
    _write_skill(specific_skill_dir, "my-skill", "Tool-specific version")

    async with LocalEnvironment(
        allowed_paths=[project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(extra_dir_names=[SHARED_SKILLS_DIR_NAME])
            instructions = await toolset.get_instructions(mock_ctx)
            instruction_text = _instruction_text(instructions)

            assert instructions is not None
            assert "Tool-specific version" in instruction_text
            assert "Shared version" not in instruction_text


@pytest.mark.asyncio
async def test_project_overrides_user_skills(tmp_path: Path):
    """Project-level skills override user-level skills with the same name."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # User-level skill
    user_skill_dir = user_dir / "skills" / "my-skill"
    _write_skill(user_skill_dir, "my-skill", "User version")

    # Project-level skill with same name (higher priority)
    project_skill_dir = project_dir / "skills" / "my-skill"
    _write_skill(project_skill_dir, "my-skill", "Project version")

    # allowed_paths order: user first (lower priority), project last (higher priority)
    async with LocalEnvironment(
        allowed_paths=[user_dir, project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset()
            instructions = await toolset.get_instructions(mock_ctx)
            instruction_text = _instruction_text(instructions)

            assert instructions is not None
            assert "Project version" in instruction_text
            assert "User version" not in instruction_text


@pytest.mark.asyncio
async def test_full_priority_chain(tmp_path: Path):
    """Full priority chain: shared_user < yaacli_user < shared_project < yaacli_project."""
    shared_user_dir = tmp_path / "agents"
    yaacli_user_dir = tmp_path / "yaacli"
    project_dir = tmp_path / "project"
    for d in (shared_user_dir, yaacli_user_dir, project_dir):
        d.mkdir()

    # 1. Shared user skill (lowest priority) - skill "alpha" only here
    _write_skill(shared_user_dir / "skills" / "alpha", "alpha", "Shared user alpha")

    # 2. YAACLI user skill - overrides "alpha", adds "beta"
    _write_skill(yaacli_user_dir / "skills" / "alpha", "alpha", "YAACLI user alpha")
    _write_skill(yaacli_user_dir / "skills" / "beta", "beta", "YAACLI user beta")

    # 3. Shared project skill - overrides "beta"
    _write_skill(project_dir / ".agents" / "skills" / "beta", "beta", "Shared project beta")

    # 4. YAACLI project skill - overrides "beta" again with highest priority
    _write_skill(project_dir / "skills" / "beta", "beta", "YAACLI project beta")

    # Order: shared_user, yaacli_user, project (lowest to highest base priority)
    async with LocalEnvironment(
        allowed_paths=[shared_user_dir, yaacli_user_dir, project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(extra_dir_names=[SHARED_SKILLS_DIR_NAME])
            instructions = await toolset.get_instructions(mock_ctx)
            instruction_text = _instruction_text(instructions)

            assert instructions is not None

            # "alpha" should come from yaacli_user (overrides shared_user)
            assert "YAACLI user alpha" in instruction_text
            assert "Shared user alpha" not in instruction_text

            # "beta" should come from yaacli_project (highest priority)
            assert "YAACLI project beta" in instruction_text
            assert "Shared project beta" not in instruction_text
            assert "YAACLI user beta" not in instruction_text


@pytest.mark.asyncio
async def test_shared_and_specific_coexist(tmp_path: Path):
    """Shared and tool-specific skills with different names coexist."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Shared skill
    _write_skill(project_dir / ".agents" / "skills" / "shared-only", "shared-only", "Only in shared")

    # Tool-specific skill
    _write_skill(project_dir / "skills" / "specific-only", "specific-only", "Only in specific")

    async with LocalEnvironment(
        allowed_paths=[project_dir],
        default_path=project_dir,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(extra_dir_names=[SHARED_SKILLS_DIR_NAME])
            instructions = await toolset.get_instructions(mock_ctx)
            instruction_text = _instruction_text(instructions)

            assert instructions is not None
            assert "shared-only" in instruction_text
            assert "Only in shared" in instruction_text
            assert "specific-only" in instruction_text
            assert "Only in specific" in instruction_text


@pytest.mark.asyncio
async def test_shared_skills_dir_name_constant():
    """SHARED_SKILLS_DIR_NAME matches the expected convention."""
    assert SHARED_SKILLS_DIR_NAME == ".agents/skills"
