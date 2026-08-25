"""Tests for ya_agent_sdk.toolsets.core.filesystem.glob module."""

import json
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import ya_agent_sdk.toolsets.core.filesystem.glob as glob_module
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem.glob import GlobTool


async def test_glob_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description, and bounded defaults."""
    assert GlobTool.name == "glob"
    assert glob_module.DEFAULT_MAX_RESULTS == 200
    assert glob_module.OUTPUT_TRUNCATE_LIMIT == 12_000
    assert "glob pattern" in GlobTool.description
    tool = GlobTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = await tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_glob_find_files(tmp_path: Path) -> None:
    """Should find files matching pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create test files
        (tmp_path / "file1.py").write_text("content")
        (tmp_path / "file2.py").write_text("content")
        (tmp_path / "file3.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py")
        assert len(result) == 2
        assert any("file1.py" in r for r in result)
        assert any("file2.py" in r for r in result)


async def test_glob_recursive_pattern(tmp_path: Path) -> None:
    """Should find files recursively with ** pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create nested structure
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.py").write_text("content")
        (tmp_path / "subdir" / "nested.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="**/*.py")
        assert len(result) >= 2


async def test_glob_no_matches(tmp_path: Path) -> None:
    """Should return empty list when no matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.nonexistent")
        assert result == []


async def test_glob_searches_managed_tmp_outside_default_root(tmp_path: Path) -> None:
    """An explicit temporary root should return absolute, reusable paths."""
    workspace = tmp_path / "workspace"
    hidden_tmp_parent = tmp_path / ".cache"
    workspace.mkdir()
    hidden_tmp_parent.mkdir()
    (workspace / ".gitignore").write_text("*.txt\n")

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(
                allowed_paths=[workspace],
                default_path=workspace,
                tmp_base_dir=hidden_tmp_parent,
            )
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tmp_dir = env.tmp_dir
        assert tmp_dir is not None
        nested = tmp_dir / "nested"
        artifact = nested / "artifact.txt"
        await env.file_operator.mkdir(str(nested))
        await env.file_operator.write_file(str(artifact), "temporary content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        tool = GlobTool()
        result = await tool.call(mock_run_ctx, pattern="nested/*.txt", root=str(tmp_dir))
        anchored_result = await tool.call(mock_run_ctx, pattern="/nested/*.txt", root=str(tmp_dir))

        assert result == [artifact.as_posix()]
        assert anchored_result == result
        assert await env.file_operator.read_file(result[0]) == "temporary content"


async def test_glob_searches_workspace_managed_tmp_from_explicit_root(tmp_path: Path) -> None:
    """A workspace tmp root matches patterns relative to the selected instance."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[workspace], default_path=workspace))
        ctx = await stack.enter_async_context(AgentContext(env=env))
        assert env.tmp_dir is not None
        artifact = env.tmp_dir / "nested" / "artifact.txt"
        await env.file_operator.mkdir(str(artifact.parent))
        await env.file_operator.write_file(str(artifact), "temporary content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        result = await GlobTool().call(mock_run_ctx, pattern="nested/*.txt", root=str(env.tmp_dir))

        assert result == [artifact.relative_to(workspace).as_posix()]
        assert await env.file_operator.read_file(result[0]) == "temporary content"


async def test_glob_matches_external_allowed_root_reached_through_alias(tmp_path: Path) -> None:
    """Path matching should retain allowed-relative coordinates through aliases."""
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    nested = external / "nested"
    alias_parent = tmp_path / "aliases"
    alias = alias_parent / "external-alias"
    workspace.mkdir()
    nested.mkdir(parents=True)
    alias_parent.mkdir()
    artifact = nested / "artifact.txt"
    artifact.write_text("aliased content")
    try:
        alias.symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(
                allowed_paths=[workspace, alias_parent, external],
                default_path=workspace,
                enable_tmp_dir=False,
            )
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await GlobTool().call(
            mock_run_ctx,
            pattern="nested/*.txt",
            root=str(alias / "nested"),
            include_ignored=True,
        )

        aliased_artifact = alias / "nested" / "artifact.txt"
        assert result == [aliased_artifact.as_posix()]
        assert await env.file_operator.read_file(result[0]) == "aliased content"


async def test_glob_specific_extension(tmp_path: Path) -> None:
    """Should match specific file extensions."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "test.json").write_text("{}")
        (tmp_path / "test.yaml").write_text("key: value")
        (tmp_path / "test.txt").write_text("text")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.json")
        assert len(result) == 1
        assert "test.json" in result[0]


async def test_glob_empty_directory(tmp_path: Path) -> None:
    """Should return empty list for empty directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # tmp_path is empty, no files created
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py")
        assert result == []


async def test_glob_matches_directories(tmp_path: Path) -> None:
    """Should include directories in glob results when pattern matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="my*")
        assert len(result) == 2
        assert any("mydir" in r for r in result)
        assert any("myfile.txt" in r for r in result)


async def test_glob_excludes_gitignored_files(tmp_path: Path) -> None:
    """Should exclude files matching .gitignore patterns by default."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create .gitignore
        (tmp_path / ".gitignore").write_text("node_modules/\n*.pyc\n")

        # Create files
        (tmp_path / "main.py").write_text("content")
        (tmp_path / "cache.pyc").write_text("content")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="**/*")

        # Result should be a dict with gitignore_excluded info
        assert isinstance(result, dict)
        assert "files" in result
        assert "gitignore_excluded" in result
        assert "note" in result

        files = result["files"]
        # Should include main.py but exclude .pyc and node_modules contents
        assert any("main.py" in f for f in files)
        assert not any(".pyc" in f for f in files)
        # node_modules directory entry may appear, but its contents (pkg.js) should be excluded
        assert not any("pkg.js" in f for f in files)

        # Summary should mention excluded paths
        summary = result["gitignore_excluded"]
        assert any("node_modules/" in s for s in summary)


async def test_glob_include_ignored_flag(tmp_path: Path) -> None:
    """Should include gitignored files when include_ignored=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "app.py").write_text("content")
        (tmp_path / "debug.log").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*", include_ignored=True)

        # Result should be a list (no gitignore filtering)
        assert isinstance(result, list)
        assert any("app.py" in f for f in result)
        assert any("debug.log" in f for f in result)


async def test_glob_hidden_files_require_include_hidden(tmp_path: Path) -> None:
    """Should include hidden dot paths only when include_hidden=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / ".env").write_text("SECRET=value")
        (tmp_path / ".config").mkdir()
        (tmp_path / ".config" / "settings.toml").write_text("key = 'value'")
        (tmp_path / "visible.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        default_result = await tool.call(mock_run_ctx, pattern="**/*")
        assert isinstance(default_result, list)
        assert "visible.txt" in default_result
        assert ".env" not in default_result
        assert ".config/settings.toml" not in default_result

        hidden_result = await tool.call(mock_run_ctx, pattern="**/*", include_hidden=True)
        assert isinstance(hidden_result, list)
        assert ".env" in hidden_result
        assert ".config/settings.toml" in hidden_result


async def test_glob_includes_agents_by_default_and_adds_skill_reminder(tmp_path: Path) -> None:
    """Should expose workspace Skills without enabling all hidden paths."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / ".gitignore").write_text("build/\n")
        skill_file = tmp_path / ".agents" / "skills" / "example" / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("# Example")
        hidden_file = tmp_path / ".hidden" / "secret.txt"
        hidden_file.parent.mkdir()
        hidden_file.write_text("secret")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="**/*")

        assert isinstance(result, dict)
        assert ".agents/skills/example/SKILL.md" in result["files"]
        assert ".hidden/secret.txt" not in result["files"]
        assert "system-reminder" in result
        assert "read each relevant SKILL.md in full" in result["system-reminder"]

        anchored_result = await tool.call(mock_run_ctx, pattern="/.agents/skills/*/SKILL.md")
        assert isinstance(anchored_result, dict)
        assert anchored_result["files"] == [".agents/skills/example/SKILL.md"]


async def test_glob_agents_exemption_is_scoped_to_search_root(tmp_path: Path) -> None:
    """Should expose only the selected root's direct .agents child across walk paths."""
    project = tmp_path / "project"
    skill_file = project / ".agents" / "skills" / "example" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("# Example")

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        assert await tool.call(mock_run_ctx, pattern="SKILL.md") == []

        project_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root="project")
        assert isinstance(project_result, dict)
        assert project_result["files"] == ["project/.agents/skills/example/SKILL.md"]

        absolute_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root=str(project))
        assert isinstance(absolute_result, dict)
        assert absolute_result["files"] == ["project/.agents/skills/example/SKILL.md"]

        alias_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root="project/.agents/..")
        assert isinstance(alias_result, dict)
        assert alias_result["files"] == ["project/.agents/skills/example/SKILL.md"]

        (tmp_path / ".gitignore").write_text("build/\n")
        assert await tool.call(mock_run_ctx, pattern="SKILL.md") == []
        fast_path_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root="project")
        assert isinstance(fast_path_result, dict)
        assert fast_path_result["files"] == project_result["files"]
        fast_absolute_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root=str(project))
        assert isinstance(fast_absolute_result, dict)
        assert fast_absolute_result["files"] == project_result["files"]
        fast_alias_result = await tool.call(mock_run_ctx, pattern="SKILL.md", root="project/.agents/..")
        assert isinstance(fast_alias_result, dict)
        assert fast_alias_result["files"] == project_result["files"]


async def test_glob_does_not_follow_agents_directory_symlink(tmp_path: Path) -> None:
    """Should keep the .agents exemption inside the selected search root."""
    project = tmp_path / "project"
    store = tmp_path / "store"
    project.mkdir()
    store.mkdir()
    (store / "SKILL.md").write_text("# External")
    (project / ".agents").symlink_to(store, target_is_directory=True)

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        tool = GlobTool()

        assert await tool.call(mock_run_ctx, pattern="SKILL.md", root="project") == []
        (tmp_path / ".gitignore").write_text("build/\n")
        assert await tool.call(mock_run_ctx, pattern="SKILL.md", root="project") == []


async def test_glob_skill_directory_does_not_trigger_reminder(tmp_path: Path) -> None:
    """Should only treat SKILL.md file entries as Skill documents."""
    (tmp_path / "SKILL.md").mkdir()

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await GlobTool().call(mock_run_ctx, pattern="SKILL.md")

    assert result == ["SKILL.md"]


async def test_glob_agents_exemption_does_not_bypass_gitignore(tmp_path: Path) -> None:
    """Should still require include_ignored when .gitignore excludes .agents."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / ".gitignore").write_text(".agents/\n")
        skill_file = tmp_path / ".agents" / "skills" / "example" / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("# Example")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        default_result = await tool.call(mock_run_ctx, pattern="SKILL.md")
        assert default_result == []

        ignored_result = await tool.call(mock_run_ctx, pattern="SKILL.md", include_ignored=True)
        assert isinstance(ignored_result, dict)
        assert ignored_result["files"] == [".agents/skills/example/SKILL.md"]
        assert "system-reminder" in ignored_result


async def test_glob_root_limits_traversal(tmp_path: Path) -> None:
    """Should traverse from the requested logical root."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("content")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", root="src")
        assert isinstance(result, list)
        assert result == ["src/main.py"]


async def test_glob_max_results_truncates(tmp_path: Path) -> None:
    """Should truncate results when exceeding max_results."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create 10 files
        for i in range(10):
            (tmp_path / f"file{i:02d}.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", max_results=3)

        assert isinstance(result, dict)
        assert len(result["files"]) == 3
        assert result["truncated"] is True
        assert result["total_matches"] == 10
        assert result["showing"] == 3
        assert "truncated" in result["note"].lower()


async def test_glob_max_results_no_truncation(tmp_path: Path) -> None:
    """Should return plain list when results are within max_results."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "file1.py").write_text("content")
        (tmp_path / "file2.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", max_results=10)

        assert isinstance(result, list)
        assert len(result) == 2


async def test_glob_max_results_unlimited(tmp_path: Path) -> None:
    """Should return all results when max_results is -1."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        for i in range(10):
            (tmp_path / f"file{i:02d}.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", max_results=-1)

        assert isinstance(result, list)
        assert len(result) == 10


async def test_glob_max_results_with_include_ignored(tmp_path: Path) -> None:
    """Should truncate results with include_ignored=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / ".gitignore").write_text("*.log\n")
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text("content")
            (tmp_path / f"file{i}.log").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*", include_ignored=True, max_results=3)

        assert isinstance(result, dict)
        assert len(result["files"]) == 3
        assert result["truncated"] is True


async def test_glob_max_results_with_gitignore(tmp_path: Path) -> None:
    """Should combine truncation note with gitignore note."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / ".gitignore").write_text("*.log\n")
        for i in range(10):
            (tmp_path / f"file{i:02d}.py").write_text("content")
        (tmp_path / "debug.log").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="**/*", max_results=3)

        assert isinstance(result, dict)
        assert len(result["files"]) == 3
        assert result["truncated"] is True
        assert "gitignore_excluded" in result
        # Note should contain both gitignore and truncation info
        assert "include_ignored" in result["note"]
        assert "truncated" in result["note"].lower()


async def test_glob_no_gitignore_returns_list(tmp_path: Path) -> None:
    """Should return list when no .gitignore exists."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "file.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py")

        # No .gitignore means no ignored files, so result is a list
        assert isinstance(result, list)
        assert any("file.py" in f for f in result)


async def test_glob_hard_output_limit_writes_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should write to temp file when serialized output exceeds hard size limit."""
    # Set a small hard limit to trigger temp file writing
    monkeypatch.setattr(glob_module, "OUTPUT_TRUNCATE_LIMIT", 1000)

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        # Create files with long names to easily exceed 100 chars
        for i in range(20):
            (tmp_path / f"very_long_filename_for_testing_truncation_{i:04d}.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", max_results=-1)

        assert isinstance(result, dict)
        assert result["truncated"] is True
        assert "output_file_path" in result
        assert result["total_matches"] == 20
        assert len(result["files"]) < 20  # Preview should be smaller than full result
        assert "too large" in result["note"].lower()

        # Hard invariant: serialized result must be within the (monkeypatched) limit
        assert len(json.dumps(result, ensure_ascii=False)) <= 1000

        # Verify the temp file exists and contains all results
        output_path = result["output_file_path"]
        content = Path(output_path).read_text()
        full_result = json.loads(content)
        assert len(full_result) == 20


async def test_glob_hard_output_limit_preserves_skill_reminder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should retain Skill guidance in both a bounded preview and saved output."""
    monkeypatch.setattr(glob_module, "OUTPUT_TRUNCATE_LIMIT", 1000)

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "SKILL.md").write_text("# Example")
        for i in range(20):
            (tmp_path / f"very_long_filename_for_skill_reminder_{i:04d}.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*", max_results=-1)

        assert isinstance(result, dict)
        assert "system-reminder" in result
        assert len(json.dumps(result, ensure_ascii=False)) <= 1000
        full_result = json.loads(Path(result["output_file_path"]).read_text())
        assert "system-reminder" in full_result
        assert "SKILL.md" in full_result["files"]


async def test_glob_hard_output_limit_not_triggered(tmp_path: Path) -> None:
    """Should not write temp file when output is within hard limit."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool()

        (tmp_path / "small.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py", max_results=-1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert any("small.py" in f for f in result)
