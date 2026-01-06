"""Tests for pai_agent_sdk.context module."""

from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment, LocalFileOperator, LocalShell


@pytest.fixture
def file_operator(tmp_path: Path) -> LocalFileOperator:
    """Create a LocalFileOperator for testing."""
    return LocalFileOperator(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
    )


@pytest.fixture
def shell(tmp_path: Path) -> LocalShell:
    """Create a LocalShell for testing."""
    return LocalShell(
        allowed_paths=[tmp_path],
        default_cwd=tmp_path,
    )


def test_agent_context_default_run_id(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should generate a unique run_id by default."""
    ctx1 = AgentContext(file_operator=file_operator, shell=shell)
    ctx2 = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx1.run_id != ctx2.run_id
    assert len(ctx1.run_id) == 32  # uuid4().hex length


def test_agent_context_no_parent_by_default(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should have no parent by default."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.parent_run_id is None


def test_agent_context_elapsed_time_before_start(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should return None before context is started."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.elapsed_time is None


def test_agent_context_elapsed_time_after_start(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should return elapsed time after start."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    ctx.start_at = datetime.now()
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() >= 0


def test_agent_context_elapsed_time_after_end(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should return final duration after end."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    start = datetime.now()
    ctx.start_at = start
    ctx.end_at = start + timedelta(seconds=5)
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert elapsed.total_seconds() == 5


@pytest.mark.asyncio
async def test_agent_context_enter_subagent(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should create child context with proper inheritance."""
    parent = AgentContext(file_operator=file_operator, shell=shell)
    parent.start_at = datetime.now()

    async with parent.enter_subagent("search") as child:
        assert child.parent_run_id == parent.run_id
        assert child.run_id != parent.run_id
        assert child._agent_name == "search"
        assert child.start_at is not None
        assert child.end_at is None

    # After exiting, end_at should be set
    assert child.end_at is not None


@pytest.mark.asyncio
async def test_agent_context_enter_subagent_with_override(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should allow field overrides in subagent context."""
    parent = AgentContext(file_operator=file_operator, shell=shell)

    async with parent.enter_subagent("reasoning", deferred_tool_metadata={"key": {}}) as child:
        assert child.deferred_tool_metadata == {"key": {}}


@pytest.mark.asyncio
async def test_agent_context_async_context_manager(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should set start/end times in async context."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.start_at is None
    assert ctx.end_at is None

    async with ctx:
        assert ctx.start_at is not None
        assert ctx.end_at is None

    assert ctx.end_at is not None
    assert ctx.end_at >= ctx.start_at


def test_agent_context_deferred_tool_metadata_default(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should have empty metadata by default."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.deferred_tool_metadata == {}


def test_agent_context_deferred_tool_metadata_storage(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should store metadata by tool_call_id."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    ctx.deferred_tool_metadata["call-1"] = {"user_choice": "option_a"}
    assert ctx.deferred_tool_metadata["call-1"]["user_choice"] == "option_a"


def test_agent_context_file_operator(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should store the provided file operator."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.file_operator is file_operator


def test_agent_context_shell(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should store the provided shell."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    assert ctx.shell is shell


@pytest.mark.asyncio
async def test_agent_context_get_environment_instructions(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should return runtime context instructions in XML format."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    instructions = await ctx.get_context_instructions()

    # Check structure
    assert "<runtime-context>" in instructions
    assert "<elapsed-time>not started</elapsed-time>" in instructions
    assert "</runtime-context>" in instructions


@pytest.mark.asyncio
async def test_agent_context_subagent_shares_environment(tmp_path: Path) -> None:
    """Subagent should share file_operator and shell with parent."""
    file_op = LocalFileOperator(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
    )
    shell = LocalShell(
        allowed_paths=[tmp_path],
        default_cwd=tmp_path,
    )
    ctx = AgentContext(file_operator=file_op, shell=shell)

    async with ctx:
        async with ctx.enter_subagent("search") as child:
            # Should share file_operator
            assert child.file_operator is ctx.file_operator

            # Should share shell
            assert child.shell is ctx.shell


# --- Environment integration tests ---


@pytest.mark.asyncio
async def test_local_environment_tmp_dir(tmp_path: Path) -> None:
    """Should create and cleanup temporary directory."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        assert env.tmp_dir is not None
        assert env.tmp_dir.exists()
        assert env.tmp_dir.is_dir()
        assert "pai_agent_" in env.tmp_dir.name
        assert env.tmp_dir.parent == tmp_path

        # Create a file to verify cleanup later
        test_file = env.tmp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()

        saved_tmp_dir = env.tmp_dir

    # After exit, tmp_dir should be cleaned up
    assert not saved_tmp_dir.exists()


@pytest.mark.asyncio
async def test_local_environment_disable_tmp_dir(tmp_path: Path) -> None:
    """Should not create tmp_dir when disabled."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        enable_tmp_dir=False,
    ) as env:
        assert env.tmp_dir is None


@pytest.mark.asyncio
async def test_local_environment_file_operator_and_shell(tmp_path: Path) -> None:
    """Should provide file_operator and shell."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
    ) as env:
        assert env.file_operator is not None
        assert env.shell is not None

        # Test file operations
        test_file = tmp_path / "test.txt"
        await env.file_operator.write_file(str(test_file), "hello")
        content = await env.file_operator.read_file(str(test_file))
        assert content == "hello"

        # Test shell execution
        exit_code, stdout, stderr = await env.shell.execute(["echo", "hello"])
        assert exit_code == 0
        assert "hello" in stdout


@pytest.mark.asyncio
async def test_local_environment_tmp_in_allowed_paths(tmp_path: Path) -> None:
    """tmp_dir should be included in file_operator and shell allowed_paths."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        assert env.tmp_dir is not None

        # Should be able to write to tmp_dir via file_operator
        tmp_file = env.tmp_dir / "data.txt"
        await env.file_operator.write_file(str(tmp_file), "test data")
        assert tmp_file.exists()

        # Should be able to use tmp_dir as shell cwd
        exit_code, stdout, stderr = await env.shell.execute(
            ["ls"],
            cwd=str(env.tmp_dir),
        )
        assert exit_code == 0


@pytest.mark.asyncio
async def test_context_with_environment(tmp_path: Path) -> None:
    """Should use Environment with AgentContext using AsyncExitStack."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(file_operator=env.file_operator, shell=env.shell))
        assert ctx.start_at is not None

        # Can use file_operator from environment
        test_file = tmp_path / "ctx_test.txt"
        await ctx.file_operator.write_file(str(test_file), "from context")
        content = await ctx.file_operator.read_file(str(test_file))
        assert content == "from context"

        # tmp_dir accessible via environment
        assert env.tmp_dir is not None
        assert env.tmp_dir.exists()

    assert ctx.end_at is not None


@pytest.mark.asyncio
async def test_multiple_contexts_share_environment(tmp_path: Path) -> None:
    """Multiple AgentContext sessions should share the same Environment."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        saved_tmp_dir = env.tmp_dir
        assert saved_tmp_dir is not None

        # First session
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx1:
            await env.file_operator.write_file(str(saved_tmp_dir / "shared.txt"), "session1")
            assert ctx1.run_id is not None
            run_id_1 = ctx1.run_id

        # Second session - tmp_dir still exists
        assert saved_tmp_dir.exists()

        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx2:
            # Different run_id
            assert ctx2.run_id != run_id_1

            # Can read file from previous session
            content = await env.file_operator.read_file(str(saved_tmp_dir / "shared.txt"))
            assert content == "session1"

    # tmp_dir cleaned up after environment exits
    assert not saved_tmp_dir.exists()
