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


async def test_agent_context_enter_subagent_with_override(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should allow field overrides in subagent context."""
    parent = AgentContext(file_operator=file_operator, shell=shell)

    async with parent.enter_subagent("reasoning", deferred_tool_metadata={"key": {}}) as child:
        assert child.deferred_tool_metadata == {"key": {}}


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


async def test_agent_context_get_environment_instructions(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should return runtime context instructions in XML format."""
    ctx = AgentContext(file_operator=file_operator, shell=shell)
    instructions = await ctx.get_context_instructions()

    # Check structure
    assert "<runtime-context>" in instructions
    assert "<elapsed-time>not started</elapsed-time>" in instructions
    assert "</runtime-context>" in instructions


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


async def test_local_environment_disable_tmp_dir(tmp_path: Path) -> None:
    """Should not create tmp_dir when disabled."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        enable_tmp_dir=False,
    ) as env:
        assert env.tmp_dir is None


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
        exit_code, stdout, stderr = await env.shell.execute("echo hello")
        assert exit_code == 0
        assert "hello" in stdout


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
            "ls",
            cwd=str(env.tmp_dir),
        )
        assert exit_code == 0


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


async def test_get_context_instructions_basic(tmp_path: Path) -> None:
    """Should return XML-formatted runtime context instructions."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            # Wait a tiny bit to get non-zero elapsed time
            import asyncio

            await asyncio.sleep(0.01)

            instructions = await ctx.get_context_instructions()

            # Should contain runtime-context element
            assert "<runtime-context>" in instructions
            assert "</runtime-context>" in instructions
            assert "<elapsed-time>" in instructions


async def test_get_context_instructions_with_model_config(tmp_path: Path) -> None:
    """Should include model config in instructions when set."""
    from inline_snapshot import snapshot

    from pai_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            instructions = await ctx.get_context_instructions()

            assert instructions == snapshot("""\
<runtime-context>
  <elapsed-time>0.0s</elapsed-time>
  <model-config>
    <context-window>200000</context-window>
  </model-config>
</runtime-context>\
""")


async def test_get_context_instructions_with_token_usage(tmp_path: Path) -> None:
    """Should include token usage when run_context with messages is provided."""
    from unittest.mock import MagicMock

    from inline_snapshot import snapshot
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    from pai_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            # Create mock run_context with messages containing usage
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
            mock_run_context.metadata = {}
            mock_run_context.messages = [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(
                    parts=[TextPart(content="Hi")],
                    usage=RequestUsage(
                        input_tokens=100,
                        output_tokens=50,
                    ),
                ),
            ]

            instructions = await ctx.get_context_instructions(mock_run_context)

            assert instructions == snapshot("""\
<runtime-context>
  <elapsed-time>0.0s</elapsed-time>
  <model-config>
    <context-window>200000</context-window>
  </model-config>
  <token-usage>
    <total-tokens>150</total-tokens>
  </token-usage>
</runtime-context>\
""")


async def test_get_context_instructions_with_handoff_warning(tmp_path: Path) -> None:
    """Should include handoff warning when threshold exceeded and enabled."""
    from unittest.mock import MagicMock

    from inline_snapshot import snapshot
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    from pai_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,  # 50% = 100000 tokens
            ),
        ) as ctx:
            # Create mock run_context with high token usage
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
            mock_run_context.metadata = {"context_manage_tool": "handoff"}
            mock_run_context.messages = [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(
                    parts=[TextPart(content="Hi")],
                    usage=RequestUsage(
                        input_tokens=80000,
                        output_tokens=30000,  # Exceeds 100000 threshold
                    ),
                ),
            ]

            instructions = await ctx.get_context_instructions(mock_run_context)

            assert instructions == snapshot("""\
<runtime-context>
  <elapsed-time>0.0s</elapsed-time>
  <model-config>
    <context-window>200000</context-window>
  </model-config>
  <token-usage>
    <total-tokens>110000</total-tokens>
  </token-usage>
</runtime-context>

<system-reminder>
  <item>IMPORTANT: **You have reached the handoff threshold, please calling the `handoff` tool to summarize then continue the task at the appropriate time.**</item>
</system-reminder>\
""")


async def test_get_context_instructions_no_handoff_warning_below_threshold(tmp_path: Path) -> None:
    """Should not include handoff warning when below threshold."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    from pai_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
            mock_run_context.metadata = {"context_manage_tool": "handoff"}
            mock_run_context.messages = [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(
                    parts=[TextPart(content="Hi")],
                    usage=RequestUsage(
                        input_tokens=40000,
                        output_tokens=10000,  # Below 100000 threshold
                    ),
                ),
            ]

            instructions = await ctx.get_context_instructions(mock_run_context)

            # Should not contain system-reminder
            assert "<system-reminder>" not in instructions
            assert "handoff" not in instructions


async def test_get_context_instructions_no_handoff_warning_when_disabled(tmp_path: Path) -> None:
    """Should not include handoff warning when context_manage_tool is False."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    from pai_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
            mock_run_context.metadata = {"context_manage_tool": False}  # Disabled
            mock_run_context.messages = [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(
                    parts=[TextPart(content="Hi")],
                    usage=RequestUsage(
                        input_tokens=80000,
                        output_tokens=30000,  # Exceeds threshold but disabled
                    ),
                ),
            ]

            instructions = await ctx.get_context_instructions(mock_run_context)

            # No system-reminder when handoff disabled
            assert "<system-reminder>" not in instructions


# =============================================================================
# ResumableState Tests
# =============================================================================


async def test_export_and_with_state_empty(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should export and restore empty state correctly."""
    async with AgentContext(file_operator=file_operator, shell=shell) as ctx:
        state = ctx.export_state()

        assert state.subagent_history == {}
        assert state.extra_usages == []
        assert state.user_prompts == []
        assert state.handoff_message is None
        assert state.deferred_tool_metadata == {}

    # Restore to new context
    async with AgentContext(file_operator=file_operator, shell=shell) as new_ctx:
        new_ctx.with_state(state)

        assert new_ctx.subagent_history == {}
        assert new_ctx.extra_usages == []


async def test_export_and_with_state_with_data(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should export and restore state with data correctly."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RunUsage

    from pai_agent_sdk.context import ExtraUsageRecord

    async with AgentContext(file_operator=file_operator, shell=shell) as ctx:
        # Set up some state
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        ctx.extra_usages.append(
            ExtraUsageRecord(uuid="test-uuid", agent="search", usage=RunUsage(input_tokens=50, output_tokens=50))
        )
        ctx.user_prompts.append("Test prompt")
        ctx.handoff_message = "Handoff summary"
        ctx.deferred_tool_metadata["tool-1"] = {"key": "value"}

        state = ctx.export_state()

    # Restore to new context
    async with AgentContext(file_operator=file_operator, shell=shell) as new_ctx:
        new_ctx.with_state(state)

        # Verify subagent_history is restored correctly
        assert "agent-1" in new_ctx.subagent_history
        assert len(new_ctx.subagent_history["agent-1"]) == 2
        request_msg = new_ctx.subagent_history["agent-1"][0]
        assert isinstance(request_msg, ModelRequest)
        assert request_msg.parts[0].content == "Hello"

        # Verify other fields
        assert len(new_ctx.extra_usages) == 1
        assert new_ctx.extra_usages[0].uuid == "test-uuid"
        assert new_ctx.user_prompts == ["Test prompt"]
        assert new_ctx.handoff_message == "Handoff summary"
        assert new_ctx.deferred_tool_metadata == {"tool-1": {"key": "value"}}


async def test_resumable_state_json_serialization(file_operator: LocalFileOperator, shell: LocalShell) -> None:
    """Should serialize and deserialize ResumableState to/from JSON."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    from pai_agent_sdk.context import ResumableState

    async with AgentContext(file_operator=file_operator, shell=shell) as ctx:
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        ctx.user_prompts.append("Test prompt")

        state = ctx.export_state()

        # Serialize to JSON string
        json_str = state.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize from JSON string
        restored_state = ResumableState.model_validate_json(json_str)

        # Verify restored state can be converted back to ModelMessage
        history = restored_state.to_subagent_history()
        assert "agent-1" in history
        assert len(history["agent-1"]) == 2
        assert history["agent-1"][0].parts[0].content == "Hello"
