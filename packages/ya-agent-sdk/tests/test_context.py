"""Tests for ya_agent_sdk.context module."""

import re
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.context import AgentContext, ModelConfig, ShellReviewConfig, TaskStatus
from ya_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture
async def env(tmp_path: Path):
    """Create a LocalEnvironment for testing."""
    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as environment:
        yield environment


async def test_agent_context_default_run_id(env: LocalEnvironment) -> None:
    """Should generate a unique run_id by default."""
    ctx1 = AgentContext(env=env)
    ctx2 = AgentContext(env=env)
    assert ctx1.run_id != ctx2.run_id
    assert len(ctx1.run_id) == 32  # uuid4().hex length


async def test_agent_context_no_parent_by_default(env: LocalEnvironment) -> None:
    """Should have no parent by default."""
    ctx = AgentContext(env=env)
    assert ctx.parent_run_id is None


async def test_model_config_image_split_defaults() -> None:
    """Should provide sensible defaults for image splitting."""
    cfg = ModelConfig()
    assert cfg.split_large_images is True
    assert cfg.image_split_max_height == 4096
    assert cfg.image_split_overlap == 50


async def test_model_config_image_split_custom_values() -> None:
    """Should allow overriding image splitting config values."""
    cfg = ModelConfig(split_large_images=False, image_split_max_height=2048, image_split_overlap=64)
    assert cfg.split_large_images is False
    assert cfg.image_split_max_height == 2048
    assert cfg.image_split_overlap == 64


async def test_agent_context_elapsed_time_before_start(env: LocalEnvironment) -> None:
    """Should return None before context is started."""
    ctx = AgentContext(env=env)
    assert ctx.elapsed_time is None


async def test_agent_context_elapsed_time_after_start(env: LocalEnvironment) -> None:
    """Should return elapsed time after start."""
    ctx = AgentContext(env=env)
    ctx.start_at = datetime.now(UTC)
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() >= 0


async def test_agent_context_elapsed_time_after_end(env: LocalEnvironment) -> None:
    """Should return final duration after end."""
    ctx = AgentContext(env=env)
    start = datetime.now(UTC)
    ctx.start_at = start
    ctx.end_at = start + timedelta(seconds=5)
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert elapsed.total_seconds() == 5


async def test_agent_context_create_subagent_context(env: LocalEnvironment) -> None:
    """Should create child context with proper inheritance."""
    parent = AgentContext(env=env)
    parent.start_at = datetime.now(UTC)

    async with parent.create_subagent_context("search") as child:
        assert child.parent_run_id == parent.run_id
        assert child.run_id != parent.run_id
        # Verify agent is registered with correct info
        assert child.agent_id in parent.agent_registry
        assert parent.agent_registry[child.agent_id].agent_name == "search"
        assert child.start_at is not None
        assert child.end_at is None

    # After exiting, end_at should be set
    assert child.end_at is not None


async def test_agent_context_create_subagent_context_with_override(env: LocalEnvironment) -> None:
    """Should allow field overrides in subagent context."""
    parent = AgentContext(env=env)

    async with parent.create_subagent_context("reasoning", deferred_tool_metadata={"key": {}}) as child:
        assert child.deferred_tool_metadata == {"key": {}}


async def test_agent_context_create_subagent_context_inherits_security(env: LocalEnvironment) -> None:
    """Subagent context should inherit runtime security policy without sharing mutable config."""
    parent = AgentContext(env=env)
    parent.security.shell_review = ShellReviewConfig(
        enabled=True,
        model="test:model",
        on_needs_approval="defer",
    )

    async with parent.create_subagent_context("search") as child:
        assert child.security.shell_review is not None
        assert child.security.shell_review.enabled is True
        assert child.security.shell_review.model == "test:model"
        assert child.security.shell_review.on_needs_approval == "defer"

        child.security.shell_review.on_needs_approval = "deny"

    assert parent.security.shell_review is not None
    assert parent.security.shell_review.on_needs_approval == "defer"


async def test_agent_context_create_subagent_context_resets_prompts(env: LocalEnvironment) -> None:
    """Should reset user_prompts and steering_messages for subagent context."""
    parent = AgentContext(env=env)
    parent.user_prompts = "Parent prompt"
    parent.steering_messages = ["Parent steering 1", "Parent steering 2"]

    async with parent.create_subagent_context("search") as child:
        # Subagent should have independent prompt tracking
        assert child.user_prompts is None  # Reset, to be set by subagent caller
        assert child.steering_messages == []  # Fresh queue for subagent

        # Parent's prompts should be unaffected
        assert parent.user_prompts == "Parent prompt"
        assert parent.steering_messages == ["Parent steering 1", "Parent steering 2"]


async def test_agent_context_async_context_manager(env: LocalEnvironment) -> None:
    """Should set start/end times in async context."""
    ctx = AgentContext(env=env)
    assert ctx.start_at is None
    assert ctx.end_at is None

    async with ctx:
        assert ctx.start_at is not None
        assert ctx.end_at is None

    assert ctx.end_at is not None
    assert ctx.end_at >= ctx.start_at


async def test_agent_context_double_enter_raises_error(env: LocalEnvironment) -> None:
    """Should raise RuntimeError when entering an already-entered context."""
    ctx = AgentContext(env=env)
    async with ctx:
        with pytest.raises(RuntimeError, match="has already been entered"):
            await ctx.__aenter__()


async def test_agent_context_can_reenter_after_exit(env: LocalEnvironment) -> None:
    """Should allow re-entering after exiting."""
    ctx = AgentContext(env=env)

    # First enter/exit cycle
    async with ctx:
        assert ctx.start_at is not None
        first_start = ctx.start_at

    # Second enter/exit cycle should work
    async with ctx:
        assert ctx.start_at is not None
        # start_at should be updated
        assert ctx.start_at >= first_start


async def test_agent_context_concurrent_enter_raises_error(env: LocalEnvironment) -> None:
    """Should raise RuntimeError when concurrently entering the same context."""
    import asyncio

    ctx = AgentContext(env=env)
    errors: list[Exception] = []
    entered_count = 0

    async def try_enter():
        nonlocal entered_count
        try:
            async with ctx:
                entered_count += 1
                await asyncio.sleep(0.1)  # Hold the context
        except RuntimeError as e:
            errors.append(e)

    # Try to enter concurrently
    await asyncio.gather(try_enter(), try_enter())

    # One should succeed, one should fail
    assert entered_count == 1
    assert len(errors) == 1
    assert "has already been entered" in str(errors[0])


async def test_agent_context_deferred_tool_metadata_default(env: LocalEnvironment) -> None:
    """Should have empty metadata by default."""
    ctx = AgentContext(env=env)
    assert ctx.deferred_tool_metadata == {}


async def test_agent_context_deferred_tool_metadata_storage(env: LocalEnvironment) -> None:
    """Should store metadata by tool_call_id."""
    ctx = AgentContext(env=env)
    ctx.deferred_tool_metadata["call-1"] = {"user_choice": "option_a"}
    assert ctx.deferred_tool_metadata["call-1"]["user_choice"] == "option_a"


async def test_agent_context_file_operator(env: LocalEnvironment) -> None:
    """Should derive file_operator from env."""
    ctx = AgentContext(env=env)
    assert ctx.file_operator is env.file_operator


async def test_agent_context_shell(env: LocalEnvironment) -> None:
    """Should derive shell from env."""
    ctx = AgentContext(env=env)
    assert ctx.shell is env.shell


async def test_agent_context_get_environment_instructions(env: LocalEnvironment) -> None:
    """Should return runtime context instructions in XML format."""
    ctx = AgentContext(env=env)
    instructions = await ctx.get_context_instructions()

    # Check structure
    assert "<runtime-context>" in instructions
    assert "<elapsed-time>not started</elapsed-time>" in instructions
    assert "</runtime-context>" in instructions


async def test_agent_context_subagent_shares_environment(env: LocalEnvironment) -> None:
    """Subagent should share env with parent."""
    ctx = AgentContext(env=env)

    async with ctx:
        async with ctx.create_subagent_context("search") as child:
            # Should share env
            assert child.env is ctx.env

            # Should share file_operator (derived from env)
            assert child.file_operator is ctx.file_operator

            # Should share shell (derived from env)
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
        assert "ya_agent_" in env.tmp_dir.name
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
        ctx = await stack.enter_async_context(AgentContext(env=env))
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
        async with AgentContext(env=env) as ctx1:
            await env.file_operator.write_file(str(saved_tmp_dir / "shared.txt"), "session1")
            assert ctx1.run_id is not None
            run_id_1 = ctx1.run_id

        # Second session - tmp_dir still exists
        assert saved_tmp_dir.exists()

        async with AgentContext(env=env) as ctx2:
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
        async with AgentContext(env=env) as ctx:
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

    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            instructions = await ctx.get_context_instructions()

            # Verify structure with regex to handle variable elapsed time
            assert "<runtime-context>" in instructions
            assert re.search(r"<elapsed-time>[\d.]+s</elapsed-time>", instructions)
            assert "<context-window>200000</context-window>" in instructions
            assert "</runtime-context>" in instructions


async def test_get_context_instructions_with_token_usage(tmp_path: Path) -> None:
    """Should include token usage when run_context with messages is provided."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage
    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
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

            # Verify structure with regex to handle variable elapsed time
            assert "<runtime-context>" in instructions
            assert re.search(r"<elapsed-time>[\d.]+s</elapsed-time>", instructions)
            assert "<context-window>200000</context-window>" in instructions
            assert "<total-tokens>150</total-tokens>" in instructions
            assert "</runtime-context>" in instructions


async def test_get_context_instructions_with_handoff_warning(tmp_path: Path) -> None:
    """Should include handoff warning when threshold exceeded and enabled."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage
    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,  # 50% = 100000 tokens
            ),
        ) as ctx:
            # Set context manage tool (normally done by create_agent auto-detection)
            ctx.context_manage_tool_names = ["summarize"]
            # Create mock run_context with high token usage
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
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

            # Verify structure with regex to handle variable elapsed time
            assert "<runtime-context>" in instructions
            assert re.search(r"<elapsed-time>[\d.]+s</elapsed-time>", instructions)
            assert "<context-window>200000</context-window>" in instructions
            assert "<total-tokens>110000</total-tokens>" in instructions
            assert "</runtime-context>" in instructions
            # Verify context management reminder
            assert "<system-reminder>" in instructions
            assert "summarize" in instructions


async def test_get_context_instructions_handoff_warning_includes_note_cleanup(tmp_path: Path) -> None:
    """Should include note cleanup guidance when threshold exceeded and notes exist."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage
    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            ctx.context_manage_tool_names = ["summarize"]
            ctx.note_manager.set("project", "large value")
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
            mock_run_context.messages = [
                ModelRequest(parts=[UserPromptPart(content="Hello")]),
                ModelResponse(
                    parts=[TextPart(content="Hi")],
                    usage=RequestUsage(input_tokens=80000, output_tokens=30000),
                ),
            ]

            instructions = await ctx.get_context_instructions(mock_run_context)

            assert "Review note keys" in instructions
            assert "note_get" in instructions
            assert "delete stale or oversized notes" in instructions


async def test_get_context_instructions_with_notes_lists_keys_only(env: LocalEnvironment) -> None:
    """Should include note keys in runtime context and omit note values."""
    async with AgentContext(env=env) as ctx:
        ctx.note_manager.set("language", "Chinese communication")
        ctx.note_manager.set("project", "Secret implementation details")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)

        assert "<notes" in instructions
        assert "note_get" in instructions
        assert 'key="language"' in instructions
        assert 'key="project"' in instructions
        assert "Chinese communication" not in instructions
        assert "Secret implementation details" not in instructions


async def test_get_context_instructions_with_notes_non_user_prompt(env: LocalEnvironment) -> None:
    """Should omit note keys on non-user prompts."""
    async with AgentContext(env=env) as ctx:
        ctx.note_manager.set("language", "Chinese communication")

        instructions = await ctx.get_context_instructions(is_user_prompt=False)

        assert "<notes" not in instructions
        assert "language" not in instructions
        assert "Chinese communication" not in instructions


async def test_get_context_instructions_no_handoff_warning_below_threshold(tmp_path: Path) -> None:
    """Should not include reminder when below threshold."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage
    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            ctx.context_manage_tool_names = ["summarize"]
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
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
            assert "summarize" not in instructions


async def test_get_context_instructions_no_handoff_warning_when_disabled(tmp_path: Path) -> None:
    """Should not include reminder when context_manage_tool_names is empty."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage
    from ya_agent_sdk.context import ModelConfig

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            # context_manage_tool_names defaults to empty list (disabled)
            mock_run_context = MagicMock()
            mock_run_context.deps = ctx
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

            # No system-reminder when context management disabled
            assert "<system-reminder>" not in instructions


# =============================================================================
# ResumableState Tests
# =============================================================================


async def test_export_and_with_state_empty(env: LocalEnvironment) -> None:
    """Should export and restore empty state correctly."""
    async with AgentContext(env=env) as ctx:
        state = ctx.export_state()

        assert state.subagent_history == {}
        assert state.usage_snapshot_entries == {}
        assert state.user_prompts is None
        assert state.steering_messages == []
        assert state.handoff_message is None
        assert state.shell_env == {}
        assert state.deferred_tool_metadata == {}

    # Restore to new context
    async with AgentContext(env=env) as new_ctx:
        new_ctx.with_state(state)

        assert new_ctx.subagent_history == {}
        assert new_ctx.usage_snapshot_entries == {}
        assert new_ctx.steering_messages == []


async def test_export_and_with_state_with_data(env: LocalEnvironment) -> None:
    """Should export and restore state with data correctly."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RunUsage

    async with AgentContext(env=env) as ctx:
        # Set up some state
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        ctx.update_usage_snapshot_entry(
            agent_id="search",
            agent_name="search",
            model_id="openai-chat:gpt-4o",
            usage=RunUsage(input_tokens=50, output_tokens=50),
            usage_id="test-uuid",
            ledger_key="test-uuid",
        )
        ctx.user_prompts = "Test prompt"
        ctx.steering_messages = ["Steering 1", "Steering 2"]
        ctx.handoff_message = "Handoff summary"
        ctx.shell_env = {"BASE_KEY": "base_value", "PATH": "/workspace/bin"}
        ctx.deferred_tool_metadata["tool-1"] = {"key": "value"}

        # Default export does NOT include usage_snapshot_entries
        state = ctx.export_state()
        assert state.usage_snapshot_entries == {}

        # Export with include_usage_ledger=True includes usage_snapshot_entries
        state_with_usages = ctx.export_state(include_usage_ledger=True)
        assert len(state_with_usages.usage_snapshot_entries) == 1
        assert state_with_usages.usage_snapshot_entries["test-uuid"].usage_id == "test-uuid"

    # Restore to new context (default state without usage_snapshot_entries)
    async with AgentContext(env=env) as new_ctx:
        new_ctx.with_state(state)

        # Verify subagent_history is restored correctly
        assert "agent-1" in new_ctx.subagent_history
        assert len(new_ctx.subagent_history["agent-1"]) == 2
        request_msg = new_ctx.subagent_history["agent-1"][0]
        assert isinstance(request_msg, ModelRequest)
        assert request_msg.parts[0].content == "Hello"

        # Verify usage_snapshot_entries is empty (not restored by default)
        assert new_ctx.usage_snapshot_entries == {}
        assert new_ctx.user_prompts == "Test prompt"
        assert new_ctx.steering_messages == ["Steering 1", "Steering 2"]
        assert new_ctx.handoff_message == "Handoff summary"
        assert new_ctx.shell_env == {"BASE_KEY": "base_value", "PATH": "/workspace/bin"}
        assert new_ctx.deferred_tool_metadata == {"tool-1": {"key": "value"}}


async def test_export_state_excludes_runtime_self_fork_agent(env: LocalEnvironment) -> None:
    """Runtime-only self fork agent should not be serialized into state."""
    sentinel = object()
    async with AgentContext(env=env) as ctx:
        ctx.self_fork_agent = sentinel
        state = ctx.export_state()

        dumped_state = state.model_dump()
        assert "self_fork_agent" not in dumped_state
        assert "self_fork_agent" not in state.model_dump_json()

    async with AgentContext(env=env) as new_ctx:
        new_ctx.with_state(state)
        assert new_ctx.self_fork_agent is None


async def test_with_state_preserves_runtime_security(env: LocalEnvironment) -> None:
    """Runtime security policy should come from the current context when restoring state."""
    async with AgentContext(env=env) as old_ctx:
        old_ctx.security.shell_review = ShellReviewConfig(
            enabled=True,
            model="old:model",
            on_needs_approval="defer",
        )
        state = old_ctx.export_state()

    async with AgentContext(env=env) as new_ctx:
        new_ctx.security.shell_review = ShellReviewConfig(
            enabled=True,
            model="current:model",
            on_needs_approval="deny",
        )
        new_ctx.with_state(state)

        assert new_ctx.security.shell_review is not None
        assert new_ctx.security.shell_review.model == "current:model"
        assert new_ctx.security.shell_review.on_needs_approval == "deny"


async def test_export_state_include_subagent_false(env: LocalEnvironment) -> None:
    """Should exclude subagent_history and agent_registry when include_subagent=False."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RunUsage

    async with AgentContext(env=env) as ctx:
        # Set up subagent-related state
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]

        # Simulate creating a subagent to populate agent_registry
        async with ctx.create_subagent_context("search") as child:
            child.subagent_history["nested-agent"] = [
                ModelRequest(parts=[UserPromptPart(content="Nested")]),
            ]

        # Set up non-subagent state
        ctx.update_usage_snapshot_entry(
            agent_id="search",
            agent_name="search",
            model_id="openai-chat:gpt-4o",
            usage=RunUsage(input_tokens=50, output_tokens=50),
            usage_id="test-uuid",
            ledger_key="test-uuid",
        )
        ctx.user_prompts = "Test prompt"
        ctx.handoff_message = "Handoff summary"
        ctx.deferred_tool_metadata["tool-1"] = {"key": "value"}
        ctx.need_user_approve_tools = ["shell", "edit"]

        # Export with include_subagent=False
        state = ctx.export_state(include_subagent=False)

        # Verify subagent-related fields are empty
        assert state.subagent_history == {}
        assert state.agent_registry == {}

        # Verify non-subagent fields are preserved (except usage_snapshot_entries which defaults to not included)
        assert state.usage_snapshot_entries == {}  # Default: not included
        assert state.user_prompts == "Test prompt"
        assert state.handoff_message == "Handoff summary"
        assert state.deferred_tool_metadata == {"tool-1": {"key": "value"}}
        assert state.need_user_approve_tools == ["shell", "edit"]

        # Export with include_usage_ledger=True
        state_with_usages = ctx.export_state(include_subagent=False, include_usage_ledger=True)
        assert len(state_with_usages.usage_snapshot_entries) == 1
        assert state_with_usages.usage_snapshot_entries["test-uuid"].usage_id == "test-uuid"


async def test_export_state_include_subagent_true_default(env: LocalEnvironment) -> None:
    """Should include subagent_history and agent_registry when include_subagent=True (default)."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    async with AgentContext(env=env) as ctx:
        # Set up subagent-related state
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]

        # Simulate creating a subagent to populate agent_registry
        async with ctx.create_subagent_context("search"):
            pass

        # Export with default (include_subagent=True)
        state_default = ctx.export_state()
        state_explicit = ctx.export_state(include_subagent=True)

        # Both should include subagent data
        for state in [state_default, state_explicit]:
            assert "agent-1" in state.subagent_history
            assert len(state.subagent_history["agent-1"]) == 2
            assert len(state.agent_registry) > 0


async def test_resumable_state_json_serialization(env: LocalEnvironment) -> None:
    """Should serialize and deserialize ResumableState to/from JSON."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from ya_agent_sdk.context import ResumableState

    async with AgentContext(env=env) as ctx:
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        ctx.user_prompts = "Test prompt"

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


class _CallableRunUsage:
    def __init__(self, usage: RunUsage) -> None:
        self._usage = usage
        self.details = {"provider_cached_tokens": 999}

    def __call__(self) -> RunUsage:
        return self._usage


async def test_update_usage_snapshot_entry_normalizes_callable_usage_wrapper(env: LocalEnvironment) -> None:
    async with AgentContext(env=env) as ctx:
        ctx.update_usage_snapshot_entry(
            agent_id="main",
            agent_name="main",
            model_id="openai-chat:gpt-4o",
            usage=_CallableRunUsage(RunUsage(input_tokens=10, output_tokens=20, details={"cached_tokens": 3})),  # type: ignore[arg-type]
        )

        entry = ctx.build_usage_snapshot().entries[0]
        assert type(entry.usage) is RunUsage
        assert entry.usage.input_tokens == 10
        assert entry.usage.output_tokens == 20
        assert entry.usage.details == {"cached_tokens": 3}


async def test_resumable_state_json_serialization_with_usage_ledger(env: LocalEnvironment) -> None:
    """Should serialize and deserialize ResumableState with usage ledger entries to/from JSON."""
    from pydantic_ai.usage import RunUsage
    from ya_agent_sdk.context import ResumableState

    async with AgentContext(env=env) as ctx:
        # Add usage ledger entries with RunUsage dataclass
        ctx.update_usage_snapshot_entry(
            agent_id="search",
            agent_name="search",
            model_id="openai-chat:gpt-4o",
            usage=RunUsage(input_tokens=100, output_tokens=200),
            usage_id="usage-1",
            ledger_key="usage-1",
        )
        ctx.update_usage_snapshot_entry(
            agent_id="compact",
            agent_name="compact",
            model_id="anthropic:claude-sonnet-4",
            usage=RunUsage(input_tokens=50, output_tokens=75, requests=1, tool_calls=2),
            usage_id="usage-2",
            ledger_key="usage-2",
        )

        # Must use include_usage_ledger=True to include usage ledger entries in state
        state = ctx.export_state(include_usage_ledger=True)

        # Serialize to JSON string
        json_str = state.model_dump_json()
        assert isinstance(json_str, str)
        assert "usage-1" in json_str
        assert "usage-2" in json_str

        # Deserialize from JSON string
        restored_state = ResumableState.model_validate_json(json_str)

        # Verify usage ledger entries restored correctly
        assert len(restored_state.usage_snapshot_entries) == 2

        # Verify first record
        rec1 = restored_state.usage_snapshot_entries["usage-1"]
        assert rec1.usage_id == "usage-1"
        assert rec1.agent_id == "search"
        assert isinstance(rec1.usage, RunUsage)
        assert rec1.usage.input_tokens == 100
        assert rec1.usage.output_tokens == 200

        # Verify second record with additional fields
        rec2 = restored_state.usage_snapshot_entries["usage-2"]
        assert rec2.usage_id == "usage-2"
        assert rec2.agent_id == "compact"
        assert rec2.usage.input_tokens == 50
        assert rec2.usage.output_tokens == 75
        assert rec2.usage.requests == 1
        assert rec2.usage.tool_calls == 2


async def test_resumable_state_with_binary_content(env: LocalEnvironment) -> None:
    """Should serialize and deserialize ResumableState with BinaryContent (images, audio) to/from JSON."""
    import base64

    from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart
    from ya_agent_sdk.context import ResumableState

    # Create a test 1x1 red PNG image
    red_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    async with AgentContext(env=env) as ctx:
        # Create BinaryContent with image data
        image_content = BinaryContent(data=red_pixel_png, media_type="image/png")

        # Add history with binary content in user prompt
        ctx.subagent_history["vision-agent"] = [
            ModelRequest(parts=[UserPromptPart(content=[image_content, "Describe this image"])]),
            ModelResponse(parts=[TextPart(content="This is a 1x1 red pixel image.")]),
        ]

        state = ctx.export_state()

        # Serialize to JSON string
        json_str = state.model_dump_json()
        assert isinstance(json_str, str)

        # Verify the binary data is encoded (should be base64 string in JSON)
        assert "image/png" in json_str

        # Deserialize from JSON string
        restored_state = ResumableState.model_validate_json(json_str)

        # Verify restored state can be converted back to ModelMessage
        history = restored_state.to_subagent_history()
        assert "vision-agent" in history
        assert len(history["vision-agent"]) == 2

        # Verify the ModelRequest with BinaryContent is properly restored
        request = history["vision-agent"][0]
        assert isinstance(request, ModelRequest)
        user_part = request.parts[0]
        assert isinstance(user_part, UserPromptPart)

        # UserPromptPart.content should be a list with BinaryContent and str
        content_list = user_part.content
        assert isinstance(content_list, list)
        assert len(content_list) == 2

        # First item should be BinaryContent with the image
        restored_image = content_list[0]
        assert isinstance(restored_image, BinaryContent)
        assert restored_image.media_type == "image/png"
        assert restored_image.data == red_pixel_png  # Binary data should match exactly

        # Second item should be the text prompt
        assert content_list[1] == "Describe this image"


async def test_resumable_state_with_multiple_binary_contents(env: LocalEnvironment) -> None:
    """Should handle multiple BinaryContent items across different messages."""
    import base64

    from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart
    from ya_agent_sdk.context import ResumableState

    # Create test images
    red_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    # Create a simple test JPEG (minimal valid JPEG)
    minimal_jpeg = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
        "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
        "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
        "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    )

    async with AgentContext(env=env) as ctx:
        image1 = BinaryContent(data=red_pixel_png, media_type="image/png")
        image2 = BinaryContent(data=minimal_jpeg, media_type="image/jpeg")

        # Multiple agents with binary content
        ctx.subagent_history["agent-1"] = [
            ModelRequest(parts=[UserPromptPart(content=[image1, "First image"])]),
            ModelResponse(parts=[TextPart(content="First response")]),
        ]
        ctx.subagent_history["agent-2"] = [
            ModelRequest(parts=[UserPromptPart(content=[image2, "Second image"])]),
            ModelResponse(parts=[TextPart(content="Second response")]),
        ]

        state = ctx.export_state()

        # Serialize and deserialize
        json_str = state.model_dump_json()
        restored_state = ResumableState.model_validate_json(json_str)
        history = restored_state.to_subagent_history()

        # Verify agent-1
        assert "agent-1" in history
        req1 = history["agent-1"][0]
        assert isinstance(req1, ModelRequest)
        content1 = req1.parts[0].content
        assert content1[0].data == red_pixel_png
        assert content1[0].media_type == "image/png"

        # Verify agent-2
        assert "agent-2" in history
        req2 = history["agent-2"][0]
        assert isinstance(req2, ModelRequest)
        content2 = req2.parts[0].content
        assert content2[0].data == minimal_jpeg
        assert content2[0].media_type == "image/jpeg"


async def test_resumable_state_with_tool_return_binary_content(env: LocalEnvironment) -> None:
    """Should restore BinaryContent inside ToolReturnPart content after JSON round-trip."""
    from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart
    from ya_agent_sdk.context import ResumableState

    image_data = b"\x89PNG\r\n\x1a\nexample"

    async with AgentContext(env=env) as ctx:
        ctx.subagent_history["fetch-agent"] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="fetch",
                        content=[BinaryContent(data=image_data, media_type="image/png")],
                        tool_call_id="call_1",
                    )
                ]
            )
        ]

        json_str = ctx.export_state().model_dump_json()
        restored_state = ResumableState.model_validate_json(json_str)
        history = restored_state.to_subagent_history()

        request = history["fetch-agent"][0]
        assert isinstance(request, ModelRequest)
        part = request.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert isinstance(part.content, list)
        restored_content = part.content[0]
        assert isinstance(restored_content, BinaryContent)
        assert restored_content.data == image_data
        assert restored_content.media_type == "image/png"


async def test_resumable_state_with_need_user_approve_tools(env: LocalEnvironment) -> None:
    """Should serialize and deserialize ResumableState with need_user_approve_tools."""
    from ya_agent_sdk.context import ResumableState

    async with AgentContext(env=env) as ctx:
        # Set need_user_approve_tools
        ctx.need_user_approve_tools = ["shell", "edit", "write"]

        state = ctx.export_state()

        # Verify state contains the tools list
        assert state.need_user_approve_tools == ["shell", "edit", "write"]

        # Serialize to JSON string
        json_str = state.model_dump_json()
        assert isinstance(json_str, str)
        assert "need_user_approve_tools" in json_str
        assert "shell" in json_str

        # Deserialize from JSON string
        restored_state = ResumableState.model_validate_json(json_str)

        # Verify restored state
        assert restored_state.need_user_approve_tools == ["shell", "edit", "write"]

        # Verify restore method works
        new_ctx = AgentContext(env=env)
        assert new_ctx.need_user_approve_tools == []  # Default is empty

        restored_state.restore(new_ctx)
        assert new_ctx.need_user_approve_tools == ["shell", "edit", "write"]


async def test_agent_context_need_user_approve_tools_default(env: LocalEnvironment) -> None:
    """Should have empty need_user_approve_tools by default."""
    ctx = AgentContext(env=env)
    assert ctx.need_user_approve_tools == []


async def test_agent_context_need_user_approve_tools_modification(env: LocalEnvironment) -> None:
    """Should allow modification of need_user_approve_tools."""
    ctx = AgentContext(env=env)
    ctx.need_user_approve_tools.append("shell")
    ctx.need_user_approve_tools.append("edit")
    assert ctx.need_user_approve_tools == ["shell", "edit"]


# =============================================================================
# ToolIdWrapper Tests
# =============================================================================


def test_tool_id_wrapper_init() -> None:
    """Should initialize with empty mapping."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    assert wrapper._prefix == "ya-"
    assert wrapper._tool_call_maps == {}


def test_tool_id_wrapper_clear() -> None:
    """Should clear all mappings."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    # Add some mappings
    wrapper.upsert_tool_call_id("original-id-1")
    wrapper.upsert_tool_call_id("original-id-2")
    assert len(wrapper._tool_call_maps) == 2

    wrapper.clear()
    assert wrapper._tool_call_maps == {}


def test_tool_id_wrapper_upsert_already_prefixed() -> None:
    """Should return ID unchanged if already has ya- prefix."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    result = wrapper.upsert_tool_call_id("ya-existing-id")
    assert result == "ya-existing-id"
    # Should not add to mapping
    assert "ya-existing-id" not in wrapper._tool_call_maps


def test_tool_id_wrapper_upsert_new_id() -> None:
    """Should create new normalized ID for non-prefixed IDs."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    result = wrapper.upsert_tool_call_id("call_abc123")
    assert result.startswith("ya-")
    assert len(result) == 3 + 32  # "ya-" + uuid4().hex
    assert "call_abc123" in wrapper._tool_call_maps


def test_tool_id_wrapper_upsert_idempotent() -> None:
    """Should return same normalized ID for same original ID."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    result1 = wrapper.upsert_tool_call_id("openai-call-id")
    result2 = wrapper.upsert_tool_call_id("openai-call-id")
    assert result1 == result2


def test_tool_id_wrapper_upsert_different_ids() -> None:
    """Should create different normalized IDs for different original IDs."""
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    result1 = wrapper.upsert_tool_call_id("id-1")
    result2 = wrapper.upsert_tool_call_id("id-2")
    assert result1 != result2
    assert result1.startswith("ya-")
    assert result2.startswith("ya-")


def test_tool_id_wrapper_wrap_event_function_tool_call() -> None:
    """Should wrap FunctionToolCallEvent with normalized ID."""
    from pydantic_ai import FunctionToolCallEvent
    from pydantic_ai.messages import ToolCallPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    part = ToolCallPart(tool_name="test_tool", tool_call_id="original-id", args={"key": "value"})
    event = FunctionToolCallEvent(part=part)

    result = wrapper.wrap_event(event)

    assert result.tool_call_id.startswith("ya-")
    assert result.part.tool_call_id.startswith("ya-")
    # Same ID should be used
    assert result.tool_call_id == result.part.tool_call_id


def test_tool_id_wrapper_wrap_event_function_tool_result() -> None:
    """Should wrap FunctionToolResultEvent with normalized ID."""
    from pydantic_ai import FunctionToolResultEvent
    from pydantic_ai.messages import ToolReturnPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    result_part = ToolReturnPart(tool_name="test_tool", tool_call_id="original-id", content="result")
    event = FunctionToolResultEvent(part=result_part)

    result = wrapper.wrap_event(event)

    assert result.tool_call_id.startswith("ya-")
    assert result.part.tool_call_id.startswith("ya-")


def test_tool_id_wrapper_wrap_event_part_start() -> None:
    """Should wrap PartStartEvent with ToolCallPart."""
    from pydantic_ai import PartStartEvent
    from pydantic_ai.messages import ToolCallPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    part = ToolCallPart(tool_name="test_tool", tool_call_id="original-id", args={})
    event = PartStartEvent(index=0, part=part)

    result = wrapper.wrap_event(event)

    assert result.part.tool_call_id.startswith("ya-")


def test_tool_id_wrapper_wrap_event_part_end() -> None:
    """Should wrap PartEndEvent with ToolReturnPart."""
    from pydantic_ai import PartEndEvent
    from pydantic_ai.messages import ToolReturnPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    part = ToolReturnPart(tool_name="test_tool", tool_call_id="original-id", content="done")
    event = PartEndEvent(index=0, part=part)

    result = wrapper.wrap_event(event)

    assert result.part.tool_call_id.startswith("ya-")


def test_tool_id_wrapper_wrap_event_part_delta() -> None:
    """Should wrap PartDeltaEvent with ToolCallPartDelta."""
    from pydantic_ai import PartDeltaEvent, ToolCallPartDelta
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    delta = ToolCallPartDelta(tool_call_id="original-id", args_delta="partial")
    event = PartDeltaEvent(index=0, delta=delta)

    result = wrapper.wrap_event(event)

    assert result.delta.tool_call_id.startswith("ya-")


def test_tool_id_wrapper_wrap_event_unrelated_event() -> None:
    """Should return unrelated events unchanged."""
    from pydantic_ai import PartDeltaEvent
    from pydantic_ai.messages import TextPartDelta
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    delta = TextPartDelta(content_delta="some text")
    event = PartDeltaEvent(index=0, delta=delta)

    result = wrapper.wrap_event(event)

    # Should be returned unchanged (same object)
    assert result is event


def test_tool_id_wrapper_wrap_deferred_tool_requests() -> None:
    """Should wrap DeferredToolRequests with normalized IDs."""
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.messages import ToolCallPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    calls = [
        ToolCallPart(tool_name="tool1", tool_call_id="call-1", args={}),
        ToolCallPart(tool_name="tool2", tool_call_id="call-2", args={}),
    ]
    approvals = [
        ToolCallPart(tool_name="tool1", tool_call_id="call-1", args={}),
    ]
    deferred = DeferredToolRequests(
        calls=calls,
        approvals=approvals,
        metadata={"call-1": {"reviewer": "shell_command_reviewer", "reason": "Deletes files"}},
    )

    result = wrapper.wrap_deferred_tool_requests(deferred)

    assert result.calls[0].tool_call_id.startswith("ya-")
    assert result.calls[1].tool_call_id.startswith("ya-")
    assert result.approvals[0].tool_call_id.startswith("ya-")
    # call-1 should map to same ID in both calls and approvals
    assert result.calls[0].tool_call_id == result.approvals[0].tool_call_id
    assert result.metadata[result.approvals[0].tool_call_id] == {
        "reviewer": "shell_command_reviewer",
        "reason": "Deletes files",
    }


def test_tool_id_wrapper_wrap_messages() -> None:
    """Should wrap message history with normalized IDs."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()

    # Create messages with tool parts
    tool_call = ToolCallPart(tool_name="test", tool_call_id="original-call", args={})
    tool_return = ToolReturnPart(tool_name="test", tool_call_id="original-return", content="result")

    messages = [
        ModelRequest(parts=[tool_call]),
        ModelRequest(parts=[tool_return]),
    ]

    # RunContext is not used but required by signature
    mock_ctx = MagicMock()

    result = wrapper.wrap_messages(mock_ctx, messages)

    assert result[0].parts[0].tool_call_id.startswith("ya-")
    assert result[1].parts[0].tool_call_id.startswith("ya-")


def test_tool_id_wrapper_wrap_messages_preserves_non_tool_parts() -> None:
    """Should preserve non-tool parts unchanged."""
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, TextPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()

    text_part = TextPart(content="Hello")
    messages = [ModelRequest(parts=[text_part])]

    mock_ctx = MagicMock()
    result = wrapper.wrap_messages(mock_ctx, messages)

    assert result[0].parts[0].content == "Hello"


def test_tool_id_wrapper_consistency_across_methods() -> None:
    """Should use consistent ID mapping across all wrap methods."""
    from unittest.mock import MagicMock

    from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
    from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
    from ya_agent_sdk.context import ToolIdWrapper

    wrapper = ToolIdWrapper()
    original_id = "consistent-id"

    # Wrap via event
    call_part = ToolCallPart(tool_name="test", tool_call_id=original_id, args={})
    call_event = FunctionToolCallEvent(part=call_part)
    wrapper.wrap_event(call_event)

    # Wrap via result event - should use same normalized ID
    result_part = ToolReturnPart(tool_name="test", tool_call_id=original_id, content="done")
    result_event = FunctionToolResultEvent(part=result_part)
    wrapper.wrap_event(result_event)

    # Wrap via messages - should use same normalized ID
    messages = [ModelRequest(parts=[ToolCallPart(tool_name="test", tool_call_id=original_id, args={})])]
    mock_ctx = MagicMock()
    wrapper.wrap_messages(mock_ctx, messages)

    # All should have same normalized ID
    assert call_event.part.tool_call_id == result_event.part.tool_call_id
    assert call_event.part.tool_call_id == messages[0].parts[0].tool_call_id


# =============================================================================
# get_context_instructions Tests
# =============================================================================


async def test_get_context_instructions_returns_xml(env: LocalEnvironment) -> None:
    """Should return runtime-context XML with elapsed time."""
    async with AgentContext(env=env) as ctx:
        instructions = await ctx.get_context_instructions()
        assert "<runtime-context>" in instructions
        assert "<elapsed-time>" in instructions


async def test_get_context_instructions_with_known_subagents(env: LocalEnvironment) -> None:
    """Should include known-subagents when agent_registry has subagents."""
    from ya_agent_sdk.context import AgentInfo

    async with AgentContext(env=env) as ctx:
        # Register some subagents
        ctx.agent_registry["sub1"] = AgentInfo(
            agent_id="sub1",
            agent_name="search_agent",
            parent_agent_id=ctx.run_id,
        )
        ctx.agent_registry["sub2"] = AgentInfo(
            agent_id="sub2",
            agent_name="reasoning_agent",
            parent_agent_id=ctx.run_id,
        )

        instructions = await ctx.get_context_instructions()

        # Check known-subagents section exists
        assert "<known-subagents" in instructions
        assert 'hint="Use subagent_info tool for more details"' in instructions
        assert 'id="sub1"' in instructions
        assert 'name="search_agent"' in instructions
        assert 'id="sub2"' in instructions
        assert 'name="reasoning_agent"' in instructions


async def test_get_context_instructions_excludes_main_agent(env: LocalEnvironment) -> None:
    """Should exclude the main agent from known-subagents."""
    from ya_agent_sdk.context import AgentInfo

    async with AgentContext(env=env) as ctx:
        # Register main agent (should be excluded by _agent_id filtering)
        ctx.agent_registry[ctx.agent_id] = AgentInfo(
            agent_id=ctx.agent_id,
            agent_name="main",
            parent_agent_id=None,
        )
        # Register a subagent (should be included)
        ctx.agent_registry["sub1"] = AgentInfo(
            agent_id="sub1",
            agent_name="search_agent",
            parent_agent_id=ctx.agent_id,
        )

        instructions = await ctx.get_context_instructions()

        # Should include subagent but not main agent itself
        assert 'id="sub1"' in instructions
        assert f'id="{ctx.agent_id}"' not in instructions


async def test_get_context_instructions_no_subagents(env: LocalEnvironment) -> None:
    """Should not include known-subagents section when no subagents."""
    async with AgentContext(env=env) as ctx:
        instructions = await ctx.get_context_instructions()

        # Should not have known-subagents section
        assert "<known-subagents" not in instructions


async def test_get_context_instructions_with_active_tasks_user_prompt(env: LocalEnvironment) -> None:
    """Should include detailed active-tasks for user prompts."""
    async with AgentContext(env=env) as ctx:
        # Create tasks
        task1 = ctx.task_manager.create("Implement API", "Create REST endpoints", active_form="Implementing API")
        task2 = ctx.task_manager.create("Write tests", "Unit tests for API")

        # Set task1 in progress and task2 blocked by task1
        ctx.task_manager.update(task1.id, status=TaskStatus.IN_PROGRESS)
        ctx.task_manager.update(task2.id, add_blocked_by=[task1.id])

        instructions = await ctx.get_context_instructions(is_user_prompt=True)

        # Should have active-tasks section with hint
        assert '<active-tasks hint="Update status with task_update tool">' in instructions
        # Should have task elements with subject as sub-element
        assert f'<task id="{task1.id}" status="in_progress">' in instructions
        assert "<subject>Implement API</subject>" in instructions
        assert "<active-form>Implementing API</active-form>" in instructions
        # Task2 should show blocked-by
        assert f'blocked-by="{task1.id}"' in instructions
        assert "<subject>Write tests</subject>" in instructions


async def test_get_context_instructions_with_active_tasks_non_user_prompt(env: LocalEnvironment) -> None:
    """Should include compact active-tasks for non-user prompts."""
    async with AgentContext(env=env) as ctx:
        # Create tasks
        task1 = ctx.task_manager.create("Implement API", "Create REST endpoints", active_form="Implementing API")
        ctx.task_manager.update(task1.id, status=TaskStatus.IN_PROGRESS)

        instructions = await ctx.get_context_instructions(is_user_prompt=False)

        # Should have active-tasks section without hint
        assert "<active-tasks>" in instructions
        assert "hint=" not in instructions.split("<active-tasks>")[1].split("</active-tasks>")[0]
        # Should have compact format: subject as text content, no active-form
        assert f'<task id="{task1.id}" status="in_progress">Implement API</task>' in instructions
        assert "<active-form>" not in instructions


async def test_get_context_instructions_no_active_tasks(env: LocalEnvironment) -> None:
    """Should not include active-tasks when no tasks exist."""
    async with AgentContext(env=env) as ctx:
        instructions = await ctx.get_context_instructions()

        # Should not have active-tasks section
        assert "<active-tasks" not in instructions


async def test_get_context_instructions_excludes_completed_tasks(env: LocalEnvironment) -> None:
    """Should exclude completed tasks from active-tasks."""
    async with AgentContext(env=env) as ctx:
        # Create and complete a task
        task = ctx.task_manager.create("Completed task", "This is done")
        ctx.task_manager.update(task.id, status=TaskStatus.COMPLETED)

        instructions = await ctx.get_context_instructions()

        # Should not have active-tasks section (no active tasks)
        assert "<active-tasks" not in instructions


async def test_get_context_instructions_excludes_completed_blockers(env: LocalEnvironment) -> None:
    """Should exclude completed blockers from blocked-by attribute."""
    async with AgentContext(env=env) as ctx:
        # Create tasks
        task1 = ctx.task_manager.create("Task 1", "First task")
        task2 = ctx.task_manager.create("Task 2", "Second task")
        ctx.task_manager.update(task2.id, add_blocked_by=[task1.id])

        # Complete task1
        ctx.task_manager.update(task1.id, status=TaskStatus.COMPLETED)

        instructions = await ctx.get_context_instructions()

        # Task2 should not show blocked-by since task1 is completed
        assert "<active-tasks" in instructions
        assert f'<task id="{task2.id}" status="pending">' in instructions
        assert "blocked-by=" not in instructions


# =============================================================================
# get_current_time Tests
# =============================================================================


async def test_get_current_time_returns_datetime_with_timezone(env: LocalEnvironment) -> None:
    """Should return datetime with timezone information."""
    ctx = AgentContext(env=env)
    current_time = ctx.get_current_time()

    # Should be a datetime
    assert isinstance(current_time, datetime)
    # Should have timezone info (not naive)
    assert current_time.tzinfo is not None


async def test_get_current_time_is_recent(env: LocalEnvironment) -> None:
    """Should return a time close to actual current time."""
    ctx = AgentContext(env=env)
    before = datetime.now(UTC)
    current_time = ctx.get_current_time()
    after = datetime.now(UTC)

    # Should be within the time window
    assert before <= current_time <= after


async def test_get_current_time_can_be_overridden(env: LocalEnvironment) -> None:
    """Should allow subclass to override time source."""

    fixed_time = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)

    class MockContext(AgentContext):
        def get_current_time(self) -> datetime:
            return fixed_time

    ctx = MockContext(env=env)
    assert ctx.get_current_time() == fixed_time


async def test_get_context_instructions_includes_current_time(env: LocalEnvironment) -> None:
    """Should include current-time element in ISO 8601 format."""
    async with AgentContext(env=env) as ctx:
        instructions = await ctx.get_context_instructions()

        # Should contain current-time element
        assert "<current-time>" in instructions
        assert "</current-time>" in instructions
        # Should be in ISO 8601 format with timezone (e.g., 2025-01-20T12:30:00+00:00)
        assert re.search(
            r"<current-time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}</current-time>", instructions
        )


async def test_get_context_instructions_uses_custom_time_source(env: LocalEnvironment) -> None:
    """Should use overridden get_current_time in context instructions."""

    fixed_time = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)

    class MockContext(AgentContext):
        def get_current_time(self) -> datetime:
            return fixed_time

    async with MockContext(env=env) as ctx:
        instructions = await ctx.get_context_instructions()

        # Should contain the fixed time in ISO format
        assert "<current-time>2025-06-15T12:30:00+00:00</current-time>" in instructions


async def test_prepare_new_run_fresh_per_run_state(env: LocalEnvironment) -> None:
    """prepare_new_run should return a fresh copy with isolated per-run state."""
    async with AgentContext(env=env) as ctx:
        from pydantic_ai.usage import RunUsage

        ctx.update_usage_snapshot_entry(
            agent_id="test",
            agent_name="test",
            model_id="m",
            usage=RunUsage(),
            usage_id="old",
            ledger_key="old",
        )
        ctx.deferred_tool_metadata["key"] = {"val": 1}

        new = ctx.prepare_new_run()

        # Per-run state is fresh
        assert new.run_id != ctx.run_id
        assert new.usage_snapshot_entries == {}
        assert new.deferred_tool_metadata == {}
        assert new.force_inject_instructions is False
        assert new.end_at is None
        assert new.start_at is not None  # set by prepare_new_run
        assert new.tool_id_wrapper is not ctx.tool_id_wrapper
        assert new.agent_stream_queues is not ctx.agent_stream_queues

        # Private state is reset
        assert new._entered is False
        assert new._stream_queue_enabled is False


async def test_prepare_new_run_shares_long_lived_state(env: LocalEnvironment) -> None:
    """prepare_new_run should share long-lived state by reference."""
    async with AgentContext(env=env) as ctx:
        # Pre-populate some shared state
        ctx.steering_messages.append("test msg")
        ctx.need_user_approve_tools.append("shell")

        new = ctx.prepare_new_run()

        # Long-lived state is same object (shared reference)
        assert new.env is ctx.env
        assert new.message_bus is ctx.message_bus
        assert new.task_manager is ctx.task_manager
        assert new.note_manager is ctx.note_manager
        assert new.model_cfg is ctx.model_cfg
        assert new.tool_config is ctx.tool_config
        assert new.subagent_history is ctx.subagent_history
        assert new.agent_registry is ctx.agent_registry
        assert new.steering_messages is ctx.steering_messages
        assert new.need_user_approve_tools is ctx.need_user_approve_tools

        # Mutations via new are visible from original (shared reference)
        new.steering_messages.append("from new")
        assert "from new" in ctx.steering_messages


async def test_prepare_new_run_preserves_subclass() -> None:
    """prepare_new_run should preserve the subclass type."""

    class CustomContext(AgentContext):
        custom_field: str = "hello"

    ctx = CustomContext()
    new = ctx.prepare_new_run()
    assert isinstance(new, CustomContext)
    assert new.custom_field == "hello"
