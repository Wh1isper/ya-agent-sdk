"""Tests for yaacli.runtime module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import ModelRequest, UserPromptPart
from ya_agent_sdk.agents.lifecycle import ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.filters.handoff import process_handoff_message
from yaacli.background import DELEGATE_BACKEND_TOOL_NAME
from yaacli.config import (
    GeneralConfig,
    MCPConfig,
    MCPServerConfig,
    ModelProfileConfig,
    ToolsConfig,
    YaacliConfig,
)
from yaacli.runtime import GoalContextHandoffExtension, create_tui_runtime
from yaacli.toolsets.background import AsyncDelegateTool, SpawnDelegateTool, SteerSubagentTool

# =============================================================================
# create_tui_runtime Tests
# =============================================================================


def test_create_tui_runtime_minimal(tmp_path: Path) -> None:
    """Test creating runtime with minimal config."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    assert runtime.env is not None
    assert runtime.ctx is not None
    assert runtime.agent is not None
    assert any(isinstance(extension, GoalContextHandoffExtension) for extension in runtime.lifecycle_extensions)


async def test_create_tui_runtime_uses_custom_config_dir_for_allowed_paths(tmp_path: Path) -> None:
    """Test runtime wiring with a custom global config directory."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )
    config_dir = tmp_path / "custom-config"

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
    )

    async with runtime:
        assert config_dir.resolve() in runtime.env.file_operator._allowed_paths


async def test_create_tui_runtime_orders_skill_paths_by_priority(tmp_path: Path) -> None:
    """Test skill path priority: global, shared, project, project config."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )
    config_dir = tmp_path / "custom-config"
    working_dir = tmp_path / "workspace"

    runtime = create_tui_runtime(
        config=config,
        working_dir=working_dir,
        config_dir=config_dir,
    )

    async with runtime:
        allowed_paths = runtime.env.file_operator._allowed_paths
        expected_prefix = [
            config_dir.resolve(),
            (Path.home() / ".agents").resolve(),
            working_dir.resolve(),
            (working_dir / ".yaacli").resolve(),
        ]
        assert allowed_paths[:4] == expected_prefix


def test_create_tui_runtime_uses_persisted_model_profile(tmp_path: Path) -> None:
    """Runtime uses the persisted model profile at startup."""
    from yaacli.model_profiles import save_selected_model_profile_id

    config_dir = tmp_path / "config"
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_cfg="claude_200k",
        ),
        model_profiles={
            "long": ModelProfileConfig(
                label="Long",
                model="openai-chat:gpt-4",
                model_cfg="gemini_1m",
            ),
        },
    )
    save_selected_model_profile_id(config_dir, "long")

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
    )

    assert runtime.ctx.model_cfg.context_window == 1_000_000


async def test_goal_context_handoff_extension_marks_active_goal() -> None:
    """YAACLI lifecycle extension should mark active goals after context handoff."""
    from yaacli.session import TUIContext

    ctx = TUIContext.model_construct()
    ctx.goal_task = "fix tests"
    ctx.goal_iteration = 0
    ctx.goal_max_iterations = 10
    ctx.goal_needs_post_restore_audit = False
    ctx.goal_last_context_handoff_source = None
    extension = GoalContextHandoffExtension()

    await extension.on_context_handoff_complete(
        ContextHandoffCompleteContext(
            event_id="handoff-1",
            deps=ctx,
            source=ContextHandoffSource.COMPACT,
            original_messages=[],
            trimmed_messages=[],
            handoff_messages=[],
            summary_markdown="summary",
        )
    )

    assert ctx.goal_needs_post_restore_audit is True
    assert ctx.goal_last_context_handoff_source == "compact"


async def test_goal_context_handoff_extension_marks_goal_through_handoff_filter() -> None:
    """The summarize handoff filter should trigger YAACLI goal post-restore audit state."""
    from yaacli.session import TUIContext

    ctx = TUIContext()
    ctx.goal_task = "fix tests"
    ctx.lifecycle_extensions = [GoalContextHandoffExtension()]
    ctx.handoff_message = "# Context Summary\n\nContinue the task."
    run_ctx = MagicMock()
    run_ctx.deps = ctx

    result = await process_handoff_message(
        run_ctx,
        [ModelRequest(parts=[UserPromptPart(content="original request")])],
    )

    assert len(result) == 1
    assert ctx.handoff_message is None
    assert ctx.goal_needs_post_restore_audit is True
    assert ctx.goal_last_context_handoff_source == "summarize_tool"


async def test_goal_context_handoff_extension_ignores_inactive_goal() -> None:
    """Inactive goal contexts should not get post-restore audit state."""
    from yaacli.session import TUIContext

    ctx = TUIContext.model_construct()
    ctx.goal_task = None
    ctx.goal_iteration = 0
    ctx.goal_max_iterations = 10
    ctx.goal_needs_post_restore_audit = False
    ctx.goal_last_context_handoff_source = None
    extension = GoalContextHandoffExtension()

    await extension.on_context_handoff_complete(
        ContextHandoffCompleteContext(
            event_id="handoff-1",
            deps=ctx,
            source=ContextHandoffSource.SUMMARIZE_TOOL,
            original_messages=[],
            trimmed_messages=[],
            handoff_messages=[],
            summary_markdown="summary",
        )
    )

    assert ctx.goal_needs_post_restore_audit is False
    assert ctx.goal_last_context_handoff_source is None


def test_create_tui_runtime_with_model_settings(tmp_path: Path) -> None:
    """Test creating runtime with model settings preset."""
    # Use openai which is more commonly mocked in tests
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_settings="openai_high",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None


def test_create_tui_runtime_with_mcp_servers(tmp_path: Path) -> None:
    """Test creating runtime with MCP servers."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )
    mcp_config = MCPConfig(
        servers={
            "test": MCPServerConfig(
                transport="stdio",
                command="echo",
                args=["test"],
            ),
        }
    )

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    mcp_proxy = next(toolset for toolset in runtime.agent.toolsets if getattr(toolset, "prefix", None) == "mcp")
    assert mcp_proxy.search_tool_name == "mcp_search_tool"
    assert mcp_proxy.call_tool_name == "mcp_call_tool"


def test_create_tui_runtime_with_need_approval(tmp_path: Path) -> None:
    """Test creating runtime with tools needing approval."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
        tools=ToolsConfig(need_approval=["shell_sandbox", "file_write"]),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None


def test_create_tui_runtime_uses_cwd_by_default() -> None:
    """Test that runtime uses cwd when working_dir not specified."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(config=config)

    assert runtime is not None


def test_create_tui_runtime_with_model_cfg_preset(tmp_path: Path) -> None:
    """Test creating runtime with model_cfg preset."""
    from ya_agent_sdk.context import ModelCapability

    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_cfg="claude_200k",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Check model_cfg was applied
    assert runtime.ctx.model_cfg.context_window == 200_000
    assert runtime.ctx.model_cfg.max_images == 20
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_with_model_cfg_gemini(tmp_path: Path) -> None:
    """Test creating runtime with gemini model_cfg preset (has video support)."""
    from ya_agent_sdk.context import ModelCapability

    # Use openai model to avoid API key requirement, but test gemini preset
    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_cfg="gemini_1m",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Check gemini preset has vision + video capabilities
    assert runtime.ctx.model_cfg.context_window == 1_000_000
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities
    assert ModelCapability.video_understanding in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_with_model_cfg_dict(tmp_path: Path) -> None:
    """Test creating runtime with custom model_cfg dict."""
    from ya_agent_sdk.context import ModelCapability

    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            model_cfg={
                "context_window": 100_000,
                "max_images": 10,
                "capabilities": ["vision"],
            },
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    assert runtime.ctx.model_cfg.context_window == 100_000
    assert runtime.ctx.model_cfg.max_images == 10
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_can_disable_async_subagents(tmp_path: Path) -> None:
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        enable_async_subagents=False,
    )

    assert runtime.core_toolset is not None
    assert "spawn_delegate" not in runtime.core_toolset._tool_classes
    assert "steer_subagent" not in runtime.core_toolset._tool_classes
    assert "shell_monitor" in runtime.core_toolset._tool_classes
    assert SpawnDelegateTool.name == "spawn_delegate"
    assert SteerSubagentTool.name == "steer_subagent"


async def test_create_tui_runtime_defaults_to_async_delegate_only(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "helper.md").write_text(
        "---\nname: helper\ndescription: Helper subagent\n---\n\nYou are a helper.\n"
    )
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
    )

    assert runtime.core_toolset is not None
    assert runtime.core_toolset._tool_classes["delegate"] is AsyncDelegateTool
    assert "spawn_delegate" not in runtime.core_toolset._tool_classes
    assert DELEGATE_BACKEND_TOOL_NAME in runtime.core_toolset._tool_classes

    async with runtime:
        runtime.env.background_monitor.set_core_toolset(runtime.core_toolset)
        run_ctx = MagicMock()
        run_ctx.deps = runtime.ctx

        visible_tools = await runtime.core_toolset.get_tools(run_ctx)
        assert "delegate" in visible_tools
        assert "spawn_delegate" not in visible_tools
        assert DELEGATE_BACKEND_TOOL_NAME not in visible_tools

        instruction_parts = await runtime.core_toolset.get_instructions(run_ctx)
        instruction_text = "\n".join(part.content for part in instruction_parts or [])
        assert '<tool-instruction name="delegate">' in instruction_text
        delegate_instruction = instruction_text.split('<tool-instruction name="delegate">', 1)[1].split(
            "</tool-instruction>", 1
        )[0]
        assert "delegate is asynchronous" in delegate_instruction
        assert "returns an agent ID immediately" in delegate_instruction
        assert '<subagent name="helper">' in delegate_instruction
        assert "Helper subagent" in delegate_instruction
        assert DELEGATE_BACKEND_TOOL_NAME not in delegate_instruction
        assert "Delegate calls are blocking" not in delegate_instruction


async def test_create_tui_runtime_can_keep_blocking_delegate_and_spawn_delegate(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True)
    (subagents_dir / "helper.md").write_text(
        "---\nname: helper\ndescription: Helper subagent\n---\n\nYou are a helper.\n"
    )
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        config_dir=config_dir,
        enable_async_subagents=True,
        enable_delegate_subagents=True,
    )

    assert runtime.core_toolset is not None
    assert runtime.core_toolset._tool_classes["spawn_delegate"] is SpawnDelegateTool
    assert runtime.core_toolset._tool_classes["delegate"] is not AsyncDelegateTool
    assert DELEGATE_BACKEND_TOOL_NAME not in runtime.core_toolset._tool_classes

    async with runtime:
        runtime.env.background_monitor.set_core_toolset(runtime.core_toolset)
        run_ctx = MagicMock()
        run_ctx.deps = runtime.ctx

        visible_tools = await runtime.core_toolset.get_tools(run_ctx)
        assert "delegate" in visible_tools
        assert "spawn_delegate" in visible_tools
        assert DELEGATE_BACKEND_TOOL_NAME not in visible_tools

        instruction_parts = await runtime.core_toolset.get_instructions(run_ctx)
        instruction_text = "\n".join(part.content for part in instruction_parts or [])
        assert '<tool-instruction name="delegate">' in instruction_text
        assert '<tool-instruction name="spawn_delegate">' in instruction_text
        delegate_instruction = instruction_text.split('<tool-instruction name="delegate">', 1)[1].split(
            "</tool-instruction>", 1
        )[0]
        spawn_instruction = instruction_text.split('<tool-instruction name="spawn_delegate">', 1)[1].split(
            "</tool-instruction>", 1
        )[0]
        assert "Delegate calls are blocking" in delegate_instruction
        assert "Use this to run a subagent asynchronously" in spawn_instruction


def test_create_tui_runtime_with_no_model_cfg(tmp_path: Path) -> None:
    """Test creating runtime without model_cfg uses defaults."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Default ModelConfig values
    assert runtime.ctx.model_cfg.context_window is None
    assert runtime.ctx.model_cfg.max_images == 20
    assert len(runtime.ctx.model_cfg.capabilities) == 0
