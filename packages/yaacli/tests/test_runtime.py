"""Tests for yaacli.runtime module."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaacli.runtime as runtime_module
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset, PrefixedToolset
from pydantic_ai.usage import RequestUsage
from ya_agent_sdk.agents.lifecycle import ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.filters.handoff import process_handoff_message
from ya_agent_sdk.mcp import NamedMCPToolset
from ya_agent_sdk.toolsets.core.base import Toolset
from ya_agent_sdk.toolsets.tool_proxy.toolset import ToolProxyToolset
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
from yaacli.toolsets.background import (
    AsyncDelegateTool,
    MonitoredShellTool,
    SpawnDelegateTool,
    SteerSubagentTool,
    WaitSubagentTool,
)

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
            Path(tempfile.gettempdir()).resolve(),
            config_dir.resolve(),
            (Path.home() / ".agents").resolve(),
            working_dir.resolve(),
            (working_dir / ".yaacli").resolve(),
        ]
        assert allowed_paths[:5] == expected_prefix
        assert runtime.env.tmp_dir is not None
        assert runtime.env.tmp_dir.resolve() in allowed_paths


async def test_create_tui_runtime_filetree_context_uses_workspace_only(tmp_path: Path) -> None:
    """TUI should keep auxiliary roots accessible without rendering them as file trees."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
    )
    config_dir = tmp_path / "custom-config"
    working_dir = tmp_path / "workspace"
    config_dir.mkdir()
    working_dir.mkdir()
    (config_dir / "config-marker.txt").write_text("config")
    (working_dir / "app.py").write_text("# app")

    runtime = create_tui_runtime(
        config=config,
        working_dir=working_dir,
        config_dir=config_dir,
    )

    async with runtime:
        instructions = await runtime.env.file_operator.get_context_instructions()
        assert instructions is not None

        assert instructions.count("<directory path=") == 1
        assert f'<directory path="{working_dir.resolve()}">' in instructions
        assert f'<directory path="{Path(tempfile.gettempdir()).resolve()}">' not in instructions
        assert f'<directory path="{config_dir.resolve()}">' not in instructions
        assert "app.py" in instructions
        assert "config-marker.txt" not in instructions

        allowed_paths = runtime.env.file_operator._allowed_paths
        assert Path(tempfile.gettempdir()).resolve() in allowed_paths
        assert config_dir.resolve() in allowed_paths
        assert runtime.env.tmp_dir is not None
        assert runtime.env.tmp_dir.resolve() in allowed_paths


async def test_model_profile_instructions_refresh_after_switch(tmp_path: Path) -> None:
    """Profile instructions are evaluated for each model request after a profile switch."""
    captured_instructions: list[str | None] = []
    captured_instruction_parts: list[list[tuple[str, bool]]] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        parts = info.model_request_parameters.instruction_parts or []
        captured_instruction_parts.append([(part.content, part.dynamic) for part in parts])
        return ModelResponse(parts=[TextPart(content="ok")])

    config = YaacliConfig(
        general=GeneralConfig(
            model="openai-chat:gpt-4",
            instructions="Use the default profile instructions.",
        )
    )
    runtime = create_tui_runtime(config=config, working_dir=tmp_path, enable_async_subagents=False)

    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            first_result = await runtime.agent.run("First turn", deps=runtime.ctx)
            runtime.ctx.model_profile_instructions = "Use the fast profile instructions."
            await runtime.agent.run("Second turn", deps=runtime.ctx, message_history=first_result.all_messages())

    assert "Use the default profile instructions." in captured_instructions[0]
    assert "Use the fast profile instructions." in captured_instructions[1]
    assert "Use the default profile instructions." not in captured_instructions[1]
    assert ("Use the default profile instructions.", True) in captured_instruction_parts[0]
    assert ("Use the fast profile instructions.", True) in captured_instruction_parts[1]
    assert ("Use the default profile instructions.", True) not in captured_instruction_parts[1]


async def test_model_profile_instructions_apply_to_legacy_history(tmp_path: Path) -> None:
    """Profile instructions do not depend on a dynamic system-prompt marker in history."""
    captured_instructions: list[str | None] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(
                model="openai-chat:gpt-4",
                instructions="Use the configured profile instructions.",
            )
        ),
        working_dir=tmp_path,
        enable_async_subagents=False,
    )
    legacy_history = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="Legacy system prompt"),
                UserPromptPart(content="Original request"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Previous answer")]),
    ]

    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            await runtime.agent.run("Continue", deps=runtime.ctx, message_history=legacy_history)

    assert len(captured_instructions) == 1
    assert "Use the configured profile instructions." in captured_instructions[0]


async def test_model_profile_instructions_apply_after_handoff(tmp_path: Path) -> None:
    """Handoff replacement does not remove profile instructions from the next model request."""
    captured_instructions: list[str | None] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(
                model="openai-chat:gpt-4",
                instructions="Use the configured profile instructions.",
            )
        ),
        working_dir=tmp_path,
        enable_async_subagents=False,
    )

    async with runtime:
        runtime.ctx.handoff_message = "# Handoff Summary\n\nContinue from this summary."
        with runtime.agent.override(model=FunctionModel(respond)):
            await runtime.agent.run("Continue", deps=runtime.ctx)

    assert len(captured_instructions) == 1
    assert "Use the configured profile instructions." in captured_instructions[0]


async def test_model_profile_instructions_apply_after_automatic_compaction(tmp_path: Path) -> None:
    """Automatic context compaction does not remove profile instructions from later requests."""
    captured_instructions: list[str | None] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="continue")])

    async def stream_respond(_messages: list[ModelMessage], info: AgentInfo):
        captured_instructions.append(info.instructions)
        yield "compacted summary"

    runtime = create_tui_runtime(
        config=YaacliConfig(
            general=GeneralConfig(
                model="openai-chat:gpt-4",
                model_cfg={"context_window": 10, "compact_threshold": 0.1},
                instructions="Use the configured profile instructions.",
            )
        ),
        working_dir=tmp_path,
        enable_async_subagents=False,
    )
    history = [
        ModelRequest(
            parts=[SystemPromptPart(content="Legacy system prompt"), UserPromptPart(content="Original request")]
        ),
        ModelResponse(parts=[TextPart(content="Previous answer")], usage=RequestUsage(input_tokens=10)),
    ]

    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond, stream_function=stream_respond)):
            await runtime.agent.run("Continue", deps=runtime.ctx, message_history=history)

    assert len(captured_instructions) >= 2
    assert all(
        instruction is not None and "Use the configured profile instructions." in instruction
        for instruction in captured_instructions
    )


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


def test_create_tui_runtime_exposes_mcp_servers_directly_by_default(tmp_path: Path) -> None:
    """Test that MCP servers use native toolsets unless proxy mode is requested."""
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
    direct_toolsets = [toolset for toolset in runtime.agent.toolsets if isinstance(toolset, PrefixedToolset)]
    assert len(direct_toolsets) == 1
    assert direct_toolsets[0].prefix == "test"
    assert isinstance(direct_toolsets[0].wrapped, NamedMCPToolset)
    assert not any(isinstance(toolset, ToolProxyToolset) for toolset in runtime.agent.toolsets)


def test_create_tui_runtime_can_proxy_mcp_servers(tmp_path: Path) -> None:
    """Test explicitly exposing MCP servers through the fixed tool proxy."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4"),
        tools=ToolsConfig(mcp_mode="proxy"),
    )
    mcp_config = MCPConfig(
        servers={
            "test": MCPServerConfig(
                transport="stdio",
                command="echo",
                args=["test"],
                prefix="ignored-in-proxy-mode",
            ),
        }
    )

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=tmp_path,
    )

    mcp_proxies = [toolset for toolset in runtime.agent.toolsets if isinstance(toolset, ToolProxyToolset)]
    assert len(mcp_proxies) == 1
    assert mcp_proxies[0].search_tool_name == "mcp_search_tool"
    assert mcp_proxies[0].call_tool_name == "mcp_call_tool"
    assert not any(isinstance(toolset, PrefixedToolset) for toolset in runtime.agent.toolsets)


async def test_create_tui_runtime_namespaces_duplicate_direct_mcp_tools(tmp_path: Path, monkeypatch) -> None:
    """Direct MCP servers may expose the same native tool name without conflicts."""

    def collide() -> str:
        return "ok"

    mcp_servers = [
        FunctionToolset([collide], id="one"),
        FunctionToolset([collide], id="two"),
    ]
    monkeypatch.setattr(runtime_module, "build_mcp_servers", lambda *_args, **_kwargs: mcp_servers)
    visible_tool_names: set[str] = set()

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible_tool_names.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4")),
        mcp_config=MCPConfig(
            servers={
                "one": MCPServerConfig(command="unused"),
                "two": MCPServerConfig(command="unused"),
            }
        ),
        working_dir=tmp_path,
        config_dir=tmp_path / "config",
        enable_async_subagents=False,
    )

    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            await runtime.agent.run("test", deps=runtime.ctx)

    assert "one_collide" in visible_tool_names
    assert "two_collide" in visible_tool_names


async def test_create_tui_runtime_honors_custom_and_empty_mcp_prefixes(tmp_path: Path, monkeypatch) -> None:
    """Direct MCP prefixes may be customized or disabled per server."""

    def collide() -> str:
        return "ok"

    mcp_servers = [
        FunctionToolset([collide], id="custom"),
        FunctionToolset([collide], id="unprefixed"),
    ]
    monkeypatch.setattr(runtime_module, "build_mcp_servers", lambda *_args, **_kwargs: mcp_servers)
    visible_tool_names: set[str] = set()

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible_tool_names.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_tui_runtime(
        config=YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4")),
        mcp_config=MCPConfig(
            servers={
                "custom": MCPServerConfig(command="unused", prefix="docs"),
                "unprefixed": MCPServerConfig(command="unused", prefix=""),
            }
        ),
        working_dir=tmp_path,
        config_dir=tmp_path / "config",
        enable_async_subagents=False,
    )

    async with runtime:
        with runtime.agent.override(model=FunctionModel(respond)):
            await runtime.agent.run("test", deps=runtime.ctx)

    assert "docs_collide" in visible_tool_names
    assert "collide" in visible_tool_names
    assert "custom_collide" not in visible_tool_names
    assert "unprefixed_collide" not in visible_tool_names
    assert "_collide" not in visible_tool_names


async def test_create_tui_runtime_skips_unavailable_optional_direct_mcp(tmp_path: Path) -> None:
    """An unavailable optional MCP server must not block direct-mode startup."""
    config = YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4"))
    mcp_config = MCPConfig(
        servers={
            "offline": MCPServerConfig(
                transport="stdio",
                command=sys.executable,
                args=["-c", "pass"],
                required=False,
            ),
        }
    )
    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=tmp_path,
        config_dir=tmp_path / "config",
        enable_async_subagents=False,
    )

    async with runtime:
        direct_toolsets = [toolset for toolset in runtime.agent.toolsets if isinstance(toolset, PrefixedToolset)]
        assert len(direct_toolsets) == 1
        assert direct_toolsets[0].prefix == "offline"


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
    assert ModelCapability.youtube_url in runtime.ctx.model_cfg.capabilities


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


async def test_create_tui_runtime_enables_codeact_by_default_and_allows_disabling(tmp_path: Path) -> None:
    visible_tools: list[set[str]] = []
    run_code_descriptions: list[str] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible_tools.append({tool.name for tool in info.function_tools})
        run_code_descriptions.extend(tool.description for tool in info.function_tools if tool.name == "run_code")
        return ModelResponse(parts=[TextPart(content="ok")])

    for enable_codeact in (True, False):
        runtime = create_tui_runtime(
            config=YaacliConfig(
                general=GeneralConfig(model="openai-chat:gpt-4"),
                tools=ToolsConfig(enable_codeact=enable_codeact),
            ),
            working_dir=tmp_path,
            enable_async_subagents=False,
        )
        async with runtime:
            with runtime.agent.override(model=FunctionModel(respond)):
                await runtime.agent.run("test", deps=runtime.ctx)

    assert {"run_code", "run_program"} <= visible_tools[0]
    assert {"run_code", "run_program"}.isdisjoint(visible_tools[1])
    assert len(run_code_descriptions) == 1
    shell_tool_names = {tool.name for tool in runtime_module.shell_tools} | {MonitoredShellTool.name}
    assert all(f"{shell_tool}(" not in run_code_descriptions[0] for shell_tool in shell_tool_names)


def test_create_tui_runtime_requires_explicit_user_input_support(tmp_path: Path) -> None:
    config = YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4"))

    default_runtime = create_tui_runtime(config=config, working_dir=tmp_path)
    interactive_runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
        enable_user_input=True,
    )

    assert default_runtime.core_toolset is not None
    assert interactive_runtime.core_toolset is not None
    assert "ask_user_question" not in default_runtime.core_toolset._tool_classes
    assert "ask_user_question" in interactive_runtime.core_toolset._tool_classes
    assert interactive_runtime.ctx.self_fork_agent is not None
    self_fork_tool_names = [
        name
        for toolset in interactive_runtime.ctx.self_fork_agent._user_toolsets
        if isinstance(toolset, Toolset)
        for name in toolset.tool_names
    ]
    assert "ask_user_question" not in self_fork_tool_names


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
    assert "wait_subagent" not in runtime.core_toolset._tool_classes
    assert "shell_monitor" in runtime.core_toolset._tool_classes
    assert SpawnDelegateTool.name == "spawn_delegate"
    assert SteerSubagentTool.name == "steer_subagent"
    assert WaitSubagentTool.name == "wait_subagent"


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
    assert runtime.core_toolset._tool_classes["wait_subagent"] is WaitSubagentTool
    assert "spawn_delegate" not in runtime.core_toolset._tool_classes
    assert DELEGATE_BACKEND_TOOL_NAME in runtime.core_toolset._tool_classes

    async with runtime:
        runtime.env.background_monitor.set_core_toolset(runtime.core_toolset)
        run_ctx = MagicMock()
        run_ctx.deps = runtime.ctx

        visible_tools = await runtime.core_toolset.get_tools(run_ctx)
        assert "delegate" in visible_tools
        assert "spawn_delegate" not in visible_tools
        assert "wait_subagent" not in visible_tools
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
    assert runtime.core_toolset._tool_classes["wait_subagent"] is WaitSubagentTool
    assert runtime.core_toolset._tool_classes["delegate"] is not AsyncDelegateTool
    assert DELEGATE_BACKEND_TOOL_NAME not in runtime.core_toolset._tool_classes

    async with runtime:
        runtime.env.background_monitor.set_core_toolset(runtime.core_toolset)
        run_ctx = MagicMock()
        run_ctx.deps = runtime.ctx

        visible_tools = await runtime.core_toolset.get_tools(run_ctx)
        assert "delegate" in visible_tools
        assert "spawn_delegate" in visible_tools
        assert "wait_subagent" not in visible_tools
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
        assert "Use asynchronous delegation only for bounded work" in spawn_instruction


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
