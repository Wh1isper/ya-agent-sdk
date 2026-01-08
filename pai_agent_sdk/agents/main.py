"""Main agent factory for creating configured agents.

This module provides the create_agent context manager for building agents
with proper environment and context lifecycle management.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import jinja2
from pydantic_ai import Agent
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.models import Model
from typing_extensions import TypeVar

from pai_agent_sdk.agents.compact import create_compact_filter
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.context import AgentContext, ModelConfig, ResumableState, ToolConfig
from pai_agent_sdk.environment.base import Environment
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter
from pai_agent_sdk.toolsets.core.base import BaseTool, GlobalHooks, Toolset

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.toolsets import AbstractToolset

# =============================================================================
# Type Variables
# =============================================================================

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext)
OutputT = TypeVar("OutputT", default=str)


# =============================================================================
# Agent Runtime
# =============================================================================


@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT]):
    """Container for agent runtime components.

    This dataclass holds all the components needed to run an agent,
    providing a clean interface for accessing the environment, context,
    and agent instance.

    Attributes:
        env: The environment instance managing resources.
        ctx: The agent context for session state.
        agent: The configured pydantic-ai Agent instance.

    Example:
        async with create_agent("openai:gpt-4") as runtime:
            result = await runtime.agent.run("Hello", deps=runtime.ctx)
            print(result.output)
    """

    env: Environment
    ctx: AgentDepsT
    agent: Agent[AgentDepsT, OutputT]
    core_toolset: Toolset[AgentDepsT] | None


# =============================================================================
# System Prompt Loading
# =============================================================================


def _load_system_prompt(template_vars: dict[str, Any] | None = None) -> str:
    """Load and render system prompt from the prompts directory.

    Args:
        template_vars: Variables to pass to Jinja2 template.

    Returns:
        Rendered system prompt string, or empty string if file not found.
    """
    prompt_path = Path(__file__).parent / "prompts" / "main.md"
    if not prompt_path.exists():
        return ""

    template_content = prompt_path.read_text()
    if not template_content.strip():
        return ""

    # Render with Jinja2
    env = jinja2.Environment(autoescape=False)  # noqa: S701
    template = env.from_string(template_content)
    return template.render(**(template_vars or {}))


# =============================================================================
# Agent Factory
# =============================================================================


@asynccontextmanager
async def create_agent(
    model: str | Model,
    *,
    # --- Model Configuration ---
    model_settings: ModelSettings | None = None,
    output_type: type[OutputT] = str,  # type: ignore[assignment]
    # --- Environment ---
    env: Environment | type[Environment] = LocalEnvironment,
    env_kwargs: dict[str, Any] | None = None,
    # --- Context ---
    context_type: type[AgentDepsT] = AgentContext,  # type: ignore[assignment]
    model_cfg: ModelConfig | None = None,
    tool_config: ToolConfig | None = None,
    extra_context_kwargs: dict[str, Any] | None = None,
    state: ResumableState | None = None,
    # --- Toolset ---
    tools: Sequence[type[BaseTool]] | None = None,
    toolsets: Sequence[AbstractToolset[Any]] | None = None,
    pre_hooks: dict[str, Any] | None = None,
    post_hooks: dict[str, Any] | None = None,
    global_hooks: GlobalHooks | None = None,
    toolset_max_retries: int = 3,
    toolset_timeout: float | None = None,
    skip_unavailable_tools: bool = True,
    # --- Compact Filter ---
    compact_model: str | Model | None = None,
    compact_model_settings: ModelSettings | None = None,
    compact_model_cfg: ModelConfig | None = None,
    # --- Agent ---
    agent_tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    system_prompt_template_vars: dict[str, Any] | None = None,
    history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    retries: int = 1,
    output_retries: int = 3,
    defer_model_check: bool = False,
    end_strategy: str = "exhaustive",
) -> AsyncIterator[AgentRuntime[AgentDepsT, OutputT]]:
    """Create and configure an agent with managed lifecycle.

    This context manager handles the full lifecycle of Environment, AgentContext,
    and Agent creation. It yields an AgentRuntime containing all three components.

    Args:
        model: Model string (e.g., "openai:gpt-4") or Model instance.

        model_settings: Optional model settings for inference configuration.
        output_type: Expected output type for the agent. Defaults to str.

        env: Environment instance or class. Defaults to LocalEnvironment.
        env_kwargs: Keyword arguments for Environment instantiation.

        context_type: AgentContext subclass to use. Defaults to AgentContext.
        model_cfg: ModelConfig for context window and capability settings.
        tool_config: ToolConfig for API keys and tool-specific settings.
        extra_context_kwargs: Additional kwargs passed to context_type constructor.
        state: ResumableState to restore session from. Defaults to None.

        tools: Sequence of BaseTool classes to include in the toolset.
        toolsets: Additional AbstractToolset instances to include.
        pre_hooks: Dict mapping tool names to pre-hook functions.
        post_hooks: Dict mapping tool names to post-hook functions.
        global_hooks: GlobalHooks instance for all tools.
        toolset_max_retries: Max retries for tool execution. Defaults to 3.
        toolset_timeout: Default timeout for tool execution.
        skip_unavailable_tools: Skip tools where is_available() returns False.

        compact_model: Model for compact filter. Falls back to AgentSettings.
        compact_model_settings: Model settings for compact filter.
        compact_model_cfg: ModelConfig for compact filter. Defaults to main model_cfg.

        agent_tools: Additional tools to pass directly to Agent (pydantic-ai Tool objects).
        system_prompt: Custom system prompt(s). If None, loads from main.md.
        system_prompt_template_vars: Variables for Jinja2 template rendering.
        history_processors: Sequence of history processor functions.
        retries: Number of retries for agent run. Defaults to 1.
        output_retries: Number of retries for output parsing. Defaults to 3.
        defer_model_check: Defer model validation. Defaults to False.
        end_strategy: Strategy for ending agent run. Defaults to "exhaustive".

    Yields:
        AgentRuntime containing env, ctx, and agent.

    Example:
        Basic usage::

            async with create_agent("openai:gpt-4") as runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
                print(result.output)

        With custom tools and configuration::

            async with create_agent(
                "anthropic:claude-3-5-sonnet",
                tools=[ReadFileTool, WriteFileTool],
                model_cfg=ModelConfig(context_window=200000),
                global_hooks=GlobalHooks(pre=my_pre_hook),
            ) as runtime:
                result = await runtime.agent.run("Read config.json", deps=runtime.ctx)

        With custom environment::

            async with create_agent(
                "openai:gpt-4",
                env=DockerEnvironment,
                env_kwargs={"image": "python:3.11"},
            ) as runtime:
                result = await runtime.agent.run("Run tests", deps=runtime.ctx)
    """
    async with AsyncExitStack() as stack:
        # --- Environment Setup ---
        if isinstance(env, Environment):
            entered_env = env
            # If already an instance, enter it
            await stack.enter_async_context(env)
        else:
            # Create and enter new environment instance
            entered_env = await stack.enter_async_context(env(**(env_kwargs or {})))

        # --- Build Configs ---
        effective_model_cfg = model_cfg or ModelConfig()
        effective_tool_config = tool_config or ToolConfig()

        # --- Context Setup ---
        ctx = await stack.enter_async_context(
            context_type(
                file_operator=entered_env.file_operator,
                shell=entered_env.shell,
                resources=entered_env.resources,
                model_cfg=effective_model_cfg,
                tool_config=effective_tool_config,
                **(extra_context_kwargs or {}),
            ).with_state(state)
        )

        # --- Toolset Setup ---
        all_toolsets: list[AbstractToolset[Any]] = []
        core_toolset: Toolset[AgentDepsT] | None = None

        # Create Toolset from BaseTool classes if provided
        if tools:
            core_toolset = Toolset(
                ctx,
                tools=tools,
                pre_hooks=pre_hooks,
                post_hooks=post_hooks,
                global_hooks=global_hooks,
                max_retries=toolset_max_retries,
                timeout=toolset_timeout,
                skip_unavailable=skip_unavailable_tools,
            )
            all_toolsets.append(core_toolset)

        # Add user-provided toolsets
        if toolsets:
            all_toolsets.extend(toolsets)

        # Add environment toolsets
        all_toolsets.extend(entered_env.toolsets)

        # --- System Prompt ---
        effective_system_prompt: str | Sequence[str]
        if system_prompt is not None:
            effective_system_prompt = system_prompt
        else:
            # Load from template
            loaded_prompt = _load_system_prompt(system_prompt_template_vars)
            effective_system_prompt = loaded_prompt if loaded_prompt else ""

        # --- History Processors ---
        # Combine context's processors with built-in and user-provided ones
        all_processors: list[HistoryProcessor[AgentDepsT]] = [
            *ctx.get_history_processors(),
            create_compact_filter(
                model=compact_model,
                model_settings=compact_model_settings,
                model_cfg=compact_model_cfg or effective_model_cfg,
            ),
            create_environment_instructions_filter(entered_env),
            create_system_prompt_filter(system_prompt=effective_system_prompt),
        ]
        if history_processors:
            all_processors.extend(history_processors)

        # --- Create Agent ---
        agent: Agent[AgentDepsT, OutputT] = Agent(
            model=infer_model(model) if isinstance(model, str) else model,
            system_prompt=effective_system_prompt,
            model_settings=model_settings,
            deps_type=context_type,
            output_type=output_type,
            tools=agent_tools or (),
            toolsets=all_toolsets if all_toolsets else None,
            history_processors=all_processors if all_processors else None,
            retries=retries,
            output_retries=output_retries,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,  # type: ignore[arg-type]
        )

        yield AgentRuntime(env=entered_env, ctx=ctx, agent=agent, core_toolset=core_toolset)
