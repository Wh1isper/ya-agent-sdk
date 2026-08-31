from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Toolset as ToolsetCapability
from pydantic_ai.capabilities import ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    FilesystemCapability,
    ShellCapability,
    ToolSupersessionCapability,
    ToolTimeoutCapability,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset


def test_tool_timeout_defaults_to_ten_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YA_AGENT_TOOL_TIMEOUT_SECONDS", raising=False)

    capability = ToolTimeoutCapability()

    assert capability.timeout == 600.0


def test_tool_timeout_reads_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_AGENT_TOOL_TIMEOUT_SECONDS", "900.5")

    capability = ToolTimeoutCapability()

    assert capability.timeout == 900.5


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "nan", "inf"])
def test_tool_timeout_rejects_invalid_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("YA_AGENT_TOOL_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="YA_AGENT_TOOL_TIMEOUT_SECONDS must be a positive finite number"):
        ToolTimeoutCapability()


def test_explicit_tool_timeout_ignores_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_AGENT_TOOL_TIMEOUT_SECONDS", "invalid")

    capability = ToolTimeoutCapability(timeout=30)

    assert capability.timeout == 30


@pytest.mark.parametrize(
    ("tool_timeout", "expected_timeout"),
    [
        (None, 600.0),
        (30.0, 30.0),
        (900.0, 600.0),
    ],
)
async def test_tool_timeout_is_a_ceiling_and_preserves_shorter_tool_deadlines(
    monkeypatch: pytest.MonkeyPatch,
    tool_timeout: float | None,
    expected_timeout: float,
) -> None:
    observed_timeouts: list[float | None] = []

    @asynccontextmanager
    async def capture_timeout(delay: float | None) -> AsyncGenerator[None, None]:
        observed_timeouts.append(delay)
        yield

    async def handler(args: ValidatedToolArgs) -> str:
        assert args == {}
        return "ok"

    monkeypatch.delenv("YA_AGENT_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.tool_policy.policy.asyncio.timeout",
        capture_timeout,
    )
    capability = ToolTimeoutCapability()
    result = await capability.wrap_tool_execute(
        cast(RunContext[AgentContext], None),
        call=ToolCallPart(tool_name="test_tool", args={}, tool_call_id="call-1"),
        tool_def=ToolDefinition(name="test_tool", timeout=tool_timeout),
        args=cast(ValidatedToolArgs, {}),
        handler=cast(WrapToolExecuteHandler, handler),
    )

    assert result == "ok"
    assert observed_timeouts == [expected_timeout]


class _TimeoutScope:
    def __init__(self, *, expired: bool) -> None:
        self._expired = expired

    async def __aenter__(self) -> _TimeoutScope:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def expired(self) -> bool:
        return self._expired


async def test_capability_owned_timeout_raises_model_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(args: ValidatedToolArgs) -> str:
        del args
        raise TimeoutError

    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.tool_policy.policy.asyncio.timeout",
        lambda delay: _TimeoutScope(expired=True),
    )
    capability = ToolTimeoutCapability(timeout=30)

    with pytest.raises(ModelRetry) as exc_info:
        await capability.wrap_tool_execute(
            cast(RunContext[AgentContext], None),
            call=ToolCallPart(tool_name="slow_tool", args={}, tool_call_id="call-1"),
            tool_def=ToolDefinition(name="slow_tool"),
            args=cast(ValidatedToolArgs, {}),
            handler=cast(WrapToolExecuteHandler, handler),
        )

    assert str(exc_info.value) == (
        "Tool 'slow_tool' exceeded the execution timeout of 30 seconds. "
        "The call was cancelled and may have produced partial side effects. Inspect current state "
        "before retrying, or continue with another approach."
    )
    assert isinstance(exc_info.value.__cause__, TimeoutError)


async def test_tool_owned_timeout_remains_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    original = TimeoutError("tool-owned timeout")

    async def handler(args: ValidatedToolArgs) -> str:
        del args
        raise original

    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.tool_policy.policy.asyncio.timeout",
        lambda delay: _TimeoutScope(expired=False),
    )
    capability = ToolTimeoutCapability(timeout=30)

    with pytest.raises(TimeoutError, match="tool-owned timeout") as exc_info:
        await capability.wrap_tool_execute(
            cast(RunContext[AgentContext], None),
            call=ToolCallPart(tool_name="slow_tool", args={}, tool_call_id="call-1"),
            tool_def=ToolDefinition(name="slow_tool"),
            args=cast(ValidatedToolArgs, {}),
            handler=cast(WrapToolExecuteHandler, handler),
        )

    assert exc_info.value is original


class _SupersededTool(BaseTool):
    name = "superseded"
    description = "A less capable tool"
    superseded_by_tags = frozenset({"advanced"})

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        del ctx
        return "superseded guidance"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        del ctx
        return "superseded"


class _SupersedingTool(BaseTool):
    name = "superseding"
    description = "A more capable tool"
    tags = frozenset({"advanced"})

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        del ctx
        return "superseding"


class _UnavailableSupersedingTool(_SupersedingTool):
    name = "unavailable_superseding"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        del ctx
        return False


class _CountingToolset(Toolset[AgentContext]):
    get_tools_calls = 0

    async def get_tools(self, ctx: RunContext[AgentContext]):
        self.get_tools_calls += 1
        return await super().get_tools(ctx)


@pytest.mark.parametrize("reverse", [False, True])
async def test_cross_toolset_supersession_requires_explicit_capability(
    tmp_path: Path,
    reverse: bool,
) -> None:
    captured_names: set[str] = set()
    captured_instructions = ""

    def model_function(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal captured_names, captured_instructions
        captured_names = {tool.name for tool in info.function_tools}
        captured_instructions = info.instructions or ""
        return ModelResponse(parts=[TextPart(content="ok")])

    capabilities = [
        ToolsetCapability(Toolset(tools=[_SupersededTool], toolset_id="basic"), id="basic"),
        ToolsetCapability(Toolset(tools=[_SupersedingTool], toolset_id="advanced"), id="advanced"),
    ]
    if reverse:
        capabilities.reverse()
    runtime = create_agent(
        FunctionModel(function=model_function),
        env=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path),
        capabilities=capabilities,
    )

    async with runtime:
        await runtime.agent.run("test local-only supersession", deps=runtime.ctx)

    assert captured_names == {"superseded", "superseding"}
    assert "superseded guidance" in captured_instructions


@pytest.mark.parametrize("reverse", [False, True])
async def test_tool_supersession_resolves_across_capability_toolsets(
    tmp_path: Path,
    reverse: bool,
) -> None:
    captured_names: set[str] = set()
    captured_instructions = ""

    def model_function(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal captured_names, captured_instructions
        captured_names = {tool.name for tool in info.function_tools}
        captured_instructions = info.instructions or ""
        return ModelResponse(parts=[TextPart(content="ok")])

    tool_capabilities = [
        ToolsetCapability(Toolset(tools=[_SupersededTool], toolset_id="basic"), id="basic"),
        ToolsetCapability(Toolset(tools=[_SupersedingTool], toolset_id="advanced"), id="advanced"),
    ]
    if reverse:
        tool_capabilities.reverse()
    runtime = create_agent(
        FunctionModel(function=model_function),
        env=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path),
        capabilities=[*tool_capabilities, ToolSupersessionCapability()],
    )

    async with runtime:
        await runtime.agent.run("test supersession", deps=runtime.ctx)

    assert captured_names == {"superseding"}
    assert "superseded guidance" not in captured_instructions
    assert runtime.ctx.tool_tags == {"advanced"}


async def test_tool_supersession_reuses_one_tool_snapshot_for_instructions(tmp_path: Path) -> None:
    toolset = _CountingToolset(tools=[_SupersededTool, _SupersedingTool], toolset_id="counting")
    runtime = create_agent(
        FunctionModel(function=lambda _messages, _info: ModelResponse(parts=[TextPart(content="ok")])),
        env=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path),
        capabilities=[
            ToolsetCapability(toolset, id="counting"),
            ToolSupersessionCapability(),
        ],
    )

    async with runtime:
        await runtime.agent.run("test one snapshot", deps=runtime.ctx)

    assert toolset.get_tools_calls == 1


async def test_unavailable_tool_does_not_supersede_other_capability_tools(tmp_path: Path) -> None:
    captured_names: set[str] = set()

    def model_function(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal captured_names
        captured_names = {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_agent(
        FunctionModel(function=model_function),
        env=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path),
        capabilities=[
            ToolsetCapability(Toolset(tools=[_SupersededTool], toolset_id="basic"), id="basic"),
            ToolsetCapability(
                Toolset(tools=[_UnavailableSupersedingTool], toolset_id="advanced"),
                id="advanced",
            ),
            ToolSupersessionCapability(),
        ],
    )

    async with runtime:
        await runtime.agent.run("test unavailable superseder", deps=runtime.ctx)

    assert captured_names == {"superseded"}
    assert runtime.ctx.tool_tags == set()


async def test_shell_supersedes_redundant_filesystem_tools_and_guidance(tmp_path: Path) -> None:
    captured_names: set[str] = set()
    captured_instructions = ""

    def model_function(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal captured_names, captured_instructions
        captured_names = {tool.name for tool in info.function_tools}
        captured_instructions = info.instructions or ""
        return ModelResponse(parts=[TextPart(content="ok")])

    runtime = create_agent(
        FunctionModel(function=model_function),
        env=LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path),
        capabilities=[
            FilesystemCapability(),
            ShellCapability(),
            ToolSupersessionCapability(),
        ],
    )

    async with runtime:
        await runtime.agent.run("test shell supersession", deps=runtime.ctx)

    assert {"delete", "move", "copy", "mkdir"}.isdisjoint(captured_names)
    assert {"glob", "grep", "ls", "view", "edit", "write", "shell_exec"} <= captured_names
    assert '<tool-instruction name="delete">' not in captured_instructions
    assert "shell" in runtime.ctx.tool_tags


class _SlowTool(BaseTool):
    name = "slow_tool"
    description = "Wait longer than the generic tool timeout"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        del ctx
        await asyncio.sleep(1)
        return "unexpected"


async def test_agent_receives_timeout_retry_prompt_and_continues(tmp_path: Path) -> None:
    saw_retry_prompt = False

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal saw_retry_prompt
        retry_prompts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart) and part.tool_name == _SlowTool.name
        ]
        if retry_prompts:
            saw_retry_prompt = True
            assert "exceeded the execution timeout" in str(retry_prompts[-1].content)
            return ModelResponse(parts=[TextPart(content="continued after timeout")])
        return ModelResponse(parts=[ToolCallPart(tool_name=_SlowTool.name, args={}, tool_call_id="slow-call")])

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        FunctionModel(function=model_function),
        env=env,
        capabilities=[
            ToolsetCapability(Toolset(tools=[_SlowTool], toolset_id="slow"), id="slow"),
            ToolTimeoutCapability(timeout=0.01),
        ],
    )

    async with runtime:
        result = await runtime.agent.run("run the slow tool", deps=runtime.ctx)

    assert result.output == "continued after timeout"
    assert saw_retry_prompt is True
