from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    BinaryContent,
    CallDeferred,
    DeferredToolResults,
    FunctionToolset,
    ModelRetry,
    RunContext,
    Tool,
    ToolFailed,
    ToolReturn,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, HandleDeferredToolCalls
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset, ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage
from ya_agent_environment import Environment
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.codeact import CodeActConfig
from ya_agent_sdk.codeact import toolset as codeact_toolset
from ya_agent_sdk.codeact.programs import validate_static_tool_references
from ya_agent_sdk.codeact.toolset import (
    CodeActToolset,
    _bounded_json_size,
    _build_catalog,
    _ExecutionBudget,
    _unwrap_tool_return,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.base import BaseTool
from ya_agent_sdk.toolsets.core.base import Toolset


class AddTool(BaseTool):
    name = "add"
    description = "Add two integers."
    codeact = True

    async def call(self, ctx: RunContext[AgentContext], *, a: int, b: int) -> int:
        return a + b


class MainOnlyCodeActTool(BaseTool):
    name = "main_only_codeact"
    description = "Available only to the root main agent."
    codeact = True
    main_agent_only = True

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "main only"


class HiddenTool(BaseTool):
    name = "hidden"
    description = "A tool that is not eligible for CodeAct."

    calls = 0

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        type(self).calls += 1
        return "hidden"


@dataclass
class _DenyMutateToolset(WrapperToolset[AgentContext]):
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentContext],
        tool: ToolsetTool[AgentContext],
    ) -> Any:
        if name == "mutate":
            raise ToolFailed("outer policy denied mutate")
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)


@dataclass
class _DenyMutateCapability(AbstractCapability[AgentContext]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost", wraps=(AbstractCapability,))

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[AgentContext],
    ) -> AbstractToolset[AgentContext]:
        return _DenyMutateToolset(wrapped=toolset)


@dataclass
class _PrefixCapability(AbstractCapability[AgentContext]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost", wraps=(AbstractCapability,))

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[AgentContext],
    ) -> AbstractToolset[AgentContext]:
        return PrefixedToolset(wrapped=toolset, prefix="p")


class _EmptyEnvironment(Environment):
    async def _setup(self) -> None:
        pass

    async def _teardown(self) -> None:
        pass


def _runtime(
    tmp_path: Path,
    model_function: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
    *,
    tools: list[type[BaseTool]] | None = None,
    toolsets: list[FunctionToolset[Any]] | None = None,
    config: CodeActConfig | None = None,
    capabilities: Sequence[AbstractCapability[AgentContext]] | None = None,
):
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    return create_agent(
        FunctionModel(function=model_function),
        env=env,
        tools=tools or [],
        toolsets=toolsets or [],
        codeact=config or CodeActConfig(),
        capabilities=capabilities,
    )


def _tool_returns(messages: list[ModelMessage], name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]


def test_codeact_config_rejects_ignored_or_invalid_limits() -> None:
    with pytest.raises(ValueError, match="must enable"):
        CodeActConfig(inline=False, programs=False)
    with pytest.raises(ValueError, match="max_concurrency"):
        CodeActConfig(max_concurrency=0)
    for timeout_seconds in (0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="timeout_seconds"):
            CodeActConfig(timeout_seconds=timeout_seconds)


async def test_run_code_orchestrates_tools_from_multiple_toolsets(tmp_path: Path) -> None:
    async def multiply(ctx: RunContext[AgentContext], *, value: int, factor: int) -> int:
        return value * factor

    external = FunctionToolset(
        [Tool(multiply, metadata={"codeact": True})],
        id="external",
    )
    model_calls = 0
    seen_tool_names: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls, seen_tool_names
        model_calls += 1
        seen_tool_names = [tool.name for tool in info.function_tools]
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "total = await add(a=2, b=3)\nawait multiply(value=total, factor=4)"},
                        tool_call_id="code-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function, tools=[AddTool], toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("orchestrate", deps=runtime.ctx)

    assert {"add", "multiply", "run_code", "run_program"} <= set(seen_tool_names)
    returned = _tool_returns(result.all_messages(), "run_code")
    assert len(returned) == 1
    assert returned[0].content == 20
    metadata = returned[0].metadata["codeact"]
    assert metadata["status"] == "completed"
    assert [call["tool_name"] for call in metadata["tool_calls"]] == ["add", "multiply"]
    assert all("args_sha256" in call and "result_sha256" in call for call in metadata["tool_calls"])


async def test_run_code_preserves_state_only_within_one_run(tmp_path: Path) -> None:
    snippets = [
        {"code": "counter = 1\ncounter"},
        {"code": "counter += 1\ncounter"},
        {"code": "counter = 10\ncounter", "restart": True},
    ]
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        if model_calls < len(snippets):
            args = snippets[model_calls]
            model_calls += 1
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_code", args=args, tool_call_id=f"code-{model_calls}")]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function)
    async with runtime:
        result = await runtime.agent.run("state", deps=runtime.ctx)

    assert [part.content for part in _tool_returns(result.all_messages(), "run_code")] == [1, 2, 10]


async def test_run_program_uses_file_operator_and_fresh_session(tmp_path: Path) -> None:
    program = tmp_path / "counter.codeact.py"
    program.write_text(
        "counter = 0\n\n"
        "async def main(inputs):\n"
        "    global counter\n"
        "    counter += 1\n"
        "    return counter + inputs['offset']\n",
        encoding="utf-8",
    )
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        if model_calls < 2:
            model_calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_program",
                        args={"path": "counter.codeact.py", "inputs": {"offset": model_calls}},
                        tool_call_id=f"program-{model_calls}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function)
    async with runtime:
        result = await runtime.agent.run("program", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_program")
    assert [part.content for part in returned] == [2, 3]
    assert all(part.metadata["codeact"]["source_path"] == "counter.codeact.py" for part in returned)
    assert returned[0].metadata["codeact"]["source_sha256"] == returned[1].metadata["codeact"]["source_sha256"]


async def test_ineligible_tool_is_not_dispatched_from_codeact(tmp_path: Path) -> None:
    HiddenTool.calls = 0
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await hidden()"},
                        tool_call_id="hidden-code",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="recovered")])

    runtime = _runtime(tmp_path, model_function, tools=[HiddenTool])
    async with runtime:
        result = await runtime.agent.run("try hidden", deps=runtime.ctx)

    assert HiddenTool.calls == 0
    assert any(
        isinstance(part, RetryPromptPart) and "unavailable function names" in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_failure_after_side_effect_is_terminal_not_retry(tmp_path: Path) -> None:
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext]) -> str:
        nonlocal mutations
        mutations += 1
        return "mutated"

    external = FunctionToolset([Tool(mutate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await mutate()\nraise RuntimeError('after mutation')"},
                        tool_call_id="mutating-code",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("mutate once", deps=runtime.ctx)

    assert mutations == 1
    returned = _tool_returns(result.all_messages(), "run_code")
    assert len(returned) == 1
    assert returned[0].outcome == "failed"
    payload = json.loads(str(returned[0].content))
    assert payload["codeact"]["status"] == "failed"
    assert payload["codeact"]["tool_call_count"] == 1
    assert not any(
        isinstance(part, RetryPromptPart)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_call_and_concurrency_limits_are_enforced(tmp_path: Path) -> None:
    active = 0
    max_active = 0
    model_calls = 0

    async def slow(ctx: RunContext[AgentContext], *, value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
            return value
        finally:
            active -= 1

    external = FunctionToolset([Tool(slow, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import asyncio\nawait asyncio.gather(*(slow(value=i) for i in range(4)))"},
                        tool_call_id="limited-code",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_tool_calls=3, max_concurrency=2),
    )
    async with runtime:
        result = await runtime.agent.run("limits", deps=runtime.ctx)

    assert max_active <= 2
    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    payload = json.loads(str(returned[0].content))
    assert payload["codeact"]["tool_call_count"] <= 3


def test_program_preflight_rejects_top_level_execution() -> None:
    from ya_agent_sdk.codeact.programs import validate_program_source

    with pytest.raises(ValueError, match="Executable module-level statement"):
        validate_program_source("print('side effect')\nasync def main(inputs):\n    return inputs\n")
    with pytest.raises(ValueError, match="exact signature"):
        validate_program_source("async def main(value, extra):\n    return value\n")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os\nasync def main(inputs):\n    return inputs\n", "ambient-capability module 'os'"),
        (
            "from pathlib import Path\nasync def main(inputs):\n    return inputs\n",
            "ambient-capability module 'pathlib'",
        ),
        (
            "async def main(inputs):\n    return open(inputs['path'])\n",
            "reserved ambient builtin name 'open'",
        ),
        (
            "async def main(inputs):\n    file_op = open\n    return file_op(inputs['path'])\n",
            "reserved ambient builtin name 'open'",
        ),
        (
            "def open(value):\n    return value\n\nasync def main(inputs):\n    return open(inputs)\n",
            "reserved ambient builtin name 'open'",
        ),
        (
            "async def main(inputs):\n    return eval(inputs['expression'])\n",
            "reserved ambient builtin name 'eval'",
        ),
    ],
)
def test_program_preflight_rejects_reserved_ambient_names_and_modules(source: str, expected: str) -> None:
    from ya_agent_sdk.codeact.programs import validate_program_source

    with pytest.raises(ValueError, match=expected):
        validate_program_source(source)


def test_program_preflight_allows_pure_python_and_supported_imports() -> None:
    from ya_agent_sdk.codeact.programs import validate_program_source

    validate_program_source(
        "import asyncio\n\n"
        "async def main(inputs):\n"
        "    values = await asyncio.gather(*(identity(value=value) for value in inputs['values']))\n"
        "    return sorted(values), len(values)\n"
    )


async def test_cancellation_awaits_nested_tool_cleanup(tmp_path: Path) -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocking(ctx: RunContext[AgentContext]) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.03)
            cleaned.set()

    external = FunctionToolset([Tool(blocking, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await blocking()"},
                        tool_call_id="blocking-code",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="unexpected")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        run_task = asyncio.create_task(runtime.agent.run("block", deps=runtime.ctx))
        await asyncio.wait_for(started.wait(), timeout=2)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert cleaned.is_set()


async def test_sanitized_tool_name_collisions_fail_closed(tmp_path: Path) -> None:
    async def first(ctx: RunContext[AgentContext]) -> int:
        return 1

    async def second(ctx: RunContext[AgentContext]) -> int:
        return 2

    colliding = FunctionToolset([
        Tool(first, name="a-b", metadata={"codeact": True}),
        Tool(second, name="a.b", metadata={"codeact": True}),
    ])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="unused")])

    runtime = _runtime(tmp_path, model_function, toolsets=[colliding])
    async with runtime:
        with pytest.raises(Exception, match="CodeAct tool-name collision"):
            await runtime.agent.run("collision", deps=runtime.ctx)


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        (
            "large.codeact.py",
            ("async def main(inputs):\n    return '" + "x" * 200 + "'\n").encode(),
            "max_source_bytes=128",
        ),
        ("invalid.codeact.py", b"async def main(inputs):\n    return '\xff'\n", "strict UTF-8"),
        ("ordinary.py", b"async def main(inputs):\n    return inputs\n", "end in .codeact.py"),
    ],
)
async def test_program_source_limits_are_checked_before_sandbox(
    tmp_path: Path,
    filename: str,
    content: bytes,
    expected: str,
) -> None:
    (tmp_path / filename).write_bytes(content)
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_program",
                        args={"path": filename},
                        tool_call_id="invalid-program",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        config=CodeActConfig(max_source_bytes=128),
    )
    async with runtime:
        result = await runtime.agent.run("invalid program", deps=runtime.ctx)

    assert any(
        isinstance(part, RetryPromptPart) and expected in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_nested_calls_preserve_outermost_policy_wrapper(tmp_path: Path) -> None:
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext]) -> str:
        nonlocal mutations
        mutations += 1
        return "mutated"

    external = FunctionToolset([Tool(mutate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="mutate", args={}, tool_call_id="direct-mutate")])
        if model_calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await mutate()"},
                        tool_call_id="nested-mutate",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        capabilities=[_DenyMutateCapability()],
    )
    async with runtime:
        result = await runtime.agent.run("try both paths", deps=runtime.ctx)

    assert mutations == 0
    direct = _tool_returns(result.all_messages(), "mutate")
    nested = _tool_returns(result.all_messages(), "run_code")
    assert direct[0].outcome == "failed"
    assert nested[0].outcome == "failed"
    assert "outer policy denied mutate" in str(direct[0].content)
    assert "outer policy denied mutate" in str(nested[0].content)


async def test_nested_approval_uses_current_capability_handler(tmp_path: Path) -> None:
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext], *, value: int) -> int:
        nonlocal mutations
        mutations += 1
        return value

    external = FunctionToolset([Tool(mutate, requires_approval=True, metadata={"codeact": True})])
    approval = HandleDeferredToolCalls[AgentContext](
        handler=lambda ctx, requests: requests.build_results(approve_all=True)
    )

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await mutate(value=7)"},
                        tool_call_id="approved-mutate",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        capabilities=[approval],
    )
    async with runtime:
        result = await runtime.agent.run("approve nested call", deps=runtime.ctx)

    assert mutations == 1
    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "success"
    assert returned[0].content == 7


async def test_nested_result_limit_is_enforced_before_sandbox_crossing(tmp_path: Path) -> None:
    calls = 0
    model_calls = 0

    async def huge(ctx: RunContext[AgentContext]) -> str:
        nonlocal calls
        calls += 1
        return "x" * 1_000_000

    external = FunctionToolset([Tool(huge, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "value = await huge()\nlen(value)"},
                        tool_call_id="huge-result",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_output_bytes=128),
    )
    async with runtime:
        result = await runtime.agent.run("do not cross a huge value", deps=runtime.ctx)

    assert calls == 1
    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    assert "nested tool result exceeds max_output_bytes=128" in str(returned[0].content)


async def test_cumulative_nested_result_limit_is_atomic(tmp_path: Path) -> None:
    model_calls = 0

    async def chunk(ctx: RunContext[AgentContext], *, value: int) -> str:
        return str(value) * 80

    external = FunctionToolset([Tool(chunk, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import asyncio\nawait asyncio.gather(chunk(value=1), chunk(value=2))"},
                        tool_call_id="cumulative-results",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_output_bytes=128),
    )
    async with runtime:
        result = await runtime.agent.run("bound cumulative values", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    assert "cumulative nested tool results exceed max_output_bytes=128" in str(returned[0].content)
    payload = json.loads(str(returned[0].content))
    assert payload["codeact"]["nested_result_bytes"] == 82


async def test_invalid_nested_arguments_do_not_mark_side_effect_started(tmp_path: Path) -> None:
    calls = 0
    model_calls = 0

    async def counted(ctx: RunContext[AgentContext], *, value: int) -> int:
        nonlocal calls
        calls += 1
        return value

    external = FunctionToolset([Tool(counted, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await counted(value='not-an-integer')"},
                        tool_call_id="invalid-args",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="corrected")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("validate first", deps=runtime.ctx)

    assert calls == 0
    assert not _tool_returns(result.all_messages(), "run_code")
    assert any(
        isinstance(part, RetryPromptPart) and "valid integer" in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def main(inputs):\n    return inputs\n\nasync def main(inputs):\n    return inputs\n",
            "exactly one async function",
        ),
        (
            "import asyncio as __ya_codeact_inputs__\nasync def main(inputs):\n    return inputs\n",
            "reserved runtime name",
        ),
        (
            "async def main(inputs):\n    return await main(inputs)\n",
            "cannot call main",
        ),
    ],
)
def test_program_preflight_rejects_ambiguous_or_reserved_entrypoints(source: str, expected: str) -> None:
    from ya_agent_sdk.codeact.programs import validate_program_source

    with pytest.raises(ValueError, match=expected):
        validate_program_source(source)


async def test_parallel_ordered_events_keeps_nested_calls_parallel(tmp_path: Path) -> None:
    active = 0
    max_active = 0
    model_calls = 0

    async def slow(ctx: RunContext[AgentContext], *, value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
            return value
        finally:
            active -= 1

    external = FunctionToolset([Tool(slow, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import asyncio\nawait asyncio.gather(*(slow(value=i) for i in range(4)))"},
                        tool_call_id="ordered-parallel",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        with ToolManager.parallel_execution_mode("parallel_ordered_events"):
            await runtime.agent.run("parallel with ordered events", deps=runtime.ctx)

    assert max_active > 1


async def test_catalog_fingerprint_covers_canonical_name_and_return_schema() -> None:
    async def int_result(value: int) -> int:
        return value

    async def str_result(value: int) -> str:
        return str(value)

    ctx = RunContext(deps=AgentContext(), model=TestModel(), usage=RunUsage())
    canonical_a = await FunctionToolset([Tool(int_result, name="a-b", metadata={"codeact": True})]).get_tools(ctx)
    canonical_b = await FunctionToolset([Tool(int_result, name="a.b", metadata={"codeact": True})]).get_tools(ctx)
    return_a = await FunctionToolset([Tool(int_result, name="value", metadata={"codeact": True})]).get_tools(ctx)
    return_b = await FunctionToolset([Tool(str_result, name="value", metadata={"codeact": True})]).get_tools(ctx)

    assert _build_catalog(canonical_a).fingerprint != _build_catalog(canonical_b).fingerprint
    assert _build_catalog(return_a).fingerprint != _build_catalog(return_b).fingerprint


async def test_failure_messages_redact_common_secret_forms(tmp_path: Path) -> None:
    model_calls = 0

    async def leak(ctx: RunContext[AgentContext]) -> None:
        raise RuntimeError(
            "token=supersecret Bearer bearer-secret https://alice:hunter2@example.test/path\n"
            "Authorization: Basic dXNlcjpwYXNz\n"
            "Cookie: session=cookie-secret; csrf=csrf-secret"
        )

    external = FunctionToolset([Tool(leak, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await leak()"},
                        tool_call_id="secret-error",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("redact failures", deps=runtime.ctx)

    rendered = str(_tool_returns(result.all_messages(), "run_code")[0].content)
    assert "supersecret" not in rendered
    assert "bearer-secret" not in rendered
    assert "alice" not in rendered
    assert "hunter2" not in rendered
    assert "dXNlcjpwYXNz" not in rendered
    assert "cookie-secret" not in rendered
    assert "csrf-secret" not in rendered
    assert rendered.count("[REDACTED]") >= 5


async def test_outer_prefixed_toolset_can_rebind_codeact_tool_owner(tmp_path: Path) -> None:
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext]) -> str:
        nonlocal mutations
        mutations += 1
        return "mutated"

    external = FunctionToolset([Tool(mutate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            assert {"p_run_code", "p_run_program", "p_mutate"} <= {tool.name for tool in info.function_tools}
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="p_run_code",
                        args={"code": "await p_mutate()"},
                        tool_call_id="prefixed-codeact",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        capabilities=[_PrefixCapability()],
    )
    async with runtime:
        result = await runtime.agent.run("use prefixed CodeAct", deps=runtime.ctx)

    assert mutations == 1
    assert _tool_returns(result.all_messages(), "p_run_code")[0].content == "mutated"


async def test_validation_failure_does_not_expose_sensitive_input_value(tmp_path: Path) -> None:
    model_calls = 0

    async def get_secret(ctx: RunContext[AgentContext]) -> dict[str, str]:
        return {"password": "supersecret-value"}

    async def sensitive(ctx: RunContext[AgentContext], *, password: int) -> int:
        return password

    external = FunctionToolset([
        Tool(get_secret, metadata={"codeact": True}),
        Tool(sensitive, metadata={"codeact": True}),
    ])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": ("secret = await get_secret()\nawait sensitive(password=secret['password'])")},
                        tool_call_id="sensitive-validation",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("validate sensitive value", deps=runtime.ctx)

    rendered = str(_tool_returns(result.all_messages(), "run_code")[0].content)
    assert "supersecret-value" not in rendered
    assert "input_value" not in rendered
    assert "valid integer" in rendered


def test_program_preflight_rejects_executable_assignment_targets() -> None:
    from ya_agent_sdk.codeact.programs import validate_program_source

    source = "x = {}\nx[await mutate()] = 1\n\nasync def main(inputs):\n    return x\n"
    with pytest.raises(ValueError, match="simple name targets"):
        validate_program_source(source)


async def test_static_missing_tool_is_rejected_before_prior_side_effect(tmp_path: Path) -> None:
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext]) -> None:
        nonlocal mutations
        mutations += 1

    external = FunctionToolset([Tool(mutate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await mutate()\nawait definitely_missing()"},
                        tool_call_id="missing-after-mutate",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="corrected")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("preflight missing names", deps=runtime.ctx)

    assert mutations == 0
    assert any(
        isinstance(part, RetryPromptPart) and "definitely_missing" in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_inline_static_preflight_remembers_successful_definitions(tmp_path: Path) -> None:
    snippets = [
        "async def helper(value):\n    return value + 1\n42",
        "await helper(41)",
    ]
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        if model_calls < len(snippets):
            code = snippets[model_calls]
            model_calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": code},
                        tool_call_id=f"definition-{model_calls}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function)
    async with runtime:
        result = await runtime.agent.run("reuse helper", deps=runtime.ctx)

    assert [part.content for part in _tool_returns(result.all_messages(), "run_code")] == [42, 42]


async def test_program_static_preflight_does_not_inherit_inline_names(tmp_path: Path) -> None:
    (tmp_path / "isolated.codeact.py").write_text(
        "async def main(inputs):\n    await mutate()\n    return await helper()\n",
        encoding="utf-8",
    )
    mutations = 0
    model_calls = 0

    async def mutate(ctx: RunContext[AgentContext]) -> None:
        nonlocal mutations
        mutations += 1

    external = FunctionToolset([Tool(mutate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "async def helper():\n    return 1\nNone"},
                        tool_call_id="define-inline-helper",
                    )
                ]
            )
        if model_calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_program",
                        args={"path": "isolated.codeact.py"},
                        tool_call_id="isolated-program",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="corrected")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("keep program fresh", deps=runtime.ctx)

    assert mutations == 0
    assert any(
        isinstance(part, RetryPromptPart) and "helper" in str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_run_program_is_hidden_without_environment_file_operator() -> None:
    seen_tools: set[str] = set()

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = create_agent(
        FunctionModel(function=model_function),
        env=_EmptyEnvironment(),
        codeact=CodeActConfig(),
    )
    async with runtime:
        await runtime.agent.run("inspect tools", deps=runtime.ctx)

    assert "run_code" in seen_tools
    assert "run_program" not in seen_tools


async def test_validation_hooks_obey_codeact_concurrency_limit(tmp_path: Path) -> None:
    active = 0
    max_active = 0
    model_calls = 0

    async def validate(ctx: RunContext[AgentContext], *, value: int) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
        finally:
            active -= 1

    async def checked(ctx: RunContext[AgentContext], *, value: int) -> int:
        return value

    external = FunctionToolset([Tool(checked, args_validator=validate, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import asyncio\nawait asyncio.gather(*(checked(value=i) for i in range(4)))"},
                        tool_call_id="bounded-validation",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_concurrency=1),
    )
    async with runtime:
        await runtime.agent.run("bound validation", deps=runtime.ctx)

    assert max_active == 1


async def test_unresolved_nested_approval_is_terminal_not_resumable(tmp_path: Path) -> None:
    model_calls = 0

    async def deferred(ctx: RunContext[AgentContext]) -> None:
        raise CallDeferred

    external = FunctionToolset([Tool(deferred, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await deferred()"},
                        tool_call_id="unresolved-approval",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("do not bubble approval", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    assert "deferred host interaction" in str(returned[0].content)


async def test_handler_retry_for_nested_approval_is_terminal(tmp_path: Path) -> None:
    model_calls = 0

    async def deferred(ctx: RunContext[AgentContext]) -> None:
        raise CallDeferred

    external = FunctionToolset([Tool(deferred, metadata={"codeact": True})])

    def retry_handler(ctx: RunContext[AgentContext], requests) -> DeferredToolResults:
        call = requests.calls[0]
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry("change arguments")})

    retry_capability = HandleDeferredToolCalls[AgentContext](handler=retry_handler)

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await deferred()"},
                        tool_call_id="handler-retry",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        capabilities=[retry_capability],
    )
    async with runtime:
        result = await runtime.agent.run("handler requests retry", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    assert "change arguments" in str(returned[0].content)


async def test_timeout_cancels_and_awaits_nested_call(tmp_path: Path) -> None:
    cleaned = asyncio.Event()
    model_calls = 0

    async def slow(ctx: RunContext[AgentContext]) -> None:
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    external = FunctionToolset([Tool(slow, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await slow()"},
                        tool_call_id="timeout-codeact",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(timeout_seconds=0.2),
    )
    async with runtime:
        result = await runtime.agent.run("time out", deps=runtime.ctx)

    assert cleaned.is_set()
    returned = _tool_returns(result.all_messages(), "run_code")
    assert returned[0].outcome == "failed"
    assert "timeout_seconds=0.2" in str(returned[0].content)


async def test_final_output_limit_is_enforced(tmp_path: Path) -> None:
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "'x' * 1000"},
                        tool_call_id="large-final-output",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        config=CodeActConfig(max_output_bytes=128),
    )
    async with runtime:
        result = await runtime.agent.run("large result", deps=runtime.ctx)

    assert not _tool_returns(result.all_messages(), "run_code")
    assert any(
        isinstance(part, RetryPromptPart)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_codeact_none_exposes_no_codeact_tools(tmp_path: Path) -> None:
    seen_tools: set[str] = set()
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = create_agent(FunctionModel(function=model_function), env=env, codeact=None)
    async with runtime:
        await runtime.agent.run("inspect tools", deps=runtime.ctx)

    assert "run_code" not in seen_tools
    assert "run_program" not in seen_tools


async def test_codeact_catalog_respects_main_agent_only_in_child_context() -> None:
    main_ctx = AgentContext()
    child_ctx = main_ctx.create_subagent_context("worker", agent_id="worker-1")
    run_ctx = RunContext(deps=child_ctx, model=TestModel(), usage=RunUsage())
    codeact = CodeActToolset(
        wrapped=Toolset(tools=[MainOnlyCodeActTool], skip_unavailable=False),
        config=CodeActConfig(),
    )

    tools = await codeact.get_tools(run_ctx)

    assert "main_only_codeact" not in tools
    assert "run_code" in tools
    assert "run_program" not in tools


async def test_monty_has_no_ambient_os_authority(tmp_path: Path) -> None:
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import os\nos.listdir('.')"},
                        tool_call_id="ambient-os",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="recovered")])

    runtime = _runtime(tmp_path, model_function)
    async with runtime:
        result = await runtime.agent.run("try ambient OS", deps=runtime.ctx)

    assert not _tool_returns(result.all_messages(), "run_code")
    assert any(
        isinstance(part, RetryPromptPart)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_nested_tool_return_content_is_propagated_on_success(tmp_path: Path) -> None:
    model_calls = 0

    async def enriched(ctx: RunContext[AgentContext]) -> ToolReturn:
        return ToolReturn(
            return_value=7,
            content="supplemental model-facing content",
            metadata={"internal": "not propagated"},
        )

    external = FunctionToolset([Tool(enriched, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await enriched()"},
                        tool_call_id="supplemental-content",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("preserve ToolReturn content", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")[0]
    assert returned.content == 7
    assert any(
        isinstance(part, UserPromptPart) and part.content == ["supplemental model-facing content"]
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    assert returned.metadata["codeact"]["tool_calls"][0]["tool_return_metadata_omitted"] is True


async def test_run_code_uses_structured_observation_while_propagating_screenshot(tmp_path: Path) -> None:
    clicked_observation_ids: list[str] = []
    screenshot = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    async def computer_observe(ctx: RunContext[AgentContext], *, include_accessibility: bool = False) -> ToolReturn:
        assert include_accessibility is False
        return ToolReturn(
            return_value={
                "catalog_version": {"major": 1, "minor": 1},
                "observation": {
                    "observation_id": "obs-123",
                    "frame_generation": 4,
                },
            },
            content=[BinaryContent(data=screenshot, media_type="image/png")],
        )

    async def computer_click(ctx: RunContext[AgentContext], *, observation_id: str) -> dict[str, Any]:
        clicked_observation_ids.append(observation_id)
        return {"operation_id": "op-456", "effect_status": "committed"}

    external = FunctionToolset([
        Tool(computer_observe, metadata={"codeact": True}),
        Tool(computer_click, metadata={"codeact": True}),
    ])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={
                            "code": (
                                "result = await computer_observe(include_accessibility=False)\n"
                                "observation = result['observation']\n"
                                "await computer_click(observation_id=observation['observation_id'])"
                            )
                        },
                        tool_call_id="observe-click",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("observe and click atomically", deps=runtime.ctx)

    assert clicked_observation_ids == ["obs-123"]
    returned = _tool_returns(result.all_messages(), "run_code")[0]
    assert returned.content == {"operation_id": "op-456", "effect_status": "committed"}
    supplemental = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert any(
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], str)
        and "<filtered-content type='image'>" in content[0]
        for content in supplemental
    )


async def test_executor_admits_calls_before_argument_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    serialized_argument_sets = 0
    original_redacted_json_bytes = codeact_toolset._redacted_json_bytes

    def tracked_redacted_json_bytes(value: Any) -> bytes:
        nonlocal serialized_argument_sets
        if isinstance(value, dict) and "payload" in value:
            serialized_argument_sets += 1
        return original_redacted_json_bytes(value)

    monkeypatch.setattr(codeact_toolset, "_redacted_json_bytes", tracked_redacted_json_bytes)

    async def blocked(ctx: RunContext[AgentContext], *, payload: str, index: int) -> int:
        started.set()
        await release.wait()
        return index + len(payload)

    external = FunctionToolset([Tool(blocked, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={
                            "code": (
                                "import asyncio\n"
                                "payload = 'x' * 1000\n"
                                "await asyncio.gather(*(blocked(payload=payload, index=i) for i in range(4)))"
                            )
                        },
                        tool_call_id="admission-before-args",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_concurrency=1),
    )
    async with runtime:
        run_task = asyncio.create_task(runtime.agent.run("bound host arguments", deps=runtime.ctx))
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert serialized_argument_sets == 1
        release.set()
        result = await asyncio.wait_for(run_task, timeout=2)

    assert _tool_returns(result.all_messages(), "run_code")[0].outcome == "success"


async def test_execute_stage_validation_error_omits_input_from_trace(tmp_path: Path) -> None:
    class Payload(BaseModel):
        count: int

    async def validates_inside_tool(ctx: RunContext[AgentContext]) -> None:
        Payload.model_validate({"count": "private-value-not-an-int"})

    external = FunctionToolset([Tool(validates_inside_tool, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await validates_inside_tool()"},
                        tool_call_id="execute-validation-error",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        result = await runtime.agent.run("validate inside tool", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")[0]
    payload = str(returned.content)
    assert returned.outcome == "failed"
    assert "private-value-not-an-int" not in payload
    assert "input_value" not in payload
    assert "valid integer" in payload


async def test_large_binary_supplemental_content_is_rejected_before_encoding(tmp_path: Path) -> None:
    async def binary(ctx: RunContext[AgentContext]) -> ToolReturn:
        return ToolReturn(
            return_value="value",
            content=[BinaryContent(data=b"x" * 1_000_000, media_type="application/octet-stream")],
        )

    external = FunctionToolset([Tool(binary, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await binary()"},
                        tool_call_id="large-binary-content",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_output_bytes=128),
    )
    async with runtime:
        result = await runtime.agent.run("bound binary", deps=runtime.ctx)

    returned = _tool_returns(result.all_messages(), "run_code")[0]
    assert returned.outcome == "failed"
    assert "supplemental content exceeds max_output_bytes=128" in str(returned.content)


def test_bounded_json_sizer_rejects_unknown_values_without_stringifying() -> None:
    class ExplosiveString:
        def __str__(self) -> str:
            raise AssertionError("unknown values must not be stringified")

    with pytest.raises(TypeError, match="Unsupported CodeAct JSON value"):
        _bounded_json_size(ExplosiveString(), 128)
    with pytest.raises(TypeError, match="Raw bytes are not supported"):
        _bounded_json_size(b"raw", 128)


async def test_supplemental_generic_sequence_fails_without_materialization() -> None:
    class UntrustedSequence(Sequence[Any]):
        iterated = False

        def __len__(self) -> int:
            return 2_000_000

        def __getitem__(self, index: int) -> Any:
            type(self).iterated = True
            raise AssertionError("CodeAct must not materialize an unbounded generic sequence")

    result = ToolReturn(return_value="value", content=UntrustedSequence())
    budget = _ExecutionBudget(CodeActConfig(max_output_bytes=128))
    with pytest.raises(TypeError, match=r"Unsupported CodeAct ToolReturn\.content sequence"):
        await _unwrap_tool_return(result, ordinal=1, budget=budget, record={})
    assert not UntrustedSequence.iterated


def test_static_preflight_respects_lexical_scope_and_binding_order() -> None:
    with pytest.raises(ValueError, match="'missing'"):
        validate_static_tool_references(
            "def decoy(missing):\n    pass\n\nawait mutate()\nawait missing()",
            valid_tool_names={"mutate"},
        )
    with pytest.raises(ValueError, match="'late'"):
        validate_static_tool_references(
            "await mutate()\nawait late()\nlate = mutate",
            valid_tool_names={"mutate"},
        )
    with pytest.raises(ValueError, match="'local'"):
        validate_static_tool_references(
            "def helper():\n    local()\n    local = print\nhelper()",
            valid_tool_names=set(),
        )
    forward_source = "async def invoke():\n    return await late()\nawait mutate()\nawait invoke()\nlate = mutate"
    with pytest.raises(ValueError, match="'late'"):
        validate_static_tool_references(
            forward_source,
            valid_tool_names={"mutate"},
        )
    validate_static_tool_references(
        forward_source,
        valid_tool_names={"mutate"},
        functions_see_complete_module=True,
    )
    validate_static_tool_references(
        "await helper()",
        valid_tool_names=set(),
        known_names={"helper"},
    )


async def test_restart_resets_inline_state_before_failed_preflight(tmp_path: Path) -> None:
    snippets = [
        {"code": "def helper():\n    return 7\nhelper()"},
        {"code": "await missing()", "restart": True},
        {"code": "1"},
        {"code": "helper()"},
    ]
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        if model_calls < len(snippets):
            args = snippets[model_calls]
            model_calls += 1
            return ModelResponse(
                parts=[ToolCallPart(tool_name="run_code", args=args, tool_call_id=f"restart-{model_calls}")]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(tmp_path, model_function)
    async with runtime:
        result = await runtime.agent.run("restart atomically", deps=runtime.ctx)

    assert [part.content for part in _tool_returns(result.all_messages(), "run_code")] == [7, 1]
    retries = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert len(retries) == 2
    assert "'missing'" in str(retries[0].content)
    assert "'helper'" in str(retries[1].content)


async def test_catalog_rejects_dangling_or_invalid_local_schema_refs() -> None:
    async def accepts_payload(ctx: RunContext[AgentContext], *, payload: str) -> str:
        return payload

    toolset = FunctionToolset([Tool(accepts_payload, metadata={"codeact": True})])
    ctx = RunContext(deps=AgentContext(), model=TestModel(), usage=RunUsage())
    original = (await toolset.get_tools(ctx))["accepts_payload"]

    for reference in ("#/$defs/Missing", "#/bad/path", "#/$defs/Bad~2Escape"):
        schema = {
            "type": "object",
            "properties": {"payload": {"$ref": reference}},
            "$defs": {},
        }
        bad = replace(
            original,
            tool_def=replace(original.tool_def, parameters_json_schema=schema),
        )
        with pytest.raises(UserError, match="invalid or dangling local \\$ref"):
            _build_catalog({"accepts_payload": bad})

    invalid_array_pointer = {
        "type": "object",
        "properties": {"payload": {"$ref": "#/prefixItems/00"}},
        "prefixItems": [{"type": "string"}],
    }
    bad = replace(
        original,
        tool_def=replace(original.tool_def, parameters_json_schema=invalid_array_pointer),
    )
    with pytest.raises(UserError, match="invalid or dangling local \\$ref"):
        _build_catalog({"accepts_payload": bad})


async def test_timeout_waits_for_temporarily_cancellation_resistant_call(tmp_path: Path) -> None:
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    async def resistant(ctx: RunContext[AgentContext]) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            cleaned.set()
            raise

    external = FunctionToolset([Tool(resistant, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "await resistant()"},
                        tool_call_id="resistant-timeout",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="handled")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(timeout_seconds=0.1),
    )
    async with runtime:
        run_task = asyncio.create_task(runtime.agent.run("wait for cleanup", deps=runtime.ctx))
        await asyncio.wait_for(cancellation_seen.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not run_task.done()
        release.set()
        result = await asyncio.wait_for(run_task, timeout=2)

    assert cleaned.is_set()
    returned = _tool_returns(result.all_messages(), "run_code")[0]
    assert returned.outcome == "failed"
    assert "timeout_seconds=0.1" in str(returned.content)


async def test_repeated_cancellation_does_not_orphan_nested_call(tmp_path: Path) -> None:
    started = asyncio.Event()
    first_cancellation = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    async def resistant(ctx: RunContext[AgentContext]) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancellation.set()
            await release.wait()
            cleaned.set()
            raise

    external = FunctionToolset([Tool(resistant, metadata={"codeact": True})])

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="run_code",
                    args={"code": "await resistant()"},
                    tool_call_id="repeated-cancellation",
                )
            ]
        )

    runtime = _runtime(tmp_path, model_function, toolsets=[external])
    async with runtime:
        run_task = asyncio.create_task(runtime.agent.run("cancel repeatedly", deps=runtime.ctx))
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            run_task.cancel()
            await asyncio.wait_for(first_cancellation.wait(), timeout=2)
            run_task.cancel()
            await asyncio.sleep(0)
            assert not run_task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=2)

    assert cleaned.is_set()


async def test_global_sequential_mode_does_not_deadlock_admission(tmp_path: Path) -> None:
    active = 0
    max_active = 0
    calls = 0

    async def slow(ctx: RunContext[AgentContext], *, value: int) -> int:
        nonlocal active, max_active, calls
        active += 1
        calls += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            return value
        finally:
            active -= 1

    external = FunctionToolset([Tool(slow, metadata={"codeact": True})])
    model_calls = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="run_code",
                        args={"code": "import asyncio\nawait asyncio.gather(*(slow(value=i) for i in range(4)))"},
                        tool_call_id="global-sequential-admission",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    runtime = _runtime(
        tmp_path,
        model_function,
        toolsets=[external],
        config=CodeActConfig(max_concurrency=1),
    )
    async with runtime:
        with ToolManager.parallel_execution_mode("sequential"):
            result = await asyncio.wait_for(
                runtime.agent.run("run sequentially", deps=runtime.ctx),
                timeout=2,
            )

    assert calls == 4
    assert max_active == 1
    returned = _tool_returns(result.all_messages(), "run_code")[0]
    assert returned.outcome == "success", returned.content
