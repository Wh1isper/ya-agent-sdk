"""Tests for SDK model correction retry budgets."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic_ai import AgentSpec, ModelRetry, RunContext, UnexpectedModelBehavior
from pydantic_ai.capabilities import Toolset as ToolsetCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import OverallRetryBudget
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset


async def test_create_agent_defaults_native_retry_budgets_to_context_config(tmp_path: Path) -> None:
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        TestModel(),
        env=env,
        context_kwargs={"retry_config": {"tools": 7, "output": 6, "toolset": 5, "tool_proxy": 5}},
    )

    async with runtime:
        assert runtime.agent._max_tool_retries == 7
        assert runtime.agent._max_output_retries == 6


async def test_create_agent_preserves_spec_retry_budgets(tmp_path: Path) -> None:
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        TestModel(),
        spec=AgentSpec(retries={"tools": 2, "output": 3}),
        env=env,
    )

    async with runtime:
        assert runtime.agent._max_tool_retries == 2
        assert runtime.agent._max_output_retries == 3


async def test_create_agent_explicit_retry_budgets_override_spec(tmp_path: Path) -> None:
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        TestModel(),
        spec=AgentSpec(retries={"tools": 2, "output": 3}),
        retries={"tools": 4, "output": 1},
        env=env,
    )

    async with runtime:
        assert runtime.agent._max_tool_retries == 4
        assert runtime.agent._max_output_retries == 1


def test_overall_retry_budget_defaults_to_five() -> None:
    assert OverallRetryBudget().max_retries == 5


class RetryAfterSuccessTool(BaseTool):
    """Fail, succeed, then fail to exercise Pydantic AI's per-tool reset."""

    name = "retry_after_success"
    description = "Retry on the first and third calls"
    calls: ClassVar[int] = 0

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        type(self).calls += 1
        if type(self).calls in {1, 3}:
            raise ModelRetry(f"correction {type(self).calls}")
        return "success"


async def test_overall_retry_budget_does_not_reset_after_tool_success(tmp_path: Path) -> None:
    RetryAfterSuccessTool.calls = 0
    model_calls = 0
    saw_successful_return = False

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls, saw_successful_return
        model_calls += 1
        saw_successful_return = saw_successful_return or any(
            isinstance(part, ToolReturnPart) and part.tool_name == RetryAfterSuccessTool.name
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=RetryAfterSuccessTool.name,
                    args={},
                    tool_call_id=f"call-{model_calls}",
                )
            ]
        )

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        FunctionModel(function=model_function),
        env=env,
        capabilities=[
            ToolsetCapability(
                Toolset(
                    tools=[RetryAfterSuccessTool],
                    toolset_id="retry_after_success",
                ),
                id="retry_after_success",
            ),
            OverallRetryBudget(max_retries=1),
        ],
        retries={"tools": 1, "output": 1},
    )

    async with runtime:
        with pytest.raises(UnexpectedModelBehavior, match="run-wide model correction retry limit of 1"):
            await runtime.agent.run("exercise retries", deps=runtime.ctx)

    assert RetryAfterSuccessTool.calls == 3
    assert model_calls == 3
    assert saw_successful_return is True


async def test_overall_retry_budget_counts_output_validator_corrections(tmp_path: Path) -> None:
    model_calls = 0

    def model_function(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"response": "bad"})])

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        FunctionModel(function=model_function),
        env=env,
        output_type=dict[str, str],
        capabilities=[OverallRetryBudget(max_retries=0)],
        retries={"tools": 1, "output": 2},
    )

    async with runtime:

        @runtime.agent.output_validator
        def reject_output(
            _ctx: RunContext[AgentContext],
            output: dict[str, str],
        ) -> dict[str, str]:
            raise ModelRetry("produce a better final result")

        with pytest.raises(UnexpectedModelBehavior, match="run-wide model correction retry limit of 0"):
            await runtime.agent.run("exercise output retry", deps=runtime.ctx)

    assert model_calls == 1


async def test_overall_retry_budget_counts_each_retry_prompt_in_one_request() -> None:
    budget = OverallRetryBudget(max_retries=1)
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.run_id = "run-1"
    request_context = MagicMock(spec=ModelRequestContext)
    request_context.messages = [
        ModelRequest(
            parts=[
                RetryPromptPart(content="first", tool_name="tool_a", tool_call_id="call-a"),
                RetryPromptPart(content="second", tool_name="tool_b", tool_call_id="call-b"),
            ],
            run_id="run-1",
        )
    ]

    with pytest.raises(UnexpectedModelBehavior, match="run-wide model correction retry limit of 1"):
        await budget.before_model_request(run_ctx, request_context)

    assert budget.retries_used == 2


async def test_overall_retry_budget_has_fresh_state_for_each_run() -> None:
    budget = OverallRetryBudget(max_retries=3, retries_used=2)
    run_ctx = MagicMock(spec=RunContext)

    first = await budget.for_run(run_ctx)
    second = await budget.for_run(run_ctx)

    assert isinstance(first, OverallRetryBudget)
    assert isinstance(second, OverallRetryBudget)
    assert first is not second
    assert first is not budget
    assert first.retries_used == 0
    assert second.retries_used == 0
    first.retries_used = 1
    assert second.retries_used == 0
    assert budget.retries_used == 2


def test_overall_retry_budget_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        OverallRetryBudget(max_retries=-1)
