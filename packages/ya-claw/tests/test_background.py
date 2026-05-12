from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import RunContext
from ya_agent_environment import Environment
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_claw.execution.background import BackgroundMonitor, BackgroundTaskAlreadyActiveError
from ya_claw.runtime_state import create_runtime_state
from ya_claw.toolsets.background import SpawnDelegateTool, SteerSubagentTool


class EmptyEnvironment(Environment):
    async def _setup(self) -> None:
        return None

    async def _teardown(self) -> None:
        return None


class FakeRunContext:
    def __init__(self, deps: AgentContext) -> None:
        self.deps = deps


class FakeDelegateTool(BaseTool):
    name = "delegate"
    description = "fake delegate"

    async def call(self, ctx: RunContext[AgentContext], /, *args: Any, **kwargs: Any) -> str:
        await asyncio.sleep(0)
        return "delegate result"


class FailingDelegateTool(BaseTool):
    name = "delegate"
    description = "failing delegate"

    async def call(self, ctx: RunContext[AgentContext], /, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("delegate boom")


class FakeToolset:
    def __init__(self, delegate: BaseTool | None = None) -> None:
        self.delegate = delegate or FakeDelegateTool()

    def _get_tool_instance(self, name: str) -> BaseTool:
        if name != "delegate":
            raise KeyError(name)
        return self.delegate


async def test_background_monitor_tracks_one_run_and_emits_events() -> None:
    runtime_state = create_runtime_state()
    runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
    monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)

    task = asyncio.create_task(asyncio.sleep(0), name="background-test-task")
    monitor.register_task("agent-1", task, subagent_name="worker", prompt="do work")

    assert monitor.has_active_tasks
    assert "agent-1" in monitor.task_infos
    await monitor.emit_subagent_spawned("agent-1", "worker", "do work")
    await task
    await asyncio.sleep(0)

    assert not monitor.has_active_tasks
    assert monitor.task_infos == {}
    events = runtime_state.get_replay_events("run-1")
    assert events[-1]["type"] == "ya_claw.subagent_spawned"
    assert events[-1]["agent_id"] == "agent-1"


async def test_background_monitor_rejects_duplicate_active_agent_id() -> None:
    runtime_state = create_runtime_state()
    runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
    monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)
    release = asyncio.Event()

    async def wait_forever() -> None:
        await release.wait()

    task = asyncio.create_task(wait_forever(), name="background-test-active")
    duplicate = asyncio.create_task(asyncio.sleep(0), name="background-test-duplicate")
    try:
        monitor.register_task("agent-1", task, subagent_name="worker", prompt="do work")
        with pytest.raises(BackgroundTaskAlreadyActiveError):
            monitor.register_task("agent-1", duplicate, subagent_name="worker", prompt="do work")
    finally:
        duplicate.cancel()
        release.set()
        await asyncio.gather(task, duplicate, return_exceptions=True)
        await monitor.close()


async def test_background_monitor_is_run_scoped() -> None:
    runtime_state = create_runtime_state()
    runtime_state.register_run("session-1", "run-a", dispatch_mode="stream")
    runtime_state.register_run("session-2", "run-b", dispatch_mode="stream")
    monitor_a = BackgroundMonitor(run_id="run-a", runtime_state=runtime_state)
    monitor_b = BackgroundMonitor(run_id="run-b", runtime_state=runtime_state)

    await monitor_a.emit_subagent_completed("agent-a", "worker", 1.0, "done a")
    await monitor_b.emit_subagent_completed("agent-b", "worker", 2.0, "done b")

    events_a = runtime_state.get_replay_events("run-a")
    events_b = runtime_state.get_replay_events("run-b")
    assert events_a[-1]["agent_id"] == "agent-a"
    assert events_b[-1]["agent_id"] == "agent-b"


async def test_background_monitor_drain_timeout_cancels_active_tasks() -> None:
    runtime_state = create_runtime_state()
    runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
    monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever(), name="background-test-cancel")
    monitor.register_task("agent-1", task, subagent_name="worker", prompt="do work")

    completed = await monitor.drain_or_cancel(timeout=0.01)

    assert completed is False
    assert task.cancelled()
    assert not monitor.has_active_tasks
    assert runtime_state.get_replay_events("run-1")[-1]["type"] == "ya_claw.subagent_cancelled"


async def test_spawn_delegate_reports_duplicate_active_agent_id() -> None:
    env = EmptyEnvironment()
    async with env:
        ctx = AgentContext(env=env, agent_id="main")
        async with ctx:
            runtime_state = create_runtime_state()
            runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
            monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)
            monitor.set_core_toolset(FakeToolset())  # type: ignore[arg-type]
            env.resources.set("background_monitor", monitor)
            release = asyncio.Event()

            async def wait_forever() -> None:
                await release.wait()

            task = asyncio.create_task(wait_forever(), name="background-test-duplicate-tool")
            monitor.register_task("agent-1", task, subagent_name="worker", prompt="do work")
            try:
                tool = SpawnDelegateTool()
                result = await tool.call(
                    FakeRunContext(ctx),  # type: ignore[arg-type]
                    subagent_name="worker",
                    prompt="do more work",
                    agent_id="agent-1",
                )

                assert result == "Error: Background task already active: agent-1"
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await monitor.close()


async def test_steer_subagent_sends_targeted_bus_message() -> None:
    env = EmptyEnvironment()
    async with env:
        ctx = AgentContext(env=env, agent_id="main")
        async with ctx:
            runtime_state = create_runtime_state()
            runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
            monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)
            env.resources.set("background_monitor", monitor)
            release = asyncio.Event()

            async def wait_forever() -> None:
                await release.wait()

            task = asyncio.create_task(wait_forever(), name="background-test-steer")
            monitor.register_task("agent-1", task, subagent_name="worker", prompt="do work")
            try:
                tool = SteerSubagentTool()
                ctx.message_bus.subscribe("agent-1")
                result = await tool.call(
                    FakeRunContext(ctx),  # type: ignore[arg-type]
                    agent_id="agent-1",
                    message="focus on tests",
                )

                assert "Steering message sent" in result
                messages = ctx.message_bus.consume("agent-1")
                assert len(messages) == 1
                assert messages[0].content == "focus on tests"
                assert messages[0].source == "main"
                assert messages[0].target == "agent-1"
                assert runtime_state.get_replay_events("run-1")[-1]["type"] == "ya_claw.subagent_steered"
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await monitor.close()


async def test_spawn_delegate_delivers_result_to_main_message_bus() -> None:
    env = EmptyEnvironment()
    async with env:
        ctx = AgentContext(env=env, agent_id="main")
        async with ctx:
            runtime_state = create_runtime_state()
            runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
            monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)
            monitor.set_core_toolset(FakeToolset())  # type: ignore[arg-type]
            env.resources.set("background_monitor", monitor)
            ctx.message_bus.subscribe("main")

            tool = SpawnDelegateTool()
            result = await tool.call(
                FakeRunContext(ctx),  # type: ignore[arg-type]
                subagent_name="worker",
                prompt="do work",
                agent_id="worker-bg-test",
            )
            drained = await monitor.drain_or_cancel(timeout=1.0)

            assert "Spawned delegate" in result
            assert drained is True
            messages = ctx.message_bus.consume("main")
            assert [message.content for message in messages] == ["delegate result"]
            event_types = [event["type"] for event in runtime_state.get_replay_events("run-1")]
            assert "ya_claw.subagent_spawned" in event_types
            assert "ya_claw.subagent_completed" in event_types


async def test_spawn_delegate_failure_delivers_error_to_main_message_bus() -> None:
    env = EmptyEnvironment()
    async with env:
        ctx = AgentContext(env=env, agent_id="main")
        async with ctx:
            runtime_state = create_runtime_state()
            runtime_state.register_run("session-1", "run-1", dispatch_mode="stream")
            monitor = BackgroundMonitor(run_id="run-1", runtime_state=runtime_state)
            monitor.set_core_toolset(FakeToolset(delegate=FailingDelegateTool()))  # type: ignore[arg-type]
            env.resources.set("background_monitor", monitor)
            ctx.message_bus.subscribe("main")

            tool = SpawnDelegateTool()
            result = await tool.call(
                FakeRunContext(ctx),  # type: ignore[arg-type]
                subagent_name="worker",
                prompt="do work",
                agent_id="worker-bg-fail",
            )
            drained = await monitor.drain_or_cancel(timeout=1.0)

            assert "Spawned delegate" in result
            assert drained is True
            messages = ctx.message_bus.consume("main")
            assert len(messages) == 1
            assert "failed" in str(messages[0].content)
            assert runtime_state.get_replay_events("run-1")[-1]["type"] == "ya_claw.subagent_failed"
