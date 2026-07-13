from __future__ import annotations

from unittest.mock import MagicMock

from pydantic_ai.usage import RunUsage
from ya_agent_sdk.context.bus import BusMessage, MessageBus
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot
from yaacli.app.tui import TUIApp
from yaacli.background import BackgroundMonitor


def _snapshot(*, run_id: str, include_subagent: bool) -> UsageSnapshot:
    agent_usages = {
        "main": UsageAgentTotal(
            agent_name="main",
            model_id="model-main",
            usage=RunUsage(requests=1, input_tokens=10, output_tokens=5),
        )
    }
    model_usages = {"model-main": RunUsage(requests=1, input_tokens=10, output_tokens=5)}
    if include_subagent:
        agent_usages["executor-bg-123"] = UsageAgentTotal(
            agent_name="executor",
            model_id="model-sub",
            usage=RunUsage(requests=1, input_tokens=20, output_tokens=7),
        )
        model_usages["model-sub"] = RunUsage(requests=1, input_tokens=20, output_tokens=7)
    return UsageSnapshot(
        run_id=run_id,
        total_usage=RunUsage(
            requests=2 if include_subagent else 1,
            input_tokens=30 if include_subagent else 10,
            output_tokens=12 if include_subagent else 5,
        ),
        agent_usages=agent_usages,
        model_usages=model_usages,
    )


def test_session_usage_late_async_subagent_snapshot_replaces_same_run_totals() -> None:
    from yaacli.usage import SessionUsage

    usage = SessionUsage()
    usage.set_run_snapshot(_snapshot(run_id="run-1", include_subagent=False))
    usage.commit_run_snapshot("run-1")

    assert usage.total_input_tokens == 10
    assert "executor-bg-123" not in usage.agent_usages

    usage.set_run_snapshot(_snapshot(run_id="run-1", include_subagent=True))
    usage.commit_run_snapshot("run-1")

    assert usage.total_input_tokens == 30
    assert usage.total_output_tokens == 12
    assert usage.total_requests == 2
    assert usage.agent_usages["executor-bg-123"].input_tokens == 20
    assert usage.model_usages["model-sub"].output_tokens == 7


def test_tui_delivers_queued_background_usage_snapshot() -> None:
    monitor = BackgroundMonitor()
    bus = MessageBus()

    env = MagicMock()
    env.resources = MagicMock()
    env.resources.get.return_value = monitor

    ctx = MagicMock()
    ctx.agent_id = "main"
    ctx.message_bus = bus

    runtime = MagicMock()
    runtime.env = env
    runtime.ctx = ctx

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()

    app = TUIApp(config=config, config_manager=config_manager)
    app._runtime = runtime

    app._session_usage.set_run_snapshot(_snapshot(run_id="run-1", include_subagent=False))
    app._session_usage.commit_run_snapshot("run-1")

    monitor.enqueue_usage_snapshot(_snapshot(run_id="run-1", include_subagent=True))
    monitor.enqueue_message(BusMessage(content="done", source="executor-bg-123", target="main"))

    assert app._deliver_background_messages() is True
    assert app._session_usage.total_input_tokens == 30
    assert app._session_usage.agent_usages["executor-bg-123"].output_tokens == 7
    assert app._session_usage._committed_run_contributions == {}
    assert bus.has_pending("main") is True
