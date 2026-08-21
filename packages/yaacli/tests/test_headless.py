from __future__ import annotations

import asyncio
import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import build_default_capability_catalog
from ya_agent_sdk.subagents import (
    SubagentDurability,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanResolver,
    SubagentSpec,
)
from yaacli.config import ConfigManager
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.models import ChildPlanManifest
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.durable.subagents import SQLiteSubagentExecutionStore
from yaacli.environment import TUIEnvironment
from yaacli.errors import safe_exception_str
from yaacli.headless import HeadlessEventSink, _run_headless_prompt
from yaacli.session import TUIContext


def _manager(tmp_path: Path) -> ConfigManager:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[general]
model = "test"

[session]
auto_restore = false
session_dir = "{session_dir}"
""".format(session_dir=(tmp_path / "sessions").as_posix()),
        encoding="utf-8",
    )
    return ConfigManager(config_dir=config_dir)


async def _seed_terminal_child_descriptor(manager: ConfigManager, tmp_path: Path) -> str:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        restart_durable=False,
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="legacy-helper",
            durability=SubagentDurability.process,
            agent=AgentSpec(name="legacy-helper", instructions="historical plan"),
        )
    )
    database_path = manager.get_session_database_path()
    product_store = SQLiteSessionStore(database_path)
    product_store.create_session(str(tmp_path), session_id="legacy-owner")
    child_store = SQLiteSubagentExecutionStore(database_path)
    child_store.put_descriptor(plan)
    await child_store.create(
        SubagentExecutionRecord(
            root_execution_id="legacy-execution",
            execution_id="legacy-execution",
            owner_scope_id="legacy-owner",
            idempotency_key="legacy-execution",
            descriptor_id=plan.descriptor_id,
            plan_fingerprint=plan.fingerprint,
            route=plan.spec.route,
            mode=SubagentExecutionMode.background,
            state=SubagentExecutionState.succeeded,
            input_state=SubagentInputState.applied,
            parent_agent_id="main",
            parent_logical_run_id="parent-run",
            prompt="historical work",
            output="done",
        )
    )
    child_store.close_sync()
    product_store.close()
    return plan.descriptor_id


def _minimal_runtime_factory(tmp_path: Path, calls: list[dict[str, object]]):
    def factory(*args: object, **kwargs: object):
        del args
        calls.append(dict(kwargs))
        return create_agent(
            TestModel(call_tools=[], custom_output_text="durable headless answer"),
            capabilities=[DurableInboxPumpCapability()],
            context_type=TUIContext,
            context_kwargs={
                "durable_binding_ref": kwargs["durable_binding_ref"],
            },
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    return factory


def test_headless_event_sink_flushes_after_every_event() -> None:
    output_stream = MagicMock()
    sink = HeadlessEventSink(output_stream=output_stream)

    sink.emit({"type": "FIRST", "value": 1})
    sink.emit({"type": "SECOND", "value": 2})

    assert output_stream.method_calls == [
        call.write('{"type":"FIRST","value":1}\n'),
        call.flush(),
        call.write('{"type":"SECOND","value":2}\n'),
        call.flush(),
    ]


class NonStringError(RuntimeError):
    def __str__(self) -> str:
        return 42  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "non-string __str__ fallback"


class UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("broken __str__")

    def __repr__(self) -> str:
        raise ValueError("broken __repr__")


def test_safe_exception_str_survives_broken_exception_formatters() -> None:
    assert safe_exception_str(NonStringError()) == "non-string __str__ fallback"
    assert safe_exception_str(UnprintableError()) == "<UnprintableError: exception text unavailable>"


@pytest.mark.asyncio
async def test_headless_testmodel_turn_commits_before_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    config = manager.load()
    retained_descriptor_id = await _seed_terminal_child_descriptor(manager, tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "yaacli.headless.create_tui_runtime",
        _minimal_runtime_factory(tmp_path, calls),
    )

    output = StringIO()
    result = await _run_headless_prompt(
        config=config,
        config_manager=manager,
        prompt="hello durable headless",
        working_dir=tmp_path,
        session_id=None,
        model_profile_id=None,
        worker=False,
        ndjson_stream=output,
    )

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1] == {
        **events[-1],
        "type": "RUN_FINISHED",
        "result": {"output_text": "durable headless answer"},
    }
    assert result.output_text == "durable headless answer"
    assert calls[0]["subagent_default_mode"] is SubagentExecutionMode.foreground
    assert isinstance(calls[0]["system_prompt"], str)
    child_manifest = calls[0]["child_plan_manifest"]
    assert isinstance(child_manifest, ChildPlanManifest)
    assert retained_descriptor_id not in {descriptor.descriptor_id for descriptor in child_manifest.descriptors}
    assert calls[0]["subagent_deferred_resolver"] is not None

    retained_store = SQLiteSubagentExecutionStore(manager.get_session_database_path())
    try:
        assert retained_store.get_descriptor(retained_descriptor_id) is not None
    finally:
        retained_store.close_sync()

    store = SQLiteSessionStore(manager.get_session_database_path())
    try:
        session = store.get_session(result.session_id)
        assert session is not None
        assert session.head_revision_id is not None
        revision = store.get_revision(session.head_revision_id)
        assert revision is not None
        assert revision.terminal == {
            "status": "completed",
            "output": "durable headless answer",
        }
        assert revision.display_projection[-1]["type"] != "RUN_FINISHED"
    finally:
        store.close()


def test_headless_restore_continues_from_durable_session_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    config = manager.load()
    monkeypatch.setattr(
        "yaacli.headless.create_tui_runtime",
        _minimal_runtime_factory(tmp_path, []),
    )
    first_output = StringIO()
    first = asyncio.run(
        _run_headless_prompt(
            config=config,
            config_manager=manager,
            prompt="first",
            working_dir=tmp_path,
            session_id=None,
            model_profile_id=None,
            worker=False,
            ndjson_stream=first_output,
        )
    )

    second_output = StringIO()
    second = asyncio.run(
        _run_headless_prompt(
            config=config,
            config_manager=manager,
            prompt="second",
            working_dir=tmp_path,
            session_id=first.session_id,
            model_profile_id=None,
            worker=False,
            ndjson_stream=second_output,
        )
    )
    second_events = [json.loads(line) for line in second_output.getvalue().splitlines()]

    assert second.session_id == first.session_id
    assert second_events[0]["type"] == "RUN_STARTED"
    assert second_events[-1]["type"] == "RUN_FINISHED"
    store = SQLiteSessionStore(manager.get_session_database_path())
    try:
        session = store.get_session(first.session_id)
        assert session is not None and session.head_revision_id is not None
        revision = store.get_revision(session.head_revision_id)
        assert revision is not None
        user_prompts = [
            part["content"]
            for message in revision.message_history
            if message.get("kind") == "request"
            for part in message.get("parts", [])
            if part.get("part_kind") == "user-prompt"
        ]
        assert "first" in user_prompts
        assert "second" in user_prompts
    finally:
        store.close()


@pytest.mark.asyncio
async def test_headless_unknown_session_does_not_create_filesystem_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    config = manager.load()
    monkeypatch.setattr(
        "yaacli.headless.create_tui_runtime",
        _minimal_runtime_factory(tmp_path, []),
    )

    output = StringIO()
    with pytest.raises(KeyError, match="missing-session"):
        await _run_headless_prompt(
            config=config,
            config_manager=manager,
            prompt="continue",
            working_dir=tmp_path,
            session_id="missing-session",
            model_profile_id=None,
            worker=False,
            ndjson_stream=output,
        )

    assert output.getvalue() == ""
    assert not (manager.get_sessions_dir() / "missing-session").exists()
