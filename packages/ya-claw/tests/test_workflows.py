from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.config import ClawSettings
from ya_claw.controller.models import RunStatus
from ya_claw.controller.schedule import ScheduleController, ScheduleCreateRequest, ScheduleUpdateRequest
from ya_claw.controller.workflow import (
    WorkflowActorContext,
    WorkflowController,
    WorkflowDefinitionCreateRequest,
    WorkflowTriggerRequest,
)
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.execution.workflow import WorkflowExecutor
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, ScheduleRecord, WorkflowNodeRunRecord, WorkflowRunRecord
from ya_claw.runtime_state import InMemoryRuntimeState


class RecordingSupervisor:
    def __init__(self, *, accepting_submissions: bool = True) -> None:
        self.submitted_run_ids: list[str] = []
        self.accepting_submissions = accepting_submissions

    def get_background_task(self, run_id: str) -> None:
        return None

    def submit_run(self, run_id: str) -> bool:
        if not self.accepting_submissions:
            return False
        self.submitted_run_ids.append(run_id)
        return True


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'workflows.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        workspace_provider_backend="local",
        bridge_dispatch_mode="manual",
        default_profile="default",
        _env_file=None,
    )


def _workflow_definition() -> dict[str, Any]:
    return {
        "schema": "ya-claw.workflow.v1",
        "name": "Research Workflow",
        "version": 1,
        "tags": ["research"],
        "inputs": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
        "policy": {"max_concurrency": 2},
        "nodes": {
            "landscape": {"profile": "Self", "prompt": "Research {{ inputs.topic }}."},
            "synthesize": {
                "profile": "Self",
                "needs": ["landscape"],
                "mode": "continue",
                "prompt": "Summarize {{ nodes.landscape.output_text }} for {{ inputs.topic }}.",
            },
        },
        "result": {"from_node": "synthesize"},
    }


async def test_workflow_controller_crud_filters_and_trigger(db_session: AsyncSession) -> None:
    controller = WorkflowController()
    workflow = await controller.create_definition(
        db_session,
        WorkflowDefinitionCreateRequest(definition=_workflow_definition()),
        actor=WorkflowActorContext(
            actor_kind="agent",
            current_session_id="session-1",
            current_run_id="run-1",
            profile_name="default",
        ),
    )

    assert workflow.scope == "session"
    assert workflow.owner_kind == "agent"
    assert workflow.owner_session_id == "session-1"

    current_session_list = await controller.list_definitions(
        db_session,
        only_current_session=True,
        current_session_id="session-1",
    )
    assert [item.id for item in current_session_list.workflows] == [workflow.id]

    broader_list = await controller.list_definitions(db_session, tags=["research"])
    assert [item.id for item in broader_list.workflows] == [workflow.id]

    run = await controller.trigger(
        db_session,
        workflow.id,
        WorkflowTriggerRequest(inputs={"topic": "workflow orchestration"}),
        actor=WorkflowActorContext(
            actor_kind="agent",
            current_session_id="session-1",
            current_run_id="run-1",
            profile_name="default",
        ),
    )

    assert run.status == "queued"
    assert run.trigger_kind == "agent"
    assert run.supervisor_session_id == "session-1"
    assert run.events[0].event_type == "workflow_queued"

    runs = await controller.list_runs(
        db_session, only_supervised_by_current_session=True, current_session_id="session-1"
    )
    assert [item.id for item in runs.workflow_runs] == [run.id]


async def test_workflow_executor_advances_dag_and_projects_result(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = WorkflowController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    workflow = await controller.create_definition(
        db_session,
        WorkflowDefinitionCreateRequest(definition=_workflow_definition()),
    )
    run = await controller.trigger(
        db_session,
        workflow.id,
        WorkflowTriggerRequest(inputs={"topic": "workflow orchestration"}, profile_name="default"),
    )
    record = await db_session.get(WorkflowRunRecord, run.id)
    assert isinstance(record, WorkflowRunRecord)

    changed = await controller.execute_once(db_session, settings, runtime_state, dispatcher, record)
    await db_session.commit()
    assert changed is True
    assert len(supervisor.submitted_run_ids) == 1

    first_node = await _node(db_session, run.id, "landscape")
    assert first_node.status == "queued"
    first_run = await db_session.get(RunRecord, first_node.run_id)
    assert isinstance(first_run, RunRecord)
    first_run.status = RunStatus.COMPLETED.value
    first_run.output_text = "landscape output"
    first_run.finished_at = datetime.now(UTC)
    first_run.committed_at = first_run.finished_at
    await db_session.commit()

    await db_session.refresh(record)
    await controller.execute_once(db_session, settings, runtime_state, dispatcher, record)
    await db_session.commit()
    assert len(supervisor.submitted_run_ids) == 2

    second_node = await _node(db_session, run.id, "synthesize")
    assert second_node.status == "queued"
    assert "landscape output" in second_node.input_parts[0]["text"]
    second_run = await db_session.get(RunRecord, second_node.run_id)
    assert isinstance(second_run, RunRecord)
    second_run.status = RunStatus.COMPLETED.value
    second_run.output_text = "final report"
    second_run.finished_at = datetime.now(UTC)
    second_run.committed_at = second_run.finished_at
    await db_session.commit()

    await db_session.refresh(record)
    await controller.execute_once(db_session, settings, runtime_state, dispatcher, record)
    await db_session.commit()
    await db_session.refresh(record)

    assert record.status == "completed"
    assert record.result == {
        "from_node": "synthesize",
        "output_text": "final report",
        "output_json": None,
        "session_id": second_node.session_id,
        "run_id": second_node.run_id,
    }


async def test_workflow_executor_service_dispatch_once(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    async with session_factory() as db_session:
        controller = WorkflowController()
        workflow = await controller.create_definition(
            db_session,
            WorkflowDefinitionCreateRequest(definition=_workflow_definition()),
        )
        await controller.trigger(
            db_session,
            workflow.id,
            WorkflowTriggerRequest(inputs={"topic": "workflow orchestration"}, profile_name="default"),
        )

    executor = WorkflowExecutor(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        run_dispatcher=dispatcher,
    )
    processed = await executor.dispatch_once()

    assert processed == 1
    assert len(supervisor.submitted_run_ids) == 1


async def test_schedule_workflow_mode_triggers_workflow_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    workflow_controller = WorkflowController()
    workflow = await workflow_controller.create_definition(
        db_session,
        WorkflowDefinitionCreateRequest(definition=_workflow_definition()),
    )
    schedule_controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    schedule = await schedule_controller.create(
        db_session,
        ScheduleCreateRequest(
            name="workflow schedule",
            prompt="",
            cron="* * * * *",
            workflow_id=workflow.id,
            workflow_inputs_template={"topic": "{{ schedule.name }}"},
            profile_name="default",
        ),
    )
    schedule_record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(schedule_record, ScheduleRecord)
    assert schedule_record.execution_mode == "workflow"

    fire = await schedule_controller.trigger(db_session, settings, runtime_state, dispatcher, schedule.id)

    assert fire.status == "submitted"
    assert fire.workflow_run_id is not None
    workflow_run = await db_session.get(WorkflowRunRecord, fire.workflow_run_id)
    assert isinstance(workflow_run, WorkflowRunRecord)
    assert workflow_run.trigger_kind == "schedule"
    assert workflow_run.inputs == {"topic": "workflow schedule"}


async def test_schedule_workflow_mode_can_be_cleared_to_prompt_schedule(
    db_session: AsyncSession,
) -> None:
    workflow_controller = WorkflowController()
    workflow = await workflow_controller.create_definition(
        db_session,
        WorkflowDefinitionCreateRequest(definition=_workflow_definition()),
    )
    schedule_controller = ScheduleController()
    schedule = await schedule_controller.create(
        db_session,
        ScheduleCreateRequest(
            name="workflow schedule",
            prompt="",
            cron="* * * * *",
            workflow_id=workflow.id,
            workflow_inputs_template={"topic": "{{ schedule.name }}"},
        ),
    )

    updated = await schedule_controller.update(
        db_session,
        schedule.id,
        ScheduleUpdateRequest(
            prompt="Run ordinary scheduled work.",
            workflow_id=None,
            workflow_inputs_template=None,
        ),
    )

    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    assert updated.execution_mode == "isolate_session"
    assert updated.workflow_id is None
    assert updated.workflow_inputs_template == {}
    assert record.workflow_id is None
    assert record.workflow_inputs_template == {}


async def _node(db_session: AsyncSession, workflow_run_id: str, node_id: str) -> WorkflowNodeRunRecord:
    from sqlalchemy import select

    result = await db_session.execute(
        select(WorkflowNodeRunRecord).where(
            WorkflowNodeRunRecord.workflow_run_id == workflow_run_id,
            WorkflowNodeRunRecord.node_id == node_id,
        )
    )
    node = result.scalar_one()
    assert isinstance(node, WorkflowNodeRunRecord)
    return node
