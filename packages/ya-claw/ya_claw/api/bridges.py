from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.bridge.controller import BridgeController
from ya_claw.bridge.models import (
    BridgeAdapterType,
    BridgeConversationListResponse,
    BridgeDispatchResult,
    BridgeEventListResponse,
    BridgeEventStatus,
    BridgeInboundAction,
    BridgeInboundMessage,
)
from ya_claw.config import ClawSettings
from ya_claw.controller.bridge import BridgeQueryController
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(prefix="/bridges", tags=["bridges"])
controller = BridgeQueryController()
inbound_controller = BridgeController()


@router.post("/inbound/messages", response_model=BridgeDispatchResult)
async def ingest_bridge_message(request: Request, payload: BridgeInboundMessage) -> BridgeDispatchResult:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    dispatcher = RunDispatcher(request.app.state.execution_supervisor)
    async with session_factory() as db_session:
        return await inbound_controller.handle_inbound_message(
            db_session,
            settings,
            runtime_state,
            dispatcher,
            payload,
        )


@router.post("/inbound/actions", response_model=BridgeDispatchResult)
async def ingest_bridge_action(request: Request, payload: BridgeInboundAction) -> BridgeDispatchResult:
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        result = await inbound_controller.handle_inbound_action(db_session, runtime_state, payload)
    notification_hub = getattr(request.app.state, "notification_hub", None)
    if notification_hub is not None and result.run_id is not None:
        await notification_hub.publish(
            "run.hitl.responded",
            {
                "session_id": result.session_id,
                "run_id": result.run_id,
                "status": "responded",
                "remaining_interaction_count": result.remaining_interaction_count,
                "current_interaction": result.current_interaction.model_dump(mode="json")
                if result.current_interaction is not None
                else None,
            },
        )
    return result


@router.get("/conversations", response_model=BridgeConversationListResponse)
async def list_bridge_conversations(
    request: Request,
    adapter: BridgeAdapterType | None = None,
    tenant_key: str | None = None,
    external_chat_id: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
) -> BridgeConversationListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_conversations(
            db_session,
            adapter=adapter,
            tenant_key=tenant_key,
            external_chat_id=external_chat_id,
            session_id=session_id,
            limit=limit,
        )


@router.get("/events", response_model=BridgeEventListResponse)
async def list_bridge_events(
    request: Request,
    adapter: BridgeAdapterType | None = None,
    tenant_key: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    external_chat_id: str | None = None,
    status: BridgeEventStatus | None = None,
    limit: int = 100,
) -> BridgeEventListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_events(
            db_session,
            adapter=adapter,
            tenant_key=tenant_key,
            conversation_id=conversation_id,
            session_id=session_id,
            run_id=run_id,
            external_chat_id=external_chat_id,
            status=status,
            limit=limit,
        )


def _get_settings(request: Request) -> ClawSettings:
    settings = request.app.state.settings
    if not isinstance(settings, ClawSettings):
        raise HTTPException(status_code=503, detail="Application settings are unavailable.")
    return settings


def _get_runtime_state(request: Request) -> InMemoryRuntimeState:
    runtime_state = request.app.state.runtime_state
    if not isinstance(runtime_state, InMemoryRuntimeState):
        raise HTTPException(status_code=503, detail="Runtime state is unavailable.")
    return runtime_state


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise HTTPException(status_code=503, detail="Database session factory is unavailable.")
    return session_factory
