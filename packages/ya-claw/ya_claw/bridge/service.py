from __future__ import annotations

import asyncio
import contextlib

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.bridge.base import BridgeAdapter, BridgeMessageHandler
from ya_claw.bridge.controller import BridgeController
from ya_claw.bridge.lark.adapter import LarkBridgeAdapter
from ya_claw.bridge.models import BridgeAdapterType, BridgeDispatchResult, BridgeInboundAction, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState


class BridgeRuntimeHandler(BridgeMessageHandler):
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        run_dispatcher: RunDispatcher,
        notification_hub: NotificationHub | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._run_dispatcher = run_dispatcher
        self._notification_hub = notification_hub
        self._controller = BridgeController()

    async def handle_message(self, message: BridgeInboundMessage) -> BridgeDispatchResult:
        async with self._session_factory() as db_session:
            return await self._controller.handle_inbound_message(
                db_session,
                self._settings,
                self._runtime_state,
                self._run_dispatcher,
                message,
            )

    async def handle_action(self, action: BridgeInboundAction) -> BridgeDispatchResult:
        async with self._session_factory() as db_session:
            result = await self._controller.handle_inbound_action(
                db_session,
                self._runtime_state,
                action,
            )
        if self._notification_hub is not None and result.run_id is not None:
            await self._notification_hub.publish(
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


class BridgeSupervisor:
    def __init__(self, *, adapters: list[BridgeAdapter]) -> None:
        self._adapters = adapters
        self._tasks: dict[BridgeAdapterType, asyncio.Task[None]] = {}

    @property
    def adapters(self) -> list[BridgeAdapterType]:
        return [adapter.adapter_type for adapter in self._adapters]

    async def startup(self) -> None:
        for adapter in self._adapters:
            if adapter.adapter_type in self._tasks:
                continue
            self._tasks[adapter.adapter_type] = asyncio.create_task(
                adapter.run(),
                name=f"ya-claw-bridge-{adapter.adapter_type}",
            )

    async def shutdown(self) -> None:
        for adapter in self._adapters:
            await adapter.stop()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()


def build_bridge_supervisor(
    *,
    settings: ClawSettings,
    session_factory: async_sessionmaker[AsyncSession],
    runtime_state: InMemoryRuntimeState,
    run_dispatcher: RunDispatcher,
    notification_hub: NotificationHub | None = None,
) -> BridgeSupervisor:
    handler = BridgeRuntimeHandler(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        run_dispatcher=run_dispatcher,
        notification_hub=notification_hub,
    )
    adapters: list[BridgeAdapter] = []
    for adapter_type in sorted(settings.resolved_bridge_enabled_adapters):
        adapters.append(
            _build_adapter(
                adapter_type,
                settings=settings,
                handler=handler,
                notification_hub=notification_hub,
                session_factory=session_factory,
            )
        )
    return BridgeSupervisor(adapters=adapters)


def _build_adapter(
    adapter_type: BridgeAdapterType,
    *,
    settings: ClawSettings,
    handler: BridgeMessageHandler,
    notification_hub: NotificationHub | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> BridgeAdapter:
    if adapter_type == BridgeAdapterType.LARK:
        return LarkBridgeAdapter(
            settings=settings,
            handler=handler,
            notification_hub=notification_hub,
            session_factory=session_factory,
        )
    raise ValueError(f"Unsupported bridge adapter: {adapter_type}")
