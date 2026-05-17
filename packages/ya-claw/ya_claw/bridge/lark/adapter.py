from __future__ import annotations

import asyncio
import base64
import contextlib
import http
import json
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeAlias

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.bridge.base import BridgeAdapter, BridgeMessageHandler
from ya_claw.bridge.context_snapshot import (
    BridgePreviousMessageSnapshotItem,
    BridgePreviousMessagesSnapshot,
    BridgeSnapshotRelation,
)
from ya_claw.bridge.lark.card import build_hitl_card, build_recovery_card
from ya_claw.bridge.lark.normalizer import normalize_lark_action, normalize_lark_event
from ya_claw.bridge.lark.snapshot import (
    int_value,
    lark_message_content_text,
    limit_snapshot_items,
    sort_snapshot_items,
    speaker_for_lark_sender,
    string_value,
    truncate_text,
)
from ya_claw.bridge.models import BridgeAdapterType, BridgeDispatchResult, BridgeEventStatus, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.controller.hitl import HitlController
from ya_claw.controller.models import ActiveInteraction
from ya_claw.notifications import NotificationHub

LarkSdkObject: TypeAlias = Any

_LARK_CARD_ACTION_ACK = {"toast": {"type": "info", "content": "YA Claw is processing your response."}}


class LarkBridgeAdapter(BridgeAdapter):
    def __init__(
        self,
        *,
        settings: ClawSettings,
        handler: BridgeMessageHandler,
        notification_hub: NotificationHub | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._settings = settings
        self._handler = handler
        self._notification_hub = notification_hub
        self._session_factory = session_factory
        self._hitl_controller = HitlController()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: LarkSdkObject | None = None
        self._app_id: str | None = None
        self._app_secret: str | None = None
        self._stopping = False
        self._pending_submissions: set[Future[object]] = set()
        self._hitl_messages: dict[str, str] = {}
        self._recovery_messages: dict[str, str] = {}

    @property
    def adapter_type(self) -> BridgeAdapterType:
        return BridgeAdapterType.LARK

    async def run(self) -> None:
        app_id = self._settings.bridge_lark_app_id
        app_secret = self._settings.bridge_lark_app_secret_value
        if app_id is None or app_id.strip() == "" or app_secret is None:
            raise RuntimeError("Lark bridge requires YA_CLAW_BRIDGE_LARK_APP_ID and YA_CLAW_BRIDGE_LARK_APP_SECRET.")
        logger.info(
            "Starting Lark bridge adapter domain={} event_types={}",
            self._settings.bridge_lark_domain,
            self._settings.resolved_bridge_lark_event_types,
        )
        self._loop = asyncio.get_running_loop()
        self._app_id = app_id.strip()
        self._app_secret = app_secret
        self._stopping = False
        notification_task: asyncio.Task[None] | None = None
        if self._notification_hub is not None:
            notification_task = asyncio.create_task(self._consume_notifications(), name="ya-claw-lark-hitl-cards")
        try:
            await asyncio.to_thread(self._run_websocket_client, self._app_id, app_secret)
        finally:
            if notification_task is not None:
                notification_task.cancel()
                await asyncio.gather(notification_task, return_exceptions=True)
            self._client = None
            self._loop = None
            self._app_id = None
            self._app_secret = None
            logger.info("Lark bridge adapter stopped")

    async def stop(self) -> None:
        logger.info("Stopping Lark bridge adapter pending_submissions={}", len(self._pending_submissions))
        self._stopping = True
        for future in list(self._pending_submissions):
            future.cancel()
        self._pending_submissions.clear()
        await asyncio.to_thread(_stop_lark_ws_loop)

    async def _consume_notifications(self) -> None:
        notification_hub = self._notification_hub
        if notification_hub is None:
            return
        async for event in notification_hub.stream():
            payload = json.loads(event["data"])
            event_type = payload.get("type")
            body = payload.get("payload") if isinstance(payload, dict) else None
            if not isinstance(body, dict):
                continue
            if event_type == "run.updated" and body.get("session_status_reason") == "hitl_pending":
                await self._present_hitl(body)
            elif event_type == "run.updated" and body.get("session_status_reason") == "run_failed":
                await self._present_recovery(body)
            elif event_type == "run.hitl.responded":
                await self._update_hitl_card(body)
            elif event_type == "run.recovery.submitted":
                await self._update_recovery_card(body)

    async def _present_hitl(self, payload: dict[str, Any]) -> None:
        interaction = _current_interaction(payload.get("session_status_detail"))
        if interaction is None:
            return
        chat_id = _chat_id_from_payload(payload)
        if chat_id is None:
            return
        await self._send_or_update_card(chat_id=chat_id, run_id=interaction.run_id, interaction=interaction)

    async def _present_recovery(self, payload: dict[str, Any]) -> None:
        run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            return
        chat_id = _chat_id_from_payload(payload)
        if chat_id is None:
            chat_id = await self._chat_id_from_run(run_id)
        if chat_id is None:
            return
        card = build_recovery_card(payload)
        existing_message_id = self._recovery_messages.get(run_id)
        if isinstance(existing_message_id, str) and existing_message_id.strip():
            await asyncio.to_thread(self._patch_lark_card, existing_message_id, card)
            return
        message_id = await asyncio.to_thread(self._send_lark_card, chat_id, card)
        if isinstance(message_id, str) and message_id.strip():
            self._recovery_messages[run_id] = message_id

    async def _update_recovery_card(self, payload: dict[str, Any]) -> None:
        action = payload.get("action")
        source_run_id = payload.get("source_run_id")
        if not isinstance(action, str) or not isinstance(source_run_id, str):
            return
        message_id = self._recovery_messages.get(source_run_id)
        if message_id is None:
            return
        card = build_recovery_card({**payload, "run_id": source_run_id}, submitted_action=action)
        await asyncio.to_thread(self._patch_lark_card, message_id, card)
        self._recovery_messages.pop(source_run_id, None)

    async def _update_hitl_card(self, payload: dict[str, Any]) -> None:
        run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            return
        current_raw = payload.get("current_interaction")
        interaction = ActiveInteraction.model_validate(current_raw) if isinstance(current_raw, dict) else None
        message_id = await self._get_hitl_message_id(run_id=run_id, interaction=interaction)
        if message_id is None:
            return
        card = build_hitl_card(interaction, completed=interaction is None)
        await asyncio.to_thread(self._patch_lark_card, message_id, card)
        if interaction is None:
            await self._mark_hitl_message_completed(run_id=run_id)

    async def _send_or_update_card(self, *, chat_id: str, run_id: str, interaction: ActiveInteraction) -> None:
        card = build_hitl_card(interaction)
        existing_message_id = await self._get_hitl_message_id(run_id=run_id, interaction=interaction)
        if existing_message_id is not None:
            await asyncio.to_thread(self._patch_lark_card, existing_message_id, card)
            await self._upsert_hitl_message(chat_id=chat_id, message_id=existing_message_id, interaction=interaction)
            return
        message_id = await asyncio.to_thread(self._send_lark_card, chat_id, card)
        if isinstance(message_id, str) and message_id.strip():
            self._hitl_messages[run_id] = message_id
            await self._upsert_hitl_message(chat_id=chat_id, message_id=message_id, interaction=interaction)

    async def _get_hitl_message_id(self, *, run_id: str, interaction: ActiveInteraction | None) -> str | None:
        cached = self._hitl_messages.get(run_id)
        if isinstance(cached, str) and cached.strip():
            return cached
        session_factory = self._session_factory
        if session_factory is None:
            return None
        tenant_key = _tenant_key_from_interaction(interaction)
        async with session_factory() as db_session:
            record = await self._hitl_controller.get_bridge_hitl_message(
                db_session,
                adapter=BridgeAdapterType.LARK,
                tenant_key=tenant_key,
                run_id=run_id,
            )
            if record is None:
                return None
            self._hitl_messages[run_id] = record.external_message_id
            return record.external_message_id

    async def _upsert_hitl_message(self, *, chat_id: str, message_id: str, interaction: ActiveInteraction) -> None:
        session_factory = self._session_factory
        if session_factory is None:
            return
        async with session_factory() as db_session:
            batch = await self._hitl_controller.get_pending_batch_for_run(db_session, interaction.run_id)
            await self._hitl_controller.upsert_bridge_hitl_message(
                db_session,
                adapter=BridgeAdapterType.LARK,
                tenant_key=_tenant_key_from_interaction(interaction),
                external_chat_id=chat_id,
                external_message_id=message_id,
                session_id=interaction.session_id,
                run_id=interaction.run_id,
                batch_id=batch.id if batch is not None else None,
                interaction_id=interaction.interaction_id,
                status="active",
            )
            await db_session.commit()

    async def _chat_id_from_run(self, run_id: str) -> str | None:
        session_factory = self._session_factory
        if session_factory is None:
            return None
        from sqlalchemy import select

        from ya_claw.orm.tables import BridgeConversationRecord, RunRecord

        async with session_factory() as db_session:
            run_record = await db_session.get(RunRecord, run_id)
            if not isinstance(run_record, RunRecord):
                return None
            result = await db_session.execute(
                select(BridgeConversationRecord)
                .where(BridgeConversationRecord.session_id == run_record.session_id)
                .limit(1)
            )
            conversation = result.scalar_one_or_none()
            if not isinstance(conversation, BridgeConversationRecord):
                return None
            return conversation.external_chat_id

    async def _mark_hitl_message_completed(self, *, run_id: str) -> None:
        session_factory = self._session_factory
        if session_factory is None:
            self._hitl_messages.pop(run_id, None)
            return
        cached_message_id = self._hitl_messages.get(run_id)
        async with session_factory() as db_session:
            record = None
            if cached_message_id is not None:
                from sqlalchemy import select

                from ya_claw.orm.tables import BridgeHitlMessageRecord

                result = await db_session.execute(
                    select(BridgeHitlMessageRecord).where(
                        BridgeHitlMessageRecord.adapter == BridgeAdapterType.LARK,
                        BridgeHitlMessageRecord.external_message_id == cached_message_id,
                    )
                )
                loaded = result.scalar_one_or_none()
                if isinstance(loaded, BridgeHitlMessageRecord):
                    record = loaded
            if record is None:
                record = await self._hitl_controller.get_bridge_hitl_message(
                    db_session,
                    adapter=BridgeAdapterType.LARK,
                    tenant_key="default",
                    run_id=run_id,
                )
            if record is not None:
                await self._hitl_controller.upsert_bridge_hitl_message(
                    db_session,
                    adapter=BridgeAdapterType.LARK,
                    tenant_key=record.tenant_key,
                    external_chat_id=record.external_chat_id,
                    external_message_id=record.external_message_id,
                    session_id=record.session_id,
                    run_id=record.run_id,
                    batch_id=record.batch_id,
                    interaction_id=record.interaction_id,
                    status="completed",
                )
                await db_session.commit()
        self._hitl_messages.pop(run_id, None)

    def _send_lark_card(self, chat_id: str, card: dict[str, Any]) -> str | None:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        client = self._openapi_client(lark)
        request = (
            CreateMessageRequest
            .builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody
                .builder()
                .receive_id(chat_id)
                .msg_type("interactive")
                .content(json.dumps(card, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = client.im.v1.message.create(request)
        data = getattr(response, "data", None)
        message_id = getattr(data, "message_id", None)
        if not getattr(response, "success", lambda: False)():
            logger.warning("Failed to send Lark HITL card response={}", response)
        return message_id if isinstance(message_id, str) else None

    def _patch_lark_card(self, message_id: str, card: dict[str, Any]) -> None:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

        client = self._openapi_client(lark)
        request = (
            PatchMessageRequest
            .builder()
            .message_id(message_id)
            .request_body(PatchMessageRequestBody.builder().content(json.dumps(card, ensure_ascii=False)).build())
            .build()
        )
        response = client.im.v1.message.patch(request)
        if not getattr(response, "success", lambda: False)():
            logger.warning("Failed to patch Lark HITL card message_id={} response={}", message_id, response)

    def _openapi_client(self, lark_module: LarkSdkObject) -> LarkSdkObject:
        if self._app_id is None or self._app_secret is None:
            raise RuntimeError("Lark OpenAPI credentials are unavailable.")
        return (
            lark_module.Client
            .builder()
            .app_id(self._app_id)
            .app_secret(self._app_secret)
            .domain(self._settings.bridge_lark_domain)
            .build()
        )

    def _run_websocket_client(self, app_id: str, app_secret: str) -> None:
        import lark_oapi as lark
        from lark_oapi.ws import Client

        def handle_event(data: object) -> None:
            self._handle_lark_payload(lark, data)

        event_handler_builder = lark.EventDispatcherHandler.builder("", "", lark.LogLevel.INFO)
        for event_type in self._settings.resolved_bridge_lark_event_types:
            event_handler_builder.register_p2_customized_event(event_type, handle_event)
        event_handler = event_handler_builder.build()
        client = Client(
            app_id=app_id,
            app_secret=app_secret,
            log_level=lark.LogLevel.INFO,
            event_handler=event_handler,
            domain=self._settings.bridge_lark_domain,
            auto_reconnect=True,
        )
        self._client = client
        self._install_card_action_handler(client, lark)
        with contextlib.suppress(RuntimeError):
            client.start()

    def _handle_lark_payload(self, lark_module: LarkSdkObject, data: object) -> None:
        raw_event = _marshal_lark_payload(lark_module, data)
        action = normalize_lark_action(raw_event)
        if action is not None:
            logger.debug("Accepted Lark HITL action event_id={}", action.event_id)
            self._submit_from_sdk_thread(self._handler.handle_action(action))
            return
        message = normalize_lark_event(raw_event)
        if message is None:
            return
        self._enrich_message_context_snapshot(lark_module, message)
        logger.debug(
            "Accepted Lark bridge event event_id={} chat_id={} message_id={}",
            message.event_id,
            message.chat_id,
            message.message_id,
        )
        self._submit_from_sdk_thread(self._handler.handle_message(message))

    def _enrich_message_context_snapshot(self, lark_module: LarkSdkObject, message: BridgeInboundMessage) -> None:
        if not self._settings.bridge_lark_previous_messages_enabled:
            return
        if message.event_type != "im.message.receive_v1":
            return
        try:
            snapshot = self._build_remote_previous_messages_snapshot(lark_module, message)
        except Exception:
            logger.exception("Failed to build Lark previous messages snapshot message_id={}", message.message_id)
            return
        if snapshot is not None:
            message.metadata["previous_messages_snapshot"] = snapshot.model_dump(mode="json")

    def _build_remote_previous_messages_snapshot(
        self,
        lark_module: LarkSdkObject,
        message: BridgeInboundMessage,
    ) -> BridgePreviousMessagesSnapshot | None:
        client = self._openapi_client(lark_module)
        limit = int(self._settings.bridge_lark_previous_messages_limit)
        max_chars = int(self._settings.bridge_lark_previous_messages_max_chars)
        item_max_chars = int(self._settings.bridge_lark_previous_message_max_chars)
        candidates: list[BridgePreviousMessageSnapshotItem] = []
        seen: set[str] = {message.message_id}

        self._append_parent_snapshot_item(
            lark_module,
            client,
            message=message,
            seen=seen,
            candidates=candidates,
            item_max_chars=item_max_chars,
        )
        self._append_container_snapshot_items(
            lark_module,
            client,
            message=message,
            seen=seen,
            candidates=candidates,
            item_max_chars=item_max_chars,
        )
        candidates = sort_snapshot_items(candidates)
        limited = candidates[-limit:]
        limited, total_truncated = limit_snapshot_items(
            limited,
            max_chars=max_chars,
            item_max_chars=item_max_chars,
        )
        if len(limited) == 0:
            return None
        return BridgePreviousMessagesSnapshot(
            items=limited, truncated=len(candidates) > len(limited) or total_truncated
        )

    def _append_parent_snapshot_item(
        self,
        lark_module: LarkSdkObject,
        client: LarkSdkObject,
        *,
        message: BridgeInboundMessage,
        seen: set[str],
        candidates: list[BridgePreviousMessageSnapshotItem],
        item_max_chars: int,
    ) -> None:
        parent_id = message.parent_id or message.root_id
        if not isinstance(parent_id, str) or not parent_id.strip() or parent_id == message.message_id:
            return
        parent_item = self._get_lark_message_snapshot_item(
            lark_module,
            client,
            parent_id,
            relation="parent",
            item_max_chars=item_max_chars,
        )
        self._append_unique_snapshot_item(candidates, seen=seen, item=parent_item)

    def _append_container_snapshot_items(
        self,
        lark_module: LarkSdkObject,
        client: LarkSdkObject,
        *,
        message: BridgeInboundMessage,
        seen: set[str],
        candidates: list[BridgePreviousMessageSnapshotItem],
        item_max_chars: int,
    ) -> None:
        container_id = message.thread_id or message.root_id
        if isinstance(container_id, str) and container_id.strip():
            thread_items = self._list_lark_message_snapshot_items(
                lark_module,
                client,
                container_id=container_id,
                container_id_type="thread",
                relation="thread",
                item_max_chars=item_max_chars,
            )
            for item in thread_items:
                self._append_unique_snapshot_item(candidates, seen=seen, item=item)
        chat_items = self._list_lark_message_snapshot_items(
            lark_module,
            client,
            container_id=message.chat_id,
            container_id_type="chat",
            relation="chat_recent",
            item_max_chars=item_max_chars,
        )
        current_create_time = int_value(message.create_time)
        for item in chat_items:
            item_create_time = int_value(item.create_time)
            if (
                current_create_time is not None
                and item_create_time is not None
                and item_create_time > current_create_time
            ):
                continue
            self._append_unique_snapshot_item(candidates, seen=seen, item=item)

    def _append_unique_snapshot_item(
        self,
        candidates: list[BridgePreviousMessageSnapshotItem],
        *,
        seen: set[str],
        item: BridgePreviousMessageSnapshotItem | None,
    ) -> None:
        if item is None:
            return
        if item.message_id in seen:
            return
        candidates.append(item)
        if item.message_id is not None:
            seen.add(item.message_id)

    def _get_lark_message_snapshot_item(
        self,
        lark_module: LarkSdkObject,
        client: LarkSdkObject,
        message_id: str,
        *,
        relation: BridgeSnapshotRelation,
        item_max_chars: int,
    ) -> BridgePreviousMessageSnapshotItem | None:
        from lark_oapi.api.im.v1 import GetMessageRequest

        request = GetMessageRequest.builder().message_id(message_id).build()
        response = client.im.v1.message.get(request)
        if not getattr(response, "success", lambda: False)():
            logger.debug("Failed to get Lark message for snapshot message_id={} response={}", message_id, response)
            return None
        data = getattr(response, "data", None)
        items = getattr(data, "items", None)
        if not isinstance(items, list) or len(items) == 0:
            return None
        return self._snapshot_item_from_lark_message(
            lark_module, items[0], relation=relation, item_max_chars=item_max_chars
        )

    def _list_lark_message_snapshot_items(
        self,
        lark_module: LarkSdkObject,
        client: LarkSdkObject,
        *,
        container_id: str,
        container_id_type: str,
        relation: BridgeSnapshotRelation,
        item_max_chars: int,
    ) -> list[BridgePreviousMessageSnapshotItem]:
        from lark_oapi.api.im.v1 import ListMessageRequest

        page_size = max(int(self._settings.bridge_lark_previous_messages_limit) * 2, 10)
        request = (
            ListMessageRequest
            .builder()
            .container_id_type(container_id_type)
            .container_id(container_id)
            .sort_type("ByCreateTimeDesc")
            .page_size(page_size)
            .build()
        )
        response = client.im.v1.message.list(request)
        if not getattr(response, "success", lambda: False)():
            logger.debug(
                "Failed to list Lark messages for snapshot container_id={} container_id_type={} response={}",
                container_id,
                container_id_type,
                response,
            )
            return []
        data = getattr(response, "data", None)
        raw_items = getattr(data, "items", None)
        if not isinstance(raw_items, list):
            return []
        items: list[BridgePreviousMessageSnapshotItem] = []
        for raw_item in raw_items:
            item = self._snapshot_item_from_lark_message(
                lark_module,
                raw_item,
                relation=relation,
                item_max_chars=item_max_chars,
            )
            if item is not None:
                items.append(item)
        return items

    def _snapshot_item_from_lark_message(
        self,
        lark_module: LarkSdkObject,
        raw_message: LarkSdkObject,
        *,
        relation: BridgeSnapshotRelation,
        item_max_chars: int,
    ) -> BridgePreviousMessageSnapshotItem | None:
        message_id = string_value(getattr(raw_message, "message_id", None))
        message_type = string_value(getattr(raw_message, "msg_type", None))
        sender = getattr(raw_message, "sender", None)
        sender_id = string_value(getattr(sender, "id", None))
        sender_type = string_value(getattr(sender, "sender_type", None))
        create_time = string_value(getattr(raw_message, "create_time", None))
        body = getattr(raw_message, "body", None)
        raw_content = string_value(getattr(body, "content", None))
        content_text = lark_message_content_text(lark_module, message_type=message_type, raw_content=raw_content)
        if content_text is None or content_text.strip() == "":
            return None
        content_text, truncated = truncate_text(content_text, item_max_chars)
        return BridgePreviousMessageSnapshotItem(
            speaker=speaker_for_lark_sender(
                sender_id=sender_id,
                sender_type=sender_type,
                app_id=self._app_id or self._settings.bridge_lark_app_id,
            ),
            relation=relation,
            message_id=message_id,
            sender_id=sender_id,
            sender_type=sender_type,
            message_type=message_type,
            create_time=create_time,
            content_text=content_text,
            truncated=truncated,
        )

    def _install_card_action_handler(self, client: LarkSdkObject, lark_module: LarkSdkObject) -> None:
        async def handle_card_action_frame(frame: LarkSdkObject, headers: LarkSdkObject, payload: bytes) -> None:
            self._handle_lark_payload(lark_module, payload)
            frame.payload = _marshal_lark_ws_response(lark_module, _LARK_CARD_ACTION_ACK)
            await client._write_message(frame.SerializeToString())

        client._handle_card_action_frame = handle_card_action_frame
        original_handle_message = client._handle_message

        async def handle_message_with_card_actions(message: bytes) -> None:
            from lark_oapi.ws.enum import MessageType
            from lark_oapi.ws.exception import HeaderNotFoundException
            from lark_oapi.ws.pb.pbbp2_pb2 import Frame

            frame = Frame()
            try:
                frame.ParseFromString(message)
                header_map = {header.key: header.value for header in frame.headers}
                if header_map.get("type") == MessageType.CARD.value:
                    await client._handle_card_action_frame(frame, frame.headers, frame.payload)
                    return
            except HeaderNotFoundException:
                raise
            except Exception:
                logger.exception("Failed to handle Lark card action frame.")
                frame.payload = _marshal_lark_ws_response(
                    lark_module, None, status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR
                )
                await client._write_message(frame.SerializeToString())
                return
            await original_handle_message(message)

        client._handle_message = handle_message_with_card_actions

    def _submit_from_sdk_thread(self, coroutine: Coroutine[Any, Any, object]) -> None:
        if self._stopping:
            coroutine.close()
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            logger.warning("Dropping Lark bridge message because the runtime loop is unavailable.")
            return
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        logger.debug("Submitted Lark bridge message pending_submissions={}", len(self._pending_submissions) + 1)
        self._pending_submissions.add(future)
        future.add_done_callback(self._complete_submission)

    def _complete_submission(self, future: Future[object]) -> None:
        self._pending_submissions.discard(future)
        if future.cancelled():
            return
        try:
            result = future.result()
            if isinstance(result, BridgeDispatchResult) and result.status == BridgeEventStatus.FAILED:
                logger.warning("Lark bridge handler returned failed result={}", result)
        except Exception:
            logger.exception("Lark bridge message handler failed.")


def _current_interaction(detail: object) -> ActiveInteraction | None:
    if not isinstance(detail, dict):
        return None
    interactions = detail.get("active_interactions")
    if not isinstance(interactions, list):
        return None
    for item in interactions:
        if isinstance(item, dict) and item.get("status") == "pending":
            return ActiveInteraction.model_validate(item)
    return None


def _tenant_key_from_interaction(interaction: ActiveInteraction | None) -> str:
    if interaction is None:
        return "default"
    bridge = interaction.metadata.get("bridge") if isinstance(interaction.metadata, dict) else None
    if isinstance(bridge, dict) and isinstance(bridge.get("tenant_key"), str) and bridge["tenant_key"].strip():
        return bridge["tenant_key"].strip()
    return "default"


def _chat_id_from_payload(payload: dict[str, Any]) -> str | None:
    detail = payload.get("session_status_detail")
    if isinstance(detail, dict):
        interactions = detail.get("active_interactions")
        if isinstance(interactions, list):
            for item in interactions:
                if isinstance(item, dict):
                    bridge = item.get("metadata", {}).get("bridge") if isinstance(item.get("metadata"), dict) else None
                    if isinstance(bridge, dict) and isinstance(bridge.get("chat_id"), str):
                        return bridge["chat_id"]
    return None


def _marshal_lark_payload(lark_module: LarkSdkObject, payload: object) -> dict[str, Any]:
    if isinstance(payload, bytes | bytearray):
        parsed_bytes = lark_module.JSON.unmarshal(bytes(payload), dict)
        return parsed_bytes if isinstance(parsed_bytes, dict) else {}
    raw_json = lark_module.JSON.marshal(payload)
    parsed = lark_module.JSON.unmarshal(raw_json, dict)
    return parsed if isinstance(parsed, dict) else {}


def _marshal_lark_ws_response(
    lark_module: LarkSdkObject,
    data: dict[str, Any] | None,
    *,
    status_code: int = http.HTTPStatus.OK,
) -> bytes:
    from lark_oapi.ws.model import Response

    response = Response(code=int(status_code))
    if data is not None:
        response.data = base64.b64encode(lark_module.JSON.marshal(data).encode("utf-8"))
    return lark_module.JSON.marshal(response).encode("utf-8")


def _stop_lark_ws_loop() -> None:
    with contextlib.suppress(Exception):
        import lark_oapi.ws.client as lark_ws_client

        lark_ws_client.loop.call_soon_threadsafe(lark_ws_client.loop.stop)
