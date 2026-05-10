from __future__ import annotations

from abc import ABC, abstractmethod

from ya_claw.bridge.models import BridgeAdapterType, BridgeDispatchResult, BridgeInboundAction, BridgeInboundMessage


class BridgeAdapter(ABC):
    @property
    @abstractmethod
    def adapter_type(self) -> BridgeAdapterType:
        raise NotImplementedError

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        return None


class BridgeMessageHandler(ABC):
    @abstractmethod
    async def handle_message(self, message: BridgeInboundMessage) -> BridgeDispatchResult:
        raise NotImplementedError

    async def handle_action(self, action: BridgeInboundAction) -> BridgeDispatchResult:
        raise NotImplementedError
