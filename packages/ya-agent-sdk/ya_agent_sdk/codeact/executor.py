"""Drive Monty's async snapshot API and dispatch external tool calls.

Portions adapted from pydantic-ai-harness 0.14.0,
Copyright (c) 2026 Pydantic Services Inc., used under the MIT License.
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Container, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic_monty import (
    AsyncFunctionSnapshot,
    AsyncFutureSnapshot,
    AsyncNameLookupSnapshot,
    ExternalException,
    ExternalReturnValue,
    ExternalSettledResult,
    MontyComplete,
)

DispatchFn: TypeAlias = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]
AsyncMontyState: TypeAlias = AsyncFunctionSnapshot | AsyncFutureSnapshot | AsyncNameLookupSnapshot | MontyComplete
PendingCall: TypeAlias = asyncio.Task[Any] | Coroutine[Any, Any, Any]


def is_sandbox_panic(exc: BaseException) -> bool:
    """Return whether a Rust-side pyo3 panic escaped the sandbox binding."""

    return type(exc).__name__ == "PanicException"


@dataclass
class MontyExecutor:
    """Single-use host loop for one Monty feed."""

    dispatch: DispatchFn
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set)
    global_sequential: bool = False
    max_concurrency: int = 1
    _pending: dict[int, PendingCall] = field(default_factory=dict, init=False)
    _admission: asyncio.Semaphore = field(init=False, repr=False)
    _global_tail: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)
    _pre_resolved: dict[int, ExternalSettledResult] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        self._admission = asyncio.Semaphore(self.max_concurrency)

    async def run(self, state: AsyncMontyState) -> MontyComplete:
        """Resume snapshots until the feed completes.

        Outstanding host calls are always cancelled and awaited before this
        method returns or propagates cancellation.
        """

        try:
            while not isinstance(state, MontyComplete):
                if isinstance(state, AsyncNameLookupSnapshot):
                    state = await state.resume()
                elif isinstance(state, AsyncFunctionSnapshot):
                    state = await self._handle_function(state)
                else:
                    state = await self._resolve_futures(state)
        finally:
            await self._cancel_pending()
        return state

    async def _cancel_pending(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        for call in self._pending.values():
            if isinstance(call, asyncio.Task):
                call.cancel()
                tasks.append(call)
            else:
                call.close()
        self._pending.clear()
        if not tasks:
            return

        # A second cancellation must not let this executor report cleanup while
        # nested calls still own host resources. Shield the ownership drain,
        # re-request child cancellation, and propagate only after every task
        # has settled. A tool that suppresses cancellation indefinitely
        # violates the CodeAct eligibility contract and necessarily blocks its
        # in-process owner; hard termination requires process isolation.
        drain = asyncio.gather(*tasks, return_exceptions=True)
        interrupted = False
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                interrupted = True
                for task in tasks:
                    if not task.done():
                        task.cancel()
        await drain
        if interrupted:
            raise asyncio.CancelledError

    async def _handle_function(self, snapshot: AsyncFunctionSnapshot) -> AsyncMontyState:
        if snapshot.is_os_function:
            # No OS handler or mount is configured by CodeAct, so Monty's
            # automatic path resolves this to its fail-closed default.
            return await snapshot.resume_auto()

        name = str(snapshot.function_name)
        if name not in self.valid_names:
            return await snapshot.resume({"exception": NameError(f"Unknown function: {name}")})

        # Admission happens before Monty converts args/kwargs into host Python
        # objects. At most max_concurrency argument sets and dispatch tasks can
        # therefore be resident outside the sandbox at once.
        await self._admission.acquire()
        admitted = True
        try:
            if snapshot.args:
                return await snapshot.resume({
                    "exception": TypeError(f"{name}() does not accept positional arguments; use keyword arguments")
                })

            kwargs = snapshot.kwargs
            if name in self.sequential_names:
                for call_id in list(self._pending):
                    self._pre_resolved[call_id] = await _await_external(self._pending.pop(call_id))
                call = self._dispatch_admitted(name, kwargs)
                admitted = False
                return await snapshot.resume(await _await_external(call))

            if self.global_sequential:
                call = self._dispatch_admitted_after(self._global_tail, name, kwargs)
            else:
                call = self._dispatch_admitted(name, kwargs)
            try:
                task = asyncio.ensure_future(call)
            except BaseException:
                call.close()
                raise
            self._pending[snapshot.call_id] = task
            if self.global_sequential:
                self._global_tail = task
            admitted = False
            return await snapshot.resume({"future": ...})
        finally:
            if admitted:
                self._admission.release()

    async def _dispatch_admitted(self, name: str, kwargs: dict[str, Any]) -> Any:
        try:
            return await self.dispatch(name, kwargs)
        finally:
            self._admission.release()

    async def _dispatch_admitted_after(
        self,
        predecessor: asyncio.Task[Any] | None,
        name: str,
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            if predecessor is not None:
                await asyncio.shield(asyncio.gather(predecessor, return_exceptions=True))
            return await self.dispatch(name, kwargs)
        finally:
            self._admission.release()

    async def _resolve_futures(self, snapshot: AsyncFutureSnapshot) -> AsyncMontyState:
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalSettledResult] = {}
        for call_id in pending_ids:
            if call_id in self._pre_resolved:
                results[call_id] = self._pre_resolved.pop(call_id)
            elif self.global_sequential:
                results[call_id] = await _await_external(self._pending.pop(call_id))

        gather_ids = [call_id for call_id in pending_ids if call_id not in results]
        if gather_ids:
            # Shield the aggregate wait so cancellation transfers ownership to
            # run()'s explicit cancel-and-drain path instead of letting
            # asyncio.gather wait indefinitely for a child that temporarily
            # suppresses cancellation.
            settled = await asyncio.shield(
                asyncio.gather(
                    *(self._pending[call_id] for call_id in gather_ids),
                    return_exceptions=True,
                )
            )
            for call_id, outcome in zip(gather_ids, settled, strict=True):
                del self._pending[call_id]
                results[call_id] = _wrap_gathered(outcome)

        return await snapshot.resume(results)


async def _await_external(call: PendingCall) -> ExternalReturnValue | ExternalException:
    try:
        result = await call
    except Exception as exc:
        return ExternalException(exception=exc)
    return ExternalReturnValue(return_value=result)


def _wrap_gathered(outcome: Any) -> ExternalReturnValue | ExternalException:
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):
        raise outcome
    return ExternalReturnValue(return_value=outcome)
