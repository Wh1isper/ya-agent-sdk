"""Worker-local runtime bindings referenced by durable workflow payloads."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TypeVar, cast

from yaacli.durable.store import SessionStore

_ContextT = TypeVar("_ContextT")


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    context: object
    store: SessionStore


class RuntimeBindingRegistry:
    """Resolve stable execution bindings to live worker authorities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bindings: dict[str, RuntimeBinding] = {}

    def register(self, binding_ref: str, context: object, store: SessionStore) -> None:
        if not binding_ref:
            raise ValueError("binding_ref cannot be empty")
        with self._lock:
            existing = self._bindings.get(binding_ref)
            if existing is not None and (existing.context is not context or existing.store is not store):
                raise ValueError(f"Runtime binding {binding_ref!r} is already registered")
            self._bindings[binding_ref] = RuntimeBinding(context=context, store=store)

    def unregister(self, binding_ref: str, context: object) -> None:
        with self._lock:
            existing = self._bindings.get(binding_ref)
            if existing is not None and existing.context is context:
                del self._bindings[binding_ref]

    def resolve(self, binding_ref: str, context_type: type[_ContextT]) -> _ContextT:
        with self._lock:
            try:
                binding = self._bindings[binding_ref]
            except KeyError as exc:
                raise RuntimeError(
                    f"Runtime binding {binding_ref!r} is unavailable; "
                    "the compatible worker plan must be active in this process"
                ) from exc
        if not isinstance(binding.context, context_type):
            raise TypeError(
                f"Runtime binding {binding_ref!r} has {type(binding.context).__name__}, "
                f"expected {context_type.__name__}"
            )
        return cast(_ContextT, binding.context)

    def get(self, binding_ref: str) -> RuntimeBinding:
        with self._lock:
            try:
                return self._bindings[binding_ref]
            except KeyError as exc:
                raise RuntimeError(f"Runtime binding {binding_ref!r} is unavailable") from exc


runtime_bindings = RuntimeBindingRegistry()
