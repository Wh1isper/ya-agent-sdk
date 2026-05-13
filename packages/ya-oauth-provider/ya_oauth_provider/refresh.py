from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ya_oauth.types import OAuthTokenSource, TokenSnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OAuthRefreshProviderStatus:
    provider_name: str
    refresh_count: int = 0
    failure_count: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error: str | None = None


@dataclass(slots=True)
class OAuthRefreshSupervisorStatus:
    running: bool
    provider_count: int
    providers: dict[str, OAuthRefreshProviderStatus] = field(default_factory=dict)


class OAuthRefreshSupervisor:
    """Periodically refresh OAuth token sources in the background."""

    def __init__(
        self,
        token_sources: dict[str, OAuthTokenSource],
        *,
        interval_seconds: float = 30 * 60,
        failure_retry_seconds: float = 60,
        refresh_on_startup: bool = True,
        name: str = "oauth-refresh",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if failure_retry_seconds <= 0:
            raise ValueError("failure_retry_seconds must be positive")
        self._token_sources = dict(token_sources)
        self._interval_seconds = interval_seconds
        self._failure_retry_seconds = failure_retry_seconds
        self._refresh_on_startup = refresh_on_startup
        self._name = name
        self._task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        self._status = OAuthRefreshSupervisorStatus(
            running=False,
            provider_count=len(self._token_sources),
            providers={
                provider_name: OAuthRefreshProviderStatus(provider_name=provider_name)
                for provider_name in self._token_sources
            },
        )

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._token_sources)

    def status(self) -> OAuthRefreshSupervisorStatus:
        return OAuthRefreshSupervisorStatus(
            running=self.is_running,
            provider_count=len(self._token_sources),
            providers={
                name: OAuthRefreshProviderStatus(
                    provider_name=status.provider_name,
                    refresh_count=status.refresh_count,
                    failure_count=status.failure_count,
                    last_success_at=status.last_success_at,
                    last_failure_at=status.last_failure_at,
                    last_error=status.last_error,
                )
                for name, status in self._status.providers.items()
            },
        )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.is_running:
                return
            self._stopped = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name=self._name)

    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            task = self._task
            if task is None:
                return
            self._stopped.set()
        await asyncio.gather(task, return_exceptions=True)
        async with self._lifecycle_lock:
            if self._task is task:
                self._task = None

    async def refresh_once(self) -> dict[str, TokenSnapshot | Exception]:
        results: dict[str, TokenSnapshot | Exception] = {}
        for provider_name, token_source in self._token_sources.items():
            try:
                snapshot = await token_source.refresh_token()
            except Exception as exc:
                self._record_failure(provider_name, exc)
                results[provider_name] = exc
            else:
                self._record_success(provider_name)
                results[provider_name] = snapshot
        return results

    async def _run(self) -> None:
        if self._refresh_on_startup:
            await self.refresh_once()
        while True:
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._next_sleep_seconds())
                return
            except TimeoutError:
                await self.refresh_once()

    def _next_sleep_seconds(self) -> float:
        if any(_last_attempt_failed(status) for status in self._status.providers.values()):
            return self._failure_retry_seconds
        return self._interval_seconds

    def _record_success(self, provider_name: str) -> None:
        status = self._status.providers[provider_name]
        status.refresh_count += 1
        status.last_success_at = datetime.now(UTC)
        status.last_error = None
        logger.debug("OAuth refresh succeeded provider=%s count=%s", provider_name, status.refresh_count)

    def _record_failure(self, provider_name: str, exc: BaseException) -> None:
        status = self._status.providers[provider_name]
        status.failure_count += 1
        status.last_failure_at = datetime.now(UTC)
        status.last_error = str(exc)
        logger.warning(
            "OAuth refresh failed provider=%s failure_count=%s error=%s", provider_name, status.failure_count, exc
        )


def _last_attempt_failed(status: OAuthRefreshProviderStatus) -> bool:
    if status.last_failure_at is None:
        return False
    if status.last_success_at is None:
        return True
    return status.last_failure_at > status.last_success_at


def oauth_provider_names_from_models(models: Iterable[str]) -> set[str]:
    providers: set[str] = set()
    for model in models:
        provider_name = oauth_provider_name_from_model(model)
        if provider_name:
            providers.add(provider_name)
    return providers


def oauth_provider_name_from_model(model: str | None) -> str | None:
    if not isinstance(model, str) or not model.startswith("oauth@"):
        return None
    provider_name, separator, model_name = model.removeprefix("oauth@").partition(":")
    if separator != ":" or provider_name == "" or model_name == "":
        return None
    return provider_name


def create_oauth_refresh_supervisor_for_models(
    models: Iterable[str],
    *,
    interval_seconds: float = 30 * 60,
    failure_retry_seconds: float = 60,
    refresh_on_startup: bool = True,
) -> OAuthRefreshSupervisor | None:
    token_sources: dict[str, OAuthTokenSource] = {}
    for provider_name in sorted(oauth_provider_names_from_models(models)):
        if provider_name == "codex":
            from ya_oauth.codex import create_codex_token_source

            token_sources[provider_name] = create_codex_token_source()
    if not token_sources:
        return None
    return OAuthRefreshSupervisor(
        token_sources,
        interval_seconds=interval_seconds,
        failure_retry_seconds=failure_retry_seconds,
        refresh_on_startup=refresh_on_startup,
    )
