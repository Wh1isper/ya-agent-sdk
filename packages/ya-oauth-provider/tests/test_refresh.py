from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from ya_oauth.types import OAuthAccount, TokenSnapshot
from ya_oauth_provider.refresh import (
    OAuthRefreshSupervisor,
    oauth_provider_name_from_model,
    oauth_provider_names_from_models,
)

ACCESS_TOKEN = "fixture-access-token"  # noqa: S105


async def wait_for_condition(condition: Callable[[], bool], *, timeout_seconds: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


class FakeTokenSource:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.refresh_count = 0

    async def get_token(self) -> TokenSnapshot:
        return TokenSnapshot(provider_name="codex", access_token=ACCESS_TOKEN, account=OAuthAccount())

    async def refresh_token(self) -> TokenSnapshot:
        self.refresh_count += 1
        if self.fail:
            raise RuntimeError("refresh failed")
        return TokenSnapshot(
            provider_name="codex",
            access_token=f"{ACCESS_TOKEN}-{self.refresh_count}",
            account=OAuthAccount(),
        )


def test_oauth_provider_name_from_model() -> None:
    assert oauth_provider_name_from_model("oauth@codex:gpt-5.5") == "codex"
    assert oauth_provider_name_from_model("openai:gpt-4o") is None
    assert oauth_provider_name_from_model("oauth@codex") is None
    assert oauth_provider_name_from_model(None) is None


def test_oauth_provider_names_from_models() -> None:
    assert oauth_provider_names_from_models(["oauth@codex:gpt-5.5", "openai:gpt-4o", "oauth@codex:gpt-5.5"]) == {
        "codex"
    }


@pytest.mark.asyncio
async def test_refresh_supervisor_refresh_once_success() -> None:
    source = FakeTokenSource()
    supervisor = OAuthRefreshSupervisor({"codex": source})

    result = await supervisor.refresh_once()
    status = supervisor.status().providers["codex"]

    assert isinstance(result["codex"], TokenSnapshot)
    assert source.refresh_count == 1
    assert status.refresh_count == 1
    assert status.failure_count == 0
    assert status.last_success_at is not None
    assert status.last_error is None


@pytest.mark.asyncio
async def test_refresh_supervisor_refresh_once_failure() -> None:
    source = FakeTokenSource(fail=True)
    supervisor = OAuthRefreshSupervisor({"codex": source})

    result = await supervisor.refresh_once()
    status = supervisor.status().providers["codex"]

    assert isinstance(result["codex"], RuntimeError)
    assert source.refresh_count == 1
    assert status.refresh_count == 0
    assert status.failure_count == 1
    assert status.last_failure_at is not None
    assert status.last_error == "refresh failed"


@pytest.mark.asyncio
async def test_refresh_supervisor_start_and_shutdown() -> None:
    source = FakeTokenSource()
    supervisor = OAuthRefreshSupervisor(
        {"codex": source},
        interval_seconds=60,
        failure_retry_seconds=1,
        refresh_on_startup=True,
    )

    await supervisor.start()
    await wait_for_condition(lambda: source.refresh_count == 1)
    await supervisor.shutdown()

    assert source.refresh_count == 1
    assert not supervisor.is_running
