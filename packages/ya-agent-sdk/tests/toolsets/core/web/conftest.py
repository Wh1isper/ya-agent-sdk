"""Compatibility helpers for HTTPX2-backed web tool tests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest


@dataclass
class _RegisteredResponse:
    url: str
    method: str | None
    response: httpx2.Response
    used: bool = False


class HTTPX2MockCompat:
    """Small pytest-httpx style facade backed by httpx2.MockTransport."""

    def __init__(self) -> None:
        self._responses: list[_RegisteredResponse] = []
        self._callbacks: list[tuple[Callable[[httpx2.Request], httpx2.Response], bool, bool]] = []

    def add_response(
        self,
        *,
        url: str,
        method: str | None = None,
        status_code: int = 200,
        content: bytes | None = None,
        text: str | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._responses.append(
            _RegisteredResponse(
                url=url,
                method=method.upper() if method else None,
                response=httpx2.Response(
                    status_code,
                    content=content,
                    text=text,
                    json=json,
                    headers=headers,
                ),
            )
        )

    def add_callback(
        self,
        callback: Callable[[httpx2.Request], httpx2.Response],
        *,
        is_reusable: bool = False,
    ) -> None:
        self._callbacks.append((callback, is_reusable, False))

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        request_url = str(request.url)
        request_method = request.method.upper()

        for registered in self._responses:
            if registered.used:
                continue
            if registered.method is not None and registered.method != request_method:
                continue
            if registered.url != request_url:
                continue
            registered.used = True
            return _bind_request(registered.response, request)

        for index, (callback, is_reusable, used) in enumerate(self._callbacks):
            if used and not is_reusable:
                continue
            response = callback(request)
            if not is_reusable:
                self._callbacks[index] = (callback, is_reusable, True)
            return _bind_request(response, request)

        raise AssertionError(f"No mocked HTTPX2 response for {request_method} {request_url}")


def _bind_request(response: httpx2.Response, request: httpx2.Request) -> httpx2.Response:
    try:
        _ = response.request
    except RuntimeError:
        return httpx2.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            extensions=response.extensions,
            request=request,
        )
    return response


@pytest.fixture
def httpx2_mock(monkeypatch: pytest.MonkeyPatch) -> HTTPX2MockCompat:
    from ya_agent_sdk.toolsets.core.web import _http_client
    from ya_agent_sdk.toolsets.core.web import search as search_module

    original_get_http_client = _http_client._get_http_client
    mock = HTTPX2MockCompat()
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(mock.handler),
        follow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ya-agent-sdk/1.0)"},
    )

    original_get_http_client.cache_clear()
    monkeypatch.setattr(_http_client, "_get_http_client", lambda: client)
    monkeypatch.setattr(_http_client, "get_http_client", lambda: client)
    monkeypatch.setattr(search_module, "get_http_client", lambda: client)
    yield mock
    original_get_http_client.cache_clear()
