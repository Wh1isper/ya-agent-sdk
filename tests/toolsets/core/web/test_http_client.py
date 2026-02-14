"""Tests for ya_agent_sdk.toolsets.core.web._http_client module."""

import pytest
from inline_snapshot import snapshot

from ya_agent_sdk.toolsets.core.web._http_client import (
    ForbiddenUrlError,
    _get_http_client,
    check_url_accessible,
    get_http_client,
    safe_request,
    verify_url,
)

# =============================================================================
# verify_url tests
# =============================================================================


def test_verify_url_valid_https() -> None:
    """Should accept valid HTTPS URLs."""
    verify_url("https://example.com")
    verify_url("https://api.github.com/users")


def test_verify_url_valid_http() -> None:
    """Should accept valid HTTP URLs."""
    verify_url("http://example.com")


def test_verify_url_rejects_unsupported_scheme() -> None:
    """Should reject non-HTTP schemes."""
    with pytest.raises(ForbiddenUrlError) as exc_info:
        verify_url("ftp://example.com")
    assert "Unsupported scheme" in str(exc_info.value)


def test_verify_url_rejects_localhost() -> None:
    """Should reject localhost URLs."""
    with pytest.raises(ForbiddenUrlError) as exc_info:
        verify_url("http://localhost:8080")
    assert "localhost" in str(exc_info.value)

    with pytest.raises(ForbiddenUrlError):
        verify_url("http://127.0.0.1:8080")


def test_verify_url_rejects_private_ip() -> None:
    """Should reject private IP ranges."""
    private_ips = [
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.168.1.1",
        "http://169.254.0.1",
    ]
    for url in private_ips:
        with pytest.raises(ForbiddenUrlError) as exc_info:
            verify_url(url)
        assert "private IP" in str(exc_info.value).lower() or "forbidden" in str(exc_info.value).lower()


def test_verify_url_accepts_public_ip() -> None:
    """Should accept public IP addresses."""
    verify_url("http://8.8.8.8")
    verify_url("https://1.1.1.1")


# =============================================================================
# get_http_client tests
# =============================================================================


def test_get_http_client_returns_client() -> None:
    """Should return an httpx.AsyncClient."""
    client = get_http_client()
    assert client is not None
    assert not client.is_closed


def test_get_http_client_caches_instance() -> None:
    """Should return the same client instance."""
    client1 = get_http_client()
    client2 = get_http_client()
    assert client1 is client2


def test_get_http_client_recreates_after_close() -> None:
    """Should create new client if previous was closed."""
    client1 = get_http_client()
    # Simulate close by clearing cache
    _get_http_client.cache_clear()
    client2 = get_http_client()
    assert client1 is not client2


# =============================================================================
# safe_request tests
# =============================================================================


async def test_safe_request_success(httpx_mock) -> None:
    """Should make successful request."""
    httpx_mock.add_response(url="https://example.com/api", json={"status": "ok"})

    response = await safe_request("https://example.com/api")
    assert response.status_code == 200
    assert response.json() == snapshot({"status": "ok"})


async def test_safe_request_follows_redirects(httpx_mock) -> None:
    """Should follow redirects safely."""
    httpx_mock.add_response(
        url="https://example.com/old",
        status_code=302,
        headers={"Location": "https://example.com/new"},
    )
    httpx_mock.add_response(url="https://example.com/new", json={"redirected": True})

    response = await safe_request("https://example.com/old")
    assert response.json() == snapshot({"redirected": True})


async def test_safe_request_blocks_redirect_to_private(httpx_mock) -> None:
    """Should block redirects to private IPs."""
    httpx_mock.add_response(
        url="https://example.com/redirect",
        status_code=302,
        headers={"Location": "http://192.168.1.1/secret"},
    )

    with pytest.raises(ForbiddenUrlError):
        await safe_request("https://example.com/redirect")


async def test_safe_request_rejects_forbidden_url() -> None:
    """Should reject forbidden URLs before making request."""
    with pytest.raises(ForbiddenUrlError):
        await safe_request("http://localhost:8080/api")


# =============================================================================
# check_url_accessible tests
# =============================================================================


async def test_check_url_accessible_success(httpx_mock) -> None:
    """Should return True for accessible URLs."""
    httpx_mock.add_response(url="https://example.com/image.png", method="HEAD")

    result = await check_url_accessible("https://example.com/image.png")
    assert result is True


async def test_check_url_accessible_fallback_to_get(httpx_mock) -> None:
    """Should fallback to GET if HEAD returns 405."""
    httpx_mock.add_response(url="https://example.com/file", method="HEAD", status_code=405)
    httpx_mock.add_response(url="https://example.com/file", method="GET", status_code=200)

    result = await check_url_accessible("https://example.com/file")
    assert result is True


async def test_check_url_accessible_failure(httpx_mock) -> None:
    """Should return False for inaccessible URLs."""
    httpx_mock.add_response(url="https://example.com/missing", method="HEAD", status_code=404)

    result = await check_url_accessible("https://example.com/missing")
    assert result is False


async def test_check_url_accessible_forbidden_url() -> None:
    """Should return False for forbidden URLs."""
    result = await check_url_accessible("http://192.168.1.1/secret")
    assert result is False
