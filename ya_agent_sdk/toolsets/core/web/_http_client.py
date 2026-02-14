"""Shared HTTP client utilities with SSRF protection."""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    pass

# Blocked IP ranges for SSRF protection
_BLOCKED_RANGES: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
)


class ForbiddenUrlError(Exception):
    """Raised when URL is forbidden due to security restrictions."""


def verify_url(url: str) -> None:
    """Verify URL is safe to access (not pointing to internal resources).

    Args:
        url: The URL to verify.

    Raises:
        ForbiddenUrlError: If URL points to internal/private resources.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ForbiddenUrlError(f"Unsupported scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ForbiddenUrlError("Missing hostname")

    # Check for localhost variations
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):  # noqa: S104
        raise ForbiddenUrlError(f"Access to localhost is forbidden: {hostname}")

    # Try to parse as IP address and check against blocked ranges
    try:
        ip = ipaddress.ip_address(hostname)
        for blocked in _BLOCKED_RANGES:
            if ip in blocked:
                raise ForbiddenUrlError(f"Access to private IP range is forbidden: {hostname}")
    except ValueError:
        # Not an IP address, hostname is allowed
        pass


@lru_cache(maxsize=1)
def _get_http_client() -> httpx.AsyncClient:
    """Internal cached client factory."""
    # Connection pool limits: max 100 connections, 20 per host
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    # Timeout: 5 minutes total, 30s connect
    timeout = httpx.Timeout(300.0, connect=30.0)

    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=False,  # Manual redirect handling for SSRF protection
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ya-agent-sdk/1.0)",
        },
    )


def get_http_client() -> httpx.AsyncClient:
    """Get shared async HTTP client instance.

    Handles closed client scenario by invalidating cache and creating new instance.

    Returns:
        Shared httpx.AsyncClient instance.
    """
    client = _get_http_client()
    if client.is_closed:
        # Invalidate cache and create new instance
        _get_http_client.cache_clear()
        client = _get_http_client()
    return client


async def safe_request(
    url: str,
    method: str = "GET",
    max_redirects: int = 10,
    skip_verification: bool = False,
    **kwargs,
) -> httpx.Response:
    """Make HTTP request with SSRF protection and safe redirect following.

    Args:
        url: Target URL.
        method: HTTP method.
        max_redirects: Maximum redirects to follow.
        skip_verification: Skip SSRF URL verification (for internal network environments).
        **kwargs: Additional arguments passed to httpx.

    Returns:
        HTTP response.

    Raises:
        ForbiddenUrlError: If URL or redirect target is forbidden.
        httpx.HTTPError: For HTTP errors.
    """
    if not skip_verification:
        verify_url(url)
    client = get_http_client()
    current_url = url

    for _ in range(max_redirects):
        response = await client.request(method, current_url, **kwargs)

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            if not location:
                raise httpx.HTTPStatusError(
                    "Redirect missing Location header",
                    request=response.request,
                    response=response,
                )

            # Handle relative redirects
            if not location.startswith(("http://", "https://")):
                location = urljoin(current_url, location)

            if not skip_verification:
                verify_url(location)
            current_url = location
            # For 303, always use GET
            if response.status_code == 303:
                method = "GET"
        else:
            return response

    raise httpx.TooManyRedirects(f"Exceeded {max_redirects} redirects")


async def check_url_accessible(url: str, timeout: float = 5.0, skip_verification: bool = False) -> bool:
    """Check if URL is accessible via HEAD request.

    Args:
        url: URL to check.
        timeout: Request timeout in seconds.
        skip_verification: Skip SSRF URL verification (for internal network environments).

    Returns:
        True if accessible, False otherwise.
    """
    try:
        response = await safe_request(url, method="HEAD", timeout=timeout, skip_verification=skip_verification)
        # Some servers don't support HEAD, try GET
        if response.status_code == 405:
            response = await safe_request(url, method="GET", timeout=timeout, skip_verification=skip_verification)
        response.raise_for_status()
        return True
    except (httpx.HTTPError, ForbiddenUrlError):
        return False
