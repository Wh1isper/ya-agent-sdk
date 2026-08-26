from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from ya_claw.bridge.github.models import GitHubNotification, GitHubNotificationPage, GitHubUser

_GITHUB_API_VERSION = "2022-11-28"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_NOTIFICATIONS_PAGE_SIZE = 50


class GitHubApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubClient(Protocol):
    async def get_authenticated_user(self) -> GitHubUser: ...

    async def list_notifications(self, *, since: datetime) -> GitHubNotificationPage: ...

    async def get_json(self, url: str) -> dict[str, Any]: ...

    async def mark_thread_read(self, thread_id: str) -> None: ...

    async def close(self) -> None: ...


class GitHubRestClient:
    def __init__(
        self,
        *,
        token: str,
        api_url: str = "https://api.github.com",
        user_agent: str = "ya-claw",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized_api_url = api_url.strip().rstrip("/")
        if normalized_api_url == "":
            raise ValueError("GitHub API URL must not be empty.")
        parsed_api_url = httpx.URL(normalized_api_url)
        if parsed_api_url.scheme != "https" or parsed_api_url.host is None:
            raise ValueError("GitHub API URL must use https and include a host.")
        normalized_token = token.strip()
        if normalized_token == "":
            raise ValueError("GitHub token must not be empty.")

        self._api_url = normalized_api_url
        self._api_origin = (parsed_api_url.scheme, parsed_api_url.host, parsed_api_url.port)
        self._client = httpx.AsyncClient(
            base_url=f"{normalized_api_url}/",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {normalized_token}",
                "User-Agent": user_agent,
                "X-GitHub-Api-Version": _GITHUB_API_VERSION,
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    async def get_authenticated_user(self) -> GitHubUser:
        payload = await self.get_json(f"{self._api_url}/user")
        return GitHubUser.model_validate(payload)

    async def list_notifications(self, *, since: datetime) -> GitHubNotificationPage:
        normalized_since = _as_utc(since)
        next_url: str | None = f"{self._api_url}/notifications"
        params: Mapping[str, str | int | bool] | None = {
            "all": "true",
            "since": _github_timestamp(normalized_since),
            "per_page": _NOTIFICATIONS_PAGE_SIZE,
        }
        headers = {"If-Modified-Since": format_datetime(normalized_since, usegmt=True)}
        notifications: list[GitHubNotification] = []
        poll_interval_seconds: int | None = None
        response_date: datetime | None = None

        while next_url is not None:
            response = await self._request("GET", next_url, params=params, headers=headers)
            params = None
            headers = None
            if response_date is None:
                response_date = _http_datetime(response.headers.get("Date"))
            response_poll_interval = _positive_int(response.headers.get("X-Poll-Interval"))
            if response_poll_interval is not None:
                poll_interval_seconds = max(poll_interval_seconds or 0, response_poll_interval)
            if response.status_code == httpx.codes.NOT_MODIFIED:
                break
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubApiError("GitHub notifications response must be a JSON array.")
            notifications.extend(GitHubNotification.model_validate(item) for item in payload)
            next_link = response.links.get("next")
            next_url = next_link.get("url") if isinstance(next_link, dict) else None

        return GitHubNotificationPage(
            notifications=notifications,
            poll_interval_seconds=poll_interval_seconds,
            response_date=response_date,
        )

    async def get_json(self, url: str) -> dict[str, Any]:
        response = await self._request("GET", url)
        payload = response.json()
        if not isinstance(payload, dict):
            raise GitHubApiError(f"GitHub API response from {response.request.url} must be a JSON object.")
        return payload

    async def mark_thread_read(self, thread_id: str) -> None:
        normalized_thread_id = thread_id.strip()
        if normalized_thread_id == "":
            raise ValueError("GitHub notification thread ID must not be empty.")
        await self._request(
            "PATCH",
            f"{self._api_url}/notifications/threads/{quote(normalized_thread_id, safe='')}",
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int | bool] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        self._require_api_origin(url)
        try:
            response = await self._client.request(method, url, params=params, headers=headers)
            if response.status_code == httpx.codes.NOT_MODIFIED:
                return response
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise GitHubApiError(
                f"GitHub API request failed: {method} {exc.request.url} returned {status_code}.",
                status_code=status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise GitHubApiError(f"GitHub API request failed: {method} {url}: {exc}") from exc

    def _require_api_origin(self, url: str) -> None:
        parsed_url = httpx.URL(url)
        origin = (parsed_url.scheme, parsed_url.host, parsed_url.port)
        if origin != self._api_origin:
            raise GitHubApiError(f"Refusing to send GitHub credentials to a foreign API origin: {parsed_url}.")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _github_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _http_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(parsedate_to_datetime(value))
    except (TypeError, ValueError):
        return None


def _positive_int(value: str | None) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
