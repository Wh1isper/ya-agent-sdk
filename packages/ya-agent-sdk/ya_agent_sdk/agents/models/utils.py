from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx
from pydantic_ai.models import get_user_agent
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.retry import RetryBaseT

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_DEFAULT_RETRY_ATTEMPTS = 5
_DEFAULT_RETRY_BACKOFF_MULTIPLIER = 1.0
_DEFAULT_RETRY_MAX_WAIT_SECONDS = 30.0
_DEFAULT_RETRY_AFTER_MAX_WAIT_SECONDS = 300.0


@dataclass(frozen=True)
class ModelRequestRetryOptions:
    """Retry policy for transient model provider HTTP requests."""

    enabled: bool = True
    attempts: int = _DEFAULT_RETRY_ATTEMPTS
    backoff_multiplier: float = _DEFAULT_RETRY_BACKOFF_MULTIPLIER
    max_wait_seconds: float = _DEFAULT_RETRY_MAX_WAIT_SECONDS
    retry_after_max_wait_seconds: float = _DEFAULT_RETRY_AFTER_MAX_WAIT_SECONDS
    status_codes: frozenset[int] = field(default_factory=lambda: DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES)

    @property
    def should_retry(self) -> bool:
        return self.enabled and self.attempts > 1


def env_model_request_retry_options() -> ModelRequestRetryOptions:
    """Read model request retry options from YA_AGENT_MODEL_REQUEST_RETRY_* env vars."""

    return ModelRequestRetryOptions(
        enabled=_env_bool("YA_AGENT_MODEL_REQUEST_RETRY_ENABLED", default=True),
        attempts=_env_int("YA_AGENT_MODEL_REQUEST_RETRY_ATTEMPTS", default=_DEFAULT_RETRY_ATTEMPTS, minimum=1),
        backoff_multiplier=_env_float(
            "YA_AGENT_MODEL_REQUEST_RETRY_BACKOFF_MULTIPLIER",
            default=_DEFAULT_RETRY_BACKOFF_MULTIPLIER,
            minimum=0.0,
        ),
        max_wait_seconds=_env_float(
            "YA_AGENT_MODEL_REQUEST_RETRY_MAX_WAIT_SECONDS",
            default=_DEFAULT_RETRY_MAX_WAIT_SECONDS,
            minimum=0.0,
        ),
        retry_after_max_wait_seconds=_env_float(
            "YA_AGENT_MODEL_REQUEST_RETRY_AFTER_MAX_WAIT_SECONDS",
            default=_DEFAULT_RETRY_AFTER_MAX_WAIT_SECONDS,
            minimum=0.0,
        ),
        status_codes=_env_status_codes(
            "YA_AGENT_MODEL_REQUEST_RETRY_STATUS_CODES",
            default=DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES,
        ),
    )


def create_async_http_client(
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 900,
    connect: int = 5,
    read: int = 300,
    retry_options: ModelRequestRetryOptions | None = None,
) -> httpx.AsyncClient:
    """Create a new httpx.AsyncClient with optional extra headers and model-request retries.

    Each call creates a new client instance. When used through a pydantic-ai Provider,
    the provider manages the client's lifecycle.
    """

    headers = {"User-Agent": get_user_agent()}
    if extra_headers:
        headers.update(extra_headers)

    request_timeout = httpx.Timeout(timeout=timeout, connect=connect, read=read)
    transport = create_model_request_retry_transport(retry_options=retry_options)
    if transport is None:
        return httpx.AsyncClient(timeout=request_timeout, headers=headers)

    return httpx.AsyncClient(
        transport=transport,
        timeout=request_timeout,
        headers=headers,
    )


def create_model_request_retry_transport(
    *,
    retry_options: ModelRequestRetryOptions | None = None,
    wrapped: httpx.AsyncBaseTransport | None = None,
) -> AsyncTenacityTransport | None:
    """Create pydantic-ai's tenacity transport for retryable model HTTP requests."""

    options = retry_options or env_model_request_retry_options()
    if not options.should_retry:
        return None
    return AsyncTenacityTransport(
        config=build_model_request_retry_config(options),
        wrapped=wrapped,
        validate_response=lambda response: validate_model_retry_response(response, options),
    )


def build_model_request_retry_config(
    options: ModelRequestRetryOptions | None = None,
    *,
    retry: RetryBaseT | None = None,
) -> RetryConfig:
    """Build a tenacity retry config shared by HTTP and WebSocket model transports."""

    resolved = options or env_model_request_retry_options()
    retry_condition = retry or retry_if_exception(lambda exc: is_retryable_model_request_exception(exc, resolved))
    return RetryConfig(
        retry=retry_condition,
        wait=wait_retry_after(
            fallback_strategy=wait_exponential(
                multiplier=resolved.backoff_multiplier,
                max=resolved.max_wait_seconds,
            ),
            max_wait=resolved.retry_after_max_wait_seconds,
        ),
        stop=stop_after_attempt(resolved.attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def validate_model_retry_response(response: httpx.Response, options: ModelRequestRetryOptions | None = None) -> None:
    """Raise HTTPStatusError only for HTTP statuses that should be retried."""

    resolved = options or env_model_request_retry_options()
    if response.status_code in resolved.status_codes:
        response.raise_for_status()


def is_retryable_model_request_exception(
    exc: BaseException,
    options: ModelRequestRetryOptions | None = None,
) -> bool:
    """Return whether a model request exception is transient enough to retry."""

    resolved = options or env_model_request_retry_options()
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in resolved.status_codes
    return isinstance(exc, httpx.RequestError | httpx.StreamError)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, *, default: int, minimum: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _env_float(name: str, *, default: float, minimum: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = float(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _env_status_codes(name: str, *, default: Iterable[int]) -> frozenset[int]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return frozenset(default)
    status_codes: set[int] = set()
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        status = int(stripped)
        if status < 100 or status > 599:
            raise ValueError(f"{name} contains invalid HTTP status code: {status}")
        status_codes.add(status)
    if not status_codes:
        raise ValueError(f"{name} must contain at least one HTTP status code")
    return frozenset(status_codes)


__all__ = [
    "DEFAULT_MODEL_REQUEST_RETRY_STATUS_CODES",
    "ModelRequestRetryOptions",
    "build_model_request_retry_config",
    "create_async_http_client",
    "create_model_request_retry_transport",
    "env_model_request_retry_options",
    "is_retryable_model_request_exception",
    "validate_model_retry_response",
]
