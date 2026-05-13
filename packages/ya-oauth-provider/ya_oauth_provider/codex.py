from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from ya_oauth.codex import CODEX_BASE_URL, create_codex_token_source
from ya_oauth.types import OAuthTokenSource

from ya_oauth_provider.http import OAuthBearerAuth


def infer_oauth_model(provider_name: str, model_name: str, *, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer an OAuth-backed model from `oauth@provider:model` parts."""
    if provider_name == "codex":
        return build_codex_model(model_name, extra_headers=extra_headers)
    raise KeyError(f"Unknown OAuth provider: {provider_name}")


def build_codex_model(
    model_name: str,
    *,
    token_source: OAuthTokenSource | None = None,
    extra_headers: dict[str, str] | None = None,
    base_url: str = CODEX_BASE_URL,
) -> Model:
    """Build a Codex OAuth-backed OpenAI Responses model."""
    import httpx
    from pydantic_ai.models import get_user_agent
    from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig
    from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

    source = token_source or create_codex_token_source()
    http_client = httpx.AsyncClient(
        auth=OAuthBearerAuth(source, provider_name="codex", extra_headers=extra_headers),
        headers={"User-Agent": get_user_agent()},
        timeout=httpx.Timeout(timeout=900, connect=5, read=300),
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type((httpx.HTTPError, httpx.StreamError)),
                wait=wait_exponential(multiplier=1, max=10),
                stop=stop_after_attempt(10),
                reraise=True,
            )
        ),
    )
    provider = OpenAIProvider(api_key="oauth-managed", base_url=base_url, http_client=http_client)
    return CodexResponsesModel(model_name, provider=provider, profile=_codex_profile())


class CodexResponsesModel(OpenAIResponsesModel):
    """Codex Responses API model that requires streaming calls."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise UserError(
            "Codex OAuth Responses API requires streaming. "
            "Use agent.run_stream(), agent.iter(), or ya_agent_sdk.stream_agent()."
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as response:
            yield response


def _codex_profile() -> OpenAIModelProfile:
    return OpenAIModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_thinking=True,
        thinking_always_enabled=True,
        openai_supports_reasoning=True,
        openai_supports_encrypted_reasoning_content=True,
        openai_supports_strict_tool_definition=True,
        openai_responses_requires_function_call_status_none=True,
    )


def build_session_headers(session_id: str | None, thread_id: str | None) -> dict[str, str]:
    """Build Codex session/thread headers with underscore and hyphen variants."""
    headers: dict[str, str] = {}
    if session_id:
        headers["session_id"] = session_id
        headers["session-id"] = session_id
    if thread_id:
        headers["thread_id"] = thread_id
        headers["thread-id"] = thread_id
        headers["x-client-request-id"] = thread_id
    return headers
