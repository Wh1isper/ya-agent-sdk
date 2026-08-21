from __future__ import annotations

import sys
import types
from typing import cast

import pytest
from pydantic_ai import ModelSettings
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import AgentContext, ModelConfig, ModelFeature


def test_agent_context_model_extra_headers_defaults_to_run_id() -> None:
    ctx = AgentContext(run_id="run-1")

    assert ctx.get_model_extra_headers() == {
        "session_id": "run-1",
        "session-id": "run-1",
        "x-session-id": "run-1",
        "thread_id": "run-1",
        "thread-id": "run-1",
        "x-client-request-id": "run-1",
    }


def test_agent_context_model_extra_headers_uses_provider_ids() -> None:
    ctx = AgentContext(run_id="run-1", provider_session_id="session-1", provider_thread_id="thread-1")

    assert ctx.get_model_extra_headers()["session_id"] == "session-1"
    assert ctx.get_model_extra_headers()["session-id"] == "session-1"
    assert ctx.get_model_extra_headers()["x-session-id"] == "session-1"
    assert ctx.get_model_extra_headers()["thread_id"] == "thread-1"
    assert ctx.get_model_extra_headers()["thread-id"] == "thread-1"
    assert ctx.get_model_extra_headers()["x-client-request-id"] == "thread-1"


def test_infer_openai_responses_rs_uses_sdk_websocket_builder(monkeypatch) -> None:
    from ya_agent_sdk.agents.models import websocket as websocket_models

    calls = []

    def fake_build(model_name: str, *, extra_headers=None):  # type: ignore[no-untyped-def]
        calls.append((model_name, extra_headers))
        return "ws-model"

    monkeypatch.setattr(websocket_models, "build_openai_responses_websocket_model", fake_build)

    assert infer_model("openai-responses-rs:gpt-5") == "ws-model"
    assert infer_model("openai-responses-ws:gpt-5-mini", extra_headers={"x-session-id": "session-1"}) == "ws-model"
    assert calls == [("gpt-5", None), ("gpt-5-mini", {"x-session-id": "session-1"})]


def test_infer_oauth_model_lazy_import(monkeypatch) -> None:
    calls = []
    module = types.ModuleType("ya_oauth_provider")

    def fake_infer(provider_name: str, model_name: str, *, extra_headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
        calls.append((provider_name, model_name, extra_headers))
        return "model"

    module.infer_oauth_model = fake_infer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ya_oauth_provider", module)

    assert infer_model("oauth@codex:gpt-5.5", extra_headers={"session_id": "s1"}) == "model"
    assert calls == [("codex", "gpt-5.5", {"session_id": "s1"})]


async def test_create_agent_passes_context_headers_to_context_header_models(monkeypatch) -> None:
    calls = []

    def fake_infer(model, extra_headers=None):  # type: ignore[no-untyped-def]
        calls.append((model, extra_headers))
        return None

    monkeypatch.setattr("ya_agent_sdk.agents.main.infer_model", fake_infer)

    models = [
        "openai-chat:gpt-4o",
        "oauth@codex:gpt-5.5",
        "openai-responses-ws:gpt-5",
        "gateway@openai-responses:gpt-5",
        "gateway@openai-responses-ws:gpt-5",
        "gateway@openai-chat:gpt-5",
        "gateway@anthropic:claude-sonnet-4",
    ]
    for model in models:
        runtime = create_agent(model, model_cfg=ModelConfig(context_window=1000))
        async with runtime:
            pass

    assert calls[0] == ("openai-chat:gpt-4o", None)
    for model, extra_headers in calls[1:5]:
        assert model in {
            "oauth@codex:gpt-5.5",
            "openai-responses-ws:gpt-5",
            "gateway@openai-responses:gpt-5",
            "gateway@openai-responses-ws:gpt-5",
        }
        assert extra_headers is not None
        assert extra_headers["session_id"]
        assert extra_headers["x-session-id"] == extra_headers["session_id"]
    assert calls[5] == ("gateway@openai-chat:gpt-5", None)
    assert calls[6] == ("gateway@anthropic:claude-sonnet-4", None)


@pytest.mark.parametrize(
    "model",
    [
        "oauth@codex:gpt-5.5",
        "openai-responses-rs:gpt-5",
        "openai-responses-ws:gpt-5",
        "gateway@openai-responses:gpt-5",
        "gateway@openai-responses-rs:gpt-5",
        "gateway@openai-responses-ws:gpt-5",
    ],
)
async def test_create_agent_binds_capable_prompt_cache_key_to_session_header(monkeypatch, model: str) -> None:
    monkeypatch.setattr(
        "ya_agent_sdk.agents.main.infer_model",
        lambda *_args, **_kwargs: __import__("pydantic_ai.models.test", fromlist=["TestModel"]).TestModel(),
    )
    configured_settings = cast(
        ModelSettings,
        {
            "temperature": 0.1,
            "openai_prompt_cache_key": "stale-session",
            "extra_headers": {"X-Session-ID": "header-session", "x-other": "value"},
            "extra_body": {"prompt_cache_key": "body-session", "other": "value"},
        },
    )

    runtime = create_agent(
        model,
        model_settings=configured_settings,
        model_cfg=ModelConfig(
            context_window=1000,
            capabilities={ModelFeature.openai_prompt_cache_key},
        ),
        context_kwargs={"provider_session_id": "session-1"},
    )

    assert runtime.ctx.get_model_extra_headers()["x-session-id"] == "session-1"
    async with runtime:
        assert runtime.agent.model_settings == {
            "temperature": 0.1,
            "openai_prompt_cache_key": "session-1",
            "extra_headers": {"x-other": "value", "x-session-id": "session-1"},
            "extra_body": {"other": "value"},
        }
    assert configured_settings["openai_prompt_cache_key"] == "stale-session"
    assert configured_settings["extra_headers"] == {
        "X-Session-ID": "header-session",
        "x-other": "value",
    }
    assert configured_settings["extra_body"] == {
        "prompt_cache_key": "body-session",
        "other": "value",
    }


async def test_create_agent_does_not_patch_prompt_cache_key_without_capability(monkeypatch) -> None:
    monkeypatch.setattr(
        "ya_agent_sdk.agents.main.infer_model",
        lambda *_args, **_kwargs: __import__("pydantic_ai.models.test", fromlist=["TestModel"]).TestModel(),
    )

    runtime = create_agent(
        "oauth@codex:gpt-5.5",
        model_cfg=ModelConfig(context_window=1000),
        context_kwargs={"provider_session_id": "session-1"},
    )

    async with runtime:
        assert runtime.agent.model_settings is None


async def test_create_agent_does_not_patch_prompt_cache_key_without_context_headers(monkeypatch) -> None:
    monkeypatch.setattr(
        "ya_agent_sdk.agents.main.infer_model",
        lambda *_args, **_kwargs: __import__("pydantic_ai.models.test", fromlist=["TestModel"]).TestModel(),
    )
    configured_settings = cast(ModelSettings, {"openai_prompt_cache_key": "configured-session"})

    runtime = create_agent(
        "openai-responses:gpt-5",
        model_settings=configured_settings,
        model_cfg=ModelConfig(
            context_window=1000,
            capabilities={ModelFeature.openai_prompt_cache_key},
        ),
    )

    async with runtime:
        assert runtime.agent.model_settings == configured_settings


def test_infer_oauth_model_rejects_invalid_string() -> None:
    with pytest.raises(ValueError, match="oauth@provider:model"):
        infer_model("oauth@codex")


def test_infer_model_rejects_ambiguous_openai_provider() -> None:
    with pytest.raises(ValueError, match=r"openai-chat:<model>.*openai-responses:<model>"):
        infer_model("openai:gpt-4o")


@pytest.mark.parametrize(
    ("legacy_model", "normalized_model"),
    [
        ("google:gemini-2.5-pro", "google:gemini-2.5-pro"),
        ("google-gla:gemini-2.5-pro", "google-cloud:gemini-2.5-pro"),
        ("google-vertex:gemini-2.5-pro", "google-cloud:gemini-2.5-pro"),
        ("google-custom:gemini-2.5-pro", "google-cloud:gemini-2.5-pro"),
    ],
)
def test_infer_model_normalizes_legacy_google_provider_aliases(
    legacy_model: str,
    normalized_model: str,
    monkeypatch,
) -> None:
    calls = []

    def fake_legacy_infer(model, *_args):  # type: ignore[no-untyped-def]
        calls.append(model)
        return model

    monkeypatch.setattr("ya_agent_sdk.agents.models.legacy_infer_model", fake_legacy_infer)

    assert infer_model(legacy_model) == normalized_model
    assert calls == [normalized_model]
