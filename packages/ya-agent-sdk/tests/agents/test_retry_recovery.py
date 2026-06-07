"""Tests for SDK retry recovery helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai import BinaryContent, ImageUrl, VideoUrl
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    CompactionPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from ya_agent_sdk.agents.retry_recovery import (
    heal_context_overflow_history,
    heal_openai_item_reference_history,
    recover_retry_message_history,
)
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.environment.local import LocalEnvironment


def test_heal_openai_item_reference_history_drops_server_side_ids() -> None:
    history = [
        ModelResponse(
            parts=[
                ThinkingPart(
                    content="summary",
                    id="rs_123",
                    signature="encrypted_sig",
                    provider_name="openai",
                    provider_details={"raw_content": ["raw"], "encrypted_content": "secret", "keep": "value"},
                ),
                TextPart(content="answer", id="msg_123", provider_name="openai"),
                ToolCallPart(
                    tool_name="lookup",
                    args={"query": "x"},
                    tool_call_id="call_123|fc_123",
                    id="fc_123",
                    provider_name="openai",
                    provider_details={"encrypted_content": "tool-secret", "namespace": "ns"},
                ),
            ],
            provider_name="openai",
            provider_response_id="resp_123",
            conversation_id="conv_123",
            provider_details={"conversation_id": "conv_123", "encrypted_content": "response-secret", "keep": "value"},
        )
    ]

    healed, changed = heal_openai_item_reference_history(history)

    assert changed is True
    response = healed[0]
    assert isinstance(response, ModelResponse)
    assert response.provider_response_id is None
    assert response.conversation_id is None
    assert response.provider_details == {"keep": "value"}

    thinking = response.parts[0]
    assert isinstance(thinking, ThinkingPart)
    assert thinking.id is None
    assert thinking.signature is None
    assert thinking.provider_details == {"keep": "value"}

    text = response.parts[1]
    assert isinstance(text, TextPart)
    assert text.id is None

    tool_call = response.parts[2]
    assert isinstance(tool_call, ToolCallPart)
    assert tool_call.id is None
    assert tool_call.tool_call_id == "call_123"
    assert tool_call.provider_details == {"namespace": "ns"}


def test_heal_openai_item_reference_history_cleans_compaction_parts() -> None:
    history = [
        ModelResponse(
            parts=[
                CompactionPart(
                    content="summary",
                    id="msg_1",
                    provider_name="openai",
                    provider_details={"encrypted_content": "secret", "keep": "value"},
                )
            ],
            provider_name="openai",
            provider_response_id="resp_1",
        )
    ]

    healed, changed = heal_openai_item_reference_history(history)

    assert changed is True
    response = healed[0]
    assert isinstance(response, ModelResponse)
    compaction = response.parts[0]
    assert isinstance(compaction, CompactionPart)
    assert compaction.id is None
    assert compaction.provider_details == {"keep": "value"}


def test_heal_openai_item_reference_history_updates_matching_tool_results() -> None:
    history = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="lookup",
                    args={"query": "x"},
                    tool_call_id="call_1|fc_1",
                    id="fc_1",
                    provider_name="openai",
                )
            ],
            provider_name="openai",
            provider_response_id="resp_1",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="lookup", tool_call_id="call_1|fc_1", content="result"),
                RetryPromptPart(tool_name="lookup", tool_call_id="call_1|fc_1", content="retry"),
            ]
        ),
    ]

    healed, changed = heal_openai_item_reference_history(history)

    assert changed is True
    response = healed[0]
    assert isinstance(response, ModelResponse)
    tool_call = response.parts[0]
    assert isinstance(tool_call, ToolCallPart)
    assert tool_call.tool_call_id == "call_1"

    request = healed[1]
    assert isinstance(request, ModelRequest)
    assert all(
        isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_call_id == "call_1" for part in request.parts
    )


async def test_recover_retry_message_history_handles_openai_item_not_found(tmp_path: Path) -> None:
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        async with AgentContext(env=env) as ctx:
            history = [
                ModelResponse(
                    parts=[ThinkingPart(content="summary", id="rs_123", signature="encrypted_sig")],
                    provider_name="openai",
                    provider_response_id="resp_123",
                )
            ]
            exc = ModelHTTPError(
                400,
                "openai-responses:gpt-5",
                body={"error": {"message": "Item 'rs_123' was not found.", "code": "item_not_found"}},
            )

            recovered = recover_retry_message_history(exc, history, ctx)

    assert recovered.changed is True
    assert recovered.reasons == ("openai_item_reference",)
    response = recovered.history[0]
    assert isinstance(response, ModelResponse)
    assert response.provider_response_id is None
    thinking = response.parts[0]
    assert isinstance(thinking, ThinkingPart)
    assert thinking.id is None
    assert thinking.signature is None


async def test_heal_context_overflow_history_trims_tool_returns_and_strips_media(tmp_path: Path) -> None:
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        async with AgentContext(env=env, model_cfg=ModelConfig(cold_start_trim_seconds=3600)) as ctx:
            history = [
                ModelRequest(parts=[UserPromptPart(content="run tool")]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name="view", args={}, tool_call_id="call_1")],
                    timestamp=datetime.now(tz=UTC),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="view",
                            tool_call_id="call_1",
                            content="A" * 2_000,
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content="processed tool output")], timestamp=datetime.now(tz=UTC)),
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "please inspect media",
                                BinaryContent(data=b"image", media_type="image/png"),
                                ImageUrl("https://example.com/image.png"),
                                UploadedFile(file_id="file_1", provider_name="openai", media_type="image/png"),
                                VideoUrl("https://example.com/video.mp4"),
                            ]
                        )
                    ]
                ),
            ]

            healed, changed = heal_context_overflow_history(history, ctx)

    assert changed is True
    tool_return = healed[2]
    assert isinstance(tool_return, ModelRequest)
    tool_part = tool_return.parts[0]
    assert isinstance(tool_part, ToolReturnPart)
    assert isinstance(tool_part.content, str)
    assert "truncated" in tool_part.content

    media_request = healed[4]
    assert isinstance(media_request, ModelRequest)
    user_part = media_request.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert user_part.content[0] == "please inspect media"
    assert all(isinstance(item, str) for item in user_part.content)
    assert sum("Media content was removed" in item for item in user_part.content if isinstance(item, str)) == 4


def test_heal_context_overflow_history_replaces_each_response_media_part() -> None:
    history = [
        ModelResponse(
            parts=[
                TextPart(content="before"),
                FilePart(content=BinaryContent(data=b"image", media_type="image/png"), id="img_1"),
                TextPart(content="after"),
                FilePart(content=BinaryContent(data=b"video", media_type="video/mp4"), id="vid_1"),
            ]
        )
    ]

    ctx = AgentContext(model_cfg=ModelConfig(cold_start_trim_seconds=3600))
    healed, changed = heal_context_overflow_history(history, ctx)

    assert changed is True
    response = healed[0]
    assert isinstance(response, ModelResponse)
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == [
        "before",
        "<system-reminder>Assistant media content was removed during retry recovery because the "
        "previous request exceeded the model context limit.</system-reminder>",
        "after",
        "<system-reminder>Assistant media content was removed during retry recovery because the "
        "previous request exceeded the model context limit.</system-reminder>",
    ]
    assert not any(isinstance(part, FilePart) for part in response.parts)


def test_heal_context_overflow_history_removes_uploaded_image_video_media() -> None:
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "uploaded files",
                        UploadedFile(file_id="file_image", provider_name="openai", media_type="image/png"),
                        UploadedFile(file_id="file_video", provider_name="openai", media_type="video/mp4"),
                        UploadedFile(file_id="file_pdf", provider_name="openai", media_type="application/pdf"),
                    ]
                )
            ]
        )
    ]

    ctx = AgentContext(model_cfg=ModelConfig(cold_start_trim_seconds=3600))
    healed, changed = heal_context_overflow_history(history, ctx)

    assert changed is True
    request = healed[0]
    assert isinstance(request, ModelRequest)
    user_part = request.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert sum("Media content was removed" in item for item in user_part.content if isinstance(item, str)) == 2
    assert any(isinstance(item, UploadedFile) and item.media_type == "application/pdf" for item in user_part.content)


async def test_recover_retry_message_history_handles_context_overflow(tmp_path: Path) -> None:
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        async with AgentContext(env=env) as ctx:
            history = [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "attached",
                                BinaryContent(data=b"video", media_type="video/mp4"),
                            ]
                        )
                    ]
                )
            ]
            exc = ModelHTTPError(
                400,
                "test-model",
                body={
                    "error": {
                        "message": "This model's maximum context length is 128000 tokens. Please reduce your prompt.",
                        "code": "context_length_exceeded",
                    }
                },
            )

            recovered = recover_retry_message_history(exc, history, ctx)

    assert recovered.changed is True
    assert recovered.reasons == ("context_overflow",)
    request = recovered.history[0]
    assert isinstance(request, ModelRequest)
    user_part = request.parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert "Media content was removed" in user_part.content[1]


async def test_recover_retry_message_history_detects_input_too_long(tmp_path: Path) -> None:
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        async with AgentContext(env=env) as ctx:
            history = [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "attached",
                                BinaryContent(data=b"image", media_type="image/png"),
                            ]
                        )
                    ]
                )
            ]
            exc = ModelHTTPError(
                400,
                "test-model",
                body={"error": {"message": "Input is too long for this model."}},
            )

            recovered = recover_retry_message_history(exc, history, ctx)

    assert recovered.changed is True
    assert recovered.reasons == ("context_overflow",)
