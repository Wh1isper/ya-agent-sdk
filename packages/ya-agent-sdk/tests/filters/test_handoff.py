"""Tests for ya_agent_sdk.filters.handoff module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.events import HandoffCompleteEvent, HandoffStartEvent
from ya_agent_sdk.filters.handoff import _build_handoff_messages, process_handoff_message


async def test_process_handoff_no_handoff_message(tmp_path: Path) -> None:
    """Should return unchanged history when no handoff message is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 1


async def test_process_handoff_with_handoff_message(tmp_path: Path) -> None:
    """Should build restored context request with handoff summary."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Previous context summary here"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue task")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            assert len(result) == 1

            restored = result[0]
            assert isinstance(restored, ModelRequest)
            assert any(isinstance(p, SystemPromptPart) for p in restored.parts)

            user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
            assert any("Previous context summary here" in p.content for p in user_parts)
            assert any("context-restored" in p.content for p in user_parts)
            assert any("summarize tool has already completed this handoff" in p.content for p in user_parts)

            assert ctx.handoff_message is None


async def test_process_handoff_retains_applied_logical_input(tmp_path: Path) -> None:
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"
            retained = ModelRequest(parts=[UserPromptPart(content="Focus on tests")])
            ctx.run_input_ledger.record_initial([retained])

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx
            history = [ModelRequest(parts=[UserPromptPart(content="Continue")])]

            result = await process_handoff_message(mock_ctx, history)

            assert len(result) == 2
            restored = result[0]
            assert isinstance(restored, ModelRequest)
            user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
            assert any("context-restored" in p.content for p in user_parts)
            assert result[1] == retained
            assert result[1] is not retained


async def test_process_handoff_without_user_prompts(tmp_path: Path) -> None:
    """Should produce valid restored request without user_prompts."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            assert len(result) == 1

            restored = result[0]
            assert isinstance(restored, ModelRequest)
            user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
            assert all("original-request" not in p.content for p in user_parts)
            assert any(p.content == "Summary" for p in user_parts)


async def test_process_handoff_empty_history(tmp_path: Path) -> None:
    """Should produce restored request even with empty history."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            result = await process_handoff_message(mock_ctx, [])

            assert len(result) == 1
            assert ctx.handoff_message is None


async def test_process_handoff_last_message_has_user_prompt(tmp_path: Path) -> None:
    """Last message should have UserPromptPart for files_to_inspect compatibility."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            result = await process_handoff_message(mock_ctx, [request])

            last_msg = result[-1]
            assert isinstance(last_msg, ModelRequest)
            assert any(isinstance(p, UserPromptPart) for p in last_msg.parts)


async def test_process_handoff_emits_events(tmp_path: Path) -> None:
    """Should emit HandoffStartEvent and HandoffCompleteEvent with handoff content."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx._stream_queue_enabled = True
            ctx.handoff_message = "Test handoff content"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            assert len(result) == 1

            events = []
            while not ctx.agent_stream_queues[ctx.agent_id].empty():
                events.append(await ctx.agent_stream_queues[ctx.agent_id].get())

            assert len(events) == 2

            start_event = events[0]
            assert isinstance(start_event, HandoffStartEvent)
            assert start_event.message_count == 1

            complete_event = events[1]
            assert isinstance(complete_event, HandoffCompleteEvent)
            assert complete_event.handoff_content == "Test handoff content"
            assert complete_event.original_message_count == 1
            assert start_event.event_id == complete_event.event_id


async def test_process_handoff_no_events_when_streaming_disabled(tmp_path: Path) -> None:
    """Should keep event queue empty when streaming is disabled."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            assert ctx._stream_queue_enabled is False
            ctx.handoff_message = "Test content"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            assert len(result) == 1
            assert ctx.agent_stream_queues[ctx._agent_id].empty()


def test_build_handoff_messages_basic() -> None:
    """Should build one restored request with summary and reminder."""
    result = _build_handoff_messages("Test summary")

    assert len(result) == 1

    restored = result[0]
    assert isinstance(restored, ModelRequest)
    assert any(isinstance(p, SystemPromptPart) for p in restored.parts)
    user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
    assert any(p.content == "Test summary" for p in user_parts)
    assert any("context-restored" in p.content for p in user_parts)
    assert any("summarize tool has already completed this handoff" in p.content for p in user_parts)


def test_build_handoff_messages_with_retained_input() -> None:
    retained = ModelRequest(parts=[UserPromptPart(content="Build a CLI tool")])

    result = _build_handoff_messages("Summary", [retained])

    restored = result[0]
    assert isinstance(restored, ModelRequest)
    user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
    assert any("context-restored" in p.content for p in user_parts)
    assert result[1] == retained
    assert result[1] is not retained


# =============================================================================
# Keep tag: _build_handoff_messages tagging
# =============================================================================


def test_build_handoff_messages_tags_with_keep_handoff() -> None:
    """Handoff restored request should have keep:handoff metadata."""
    from ya_agent_sdk.filters._builders import KEEP_HANDOFF, KEEP_TAG_KEY

    result = _build_handoff_messages("Summary")

    assert len(result) == 1
    restored = result[0]
    assert isinstance(restored, ModelRequest)
    assert restored.metadata is not None
    assert restored.metadata[KEEP_TAG_KEY] == KEEP_HANDOFF


def test_build_handoff_messages_uses_restoration_marker() -> None:
    result = _build_handoff_messages("Summary")

    restored = result[0]
    assert isinstance(restored, ModelRequest)
    user_parts = [p for p in restored.parts if isinstance(p, UserPromptPart)]
    assert any("context-restored" in p.content for p in user_parts)
