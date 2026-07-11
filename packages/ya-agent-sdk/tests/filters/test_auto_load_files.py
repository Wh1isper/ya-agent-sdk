"""Tests for ya_agent_sdk.filters.auto_load_files module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.filters.auto_load_files import process_auto_load_files
from ya_agent_sdk.filters.handoff import process_handoff_message
from ya_agent_sdk.toolsets.core.context.handoff import HandoffTool


async def test_no_auto_load_files(tmp_path: Path) -> None:
    """Should return unchanged history when no auto_load_files is set."""
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

            result = await process_auto_load_files(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 1


async def test_auto_load_files_injects_prompt_without_reading_contents() -> None:
    """Should inject only file paths and never read file contents."""
    deps = MagicMock()
    deps.auto_load_files = ["test.txt"]
    deps.file_operator.read_file = AsyncMock(return_value="secret file content")
    mock_ctx = MagicMock()
    mock_ctx.deps = deps

    request = ModelRequest(parts=[UserPromptPart(content="Continue")])
    history = [request]

    result = await process_auto_load_files(mock_ctx, history)

    assert result == history
    assert len(request.parts) == 2
    assert isinstance(request.parts[1], UserPromptPart)
    content = request.parts[1].content
    assert isinstance(content, str)
    assert '<files-to-inspect contents-loaded="false">' in content
    assert '<file path="test.txt"' in content
    assert "secret file content" not in content
    assert "untrusted inert data" in content
    deps.file_operator.read_file.assert_not_awaited()
    assert deps.auto_load_files == []


async def test_auto_load_files_injects_multiple_paths(tmp_path: Path) -> None:
    """Should include every requested path without including file contents."""
    (tmp_path / "file1.txt").write_text("Content 1")
    (tmp_path / "file2.txt").write_text("Content 2")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["file1.txt", "file2.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            await process_auto_load_files(mock_ctx, history)

            assert len(request.parts) == 2
            content = request.parts[1].content
            assert isinstance(content, str)
            assert 'path="file1.txt"' in content
            assert 'path="file2.txt"' in content
            assert "Content 1" not in content
            assert "Content 2" not in content


async def test_auto_load_files_does_not_require_existing_file() -> None:
    """Should preserve a missing path as a reminder without trying to open it."""
    async with AgentContext() as ctx:
        ctx.auto_load_files = ["nonexistent.txt"]

        mock_ctx = MagicMock()
        mock_ctx.deps = ctx

        request = ModelRequest(parts=[UserPromptPart(content="Continue")])
        history = [request]

        await process_auto_load_files(mock_ctx, history)

        assert len(request.parts) == 2
        content = request.parts[1].content
        assert isinstance(content, str)
        assert 'path="nonexistent.txt"' in content
        assert "Failed to load" not in content
        assert ctx.auto_load_files == []


async def test_auto_load_files_escapes_paths() -> None:
    """Should XML-escape paths instead of allowing prompt structure injection."""
    async with AgentContext() as ctx:
        ctx.auto_load_files = ['src/<unsafe>&"file.py']

        mock_ctx = MagicMock()
        mock_ctx.deps = ctx

        request = ModelRequest(parts=[UserPromptPart(content="Continue")])
        await process_auto_load_files(mock_ctx, [request])

        content = request.parts[1].content
        assert isinstance(content, str)
        assert 'path="src/&lt;unsafe&gt;&amp;&quot;file.py"' in content


async def test_auto_load_files_injects_into_tool_return(tmp_path: Path) -> None:
    """Should inject into the last request even if it contains tool return parts."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["test.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            user_request = ModelRequest(parts=[UserPromptPart(content="Do something")])
            response = ModelResponse(parts=[TextPart(content="Response")])
            tool_return = ModelRequest(parts=[ToolReturnPart(tool_call_id="tc1", tool_name="tool", content="result")])

            history = [user_request, response, tool_return]
            await process_auto_load_files(mock_ctx, history)

            assert len(tool_return.parts) == 2
            assert isinstance(tool_return.parts[1], UserPromptPart)
            assert "files-to-inspect" in tool_return.parts[1].content
            assert ctx.auto_load_files == []


async def test_auto_load_files_without_file_operator() -> None:
    """Should inject the path reminder even when no file operator is available."""
    async with AgentContext() as ctx:
        ctx.auto_load_files = ["test.txt"]

        mock_ctx = MagicMock()
        mock_ctx.deps = ctx

        request = ModelRequest(parts=[UserPromptPart(content="Hello")])
        history = [request]

        result = await process_auto_load_files(mock_ctx, history)

        assert result == history
        assert len(request.parts) == 2
        assert isinstance(request.parts[1], UserPromptPart)
        assert 'path="test.txt"' in request.parts[1].content
        assert ctx.auto_load_files == []


async def test_auto_load_files_empty_history() -> None:
    """Should keep pending paths when no request exists for reminder injection."""
    async with AgentContext() as ctx:
        ctx.auto_load_files = ["test.txt"]

        mock_ctx = MagicMock()
        mock_ctx.deps = ctx

        result = await process_auto_load_files(mock_ctx, [])

        assert result == []
        assert ctx.auto_load_files == ["test.txt"]


async def test_handoff_pipeline_injects_paths_without_file_contents(tmp_path: Path) -> None:
    """Should carry summary file hints through restore as prompt-only paths."""
    (tmp_path / "important.py").write_text("sensitive implementation details")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            run_ctx = MagicMock()
            run_ctx.deps = ctx
            ctx.user_prompts = "Continue implementation"

            await HandoffTool().call(
                run_ctx,
                content="## Current State\nImplementation is in progress.",
                auto_load_files=["important.py"],
            )
            restored = await process_handoff_message(
                run_ctx,
                [ModelRequest(parts=[UserPromptPart(content="old history")])],
            )
            result = await process_auto_load_files(run_ctx, restored)

            assert len(result) == 1
            restored_request = result[0]
            assert isinstance(restored_request, ModelRequest)
            contents = [part.content for part in restored_request.parts if isinstance(part, UserPromptPart)]
            text_contents = [content for content in contents if isinstance(content, str)]
            assert any("Implementation is in progress" in content for content in text_contents)
            assert any('path="important.py"' in content for content in text_contents)
            assert all("sensitive implementation details" not in content for content in text_contents)
            assert ctx.handoff_message is None
            assert ctx.auto_load_files == []
