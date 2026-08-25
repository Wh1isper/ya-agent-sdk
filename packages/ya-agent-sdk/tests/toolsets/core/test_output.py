"""Tests for shared bounded-output helpers."""

from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core._output import write_tmp_output


async def test_write_tmp_output_preserves_text_bytes() -> None:
    """Text spills should bypass platform newline translation."""
    context = MagicMock(spec=AgentContext)
    file_operator = AsyncMock()
    output_path = PurePosixPath("workspace/.tmp/shell-stdout.log")
    context.file_operator = file_operator
    context.tmp_dir = output_path.parent
    context.resolve_tmp_path.return_value = output_path

    result = await write_tmp_output(
        context,
        prefix="shell-stdout",
        content="first\r\nsecond\r\n",
        extension="log",
    )

    assert result == str(output_path)
    written_path, written_content = file_operator.write_file.await_args.args
    assert written_path == str(output_path)
    assert written_content == b"first\r\nsecond\r\n"
