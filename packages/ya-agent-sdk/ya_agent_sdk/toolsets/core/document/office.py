"""Office/EPub to Markdown conversion tool.

Converts Office documents (Word, PowerPoint, Excel) and EPub files to markdown.
Requires optional dependency: markitdown.

Install with: pip install ya-agent-sdk[document]
"""

import base64
import functools
import importlib.util
import re
import tempfile
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

import anyio.to_thread
from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SUPPORTED_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".epub"}


@cache
def _markitdown_available() -> bool:
    """Return whether the Office conversion optional dependency is installed."""
    return importlib.util.find_spec("markitdown") is not None


@cache
def _load_markitdown_class() -> type[Any]:
    """Import MarkItDown on first actual use."""
    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise ImportError(
            "The 'markitdown' package is required for OfficeConvertTool. Install with: pip install ya-agent-sdk[document]"
        ) from e

    return MarkItDown


@cache
def _load_instruction() -> str:
    """Load office convert instruction from prompts/office.md."""
    prompt_file = _PROMPTS_DIR / "office.md"
    return prompt_file.read_text()


async def _run_in_threadpool(func, *args, **kwargs):
    """Run a sync function in a thread pool."""
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


class OfficeConvertTool(BaseTool):
    """Tool for converting Office documents and EPub to markdown."""

    name = "office_to_markdown"
    description = "Convert Office documents (Word, PowerPoint, Excel) and EPub to markdown."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available without importing MarkItDown."""
        if ctx.deps.file_operator is None:
            logger.debug("OfficeConvertTool unavailable: file_operator is not configured")
            return False
        if not _markitdown_available():
            logger.debug("OfficeConvertTool unavailable: markitdown is not installed")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/office.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the document file to convert.")],
    ) -> dict[str, Any]:
        file_op = cast(FileOperator, ctx.deps.file_operator)

        # Check file exists
        if not await file_op.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        # Get file extension from path
        ext = self._get_extension(file_path)
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "error": f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                "success": False,
            }

        # Get file stem from path
        stem = self._get_stem(file_path)

        # Check file size before reading into memory
        max_file_size = ctx.deps.tool_config.document_max_file_size
        try:
            file_stat = await file_op.stat(file_path)
            if file_stat["size"] > max_file_size:
                size_mb = file_stat["size"] / (1024 * 1024)
                limit_mb = max_file_size / (1024 * 1024)
                return {
                    "error": f"File too large: {size_mb:.1f} MB. Maximum supported size is {limit_mb:.0f} MB.",
                    "success": False,
                }
        except Exception as e:
            return {"error": f"Failed to stat file: {e}", "success": False}

        # Step 1: Read source file via file_op into memory
        try:
            file_content = await file_op.read_bytes(file_path)
        except Exception as e:
            return {"error": f"Failed to read file: {e}", "success": False}

        # Step 2: Process in a private local tempdir (markitdown needs local filesystem access)
        # Write-back happens inside the with block so images are read from local disk
        # one-by-one rather than collected into memory all at once.
        with tempfile.TemporaryDirectory() as local_tmp:
            local_tmp_path = Path(local_tmp)
            local_source = local_tmp_path / f"source_{stem}{ext}"
            local_images_dir = local_tmp_path / "images"

            await _run_in_threadpool(local_source.write_bytes, file_content)
            del file_content  # Free source bytes after writing to local disk
            await _run_in_threadpool(local_images_dir.mkdir, exist_ok=True)

            try:
                markitdown_cls = _load_markitdown_class()
                md = markitdown_cls(enable_plugins=True)
                result = await _run_in_threadpool(md.convert, local_source.as_uri(), keep_data_uris=True)
                content = result.text_content
            except Exception as e:
                return {"error": f"Failed to convert document: {e}", "success": False}

            # Extract base64 images and save to local tempdir
            content = self._extract_images(content, local_images_dir)

            # Step 3: Write results back via file_op (inside with block so we can
            # read images from local disk one-by-one without buffering all in memory)
            source_dir = self._get_dir(file_path)
            target_export_dir = f"{source_dir}/export_{stem}" if source_dir else f"export_{stem}"
            md_filename = f"{stem}.md"
            target_md_path = f"{target_export_dir}/{md_filename}"
            target_images_dir = f"{target_export_dir}/images"

            try:
                await file_op.mkdir(target_export_dir, parents=True)
                await file_op.mkdir(target_images_dir, parents=True)

                # Write markdown
                await file_op.write_file(target_md_path, content)

                # Write images one-by-one from local disk
                def list_image_names():
                    return [f.name for f in local_images_dir.iterdir() if f.is_file()]

                image_names = await _run_in_threadpool(list_image_names)
                for img_name in image_names:
                    img_bytes = await _run_in_threadpool((local_images_dir / img_name).read_bytes)
                    await file_op.write_file(f"{target_images_dir}/{img_name}", img_bytes)
            except Exception as e:
                return {"error": f"Failed to write results: {e}", "success": False}

        return {
            "success": True,
            "export_path": target_export_dir,
            "markdown_path": target_md_path,
        }

    def _get_extension(self, file_path: str) -> str:
        """Extract file extension from path string."""
        idx = file_path.rfind(".")
        if idx == -1:
            return ""
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        if idx < last_sep:
            return ""
        return file_path[idx:].lower()

    def _get_stem(self, file_path: str) -> str:
        """Extract file stem (name without extension) from path string."""
        # Get basename
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        basename = file_path[last_sep + 1 :] if last_sep >= 0 else file_path
        # Remove extension
        idx = basename.rfind(".")
        return basename[:idx] if idx > 0 else basename

    def _get_dir(self, file_path: str) -> str:
        """Extract directory part from path string."""
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        if last_sep < 0:
            return ""
        return file_path[:last_sep]

    def _extract_images(self, content: str, images_dir: Path) -> str:
        """Extract base64 image data URIs and save to files.

        Args:
            content: Markdown content with base64 images.
            images_dir: Directory to save extracted images.

        Returns:
            Modified markdown with file paths instead of data URIs.
        """
        pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
        prefix = uuid.uuid4().hex[:8]
        counter = [0]

        def replace_image(match):
            alt_text, image_format, base64_data = match.groups()
            counter[0] += 1

            try:
                image_data = base64.b64decode(base64_data)
                ext = f".{image_format.lower()}" if image_format else ".png"
                filename = f"{prefix}_{counter[0]}{ext}"
                image_file = images_dir / filename

                with open(image_file, "wb") as f:
                    f.write(image_data)

                return f"![{alt_text}](./images/{filename})"
            except Exception:
                return match.group(0)

        return re.sub(pattern, replace_image, content)
