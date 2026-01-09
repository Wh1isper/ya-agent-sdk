"""Office/EPub to Markdown conversion tool.

Converts Office documents (Word, PowerPoint, Excel) and EPub files to markdown.
Requires optional dependency: markitdown.

Install with: pip install pai-agent-sdk[document]
"""

import base64
import functools
import re
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

import anyio.to_thread
from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.base import FileOperator
from pai_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

# Optional dependency check
try:
    from markitdown import MarkItDown
except ImportError as e:
    raise ImportError(
        "The 'markitdown' package is required for OfficeConvertTool. Install with: pip install pai-agent-sdk[document]"
    ) from e

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SUPPORTED_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".epub"}


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

    def is_available(self) -> bool:
        """Check if tool is available (requires file_operator)."""
        if self.ctx.file_operator is None:
            logger.debug("OfficeConvertTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
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

        # Step 1: Copy source file to tmp directory
        try:
            file_content = await file_op.read_bytes(file_path)
            tmp_source_path = await file_op.write_tmp_file(f"source_{stem}{ext}", file_content)
        except Exception as e:
            return {"error": f"Failed to read file: {e}", "success": False}

        # Step 2: Convert document in tmp directory
        # markitdown needs a local file path
        tmp_source = Path(tmp_source_path)
        tmp_export_dir = tmp_source.parent / f"export_{stem}"
        tmp_images_dir = tmp_export_dir / "images"

        try:
            await _run_in_threadpool(tmp_export_dir.mkdir, exist_ok=True)
            await _run_in_threadpool(tmp_images_dir.mkdir, exist_ok=True)
        except Exception as e:
            return {"error": f"Failed to create export directory: {e}", "success": False}

        try:
            md = MarkItDown(enable_plugins=True)
            result = await _run_in_threadpool(md.convert, f"file://{tmp_source.as_posix()}", keep_data_uris=True)
            content = result.text_content
        except Exception as e:
            return {"error": f"Failed to convert document: {e}", "success": False}

        # Extract base64 images and save to tmp directory
        content = self._extract_images(content, tmp_images_dir)

        # Write markdown file to tmp
        md_filename = f"{stem}.md"
        tmp_md_path = tmp_export_dir / md_filename
        try:
            await _run_in_threadpool(tmp_md_path.write_text, content, encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to write markdown: {e}", "success": False}

        # Step 3: Copy results back to target directory
        # Determine target directory (same as source file's directory)
        source_dir = self._get_dir(file_path)
        target_export_dir = f"{source_dir}/export_{stem}" if source_dir else f"export_{stem}"
        target_md_path = f"{target_export_dir}/{md_filename}"
        target_images_dir = f"{target_export_dir}/images"

        try:
            # Create target directories first
            await file_op.mkdir(target_export_dir, parents=True)
            await file_op.mkdir(target_images_dir, parents=True)

            # Copy markdown file
            await file_op.copy(str(tmp_md_path), target_md_path)

            # Copy images (use thread pool for directory iteration)
            def list_image_files():
                return [f for f in tmp_images_dir.iterdir() if f.is_file()]

            image_files = await _run_in_threadpool(list_image_files)
            for img_file in image_files:
                target_img_path = f"{target_images_dir}/{img_file.name}"
                await file_op.copy(str(img_file), target_img_path)
        except Exception as e:
            return {"error": f"Failed to copy results to target: {e}", "success": False}

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
