"""PDF to Markdown conversion tool.

Converts PDF files to markdown format with embedded images extracted.
Requires optional dependencies: pymupdf, pymupdf4llm.

Install with: pip install pai-agent-sdk[document]
"""

import functools
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
    import pymupdf
    import pymupdf4llm
except ImportError as e:
    raise ImportError(
        "The 'pymupdf' and 'pymupdf4llm' packages are required for PdfConvertTool. "
        "Install with: pip install pai-agent-sdk[document]"
    ) from e

_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_MAX_PAGES = 20


def _validate_page_params(
    page_start: int | None, page_end: int | None, total_pages: int
) -> tuple[int, int, str | None]:
    """Validate and calculate page range.

    Returns:
        (start_page, end_page, error_message) - error_message is None if valid
    """
    # Validate inputs
    if page_start is not None and page_start <= 0:
        return 0, 0, f"Invalid page_start: {page_start}. Must be >= 1."
    if page_end is not None and page_end != -1 and page_end <= 0:
        return 0, 0, f"Invalid page_end: {page_end}. Must be >= 1 or -1."

    # Calculate page range (convert to 0-based for pymupdf)
    start_page = (page_start - 1) if page_start else 0
    if page_end == -1:
        end_page = total_pages - 1
    elif page_end:
        end_page = page_end - 1
    else:
        end_page = min(start_page + DEFAULT_MAX_PAGES - 1, total_pages - 1)

    # Validate range against PDF size
    if start_page >= total_pages:
        return 0, 0, f"Invalid page_start: PDF has only {total_pages} pages."
    if end_page < start_page:
        return 0, 0, "Invalid range: page_end must be >= page_start."

    return start_page, min(end_page, total_pages - 1), None


@cache
def _load_instruction() -> str:
    """Load PDF convert instruction from prompts/pdf.md."""
    prompt_file = _PROMPTS_DIR / "pdf.md"
    return prompt_file.read_text()


async def _run_in_threadpool(func, *args, **kwargs):
    """Run a sync function in a thread pool."""
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


class PdfConvertTool(BaseTool):
    """Tool for converting PDF files to markdown."""

    name = "pdf_convert"
    description = "Convert PDF to markdown with image extraction."

    def is_available(self) -> bool:
        """Check if tool is available (requires file_operator)."""
        if self.ctx.file_operator is None:
            logger.debug("PdfConvertTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/pdf.md."""
        return _load_instruction()

    async def call(  # noqa: C901
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the PDF file to convert.")],
        page_start: Annotated[
            int | None,
            Field(description="Starting page number (1-based). Default: 1."),
        ] = None,
        page_end: Annotated[
            int | None,
            Field(description="Ending page number (1-based, inclusive). Default: 20. Use -1 for all pages."),
        ] = None,
    ) -> dict[str, Any]:
        file_op = cast(FileOperator, ctx.deps.file_operator)

        # Check file exists
        if not await file_op.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        # Get file extension and stem from path
        ext = self._get_extension(file_path)
        if ext != ".pdf":
            return {"error": f"Not a PDF file: {file_path}", "success": False}

        stem = self._get_stem(file_path)

        # Step 1: Copy source file to tmp directory
        try:
            file_content = await file_op.read_bytes(file_path)
            tmp_source_path = await file_op.write_tmp_file(f"source_{stem}.pdf", file_content)
        except Exception as e:
            return {"error": f"Failed to read file: {e}", "success": False}

        # Step 2: Process in tmp directory
        tmp_source = Path(tmp_source_path)
        tmp_export_dir = tmp_source.parent / f"export_{stem}"
        tmp_images_dir = tmp_export_dir / "images"

        try:
            await _run_in_threadpool(tmp_export_dir.mkdir, exist_ok=True)
            await _run_in_threadpool(tmp_images_dir.mkdir, exist_ok=True)
        except Exception as e:
            return {"error": f"Failed to create export directory: {e}", "success": False}

        # Get total page count
        try:

            def get_page_count(path):
                with pymupdf.open(path) as doc:
                    return len(doc)

            total_pages = await _run_in_threadpool(get_page_count, tmp_source)
        except Exception as e:
            logger.exception("Failed to read PDF file")
            return {"error": f"Failed to read PDF file: {e}", "success": False}

        # Validate and calculate page range
        start_page, actual_end_page, error = _validate_page_params(page_start, page_end, total_pages)
        if error:
            return {"error": error, "success": False}

        converted_pages = actual_end_page - start_page + 1

        # Convert PDF to markdown
        try:
            content = await _run_in_threadpool(
                pymupdf4llm.to_markdown,
                str(tmp_source),
                write_images=True,
                image_path=str(tmp_images_dir),
                pages=list(range(start_page, actual_end_page + 1)),
            )
        except Exception as e:
            return {"error": f"Failed to convert PDF: {e}", "success": False}

        # Fix image paths in markdown (pymupdf4llm uses absolute paths)
        content = content.replace(str(tmp_images_dir) + "/", "./images/")

        # Write markdown file to tmp
        md_filename = f"{stem}.md"
        tmp_md_path = tmp_export_dir / md_filename
        try:
            await _run_in_threadpool(tmp_md_path.write_text, content, encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to write markdown: {e}", "success": False}

        # Step 3: Copy results back to target directory
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
            "total_pages": total_pages,
            "converted_pages": converted_pages,
            "page_range": f"{start_page + 1}-{actual_end_page + 1}",
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
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        basename = file_path[last_sep + 1 :] if last_sep >= 0 else file_path
        idx = basename.rfind(".")
        return basename[:idx] if idx > 0 else basename

    def _get_dir(self, file_path: str) -> str:
        """Extract directory part from path string."""
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        if last_sep < 0:
            return ""
        return file_path[:last_sep]
