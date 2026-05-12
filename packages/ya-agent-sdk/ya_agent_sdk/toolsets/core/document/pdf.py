"""PDF to Markdown conversion tool.

Converts PDF files to markdown format with embedded images extracted.
Requires optional dependencies: pymupdf, pymupdf4llm.

Install with: pip install ya-agent-sdk[document]
"""

import functools
import tempfile
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

# Optional dependency check
try:
    import pymupdf
    import pymupdf.layout
    import pymupdf4llm
except ImportError as e:
    raise ImportError(
        "The 'pymupdf' and 'pymupdf4llm' packages are required for PdfConvertTool. "
        "Install with: pip install ya-agent-sdk[document]"
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

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("PdfConvertTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
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

        # Step 2: Process in a private local tempdir (pymupdf needs local filesystem access)
        # Write-back happens inside the with block so images are read from local disk
        # one-by-one rather than collected into memory all at once.
        with tempfile.TemporaryDirectory() as local_tmp:
            local_tmp_path = Path(local_tmp)
            local_source = local_tmp_path / f"source_{stem}.pdf"
            local_images_dir = local_tmp_path / "images"

            await _run_in_threadpool(local_source.write_bytes, file_content)
            del file_content  # Free source bytes after writing to local disk
            await _run_in_threadpool(local_images_dir.mkdir, exist_ok=True)

            # Get total page count
            try:

                def get_page_count(path):
                    with pymupdf.open(path) as doc:
                        return len(doc)

                total_pages = await _run_in_threadpool(get_page_count, local_source)
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
                    str(local_source),
                    write_images=True,
                    image_path=str(local_images_dir),
                    pages=list(range(start_page, actual_end_page + 1)),
                )
            except Exception as e:
                return {"error": f"Failed to convert PDF: {e}", "success": False}

            # Fix image paths in markdown (pymupdf4llm uses absolute paths)
            content = content.replace(str(local_images_dir) + "/", "./images/")

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
