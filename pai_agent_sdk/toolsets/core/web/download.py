"""Download tool for saving web files to local filesystem."""

from __future__ import annotations

import asyncio
import mimetypes
from functools import cache
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.web._http_client import ForbiddenUrlError, safe_request, verify_url

logger = get_logger(__name__)
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    return (_PROMPTS_DIR / "download.md").read_text()


class DownloadTool(BaseTool):
    """Download files from the web and save to local filesystem."""

    name = "download"
    description = "Download files from URLs and save to local filesystem."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        urls: Annotated[list[str], Field(description="List of URLs to download")],
        save_dir: Annotated[
            str,
            Field(description="Directory where files should be saved (relative path)"),
        ],
    ) -> list[dict[str, Any]]:
        """Download files from URLs and save to local directory."""
        file_operator = ctx.deps.file_operator

        # Ensure directory exists
        await file_operator.mkdir(save_dir, parents=True)

        # Download all URLs in parallel
        tasks = [self._download_one(ctx, url, save_dir) for url in urls]
        return await asyncio.gather(*tasks)

    async def _download_one(
        self,
        ctx: RunContext[AgentContext],
        url: str,
        save_dir: str,
    ) -> dict[str, Any]:
        """Download a single file."""
        file_operator = ctx.deps.file_operator
        skip_verification = ctx.deps.model_cfg.tool_config.skip_url_verification

        # Verify URL security
        if not skip_verification:
            try:
                verify_url(url)
            except ForbiddenUrlError as e:
                logger.warning(f"URL access forbidden: {url} - {e}")
                return {"success": False, "url": url, "error": f"URL access forbidden: {e}"}

        try:
            response = await safe_request(url, method="GET", timeout=60.0, skip_verification=skip_verification)
            response.raise_for_status()

            # Determine filename
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            original_name = Path(url.split("?")[0]).name or "downloaded"

            # Get extension from content type if missing
            extension = self._get_extension(content_type, original_name)
            filename = f"{uuid4().hex}{extension}"
            save_path = f"{save_dir}/{filename}"

            # Write file using file_operator
            await file_operator.write_file(save_path, response.content)

            return {
                "success": True,
                "url": url,
                "save_path": save_path,
                "size": len(response.content),
                "content_type": content_type,
                "message": f"Downloaded to {save_path}. Use `move` tool to rename if needed.",
            }

        except ForbiddenUrlError as e:
            return {"success": False, "url": url, "error": f"Redirect forbidden: {e}"}
        except Exception:
            logger.exception(f"Failed to download {url}")
            return {"success": False, "url": url, "error": "Download failed"}

    def _get_extension(self, content_type: str, original_name: str) -> str:
        """Get file extension from content type or original name."""
        # Try to get extension from original name
        ext = Path(original_name).suffix
        if ext:
            return ext

        # Fallback to content type
        ext = mimetypes.guess_extension(content_type) or ""
        return ext
