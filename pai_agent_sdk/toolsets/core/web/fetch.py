"""Fetch tool for viewing web files with optional HEAD-only mode."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import BinaryContent, RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.web._http_client import ForbiddenUrlError, safe_request, verify_url

logger = get_logger(__name__)

CONTENT_TRUNCATE_THRESHOLD = 60000
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    return (_PROMPTS_DIR / "fetch.md").read_text()


class FetchTool(BaseTool):
    """Fetch web files with optional HEAD-only mode for checking existence."""

    name = "fetch"
    description = "Read web files or check resource availability via HTTP."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[str, Field(description="URL of the web resource to fetch")],
        head_only: Annotated[
            bool,
            Field(description="Only check existence without downloading content", default=False),
        ] = False,
    ) -> str | dict[str, Any] | BinaryContent:
        """Fetch web resource or check its existence."""
        skip_verification = ctx.deps.tool_config.skip_url_verification

        # Verify URL security
        if not skip_verification:
            try:
                verify_url(url)
            except ForbiddenUrlError as e:
                logger.warning(f"URL access forbidden: {url} - {e}")
                return {"success": False, "error": f"URL access forbidden - {e}"}

        if head_only:
            return await self._head_request(url, skip_verification)
        else:
            return await self._get_request(url, skip_verification)

    async def _head_request(self, url: str, skip_verification: bool = False) -> dict[str, Any]:
        """Make HEAD request to check resource info."""
        try:
            response = await safe_request(url, method="HEAD", timeout=10.0, skip_verification=skip_verification)

            # Some servers don't support HEAD
            if response.status_code == 405:
                response = await safe_request(url, method="GET", timeout=10.0, skip_verification=skip_verification)

            return {
                "exists": response.status_code < 400,
                "accessible": response.status_code < 400,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "last_modified": response.headers.get("Last-Modified"),
                "url": url,
            }
        except ForbiddenUrlError as e:
            return {
                "exists": False,
                "accessible": False,
                "error": f"URL forbidden: {e}",
                "url": url,
            }
        except Exception as e:
            return {
                "exists": False,
                "accessible": False,
                "error": str(e),
                "url": url,
            }

    async def _get_request(self, url: str, skip_verification: bool = False) -> str | dict[str, Any] | BinaryContent:
        """Make GET request and return content."""
        try:
            response = await safe_request(url, method="GET", timeout=60.0, skip_verification=skip_verification)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            # Return images as BinaryContent
            if "image" in content_type:
                return BinaryContent(data=response.content, media_type=content_type)

            # Return text content
            text = response.text
            if len(text) > CONTENT_TRUNCATE_THRESHOLD:
                return {
                    "content": text[:CONTENT_TRUNCATE_THRESHOLD] + "\n\n... (truncated)",
                    "truncated": True,
                    "total_length": len(text),
                    "tips": "Content truncated. Use `download` to save the full file.",
                }

            return text

        except ForbiddenUrlError as e:
            return {"success": False, "error": f"URL forbidden: {e}"}
        except Exception:
            logger.exception(f"Failed to fetch {url}")
            return {"success": False, "error": "Failed to fetch resource"}
