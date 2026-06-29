"""Web scraping tool using Firecrawl with MarkItDown fallback."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Annotated, Any

import anyio.to_thread
from pydantic import Field
from pydantic_ai import RunContext

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core._output import (
    DEFAULT_OUTPUT_TRUNCATE_LIMIT,
    fit_text_fields_to_limit,
    output_too_large_message,
    tool_output_size,
    write_tmp_output,
)
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.web._http_client import ForbiddenUrlError, verify_url

logger = get_logger(__name__)

CONTENT_TRUNCATE_THRESHOLD = 60000
CONTENT_PREVIEW_LIMIT = DEFAULT_OUTPUT_TRUNCATE_LIMIT
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    return (_PROMPTS_DIR / "scrape.md").read_text()


class ScrapeTool(BaseTool):
    """Web scraping tool that converts websites to Markdown."""

    name = "scrape"
    description = "Convert websites to Markdown format for content analysis."

    def __init__(self) -> None:
        self._md: Any | None = None

    def _get_markitdown(self) -> Any:
        """Create the MarkItDown converter on first fallback scrape use."""
        if self._md is None:
            from markitdown import MarkItDown

            self._md = MarkItDown(enable_plugins=True)
        return self._md

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[str, Field(description="URL of the web page to scrape. e.g. https://example.com")],
    ) -> dict[str, Any]:
        """Scrape webpage and return content as Markdown."""
        skip_verification = ctx.deps.tool_config.skip_url_verification

        # Verify URL security
        if not skip_verification:
            try:
                verify_url(url)
            except ForbiddenUrlError as e:
                logger.warning(f"URL access forbidden: {url} - {e}")
                return {"success": False, "error": f"URL access forbidden - {e}"}

        cfg = ctx.deps.tool_config

        # Try Firecrawl first if available
        if cfg.firecrawl_api_key:
            try:
                from firecrawl import AsyncFirecrawlApp

                logger.info(f"Scraping webpage with Firecrawl: {url}")
                app = AsyncFirecrawlApp(api_key=cfg.firecrawl_api_key)
                result = await app.scrape(url=url, formats=["markdown"])

                if result.markdown:
                    return await self._build_success_response(ctx, result.markdown)
                logger.warning(f"Firecrawl returned empty result for {url}, falling back")
            except Exception:
                logger.exception(f"Firecrawl failed for {url}, falling back")

        # Fallback to MarkItDown
        return await self._fallback_scrape(ctx, url)

    async def _fallback_scrape(self, ctx: RunContext[AgentContext], url: str) -> dict[str, Any]:
        """Fallback scraping using MarkItDown."""
        try:
            md = self._get_markitdown()
            result = await anyio.to_thread.run_sync(md.convert, url)
            return await self._build_success_response(ctx, result.text_content)
        except Exception:
            logger.exception(f"Fallback scrape failed for {url}")
            return {"success": False, "error": "Failed to scrape webpage"}

    async def _build_success_response(self, ctx: RunContext[AgentContext], content: str) -> dict[str, Any]:
        """Build success response with optional truncation."""
        total_length = len(content)
        response: dict[str, Any] = {
            "success": True,
            "markdown_content": content,
            "truncated": False,
            "total_length": total_length,
            "tips": "All content is returned.",
        }
        if tool_output_size(response) <= CONTENT_PREVIEW_LIMIT:
            return response

        output_path = await write_tmp_output(
            ctx.deps.file_operator,
            prefix="scrape",
            content=content,
            extension="md",
        )
        response["truncated"] = True
        response["tips"] = output_too_large_message(
            size=total_length,
            output_path=output_path,
            noun="Markdown",
        )
        if output_path is not None:
            response["output_file_path"] = output_path

        suffix = "\n\n... (truncated; full Markdown saved in `output_file_path`)"
        return fit_text_fields_to_limit(
            response,
            text_fields=("markdown_content",),
            limit=CONTENT_PREVIEW_LIMIT,
            suffix=suffix,
        )
