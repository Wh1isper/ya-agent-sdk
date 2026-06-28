"""Fetch tool for viewing web files with optional HEAD-only mode."""

from __future__ import annotations

import contextlib
from functools import cache
from pathlib import Path
from typing import Annotated, Any

import httpx2 as httpx
from pydantic import Field
from pydantic_ai import BinaryContent, RunContext, ToolReturn

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.web._http_client import (
    ForbiddenUrlError,
    safe_request,
    safe_stream_request,
    verify_url,
)
from ya_agent_sdk.utils import compress_image_to_model_limit, detect_image_media_type, raw_bytes_limit_for_base64

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

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[str, Field(description="URL of the web resource to fetch")],
        head_only: Annotated[
            bool,
            Field(description="Only check existence without downloading content", default=False),
        ] = False,
    ) -> str | dict[str, Any] | ToolReturn:
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
            return await self._get_request(ctx, url, skip_verification)

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

    def _build_binary_size_error(self, max_bytes: int, size: int, *, downloaded: bool = False) -> dict[str, Any]:
        """Build a standardized error for oversized binary fetches."""
        if downloaded:
            return {
                "success": False,
                "error": (
                    f"Resource exceeded inline limit while downloading ({size} bytes). "
                    f"Maximum supported size is {max_bytes} bytes."
                ),
            }
        return {
            "success": False,
            "error": (f"Resource too large to inline ({size} bytes). Maximum supported size is {max_bytes} bytes."),
        }

    async def _compress_image_for_model_limit(
        self,
        ctx: RunContext[AgentContext],
        image_data: bytes,
        media_type: str,
        *,
        source: str,
    ) -> tuple[bytes, str] | dict[str, Any]:
        """Compress fetched image data to the configured model API image limit."""
        max_encoded_bytes = ctx.deps.model_cfg.max_image_bytes if ctx.deps.model_cfg else 0
        if max_encoded_bytes <= 0:
            return image_data, media_type

        max_raw_bytes = raw_bytes_limit_for_base64(max_encoded_bytes)
        if len(image_data) <= max_raw_bytes:
            return image_data, media_type

        try:
            compressed_data, compressed_media_type = await compress_image_to_model_limit(
                image_data,
                max_encoded_bytes=max_encoded_bytes,
                media_type=media_type,
            )
        except Exception:
            logger.exception("Failed to compress fetched image from %s before inlining", source)
            return {
                "success": False,
                "error": "Fetched image could not be compressed for inline model input.",
            }

        if len(compressed_data) > max_raw_bytes:
            return {
                "success": False,
                "error": (
                    f"Fetched image could not be compressed below the {max_encoded_bytes} byte API limit "
                    "after accounting for base64 encoding."
                ),
            }

        logger.info(
            "Compressed fetched image from %s from %d bytes to %d bytes before inlining",
            source,
            len(image_data),
            len(compressed_data),
        )
        return compressed_data, compressed_media_type

    async def _read_binary_response(
        self,
        ctx: RunContext[AgentContext],
        response: httpx.Response,
        content_type: str,
        url: str,
    ) -> ToolReturn | dict[str, Any]:
        """Read an image response with a hard in-memory size limit."""
        max_bytes = ctx.deps.tool_config.fetch_max_inline_binary_bytes
        chunk_size = ctx.deps.tool_config.fetch_stream_chunk_size
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except (ValueError, OverflowError):
                declared_size = None
            if declared_size is not None and declared_size > max_bytes:
                return self._build_binary_size_error(max_bytes, declared_size)

        image_data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            image_data.extend(chunk)
            if len(image_data) > max_bytes:
                return self._build_binary_size_error(max_bytes, len(image_data), downloaded=True)

        data = bytes(image_data)
        # Verify declared content type against actual content for images
        # (some servers declare wrong MIME types; API providers reject mismatches)
        if content_type.startswith("image/"):
            detected = detect_image_media_type(data)
            if detected:
                content_type = detected
            compressed = await self._compress_image_for_model_limit(ctx, data, content_type, source=url)
            if isinstance(compressed, dict):
                return compressed
            data, content_type = compressed
        return ToolReturn(
            return_value="The image is attached in the user message.",
            content=[BinaryContent(data=data, media_type=content_type)],
        )

    async def _read_text_response(
        self, ctx: RunContext[AgentContext], response: httpx.Response
    ) -> str | dict[str, Any]:
        """Read a text response incrementally and truncate by character budget."""
        chunk_size = ctx.deps.tool_config.fetch_stream_chunk_size
        content_parts: list[str] = []
        current_length = 0
        truncated = False

        async for chunk in response.aiter_text(chunk_size=chunk_size):
            remaining = CONTENT_TRUNCATE_THRESHOLD - current_length
            if remaining <= 0:
                truncated = True
                break

            if len(chunk) <= remaining:
                content_parts.append(chunk)
                current_length += len(chunk)
            else:
                content_parts.append(chunk[:remaining])
                current_length += remaining
                truncated = True
                break

        text = "".join(content_parts)
        if not truncated:
            return text

        # Use Content-Length header for total size estimate when available
        total_length: int | None = None
        content_length = response.headers.get("Content-Length")
        if content_length:
            with contextlib.suppress(ValueError, OverflowError):
                total_length = int(content_length)

        result: dict[str, Any] = {
            "content": text + "\n\n... (truncated)",
            "truncated": True,
            "tips": "Content truncated. Use `download` to save the full file.",
        }
        if total_length is not None:
            result["total_length"] = total_length
        return result

    async def _get_request(
        self,
        ctx: RunContext[AgentContext],
        url: str,
        skip_verification: bool = False,
    ) -> str | dict[str, Any] | ToolReturn:
        """Make GET request and return content."""
        try:
            async with safe_stream_request(
                url, method="GET", timeout=60.0, skip_verification=skip_verification
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "image" in content_type:
                    return await self._read_binary_response(ctx, response, content_type, url)

                return await self._read_text_response(ctx, response)

        except ForbiddenUrlError as e:
            return {"success": False, "error": f"URL forbidden: {e}"}
        except Exception:
            logger.exception(f"Failed to fetch {url}")
            return {"success": False, "error": "Failed to fetch resource"}
