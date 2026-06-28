"""Web search tools: search, search_stock_image, search_image."""

from __future__ import annotations

import asyncio
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field
from pydantic_ai import RunContext

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core._output import (
    DEFAULT_OUTPUT_TRUNCATE_LIMIT,
    append_guidance,
    dump_tool_output,
    output_too_large_message,
    tool_output_size,
    write_tmp_output,
)
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.web._http_client import check_url_accessible, get_http_client

if TYPE_CHECKING:
    pass

URL_CHECK_TIMEOUT = 5.0
URL_VALIDATION_CONCURRENCY = 5
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SEARCH_LIST_KEYS = ("items", "results", "hits", "data")


@cache
def _load_search_instruction() -> str:
    return (_PROMPTS_DIR / "search.md").read_text()


@cache
def _load_search_stock_image_instruction() -> str:
    return (_PROMPTS_DIR / "search_stock_image.md").read_text()


@cache
def _load_search_image_instruction() -> str:
    return (_PROMPTS_DIR / "search_image.md").read_text()


@cache
def _is_tavily_available() -> bool:
    """Check if tavily package is installed."""
    try:
        import tavily  # noqa: F401

        return True
    except ImportError:
        return False


def _build_list_search_preview(result: list[dict[str, Any]], output_path: str | None, note: str) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "results": [],
        "truncated": True,
        "total_results": len(result),
        "showing": 0,
        "note": note,
    }
    if output_path is not None:
        preview["output_file_path"] = output_path

    for item in result:
        candidate_results = [*preview["results"], item]
        candidate = {**preview, "results": candidate_results, "showing": len(candidate_results)}
        if tool_output_size(candidate) > DEFAULT_OUTPUT_TRUNCATE_LIMIT:
            break
        preview = candidate
    return preview


def _build_dict_search_base(
    result: dict[str, Any],
    output_path: str | None,
    note: str,
) -> tuple[dict[str, Any], list[str]]:
    list_keys = [key for key in _SEARCH_LIST_KEYS if isinstance(result.get(key), list)]
    preview = dict(result)
    preview["truncated"] = True
    preview["note"] = append_guidance(preview.get("note") if isinstance(preview.get("note"), str) else None, note)
    if output_path is not None:
        preview["output_file_path"] = output_path

    for key in list_keys:
        items = result[key]
        preview[key] = []
        preview[f"{key}_total"] = len(items)
        preview[f"{key}_showing"] = 0

    return preview, list_keys


def _minimal_search_preview(output_path: str | None, note: str) -> dict[str, Any]:
    minimal: dict[str, Any] = {"truncated": True, "note": note}
    if output_path is not None:
        minimal["output_file_path"] = output_path
    return minimal


def _fill_dict_search_preview(
    result: dict[str, Any],
    preview: dict[str, Any],
    list_keys: list[str],
) -> dict[str, Any]:
    for key in list_keys:
        items = result[key]
        for item in items:
            candidate_items = [*preview[key], item]
            candidate = {
                **preview,
                key: candidate_items,
                f"{key}_showing": len(candidate_items),
            }
            if tool_output_size(candidate) > DEFAULT_OUTPUT_TRUNCATE_LIMIT:
                break
            preview = candidate
    return preview


async def _guard_search_result(
    ctx: RunContext[AgentContext],
    result: list[dict[str, Any]] | dict[str, Any],
    *,
    prefix: str,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Spill oversized search API payloads and return a bounded preview."""
    if tool_output_size(result) <= DEFAULT_OUTPUT_TRUNCATE_LIMIT:
        return result

    serialized = dump_tool_output(result)
    output_path = await write_tmp_output(
        ctx.deps.file_operator,
        prefix=prefix,
        content=serialized,
        extension="json",
    )
    note = output_too_large_message(size=len(serialized), output_path=output_path, noun="search results")

    if isinstance(result, list):
        return _build_list_search_preview(result, output_path, note)

    preview, list_keys = _build_dict_search_base(result, output_path, note)
    if tool_output_size(preview) > DEFAULT_OUTPUT_TRUNCATE_LIMIT and not list_keys:
        return _minimal_search_preview(output_path, note)

    return _fill_dict_search_preview(result, preview, list_keys)


class SearchTool(BaseTool):
    """Web search tool using Google, Brave, or Tavily."""

    name = "search"
    description = "Search the web for information using search APIs."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available if Google, Brave, or Tavily API keys are configured."""
        cfg = ctx.deps.tool_config
        has_google = bool(cfg.google_search_api_key and cfg.google_search_cx)
        has_brave = bool(cfg.brave_search_api_key)
        has_tavily = bool(cfg.tavily_api_key) and _is_tavily_available()
        return has_google or has_brave or has_tavily

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if not self.is_available(ctx):
            return None
        return _load_search_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        query: Annotated[str, Field(description="The search query")],
        num: Annotated[int, Field(description="Number of results to return (1-10)", default=10)] = 10,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Execute web search."""
        cfg = ctx.deps.tool_config
        has_brave = bool(cfg.brave_search_api_key)
        has_tavily = bool(cfg.tavily_api_key) and _is_tavily_available()

        # Priority: Google > Brave > Tavily
        if cfg.google_search_api_key and cfg.google_search_cx:
            result = await self._search_google(query, num, cfg.google_search_api_key, cfg.google_search_cx)
            if isinstance(result, dict) and result.get("success") is False:
                # Fallback to Brave or Tavily
                if has_brave:
                    result = await self._search_brave(query, num, cfg.brave_search_api_key)  # type: ignore[arg-type]
                    return await _guard_search_result(ctx, result, prefix="search")
                if has_tavily:
                    result = await self._search_tavily(query, cfg.tavily_api_key)  # type: ignore[arg-type]
                    return await _guard_search_result(ctx, result, prefix="search")
            return await _guard_search_result(ctx, result, prefix="search")
        elif has_brave:
            result = await self._search_brave(query, num, cfg.brave_search_api_key)  # type: ignore[arg-type]
            return await _guard_search_result(ctx, result, prefix="search")
        elif has_tavily:
            result = await self._search_tavily(query, cfg.tavily_api_key)  # type: ignore[arg-type]
            return await _guard_search_result(ctx, result, prefix="search")
        else:
            return {"success": False, "error": "No search API available"}

    async def _search_google(
        self, query: str, num: int, api_key: str, cx: str
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Search using Google Custom Search API."""
        if not 1 <= num <= 10:
            return {"success": False, "error": "num must be between 1 and 10"}

        client = get_http_client()
        params = {
            "q": query,
            "num": num,
            "key": api_key,
            "cx": cx,
        }

        response = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=60,
        )

        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

        return response.json()

    async def _search_brave(self, query: str, num: int, api_key: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Search using Brave Search API."""
        client = get_http_client()
        # Brave API max count is 20
        params = {"q": query, "count": min(num, 20)}
        headers = {"X-Subscription-Token": api_key}

        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

        data = response.json()
        # Extract web results from Brave response
        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            return {"success": False, "error": "No search results found."}

        # Normalize to simple format
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
            for r in web_results
        ]

    async def _search_tavily(self, query: str, api_key: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Search using Tavily API."""
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key)
        results = await client.search(query, search_depth="advanced")  # type: ignore[arg-type]

        if not results.get("results"):
            return {"success": False, "error": "No search results found."}

        return results["results"]


class SearchStockImageTool(BaseTool):
    """Stock image search using Pixabay."""

    name = "search_stock_image"
    description = "Search royalty-free stock images from Pixabay for design work."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available if Pixabay API key is configured."""
        return bool(ctx.deps.tool_config.pixabay_api_key)

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_search_stock_image_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        query: Annotated[
            str,
            Field(description="Search term (max 100 chars). E.g., 'business team', 'nature landscape'"),
        ],
    ) -> dict[str, Any]:
        """Search Pixabay for stock images."""
        cfg = ctx.deps.tool_config
        client = get_http_client()

        params = {"q": query, "key": cfg.pixabay_api_key}
        response = await client.get("https://pixabay.com/api/", params=params, follow_redirects=True)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and "hits" in data:
            await self._validate_results(data["hits"])

        data["system-reminder"] = (
            "All image URLs have been verified for accessibility. "
            "You can use the `download` tool to save the images you need."
        )
        return cast(dict[str, Any], await _guard_search_result(ctx, data, prefix="search-stock-image"))

    async def _validate_results(self, results: list[dict[str, Any]]) -> None:
        """Validate image URLs in parallel."""
        semaphore = asyncio.Semaphore(URL_VALIDATION_CONCURRENCY)

        async def validate_one(idx: int) -> None:
            item = results[idx]
            urls = [item.get("webformatURL"), item.get("previewURL"), item.get("largeImageURL")]
            urls = [u for u in urls if u]

            async with semaphore:
                accessible = False
                for url in urls:
                    if await check_url_accessible(url, URL_CHECK_TIMEOUT):
                        accessible = True
                        break

                if not accessible:
                    results[idx] = {
                        "id": item.get("id"),
                        "tags": item.get("tags"),
                        "accessible": False,
                        "unavailable_reason": "Image URLs could not be reached during verification.",
                    }

        tasks = [validate_one(i) for i in range(len(results)) if isinstance(results[i], dict)]
        await asyncio.gather(*tasks)


class SearchImageTool(BaseTool):
    """Real-time image search using RapidAPI."""

    name = "search_image"
    description = "Search real-time images via RapidAPI (similar to Google Images)."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available if RapidAPI key is configured."""
        return bool(ctx.deps.tool_config.rapidapi_api_key)

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_search_image_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        query: Annotated[str, Field(description="Search query/keywords")],
        limit: Annotated[int, Field(description="Maximum results to return", default=10)] = 10,
        size: Annotated[
            str,
            Field(description="Image size: any, large, medium, icon, etc.", default="any"),
        ] = "any",
    ) -> dict[str, Any]:
        """Search images via RapidAPI."""
        cfg = ctx.deps.tool_config
        client = get_http_client()

        params = {
            "query": query,
            "limit": limit,
            "size": size,
            "color": "any",
            "type": "any",
            "time": "any",
            "usage_rights": "any",
            "file_type": "any",
            "aspect_ratio": "any",
            "safe_search": "off",
            "region": "us",
        }
        headers = {
            "x-rapidapi-host": "real-time-image-search.p.rapidapi.com",
            "x-rapidapi-key": cfg.rapidapi_api_key,
        }

        response = await client.get(
            "https://real-time-image-search.p.rapidapi.com/search",
            params=params,
            headers=headers,
            follow_redirects=True,
        )
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "ERROR":
            return {"success": False, "error": data.get("message", "Unknown error")}

        if isinstance(data, dict) and "data" in data:
            await self._validate_results(data["data"])

        data["system-reminder"] = (
            "All image URLs have been verified for accessibility. "
            "You can use the `download` tool to save the images you need."
        )
        return cast(dict[str, Any], await _guard_search_result(ctx, data, prefix="search-image"))

    async def _validate_results(self, results: list[dict[str, Any]]) -> None:
        """Validate image URLs in parallel."""
        semaphore = asyncio.Semaphore(URL_VALIDATION_CONCURRENCY)

        async def validate_one(idx: int) -> None:
            item = results[idx]
            url = item.get("url")

            if not url:
                results[idx] = self._build_inaccessible(item)
                return

            async with semaphore:
                if not await check_url_accessible(url, URL_CHECK_TIMEOUT):
                    results[idx] = self._build_inaccessible(item)

        tasks = [validate_one(i) for i in range(len(results)) if isinstance(results[i], dict)]
        await asyncio.gather(*tasks)

    def _build_inaccessible(self, item: dict[str, Any]) -> dict[str, Any]:
        """Build response for inaccessible image."""
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "accessible": False,
            "unavailable_reason": "Image URL could not be reached during verification.",
        }
