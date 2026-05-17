"""ToolSearchToolSet: Dynamic tool loading via search.

Wraps multiple AbstractToolsets and provides a ``tool_search`` tool that
allows the model to discover and load tools on demand. Tools are organized
by namespace (toolsets with ``id``) for atomic loading, or as loose individual
tools (toolsets without ``id``).

State is stored in AgentContext for automatic session restore via ResumableState.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext, Tool
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import NamespaceStatus, ToolSearchInitEvent
from ya_agent_sdk.toolsets.base import BaseToolset, InstructableToolset, collect_instruction_parts
from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata, extract_metadata_from_schema
from ya_agent_sdk.toolsets.tool_search.strategies.keyword import KeywordSearchStrategy

if TYPE_CHECKING:
    from ya_agent_sdk.toolsets.tool_search.strategies import SearchStrategy

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_TOOL_SEARCH_NAME = "tool_search"
_NS_KEY_PREFIX = "__ns__"


class ToolSearchToolSet(BaseToolset[AgentContext]):
    """Dynamic tool loading via search with namespace support.

    Wraps multiple toolsets and provides a ``tool_search`` tool for the model
    to discover tools on demand. No tools are visible initially except
    ``tool_search`` itself; discovered tools become visible on the next turn.

    Toolsets are organized into:

    - **Namespaces** (toolsets with ``id``): All tools load atomically when
      any tool or the namespace itself matches a search query.
    - **Loose tools** (toolsets without ``id``): Individual tools load independently.

    State is stored in ``AgentContext.tool_search_loaded_tools`` and
    ``AgentContext.tool_search_loaded_namespaces``, enabling automatic
    session restore via ``ResumableState``.

    Example::

        from ya_agent_sdk.toolsets import Toolset
        from ya_agent_sdk.toolsets.tool_search import ToolSearchToolSet

        arxiv_toolset = Toolset(tools=[...], toolset_id="arxiv")
        github_toolset = Toolset(tools=[...], toolset_id="github")
        misc_toolset = Toolset(tools=[MiscTool1, MiscTool2])

        search_toolset = ToolSearchToolSet(
            toolsets=[arxiv_toolset, github_toolset, misc_toolset],
            namespace_descriptions={
                "arxiv": "Search academic papers on arXiv",
                "github": "GitHub repository operations",
            },
            max_results=5,
        )
    """

    def __init__(
        self,
        toolsets: Sequence[AbstractToolset[AgentContext]],
        *,
        namespace_descriptions: dict[str, str] | None = None,
        search_strategy: SearchStrategy | None = None,
        max_results: int = 5,
        optional_namespaces: set[str] | None = None,
    ) -> None:
        """Initialize ToolSearchToolSet.

        Args:
            toolsets: The wrapped toolsets containing all tools.
                Toolsets with ``id`` are treated as namespaces (atomic loading).
                Toolsets without ``id`` provide loose tools (individual loading).
            namespace_descriptions: Optional descriptions for namespaces, keyed
                by toolset ID. Highest priority source for namespace descriptions.
            search_strategy: Pluggable search implementation.
                Defaults to KeywordSearchStrategy.
            max_results: Maximum results returned per search.
            optional_namespaces: Set of toolset IDs that are optional. If a
                toolset with an ID in this set fails to initialize during
                ``__aenter__``, it will be skipped with a warning instead of
                raising. Toolsets not in this set (or without an ID) are
                required and will raise on failure.
        """
        self._toolsets = list(toolsets)
        self._namespace_descriptions = namespace_descriptions or {}
        self._strategy: SearchStrategy = search_strategy or KeywordSearchStrategy()
        self._max_results = max_results
        self._optional_namespaces = optional_namespaces or set()

        # Init report: namespace_id -> status (populated during __aenter__)
        self._init_report: dict[str, NamespaceStatus] = {}

        # Built on each get_tools() call
        self._search_entries: dict[str, ToolMetadata] = {}
        self._toolset_tools_cache: dict[str, tuple[AbstractToolset[AgentContext], ToolsetTool[AgentContext]]] = {}
        self._namespace_tools: defaultdict[str, list[str]] = defaultdict(list)

        # Create the tool_search pydantic-ai Tool
        self._search_pydantic_tool = self._create_search_tool()

        logger.debug(f"ToolSearchToolSet initialized: {len(self._toolsets)} toolsets, max_results={max_results}")

    @property
    def id(self) -> str | None:
        return None

    @property
    def init_report(self) -> dict[str, NamespaceStatus]:
        """Namespace initialization status after __aenter__.

        Returns:
            Mapping of namespace ID to NamespaceStatus.
            Only populated after __aenter__ completes.
        """
        return dict(self._init_report)

    # -------------------------------------------------------------------------
    # AbstractToolset interface
    # -------------------------------------------------------------------------

    async def get_tools(self, ctx: RunContext[AgentContext]) -> dict[str, ToolsetTool[AgentContext]]:
        """Return visible tools: tool_search first, then loaded tools.

        Collects all tools from wrapped toolsets, builds the search index,
        but only exposes tools that have been loaded via search (stored in
        AgentContext) and the ``tool_search`` tool itself.

        ``tool_search`` is always the **first** entry in the returned dict so
        that it occupies a stable position in the model's tool list regardless
        of how many tools have been dynamically loaded.  Newly loaded tools
        are appended after it.

        On each call, emits a ``ToolSearchInitEvent`` via the context's
        sideband stream to report current namespace status, since namespace
        availability can change at runtime (e.g., MCP server disconnection).
        """
        all_tools = await self._collect_and_index_tools(ctx)

        # Emit namespace status event on every call (status can change dynamically)
        if self._init_report:
            await ctx.deps.emit_event(
                ToolSearchInitEvent(
                    event_id=f"tool-search-init-{ctx.deps.run_id[:8]}",
                    namespace_status=dict(self._init_report),
                )
            )

        # Rebuild search index
        all_metadata = list(self._search_entries.values())
        await self._strategy.build_index(all_metadata)

        # Get loaded state from context
        loaded_tools = set(ctx.deps.tool_search_loaded_tools)
        loaded_namespaces = set(ctx.deps.tool_search_loaded_namespaces)

        # tool_search is always FIRST for stable positioning in the tool list.
        # Dynamically loaded tools are appended after it.
        visible: dict[str, ToolsetTool[AgentContext]] = {}

        # Add tool_search tool first
        search_tool_def = await self._search_pydantic_tool.prepare_tool_def(ctx)
        if search_tool_def:
            visible[_TOOL_SEARCH_NAME] = ToolsetTool(
                toolset=self,
                tool_def=search_tool_def,
                max_retries=3,
                args_validator=self._search_pydantic_tool.function_schema.validator,
            )

        # Append loaded tools after tool_search
        for name, tool in all_tools.items():
            meta = self._search_entries.get(name)
            if not meta:
                continue
            if (meta.namespace and meta.namespace in loaded_namespaces) or (
                not meta.namespace and name in loaded_tools
            ):
                visible[name] = tool

        deferred_count = len(all_tools) - (len(visible) - 1)  # -1 excludes tool_search from visible count
        logger.debug(
            f"get_tools: {len(visible)} visible ({len(loaded_namespaces)} namespaces, "
            f"{len(loaded_tools)} loose, 1 tool_search), {deferred_count} deferred"
        )

        return visible

    async def _collect_and_index_tools(self, ctx: RunContext[AgentContext]) -> dict[str, ToolsetTool[AgentContext]]:
        """Collect tools from all wrapped toolsets and build the search index.

        For optional namespaces, if ``get_tools()`` fails at runtime (e.g., MCP
        server disconnected), the namespace is skipped with a warning and its
        status in ``_init_report`` is updated to ``NamespaceStatus.error``.
        """
        all_tools: dict[str, ToolsetTool[AgentContext]] = {}
        self._search_entries.clear()
        self._toolset_tools_cache.clear()
        self._namespace_tools.clear()

        for ts in self._toolsets:
            namespace_id = ts.id
            try:
                ts_tools = await ts.get_tools(ctx)
            except Exception:
                if namespace_id and namespace_id in self._optional_namespaces:
                    logger.warning(
                        "Optional toolset %r failed during get_tools, skipping",
                        namespace_id,
                        exc_info=True,
                    )
                    self._init_report[namespace_id] = NamespaceStatus.error
                    continue
                raise

            for name, tool in ts_tools.items():
                if name == _TOOL_SEARCH_NAME:
                    logger.debug("Skipping tool_search from wrapped toolset")
                    continue
                if name in all_tools:
                    logger.warning(f"Duplicate tool name {name!r} across wrapped toolsets, last one wins")
                all_tools[name] = tool
                self._toolset_tools_cache[name] = (ts, tool)

                if namespace_id:
                    self._namespace_tools[namespace_id].append(name)

                # Create individual tool metadata entry
                td = tool.tool_def
                metadata = extract_metadata_from_schema(
                    name=td.name,
                    description=td.description,
                    parameters_json_schema=td.parameters_json_schema,
                    namespace=namespace_id,
                )
                self._search_entries[name] = metadata

            # Create namespace-level entry for toolsets with id
            if namespace_id and self._namespace_tools.get(namespace_id):
                ns_desc = self._resolve_namespace_description(ts)
                ns_metadata = ToolMetadata(
                    name=namespace_id,
                    description=ns_desc,
                    is_namespace_entry=True,
                    namespace=namespace_id,
                    namespace_tool_names=list(self._namespace_tools[namespace_id]),
                )
                self._search_entries[f"{_NS_KEY_PREFIX}{namespace_id}"] = ns_metadata

        return all_tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentContext],
        tool: ToolsetTool[AgentContext],
    ) -> object:
        """Call a tool by name.

        If the tool is tool_search, execute the search and update loaded state
        in AgentContext. Otherwise, delegate to the owning wrapped toolset.
        """
        if name == _TOOL_SEARCH_NAME:
            return await self._execute_search_tool(ctx, tool_args)

        if name not in self._toolset_tools_cache:
            return f"Error: Tool {name!r} not found. Use tool_search to discover available tools."

        ts, original_tool = self._toolset_tools_cache[name]
        return await ts.call_tool(name, tool_args, ctx, original_tool)

    async def get_instructions(self, ctx: RunContext[AgentContext]) -> str | list[str] | None:
        """Get instructions for tool_search and delegate to loaded toolsets.

        Only collects instructions from toolsets whose tools are currently
        loaded (by namespace or individual tool).
        """
        parts: list[str] = []

        loaded_namespaces = set(ctx.deps.tool_search_loaded_namespaces)
        loaded_tools = set(ctx.deps.tool_search_loaded_tools)

        for ts in self._toolsets:
            if not isinstance(ts, InstructableToolset):
                continue
            ns_id = ts.id
            # Only include instructions for loaded toolsets
            if ns_id:
                if ns_id not in loaded_namespaces:
                    continue
            else:
                # For loose toolsets, check if any of its tools are loaded
                ts_tool_names = [n for n, (cached_ts, _) in self._toolset_tools_cache.items() if cached_ts is ts]
                if not any(t in loaded_tools for t in ts_tool_names):
                    continue
            instructions = await ts.get_instructions(ctx)
            if instructions:
                collect_instruction_parts(instructions, parts)

        # Add tool_search instruction
        search_instruction = self._build_search_instruction()
        if search_instruction:
            parts.append(f'<tool-instruction name="tool_search">{search_instruction}</tool-instruction>')

        return "\n".join(parts) if parts else None

    # -------------------------------------------------------------------------
    # Context manager delegation
    # -------------------------------------------------------------------------

    async def __aenter__(self):
        entered: list[AbstractToolset[AgentContext]] = []
        failed: list[AbstractToolset[AgentContext]] = []
        self._init_report.clear()
        try:
            for ts in self._toolsets:
                try:
                    await ts.__aenter__()
                    entered.append(ts)
                    if ts.id:
                        self._init_report[ts.id] = NamespaceStatus.connected
                except Exception:
                    ts_id = ts.id
                    if ts_id and ts_id in self._optional_namespaces:
                        logger.warning(
                            "Optional toolset %r failed to initialize, skipping",
                            ts_id,
                            exc_info=True,
                        )
                        self._init_report[ts_id] = NamespaceStatus.skipped
                        failed.append(ts)
                    else:
                        raise
        except BaseException:
            # Rollback already-entered toolsets on failure
            for ts in reversed(entered):
                try:
                    await ts.__aexit__(None, None, None)
                except Exception:
                    logger.warning(f"Error during rollback of {ts!r}", exc_info=True)
            raise

        # Remove failed optional toolsets from active list
        if failed:
            self._toolsets = [ts for ts in self._toolsets if ts not in failed]
            logger.warning(
                "Skipped %d optional toolset(s), continuing with %d",
                len(failed),
                len(self._toolsets),
            )

        return self

    async def __aexit__(self, *args):
        first_exc: BaseException | None = None
        for ts in reversed(self._toolsets):
            try:
                await ts.__aexit__(*args)
            except Exception as exc:
                logger.warning(f"Error during __aexit__ of {ts!r}", exc_info=True)
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc
        return None

    # -------------------------------------------------------------------------
    # Internal implementation
    # -------------------------------------------------------------------------

    def _resolve_namespace_description(self, ts: AbstractToolset[AgentContext]) -> str:
        """Resolve description for a namespace toolset.

        Priority:
        1. namespace_descriptions[ts.id] (user explicit)
        2. ts.description (BaseToolset subclasses)
        3. ts.instructions first line (MCPServer after initialization)
        4. ts.label (pydantic-ai, if not default class name)
        5. Auto: "Toolset: {ts.id}"
        """
        ns_id = ts.id
        if ns_id and ns_id in self._namespace_descriptions:
            return self._namespace_descriptions[ns_id]

        desc = getattr(ts, "description", None)
        if desc:
            return desc

        # Check MCP server instructions (available after __aenter__)
        try:
            instructions = getattr(ts, "instructions", None)
            if instructions:
                first_line = instructions.strip().split("\n", 1)[0].strip()
                if first_line:
                    return first_line
        except (AttributeError, RuntimeError):
            pass

        label = getattr(ts, "label", None)
        if label and label != type(ts).__name__:
            return str(label)

        if ns_id:
            return f"Toolset: {ns_id}"
        return "Unknown toolset"

    def _create_search_tool(self) -> Tool[AgentContext]:
        """Create the pydantic-ai Tool for tool_search."""
        toolset_ref = self

        async def _tool_search(
            ctx: RunContext[AgentContext],
            query: Annotated[str, Field(description="Natural language or keyword query to search for tools")],
        ) -> str:
            return await toolset_ref._execute_search_tool(ctx, {"query": query})

        return Tool(
            function=_tool_search,
            name=_TOOL_SEARCH_NAME,
            description=(
                "Search for available tools by keyword or description. "
                "Use this when you need a capability that is not currently available. "
                "Discovered tools become usable in subsequent turns."
            ),
            takes_ctx=True,
            max_retries=3,
        )

    async def _execute_search_tool(self, ctx: RunContext[AgentContext], tool_args: dict[str, Any]) -> str:
        """Execute a tool search query and update loaded state in AgentContext."""
        query = tool_args.get("query", "")
        if not query:
            return "Error: query parameter is required."

        loaded_tools = set(ctx.deps.tool_search_loaded_tools)
        loaded_namespaces = set(ctx.deps.tool_search_loaded_namespaces)

        candidates = self._get_unloaded_candidates(loaded_tools, loaded_namespaces)
        if not candidates:
            return "All available tools are already loaded. No additional tools to discover."

        results = await self._strategy.search(query, candidates, max_results=self._max_results)
        if not results:
            return f"No tools found matching query: {query!r}. Try different keywords."

        new_namespaces, new_tools = self._apply_search_results(results, query, ctx, loaded_tools, loaded_namespaces)
        return self._format_search_results(query, new_namespaces, new_tools)

    def _get_unloaded_candidates(
        self,
        loaded_tools: set[str],
        loaded_namespaces: set[str],
    ) -> list[ToolMetadata]:
        """Get search candidates that are not already loaded."""
        candidates: list[ToolMetadata] = []
        for meta in self._search_entries.values():
            if meta.is_namespace_entry or meta.namespace:
                if meta.namespace and meta.namespace not in loaded_namespaces:
                    candidates.append(meta)
            elif meta.name not in loaded_tools:
                candidates.append(meta)
        return candidates

    def _apply_search_results(
        self,
        results: list[ToolMetadata],
        query: str,
        ctx: RunContext[AgentContext],
        loaded_tools: set[str],
        loaded_namespaces: set[str],
    ) -> tuple[set[str], set[str]]:
        """Determine what to load from search results and update context state."""
        new_namespaces: set[str] = set()
        new_tools: set[str] = set()

        for meta in results:
            if meta.namespace:
                new_namespaces.add(meta.namespace)
            else:
                new_tools.add(meta.name)

        for ns in new_namespaces:
            if ns not in loaded_namespaces:
                ctx.deps.tool_search_loaded_namespaces.append(ns)
                logger.debug(f"Namespace {ns!r} loaded via search query {query!r}")
        for tool_name in new_tools:
            if tool_name not in loaded_tools:
                ctx.deps.tool_search_loaded_tools.append(tool_name)
                logger.debug(f"Tool {tool_name!r} loaded via search query {query!r}")

        return new_namespaces, new_tools

    def _format_search_results(
        self,
        query: str,
        loaded_namespaces: set[str],
        loaded_tools: set[str],
    ) -> str:
        """Format search results, deduplicating namespace and tool entries."""
        lines: list[str] = []

        # Show namespace-level results
        for ns_id in sorted(loaded_namespaces):
            ns_key = f"{_NS_KEY_PREFIX}{ns_id}"
            ns_meta = self._search_entries.get(ns_key)
            if ns_meta and ns_meta.namespace_tool_names:
                tool_list = ", ".join(ns_meta.namespace_tool_names)
                lines.append(f"- **[{ns_id}]** {ns_meta.description}")
                lines.append(f"  Tools: {tool_list}")

        # Show loose tool results
        for tool_name in sorted(loaded_tools):
            meta = self._search_entries.get(tool_name)
            if meta:
                lines.append(f"- **{meta.name}**: {meta.description}")
                if meta.parameter_names:
                    param_parts = []
                    for pname in meta.parameter_names:
                        pdesc = meta.parameter_descriptions.get(pname, "")
                        param_parts.append(f"{pname}: {pdesc}" if pdesc else pname)
                    lines.append(f"  Parameters: {', '.join(param_parts)}")

        total = len(loaded_namespaces) + len(loaded_tools)
        header = f"Found {total} result(s) matching {query!r}. They are now available for use:\n"
        return header + "\n".join(lines)

    def _build_search_instruction(self) -> str:
        """Build the tool_search instruction with namespace and strategy info."""
        parts: list[str] = []

        # Load base instruction
        prompt_file = _PROMPTS_DIR / "tool_search.md"
        if prompt_file.exists():
            parts.append(prompt_file.read_text().strip())

        # Add strategy-specific search hint
        search_hint = self._strategy.get_search_hint()
        if search_hint:
            parts.append(f"\n{search_hint}")

        # Add deferred tool count
        total_tools = len([k for k in self._search_entries if not k.startswith(_NS_KEY_PREFIX)])
        if total_tools > 0:
            parts.append(f"\nThere are {total_tools} additional tools available via search.")

        # Add namespace information
        namespace_entries = [m for m in self._search_entries.values() if m.is_namespace_entry]
        if namespace_entries:
            parts.append("\nAvailable tool namespaces:")
            for ns_meta in sorted(namespace_entries, key=lambda m: m.name):
                tool_count = len(ns_meta.namespace_tool_names) if ns_meta.namespace_tool_names else 0
                parts.append(f"- {ns_meta.name}: {ns_meta.description} ({tool_count} tools)")

        return "\n".join(parts)
