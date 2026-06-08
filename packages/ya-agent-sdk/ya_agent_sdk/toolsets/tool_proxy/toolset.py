"""ToolProxyToolset: Fixed two-tool proxy for dynamic tool invocation.

Wraps multiple AbstractToolsets and exposes exactly two tools:
- ``search_tools`` or ``{prefix}_search_tool``: discover tools via search (returns XML with full schemas)
- ``call_tool`` or ``{prefix}_call_tool``: invoke any discovered tool by name

Unlike ToolSearchToolSet which dynamically adds tools to the model's tool list,
ToolProxyToolset keeps the tool list constant (always two tools), maximizing
prompt cache hit rates for providers that cache based on tool definitions.

State is stored in AgentContext for automatic session restore via ResumableState.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from html import escape as _html_escape
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


def xml_escape(s: str) -> str:
    """Escape XML-special characters without escaping quotes.

    Only escapes ``&``, ``<``, ``>`` so that JSON embedded in XML
    retains readable double-quote characters.
    """
    return _html_escape(s, quote=False)


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SEARCH_TOOLS_NAME = "search_tools"
_CALL_TOOL_NAME = "call_tool"
_PREFIXED_SEARCH_TOOL_SUFFIX = "search_tool"
_PREFIXED_CALL_TOOL_SUFFIX = "call_tool"
_NS_KEY_PREFIX = "__ns__"
_INSTRUCTION_GROUP = "tool-proxy"
_PREFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TOOL_NAME_PATTERN = re.compile(rf"\b({_SEARCH_TOOLS_NAME}|{_CALL_TOOL_NAME})\b")


def _normalize_prefix(prefix: str | None) -> str | None:
    """Normalize and validate a proxy tool name prefix."""
    if prefix is None:
        return None

    normalized = prefix.strip().strip("_")
    if not normalized:
        return None
    if not _PREFIX_PATTERN.fullmatch(normalized):
        raise ValueError(
            "ToolProxyToolset prefix must start with a letter and contain only letters, numbers, and underscores"
        )
    return normalized


class ToolProxyToolset(BaseToolset[AgentContext]):
    """Fixed two-tool proxy for dynamic tool invocation.

    Wraps multiple toolsets and exposes exactly two tools: ``search_tools``
    for discovery and ``call_tool`` for invocation by default, or
    ``{prefix}_search_tool`` and ``{prefix}_call_tool`` when ``prefix`` is
    configured. The tool list never changes, which maximizes prompt cache hit
    rates.

    Toolsets are organized into:

    - **Namespaces** (toolsets with ``id``): All tools load atomically when
      any tool or the namespace itself matches a search query.
    - **Loose tools** (toolsets without ``id``): Individual tools load independently.

    State is stored in ``AgentContext.tool_search_loaded_tools`` and
    ``AgentContext.tool_search_loaded_namespaces``, enabling automatic
    session restore via ``ResumableState``.

    Example::

        from ya_agent_sdk.toolsets import Toolset
        from ya_agent_sdk.toolsets.tool_proxy import ToolProxyToolset

        arxiv_toolset = Toolset(tools=[...], toolset_id="arxiv")
        github_toolset = Toolset(tools=[...], toolset_id="github")
        misc_toolset = Toolset(tools=[MiscTool1, MiscTool2])

        proxy_toolset = ToolProxyToolset(
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
        prefix: str | None = None,
        include_legacy_unprefixed_state: bool = False,
    ) -> None:
        """Initialize ToolProxyToolset.

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
            prefix: Optional prefix for the two visible proxy tools. When set,
                tools are exposed as ``{prefix}_search_tool`` and
                ``{prefix}_call_tool``. When omitted, the legacy names
                ``search_tools`` and ``call_tool`` are used.
            include_legacy_unprefixed_state: When True with a prefix, restore
                unprefixed loaded tool/namespace state for names that belong to
                this proxy and write back prefixed state entries. This is useful
                when migrating an existing unprefixed proxy block to a prefixed
                one, such as YAACLI/YA Claw MCP proxies.
        """
        self._toolsets = list(toolsets)
        self._namespace_descriptions = namespace_descriptions or {}
        self._strategy: SearchStrategy = search_strategy or KeywordSearchStrategy()
        self._max_results = max_results
        self._optional_namespaces = optional_namespaces or set()
        self._prefix = _normalize_prefix(prefix)
        self._search_tool_name = (
            f"{self._prefix}_{_PREFIXED_SEARCH_TOOL_SUFFIX}" if self._prefix else _SEARCH_TOOLS_NAME
        )
        self._call_tool_name = f"{self._prefix}_{_PREFIXED_CALL_TOOL_SUFFIX}" if self._prefix else _CALL_TOOL_NAME
        self._instruction_group = f"{self._prefix}-tool-proxy" if self._prefix else _INSTRUCTION_GROUP
        self._state_key_prefix = f"{self._prefix}:" if self._prefix else ""
        self._include_legacy_unprefixed_state = bool(self._prefix and include_legacy_unprefixed_state)
        self._event_id_prefix = f"tool-proxy-{self._prefix}-init" if self._prefix else "tool-proxy-init"

        # Init report: namespace_id -> status (populated during __aenter__)
        self._init_report: dict[str, NamespaceStatus] = {}

        # Built on each get_tools() call
        self._search_entries: dict[str, ToolMetadata] = {}
        self._toolset_tools_cache: dict[str, tuple[AbstractToolset[AgentContext], ToolsetTool[AgentContext]]] = {}
        self._namespace_tools: defaultdict[str, list[str]] = defaultdict(list)

        # Create the two fixed pydantic-ai Tools
        self._search_pydantic_tool = self._create_search_tool()
        self._call_pydantic_tool = self._create_call_tool()

        logger.debug(
            "ToolProxyToolset initialized: %d toolsets, max_results=%d, tools=(%s, %s)",
            len(self._toolsets),
            max_results,
            self._search_tool_name,
            self._call_tool_name,
        )

    @property
    def id(self) -> str | None:
        return None

    @property
    def prefix(self) -> str | None:
        """Optional prefix used for visible proxy tool names."""
        return self._prefix

    @property
    def search_tool_name(self) -> str:
        """Visible name of the search proxy tool."""
        return self._search_tool_name

    @property
    def call_tool_name(self) -> str:
        """Visible name of the call proxy tool."""
        return self._call_tool_name

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
        """Return the two fixed proxy tools.

        Internally collects all tools from wrapped toolsets and builds the
        search index, but only exposes the two proxy tools. The tool list
        never changes across calls.

        On each call, emits a ``ToolSearchInitEvent`` via the context's
        sideband stream to report current namespace status.
        """
        await self._collect_and_index_tools(ctx)

        # Emit namespace status event (status can change dynamically)
        if self._init_report:
            await ctx.deps.emit_event(
                ToolSearchInitEvent(
                    event_id=f"{self._event_id_prefix}-{ctx.deps.run_id[:8]}",
                    namespace_status=dict(self._init_report),
                )
            )

        # Rebuild search index
        all_metadata = list(self._search_entries.values())
        await self._strategy.build_index(all_metadata)

        # Always return exactly 2 tools in stable order
        visible: dict[str, ToolsetTool[AgentContext]] = {}

        search_tool_def = await self._search_pydantic_tool.prepare_tool_def(ctx)
        if search_tool_def:
            visible[self._search_tool_name] = ToolsetTool(
                toolset=self,
                tool_def=search_tool_def,
                max_retries=3,
                args_validator=self._search_pydantic_tool.function_schema.validator,
            )

        call_tool_def = await self._call_pydantic_tool.prepare_tool_def(ctx)
        if call_tool_def:
            visible[self._call_tool_name] = ToolsetTool(
                toolset=self,
                tool_def=call_tool_def,
                max_retries=3,
                args_validator=self._call_pydantic_tool.function_schema.validator,
            )

        logger.debug(f"get_tools: 2 proxy tools, {len(self._toolset_tools_cache)} underlying tools indexed")

        return visible

    async def _collect_and_index_tools(self, ctx: RunContext[AgentContext]) -> None:
        """Collect tools from all wrapped toolsets and build the search index.

        For optional namespaces, if ``get_tools()`` fails at runtime (e.g., MCP
        server disconnected), the namespace is skipped with a warning and its
        status in ``_init_report`` is updated to ``NamespaceStatus.error``.
        """
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
                if name in (self._search_tool_name, self._call_tool_name):
                    logger.debug("Skipping proxy tool name %r from wrapped toolset", name)
                    continue
                if name in self._toolset_tools_cache:
                    logger.warning(f"Duplicate tool name {name!r} across wrapped toolsets, last one wins")
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

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentContext],
        tool: ToolsetTool[AgentContext],
    ) -> object:
        """Dispatch to search_tools or call_tool implementation."""
        if name == self._search_tool_name:
            return await self._execute_search(ctx, tool_args)
        if name == self._call_tool_name:
            return await self._execute_call(ctx, tool_args)
        return f"<error>Unknown proxy tool: {xml_escape(name)}</error>"

    async def get_instructions(self, ctx: RunContext[AgentContext]) -> str | list[str] | None:
        """Get instructions for proxy tools and delegate to loaded toolsets.

        Only collects instructions from toolsets whose tools have been
        discovered (by namespace or individual tool).
        """
        self._restore_legacy_state_entries(ctx)
        parts: list[str] = []

        loaded_namespaces = self._loaded_namespaces(ctx)
        loaded_tools = self._loaded_tools(ctx)

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

        # Add tool-proxy instruction
        proxy_instruction = self._build_proxy_instruction(ctx)
        if proxy_instruction:
            parts.append(f'<tool-instruction name="{self._instruction_group}">{proxy_instruction}</tool-instruction>')

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
            for ts in reversed(entered):
                try:
                    await ts.__aexit__(None, None, None)
                except Exception:
                    logger.warning(f"Error during rollback of {ts!r}", exc_info=True)
            raise

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
    # Tool creation
    # -------------------------------------------------------------------------

    def _create_search_tool(self) -> Tool[AgentContext]:
        """Create the pydantic-ai Tool for tool discovery."""
        toolset_ref = self

        async def _search_tools(
            ctx: RunContext[AgentContext],
            query: Annotated[str, Field(description="Natural language or keyword query to search for tools")],
        ) -> str:
            return await toolset_ref._execute_search(ctx, {"query": query})

        return Tool(
            function=_search_tools,
            name=self._search_tool_name,
            description=(
                "Search for available tools by keyword or description. "
                "Returns tool names, descriptions, and full parameter schemas. "
                "Use this when you need a capability that is not currently known."
            ),
            takes_ctx=True,
            max_retries=3,
        )

    def _create_call_tool(self) -> Tool[AgentContext]:
        """Create the pydantic-ai Tool for proxied invocation."""
        toolset_ref = self

        async def _call_tool(
            ctx: RunContext[AgentContext],
            name: Annotated[str, Field(description="Name of the tool to invoke")],
            arguments: Annotated[
                dict[str, Any],
                Field(description="Arguments to pass to the tool, matching its parameter schema"),
            ],
        ) -> object:
            return await toolset_ref._execute_call(ctx, {"name": name, "arguments": arguments})

        return Tool(
            function=_call_tool,
            name=self._call_tool_name,
            description=(
                "Invoke a tool by name with the given arguments. "
                "Pass the tool name and a JSON object of arguments matching "
                "the tool's parameter schema."
            ),
            takes_ctx=True,
            max_retries=3,
        )

    # -------------------------------------------------------------------------
    # search_tools implementation
    # -------------------------------------------------------------------------

    async def _execute_search(self, ctx: RunContext[AgentContext], tool_args: dict[str, Any]) -> str:
        """Execute a tool search query and update loaded state in AgentContext."""
        query = tool_args.get("query", "")
        if not query:
            return "<error>Parameter 'query' is required.</error>"

        self._restore_legacy_state_entries(ctx)
        loaded_tools = self._loaded_tools(ctx)
        loaded_namespaces = self._loaded_namespaces(ctx)

        candidates = self._get_unloaded_candidates(loaded_tools, loaded_namespaces)
        if not candidates:
            return f'<search-results query="{xml_escape(query)}" count="0">All available tools are already discovered.</search-results>'

        results = await self._strategy.search(query, candidates, max_results=self._max_results)
        if not results:
            return (
                f'<search-results query="{xml_escape(query)}" count="0">'
                f"No tools found matching query. Try different keywords."
                f"</search-results>"
            )

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
                ctx.deps.tool_search_loaded_namespaces.append(self._state_key(ns))
                logger.debug(f"Namespace {ns!r} discovered via search query {query!r}")
        for tool_name in new_tools:
            if tool_name not in loaded_tools:
                ctx.deps.tool_search_loaded_tools.append(self._state_key(tool_name))
                logger.debug(f"Tool {tool_name!r} discovered via search query {query!r}")

        return new_namespaces, new_tools

    def _format_search_results(
        self,
        query: str,
        loaded_namespaces: set[str],
        loaded_tools: set[str],
    ) -> str:
        """Format search results as XML with full parameter schemas."""
        tools_to_show: list[tuple[str, str | None]] = []  # (tool_name, namespace)

        # Collect tools from discovered namespaces
        for ns_id in sorted(loaded_namespaces):
            ns_key = f"{_NS_KEY_PREFIX}{ns_id}"
            ns_meta = self._search_entries.get(ns_key)
            if ns_meta and ns_meta.namespace_tool_names:
                for tool_name in ns_meta.namespace_tool_names:
                    if tool_name in self._toolset_tools_cache:
                        tools_to_show.append((tool_name, ns_id))

        # Collect loose tools
        for tool_name in sorted(loaded_tools):
            if tool_name in self._toolset_tools_cache:
                tools_to_show.append((tool_name, None))

        lines: list[str] = [f'<search-results query="{xml_escape(query)}" count="{len(tools_to_show)}">']

        for tool_name, namespace in tools_to_show:
            meta = self._search_entries.get(tool_name)
            _, tool = self._toolset_tools_cache[tool_name]
            schema = tool.tool_def.parameters_json_schema
            desc = meta.description if meta else ""

            ns_attr = f' namespace="{xml_escape(namespace)}"' if namespace else ""
            lines.append(f'<tool name="{xml_escape(tool_name)}"{ns_attr}>')
            lines.append(f"<description>{xml_escape(desc)}</description>")
            lines.append(f"<parameters>{xml_escape(json.dumps(schema))}</parameters>")
            lines.append("</tool>")

        lines.append("</search-results>")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # call_tool implementation
    # -------------------------------------------------------------------------

    async def _execute_call(self, ctx: RunContext[AgentContext], tool_args: dict[str, Any]) -> object:
        """Proxy a tool call to the underlying toolset.

        Validates the tool exists, delegates to the owning toolset, and
        catches errors to return XML-formatted diagnostics with the
        parameter schema so the model can self-correct.

        ApprovalRequired and CallDeferred are re-raised for pydantic-ai
        control flow (HITL approval, deferred execution).
        """
        # Import here to use in except clause without circular import at module level
        from pydantic_ai import ApprovalRequired, CallDeferred

        tool_name = tool_args.get("name", "")
        arguments = tool_args.get("arguments") or {}

        if not tool_name:
            return '<error>Parameter "name" is required.</error>'

        if tool_name not in self._toolset_tools_cache:
            return (
                f'<error>Tool "{xml_escape(tool_name)}" not found. '
                f"Use {self._search_tool_name} to discover available tools.</error>"
            )

        ts, original_tool = self._toolset_tools_cache[tool_name]

        try:
            return await ts.call_tool(tool_name, arguments, ctx, original_tool)
        except (ApprovalRequired, CallDeferred):
            raise
        except Exception as e:
            schema = original_tool.tool_def.parameters_json_schema
            error_lines = [
                f'<tool-call-error tool="{xml_escape(tool_name)}">',
                f"<message>{xml_escape(str(e))}</message>",
                f"<parameters>{xml_escape(json.dumps(schema))}</parameters>",
                "</tool-call-error>",
            ]
            return "\n".join(error_lines)

    # -------------------------------------------------------------------------
    # Instruction building
    # -------------------------------------------------------------------------

    def _build_proxy_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Build the tool-proxy instruction with namespace and discovered tools info."""
        parts: list[str] = []

        # Load base instruction
        prompt_file = _PROMPTS_DIR / "tool_proxy.md"
        if prompt_file.exists():
            parts.append(self._render_base_instruction(prompt_file.read_text().strip()))

        # Add strategy-specific search hint
        search_hint = self._strategy.get_search_hint()
        if search_hint:
            parts.append(f"\n{search_hint}")

        # Add deferred tool count
        total_tools = len([k for k in self._search_entries if not k.startswith(_NS_KEY_PREFIX)])
        if total_tools > 0:
            parts.append(f"\nThere are {total_tools} tools available via {self._search_tool_name}.")

        # Add namespace information
        namespace_entries = [m for m in self._search_entries.values() if m.is_namespace_entry]
        if namespace_entries:
            parts.append("\nAvailable tool namespaces:")
            for ns_meta in sorted(namespace_entries, key=lambda m: m.name):
                tool_count = len(ns_meta.namespace_tool_names) if ns_meta.namespace_tool_names else 0
                parts.append(f"- {ns_meta.name}: {ns_meta.description} ({tool_count} tools)")

        # Add previously discovered tools summary (from context state)
        discovered = self._get_discovered_tools_summary(ctx)
        if discovered:
            parts.append("\nPreviously discovered tools:")
            parts.extend(discovered)

        return "\n".join(parts)

    def _get_discovered_tools_summary(self, ctx: RunContext[AgentContext]) -> list[str]:
        """Build a summary of previously discovered tools from context state."""
        loaded_namespaces = self._loaded_namespaces(ctx)
        loaded_tools = self._loaded_tools(ctx)
        lines: list[str] = []

        # Namespace tools
        for ns_id in sorted(loaded_namespaces):
            ns_key = f"{_NS_KEY_PREFIX}{ns_id}"
            ns_meta = self._search_entries.get(ns_key)
            if ns_meta and ns_meta.namespace_tool_names:
                for tool_name in ns_meta.namespace_tool_names:
                    meta = self._search_entries.get(tool_name)
                    if meta:
                        params = ", ".join(meta.parameter_names) if meta.parameter_names else ""
                        lines.append(f"- {meta.name}({params}): {meta.description} [{ns_id}]")

        # Loose tools
        for tool_name in sorted(loaded_tools):
            meta = self._search_entries.get(tool_name)
            if meta:
                params = ", ".join(meta.parameter_names) if meta.parameter_names else ""
                lines.append(f"- {meta.name}({params}): {meta.description}")

        return lines

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _state_key(self, name: str) -> str:
        """Return the AgentContext state key for a tool or namespace name."""
        return f"{self._state_key_prefix}{name}" if self._state_key_prefix else name

    def _loaded_tools(self, ctx: RunContext[AgentContext]) -> set[str]:
        """Return loaded tool names scoped to this proxy instance."""
        return self._loaded_state_names(ctx.deps.tool_search_loaded_tools, self._valid_tool_state_names())

    def _loaded_namespaces(self, ctx: RunContext[AgentContext]) -> set[str]:
        """Return loaded namespace IDs scoped to this proxy instance."""
        return self._loaded_state_names(ctx.deps.tool_search_loaded_namespaces, self._valid_namespace_state_names())

    def _loaded_state_names(self, entries: Sequence[str], valid_names: set[str]) -> set[str]:
        if not self._state_key_prefix:
            return set(entries)
        prefix_len = len(self._state_key_prefix)
        loaded = {entry[prefix_len:] for entry in entries if entry.startswith(self._state_key_prefix)}
        if self._include_legacy_unprefixed_state:
            loaded.update(entry for entry in entries if entry in valid_names)
        return loaded

    def _restore_legacy_state_entries(self, ctx: RunContext[AgentContext]) -> None:
        """Copy legacy unprefixed loaded state into prefixed state keys."""
        if not self._include_legacy_unprefixed_state:
            return

        valid_tools = self._valid_tool_state_names()
        valid_namespaces = self._valid_namespace_state_names()
        self._restore_legacy_entries(ctx.deps.tool_search_loaded_tools, valid_tools)
        self._restore_legacy_entries(ctx.deps.tool_search_loaded_namespaces, valid_namespaces)

    def _restore_legacy_entries(self, entries: list[str], valid_names: set[str]) -> None:
        existing = set(entries)
        for entry in list(entries):
            if entry not in valid_names:
                continue
            prefixed_entry = self._state_key(entry)
            if prefixed_entry not in existing:
                entries.append(prefixed_entry)
                existing.add(prefixed_entry)

    def _valid_tool_state_names(self) -> set[str]:
        return {name for name in self._toolset_tools_cache if not name.startswith(_NS_KEY_PREFIX)}

    def _valid_namespace_state_names(self) -> set[str]:
        return {meta.namespace for meta in self._search_entries.values() if meta.is_namespace_entry and meta.namespace}

    def _render_base_instruction(self, instruction: str) -> str:
        """Render base instruction text with the configured proxy tool names."""
        if self._search_tool_name == _SEARCH_TOOLS_NAME and self._call_tool_name == _CALL_TOOL_NAME:
            return instruction

        def replace_match(match: re.Match[str]) -> str:
            if match.group(0) == _SEARCH_TOOLS_NAME:
                return self._search_tool_name
            return self._call_tool_name

        return _TOOL_NAME_PATTERN.sub(replace_match, instruction)

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
