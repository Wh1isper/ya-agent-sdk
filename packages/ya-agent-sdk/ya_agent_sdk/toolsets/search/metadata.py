"""Tool metadata for search indexing.

ToolMetadata is a lightweight representation of a tool that contains
the searchable fields (name, description, parameter info) without
the full JSON schema overhead. Supports both individual tools and
namespace entries for atomic toolset loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolMetadata:
    """Lightweight tool metadata for search indexing.

    Extracted from ToolDefinition objects returned by wrapped toolsets.
    Contains only the fields needed for search: name, description,
    and parameter names/descriptions.

    There are two kinds of entries:
    - **Tool entries** (is_namespace_entry=False): represent individual tools.
      If ``namespace`` is set, loading this tool also loads the entire namespace.
    - **Namespace entries** (is_namespace_entry=True): represent a whole namespace.
      Matching a namespace entry loads all tools in that namespace at once.
    """

    name: str
    """The tool name, or namespace ID for namespace entries."""

    description: str
    """The tool description, or namespace description for namespace entries."""

    parameter_names: list[str] = field(default_factory=list)
    """List of parameter names from the tool's input schema."""

    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    """Mapping of parameter name to description."""

    namespace: str | None = None
    """Namespace this tool belongs to. None for loose (non-namespaced) tools."""

    is_namespace_entry: bool = False
    """Whether this is a namespace-level entry (vs individual tool entry)."""

    namespace_tool_names: list[str] | None = None
    """Tool names contained in this namespace. Only set for namespace entries."""

    @property
    def searchable_text(self) -> str:
        """Combined text for search indexing.

        Used by both keyword and embedding strategies.
        """
        if self.is_namespace_entry:
            parts = [f"Namespace: {self.name}", f"Description: {self.description}"]
            if self.namespace_tool_names:
                parts.append(f"Tools: {', '.join(self.namespace_tool_names)}")
            return "\n".join(parts)

        parts = [f"Tool: {self.name}", f"Description: {self.description}"]
        for pname in self.parameter_names:
            pdesc = self.parameter_descriptions.get(pname, "")
            if pdesc:
                parts.append(f"Parameter {pname}: {pdesc}")
            else:
                parts.append(f"Parameter: {pname}")
        if self.namespace:
            parts.append(f"Namespace: {self.namespace}")
        return "\n".join(parts)

    @property
    def brief(self) -> str:
        """Brief one-line summary for search results."""
        if self.is_namespace_entry:
            tool_list = ", ".join(self.namespace_tool_names) if self.namespace_tool_names else "none"
            return f"[{self.name}] {self.description}\n  Tools: {tool_list}"

        params = ", ".join(self.parameter_names) if self.parameter_names else "none"
        return f"{self.name}: {self.description} (params: {params})"


def extract_metadata_from_schema(
    name: str,
    description: str | None,
    parameters_json_schema: dict,
    *,
    namespace: str | None = None,
) -> ToolMetadata:
    """Extract ToolMetadata from a tool's JSON schema.

    Args:
        name: The tool name.
        description: The tool description.
        parameters_json_schema: The JSON schema for the tool's parameters.
        namespace: Optional namespace this tool belongs to.

    Returns:
        A ToolMetadata instance with extracted parameter info.
    """
    param_names: list[str] = []
    param_descriptions: dict[str, str] = {}

    properties = parameters_json_schema.get("properties", {})
    for pname, pinfo in properties.items():
        param_names.append(pname)
        pdesc = pinfo.get("description", "")
        if pdesc:
            param_descriptions[pname] = pdesc

    return ToolMetadata(
        name=name,
        description=description or "",
        parameter_names=param_names,
        parameter_descriptions=param_descriptions,
        namespace=namespace,
    )
