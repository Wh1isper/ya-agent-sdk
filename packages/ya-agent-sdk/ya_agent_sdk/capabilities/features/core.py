"""Feature-owned wrappers for the SDK's existing BaseTool implementations."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.capabilities.foundation.deferred import SupportsDeferredOutput
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset


def _adapter(tools: list[type[BaseTool]], feature_id: str) -> AbstractToolset[AgentContext]:
    """Build the private BaseTool adapter for one owning capability."""
    return Toolset(tools=tools, toolset_id=feature_id)


@dataclass(kw_only=True)
class FilesystemCapability(AbstractCapability[AgentContext]):
    """Own filesystem inspection/mutation tools and their guidance."""

    writable: bool = True
    id: str | None = "filesystem"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.filesystem import tools

        selected = tools if self.writable else [tool for tool in tools if tool.name in {"glob", "grep", "ls", "view"}]
        return _adapter(selected, self.id or "filesystem")


@dataclass(kw_only=True)
class DocumentConversionCapability(AbstractCapability[AgentContext]):
    """Own PDF and Office document conversion tools."""

    id: str | None = "document_conversion"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.document import tools

        return _adapter(tools, self.id or "document_conversion")


@dataclass(kw_only=True)
class MediaReadCapability(AbstractCapability[AgentContext]):
    """Own the canonical image/video/audio inspection surface."""

    id: str | None = "media_read"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.content import tools

        return _adapter(tools, self.id or "media_read")


@dataclass(kw_only=True)
class WebSearchCapability(AbstractCapability[AgentContext]):
    """Own general, image, and stock-image search tools."""

    id: str | None = "web_search"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.web import SearchImageTool, SearchStockImageTool, SearchTool

        return _adapter([SearchTool, SearchStockImageTool, SearchImageTool], self.id or "web_search")


@dataclass(kw_only=True)
class WebContentCapability(AbstractCapability[AgentContext]):
    """Own fetch, scrape, and download tools."""

    id: str | None = "web_content"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.web import DownloadTool, FetchTool, ScrapeTool

        return _adapter([ScrapeTool, FetchTool, DownloadTool], self.id or "web_content")


@dataclass(kw_only=True)
class TaskCapability(AbstractCapability[AgentContext]):
    """Own dependency-aware task tools backed by the context task store."""

    id: str | None = "tasks"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.enhance import task_tools

        return _adapter(task_tools, self.id or "tasks")


@dataclass(kw_only=True)
class NoteCapability(AbstractCapability[AgentContext]):
    """Own note tools backed by the context note store."""

    id: str | None = "notes"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.enhance import note_tools

        return _adapter(note_tools, self.id or "notes")


@dataclass(kw_only=True)
class ThinkingCapability(AbstractCapability[AgentContext]):
    """Own the optional explicit thinking tool."""

    id: str | None = "thinking"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.enhance import thinking_tools

        return _adapter(thinking_tools, self.id or "thinking")


@dataclass(kw_only=True)
class UserInteractionCapability(SupportsDeferredOutput, AbstractCapability[AgentContext]):
    """Own structured deferred user questions."""

    id: str | None = "user_interaction"

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.interaction import tools

        return _adapter(tools, self.id or "user_interaction")
