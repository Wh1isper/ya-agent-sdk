"""Video reading tool for models without native video understanding support.

This tool allows processing videos when the model does not support
native video understanding capabilities.
"""

from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext

from ya_agent_sdk.agents.video_understanding import get_video_description
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool


class ReadVideoTool(BaseTool):
    """Tool for reading and analyzing videos.

    Use this tool when the model does not support native video understanding
    but needs to process video content from URLs.
    """

    name = "read_video"
    description = "Read and analyze a video from a URL. Use when native video understanding is unavailable."

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """No instruction needed for this tool."""
        return None

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return not ctx.deps.model_cfg.has_video_understanding

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[
            str,
            Field(description="The URL of the video to read and analyze."),
        ],
    ) -> str:
        """Read and analyze a video from a URL.

        Args:
            ctx: The run context containing the agent context.
            url: The URL of the video to analyze.

        Returns:
            A text description or analysis of the video content in XML format.
        """
        agent_ctx = ctx.deps

        # Get model and settings from tool_config if available
        model = None
        model_settings = None
        if agent_ctx.tool_config:
            tool_config = agent_ctx.tool_config
            model = tool_config.video_understanding_model
            model_settings = tool_config.video_understanding_model_settings

        description, model_id, usage = await get_video_description(
            video_url=url,
            model=model,
            model_settings=model_settings,
            model_wrapper=agent_ctx.model_wrapper,
            wrapper_metadata=agent_ctx.get_wrapper_metadata(),
        )

        if ctx.tool_call_id:
            agent_ctx.update_usage_snapshot_entry(
                agent_id="video_understanding",
                agent_name="video_understanding",
                model_id=model_id,
                usage=usage,
                source="video_understanding",
                usage_id=ctx.tool_call_id,
                ledger_key=ctx.tool_call_id,
            )

        return description
