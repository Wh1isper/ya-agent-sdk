"""Tests for Instruction class and group-based deduplication."""

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import InstructionPart
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets import Instruction, Toolset
from ya_agent_sdk.toolsets.base import BaseTool

from ._instruction_helpers import instruction_text as _instruction_text


class ToolWithStringInstruction(BaseTool):
    """Tool returning plain string instruction."""

    name = "string_tool"
    description = "Test tool with string instruction"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "String instruction content"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithGroupedInstruction(BaseTool):
    """Tool returning Instruction with group."""

    name = "grouped_tool"
    description = "Test tool with grouped instruction"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction:
        return Instruction(group="my-group", content="Grouped instruction content")

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithSameGroup(BaseTool):
    """Another tool with the same group (should be deduplicated)."""

    name = "same_group_tool"
    description = "Test tool with same group"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction:
        return Instruction(group="my-group", content="This should be skipped")

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithNoInstruction(BaseTool):
    """Tool with no instruction."""

    name = "no_instruction_tool"
    description = "Test tool without instruction"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithBlankStringInstruction(BaseTool):
    """Tool returning blank plain string instruction."""

    name = "blank_string_tool"
    description = "Test tool with blank string instruction"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "   \n"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithBlankGroupedInstruction(BaseTool):
    """Tool returning blank grouped instruction."""

    name = "blank_grouped_tool"
    description = "Test tool with blank grouped instruction"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction:
        return Instruction(group="blank-group", content="   \n")

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class UnavailableTool(BaseTool):
    """Tool that is not available."""

    name = "unavailable_tool"
    description = "This tool is unavailable"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return False

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "This instruction should NOT be injected"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class SupersededTool(BaseTool):
    """Tool that is superseded by a capability tag."""

    name = "superseded_tool"
    description = "This tool is superseded"
    superseded_by_tags = frozenset({"advanced"})

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "This instruction should NOT be injected when superseded"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class AdvancedTool(BaseTool):
    """Tool that provides the 'advanced' capability tag."""

    name = "advanced_tool"
    description = "Advanced tool"
    tags = frozenset({"advanced"})

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "Advanced tool instruction"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


# Tests for Instruction model


def test_instruction_model():
    """Test Instruction model creation."""
    instr = Instruction(group="test-group", content="test content")
    assert instr.group == "test-group"
    assert instr.content == "test content"


# Tests for Toolset.get_instructions deduplication


@pytest.mark.asyncio
async def test_get_instructions_string_tool(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with tool returning string."""
    toolset = Toolset(tools=[ToolWithStringInstruction])
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    assert 'name="string_tool"' in instruction_text
    assert "String instruction content" in instruction_text
    assert isinstance(instructions[0], InstructionPart)
    assert instructions[0].dynamic is False


@pytest.mark.asyncio
async def test_get_instructions_grouped_tool(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with tool returning Instruction."""
    toolset = Toolset(tools=[ToolWithGroupedInstruction])
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    assert 'name="my-group"' in instruction_text
    assert "Grouped instruction content" in instruction_text
    assert isinstance(instructions[0], InstructionPart)
    assert instructions[0].dynamic is False


@pytest.mark.asyncio
async def test_get_instructions_deduplication(mock_run_context: RunContext[AgentContext]):
    """Test that tools with same group are deduplicated."""
    toolset = Toolset(tools=[ToolWithGroupedInstruction, ToolWithSameGroup])
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    # First tool's content should be present
    assert "Grouped instruction content" in instruction_text
    # Second tool's content should be skipped
    assert "This should be skipped" not in instruction_text
    # Only one instruction block for the group
    assert instruction_text.count('name="my-group"') == 1


@pytest.mark.asyncio
async def test_get_instructions_mixed_tools(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with mixed tool types."""
    toolset = Toolset(
        tools=[
            ToolWithStringInstruction,
            ToolWithGroupedInstruction,
            ToolWithSameGroup,
            ToolWithNoInstruction,
        ]
    )
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    # String tool uses tool name as group
    assert 'name="string_tool"' in instruction_text
    # Grouped tools share one instruction
    assert 'name="my-group"' in instruction_text
    assert instruction_text.count('name="my-group"') == 1
    # No instruction tool contributes nothing
    assert 'name="no_instruction_tool"' not in instruction_text


@pytest.mark.asyncio
async def test_get_instructions_empty(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with only no-instruction tools."""
    toolset = Toolset(tools=[ToolWithNoInstruction])
    instructions = await toolset.get_instructions(mock_run_context)
    assert instructions is None


@pytest.mark.asyncio
async def test_get_instructions_skips_blank_content(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with tools returning blank instruction content."""
    toolset = Toolset(tools=[ToolWithBlankStringInstruction, ToolWithBlankGroupedInstruction])
    instructions = await toolset.get_instructions(mock_run_context)
    assert instructions is None


# Tests for unavailable/superseded tool instruction filtering


@pytest.mark.asyncio
async def test_get_instructions_skips_unavailable_tools(mock_run_context: RunContext[AgentContext]):
    """Test that unavailable tools do not inject instructions when skip_unavailable=True."""
    toolset = Toolset(tools=[ToolWithStringInstruction, UnavailableTool], skip_unavailable=True)
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    assert "String instruction content" in instruction_text
    assert "should NOT be injected" not in instruction_text
    assert 'name="unavailable_tool"' not in instruction_text


@pytest.mark.asyncio
async def test_get_instructions_includes_unavailable_when_skip_disabled(mock_run_context: RunContext[AgentContext]):
    """Test that unavailable tools still inject instructions when skip_unavailable=False."""
    toolset = Toolset(tools=[UnavailableTool], skip_unavailable=False)
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    assert 'name="unavailable_tool"' in instruction_text


@pytest.mark.asyncio
async def test_get_instructions_skips_superseded_tools(mock_run_context: RunContext[AgentContext]):
    """Test that superseded tools do not inject instructions."""
    toolset = Toolset(tools=[SupersededTool, AdvancedTool])
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    # Advanced tool instruction should be present
    assert "Advanced tool instruction" in instruction_text
    # Superseded tool instruction should NOT be present
    assert "should NOT be injected when superseded" not in instruction_text
    assert 'name="superseded_tool"' not in instruction_text


@pytest.mark.asyncio
async def test_get_instructions_includes_superseded_when_no_superseding_tag(mock_run_context: RunContext[AgentContext]):
    """Test that superseded tools inject instructions when the superseding tag is absent."""
    toolset = Toolset(tools=[SupersededTool])  # No AdvancedTool, so no "advanced" tag
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    assert 'name="superseded_tool"' in instruction_text


@pytest.mark.asyncio
async def test_get_instructions_skips_both_unavailable_and_superseded(mock_run_context: RunContext[AgentContext]):
    """Test combined filtering: unavailable and superseded tools are both excluded."""
    toolset = Toolset(
        tools=[ToolWithStringInstruction, UnavailableTool, SupersededTool, AdvancedTool],
        skip_unavailable=True,
    )
    instructions = await toolset.get_instructions(mock_run_context)
    instruction_text = _instruction_text(instructions)

    assert instructions is not None
    # Available, non-superseded tools should be present
    assert "String instruction content" in instruction_text
    assert "Advanced tool instruction" in instruction_text
    # Unavailable and superseded tools should be excluded
    assert 'name="unavailable_tool"' not in instruction_text
    assert 'name="superseded_tool"' not in instruction_text
