"""Tests for NoteManager and note integration in AgentContext."""

from ya_agent_sdk.context.note import NoteManager


def test_note_manager_set_and_get():
    """Test basic set and get operations."""
    manager = NoteManager()
    manager.set("lang", "Chinese")
    assert manager.get("lang") == "Chinese"
    assert manager.get("missing") is None


def test_note_manager_update_existing():
    """Test that set overwrites existing value."""
    manager = NoteManager()
    manager.set("lang", "Chinese")
    manager.set("lang", "English")
    assert manager.get("lang") == "English"


def test_note_manager_delete():
    """Test delete operation."""
    manager = NoteManager()
    manager.set("lang", "Chinese")
    assert manager.delete("lang") is True
    assert manager.get("lang") is None
    assert manager.delete("lang") is False


def test_note_manager_list_all():
    """Test list_all returns sorted entries."""
    manager = NoteManager()
    manager.set("z-key", "last")
    manager.set("a-key", "first")
    manager.set("m-key", "middle")
    entries = manager.list_all()
    assert entries == [("a-key", "first"), ("m-key", "middle"), ("z-key", "last")]


def test_note_manager_list_all_empty():
    """Test list_all with no entries."""
    manager = NoteManager()
    assert manager.list_all() == []


def test_note_manager_export_and_restore():
    """Test export and restore round-trip."""
    manager = NoteManager()
    manager.set("lang", "Chinese")
    manager.set("os", "macOS")

    exported = manager.export_notes()
    assert exported == {"lang": "Chinese", "os": "macOS"}

    restored = NoteManager.from_exported(exported)
    assert restored.get("lang") == "Chinese"
    assert restored.get("os") == "macOS"


def test_note_manager_from_exported_empty():
    """Test restore from empty data."""
    restored = NoteManager.from_exported({})
    assert restored.list_all() == []


def test_notes_in_resumable_state():
    """Test that notes are included in ResumableState export/restore."""
    from ya_agent_sdk.context import ResumableState

    state = ResumableState(notes={"lang": "Chinese", "os": "macOS"})
    assert state.notes == {"lang": "Chinese", "os": "macOS"}

    # Verify it can be serialized to JSON and back
    json_str = state.model_dump_json()
    restored_state = ResumableState.model_validate_json(json_str)
    assert restored_state.notes == {"lang": "Chinese", "os": "macOS"}


async def test_notes_in_export_state():
    """Test that AgentContext.export_state includes notes."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("lang", "Chinese")
        ctx.note_manager.set("os", "macOS")

        state = ctx.export_state()
        assert state.notes == {"lang": "Chinese", "os": "macOS"}


async def test_notes_restore_via_with_state():
    """Test full round-trip: export -> restore via with_state."""
    from ya_agent_sdk.context import AgentContext

    # Create and populate
    async with AgentContext() as ctx1:
        ctx1.note_manager.set("lang", "Chinese")
        state = ctx1.export_state()

    # Restore into new context
    async with AgentContext().with_state(state) as ctx2:
        assert ctx2.note_manager.get("lang") == "Chinese"


async def test_notes_in_context_instructions():
    """Test that note entries appear in runtime instructions XML."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("lang", "Chinese")
        ctx.note_manager.set("os", "macOS")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        assert "<notes " in instructions
        assert 'key="lang"' in instructions
        assert 'key="os"' in instructions
        assert "Chinese" not in instructions
        assert "macOS" not in instructions


async def test_notes_not_in_instructions_when_empty():
    """Test that empty notes does not produce notes element."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        assert "<notes>" not in instructions


async def test_notes_not_in_instructions_for_tool_response():
    """Test that notes are excluded from tool response instructions."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("lang", "Chinese")

        instructions = await ctx.get_context_instructions(is_user_prompt=False)
        assert "<notes>" not in instructions
