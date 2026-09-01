"""Tests for NoteManager and note integration in AgentContext."""

import json
from xml.etree.ElementTree import fromstring

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
    """Test that runtime context contains a compact key-only note index."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("lang", "Chinese")
        ctx.note_manager.set("os", "macOS")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        notes = fromstring(instructions).find("notes")  # noqa: S314

        assert notes is not None
        assert notes.attrib == {
            "total": "2",
            "shown": "2",
            "omitted": "0",
            "order": "key",
            "hint": "Note values are omitted. Use note_get with a key to read one note; omit the key to list all notes.",
        }
        assert json.loads(notes.text or "") == ["lang", "os"]
        assert "Chinese" not in instructions
        assert "macOS" not in instructions


async def test_notes_context_index_limits_key_count():
    """Test that runtime context lists at most fifty sorted note keys."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        for index in reversed(range(60)):
            ctx.note_manager.set(f"key-{index:02d}", f"value-{index}")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        notes = fromstring(instructions).find("notes")  # noqa: S314

        assert notes is not None
        assert notes.get("total") == "60"
        assert notes.get("shown") == "50"
        assert notes.get("omitted") == "10"
        assert json.loads(notes.text or "") == [f"key-{index:02d}" for index in range(50)]
        assert "value-" not in instructions


async def test_notes_context_index_respects_serialized_byte_budget():
    """Test that XML escaping and UTF-8 keys stay within the hint budget."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("00-" + "&" * 700, "first value")
        ctx.note_manager.set("01-" + "界" * 1_000, "second value")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        notes = fromstring(instructions).find("notes")  # noqa: S314
        notes_line = next(line for line in instructions.splitlines() if line.lstrip().startswith("<notes "))
        compact_hint = ctx.get_notes_context_hint()

        assert notes is not None
        assert compact_hint is not None
        assert len(notes_line.encode("utf-8")) <= 4_000
        assert len(compact_hint.encode("utf-8")) <= 4_000
        assert fromstring(compact_hint).attrib == notes.attrib  # noqa: S314
        assert notes.get("total") == "2"
        assert notes.get("shown") == "1"
        assert notes.get("omitted") == "1"
        assert json.loads(notes.text or "") == ["00-" + "&" * 700]
        assert "first value" not in instructions
        assert "second value" not in instructions


async def test_notes_context_index_round_trips_xml_unsafe_keys():
    """Test that every stored string key is represented through XML-safe JSON escapes."""
    from ya_agent_sdk.context import AgentContext

    keys = [
        'quote-"',
        "slash-\\",
        "line-\n",
        "cjk-界",
        "emoji-😀",
        "xml-<&>",
        "noncharacter-\ufffe",
        "surrogate-\ud800",
    ]
    async with AgentContext() as ctx:
        for key in keys:
            ctx.note_manager.set(key, "private value")

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        notes = fromstring(instructions).find("notes")  # noqa: S314
        compact_hint = ctx.get_notes_context_hint()

        assert notes is not None
        assert compact_hint is not None
        assert json.loads(notes.text or "") == sorted(keys)
        assert json.loads(fromstring(compact_hint).text or "") == sorted(keys)  # noqa: S314
        assert "private value" not in instructions
        assert "private value" not in compact_hint


async def test_notes_not_in_instructions_when_empty():
    """Test that empty notes does not produce notes element."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        assert "<notes" not in instructions


async def test_notes_not_in_instructions_for_tool_response():
    """Test that notes are excluded from tool response instructions."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("lang", "Chinese")

        instructions = await ctx.get_context_instructions(is_user_prompt=False)
        assert "<notes" not in instructions


async def test_notes_use_explicit_hint_during_nested_compact():
    """Test that nested compact receives one explicit note index instead of a duplicate runtime hint."""
    from ya_agent_sdk.context import AgentContext

    async with AgentContext() as ctx:
        ctx.note_manager.set("resume-plan", "private value")
        object.__setattr__(ctx, "_compact_depth", 1)

        instructions = await ctx.get_context_instructions(is_user_prompt=True)
        explicit_hint = ctx.get_notes_context_hint()

        assert "<notes" not in instructions
        assert explicit_hint is not None
        assert "resume-plan" in explicit_hint
        assert "private value" not in explicit_hint
