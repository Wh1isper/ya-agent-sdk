"""Tests for protocol detection."""

from ya_agent_environment import (
    InstructableResource,
    Resource,
    ResumableResource,
)

from .conftest import (
    MinimalBaseResource,
    ResourceWithInstructions,
    ResumableMockResource,
    SimpleResource,
)


def test_resumable_resource_protocol_detection() -> None:
    """Should detect ResumableResource implementation via isinstance."""
    resumable = ResumableMockResource()
    simple = SimpleResource()

    assert isinstance(resumable, Resource)
    assert isinstance(resumable, ResumableResource)
    assert isinstance(simple, Resource)
    assert not isinstance(simple, ResumableResource)


def test_instructable_resource_protocol_detection() -> None:
    """Should detect InstructableResource implementation via isinstance."""
    instructable = ResourceWithInstructions("test")
    simple = SimpleResource()
    minimal = MinimalBaseResource()

    assert isinstance(instructable, InstructableResource)
    assert not isinstance(simple, InstructableResource)
    # MinimalBaseResource has get_context_instructions from BaseResource
    assert isinstance(minimal, InstructableResource)
