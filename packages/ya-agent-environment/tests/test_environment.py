"""Tests for Environment class."""

import pytest
from ya_agent_environment import (
    Environment,
    ResourceEntry,
    ResourceRegistryState,
)

from .conftest import (
    MockEnvironment,
    ResourceWithInstructions,
    ResumableMockResource,
    SimpleResource,
)


async def test_environment_constructor_with_state() -> None:
    """Should accept resource_state and resource_factories in constructor."""
    state = ResourceRegistryState(entries={"cache": ResourceEntry(state={"data": "cached"})})

    async def create_cache(env: Environment) -> ResumableMockResource:
        return ResumableMockResource()

    async with MockEnvironment(
        resource_state=state,
        resource_factories={"cache": create_cache},
    ) as env:
        # Resource should be restored on enter
        cache = env.resources.get_typed("cache", ResumableMockResource)
        assert cache is not None
        assert cache.data == "cached"


async def test_environment_chaining_api() -> None:
    """Should support chaining API for factories and state."""
    state = ResourceRegistryState(entries={"session": ResourceEntry(state={"data": "user_123"})})

    async def create_session(env: Environment) -> ResumableMockResource:
        return ResumableMockResource()

    env = MockEnvironment().with_resource_factory("session", create_session).with_resource_state(state)

    async with env:
        session = env.resources.get_typed("session", ResumableMockResource)
        assert session is not None
        assert session.data == "user_123"


async def test_environment_export_resource_state() -> None:
    """Should export resource state via environment method."""

    async def create_session(env: Environment) -> ResumableMockResource:
        r = ResumableMockResource()
        r.data = "session_data"
        return r

    async with MockEnvironment().with_resource_factory("session", create_session) as env:
        await env.resources.get_or_create("session")
        state = await env.export_resource_state()

        assert "session" in state.entries
        assert state.entries["session"].state == {"data": "session_data"}


async def test_environment_full_roundtrip() -> None:
    """Should support full export -> JSON -> restore cycle."""

    async def create_browser(env: Environment) -> ResumableMockResource:
        return ResumableMockResource()

    # First session: create and use resource
    async with MockEnvironment().with_resource_factory("browser", create_browser) as env1:
        browser = await env1.resources.get_or_create_typed("browser", ResumableMockResource)
        browser.data = "session_cookies_data"

        # Export state
        state1 = await env1.export_resource_state()
        json_data = state1.model_dump_json()

    # Second session: restore from JSON
    state2 = ResourceRegistryState.model_validate_json(json_data)

    async with MockEnvironment(
        resource_state=state2,
        resource_factories={"browser": create_browser},
    ) as env2:
        # Resource should be restored automatically
        browser2 = env2.resources.get_typed("browser", ResumableMockResource)
        assert browser2 is not None
        assert browser2.data == "session_cookies_data"


async def test_environment_backward_compatible() -> None:
    """Should preserve existing set/get API."""
    async with MockEnvironment() as env:
        # Old API should still work
        resource = SimpleResource()
        env.resources.set("legacy", resource)

        retrieved = env.resources.get_typed("legacy", SimpleResource)
        assert retrieved is resource


async def test_environment_context_instructions_includes_resources() -> None:
    """Environment.get_context_instructions includes resource instructions."""

    async def create_browser(env: Environment) -> ResourceWithInstructions:
        return ResourceWithInstructions("Browser session is active.")

    async with MockEnvironment().with_resource_factory("browser", create_browser) as env:
        await env.resources.get_or_create("browser")
        result = await env.get_context_instructions()
        assert "Browser session is active." in result


async def test_environment_double_enter() -> None:
    """Should raise RuntimeError when entering twice."""
    env = MockEnvironment()
    async with env:
        with pytest.raises(RuntimeError, match="has already been entered"):
            async with env:
                pass


async def test_environment_properties_before_enter() -> None:
    """Accessing file_operator/shell before enter should raise; entered should be False."""
    from ya_agent_environment import EnvironmentNotEnteredError

    env = MockEnvironment()

    assert env.entered is False
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.file_operator
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.shell


async def test_environment_entered_flag() -> None:
    """entered should be True inside context, False after exit."""
    env = MockEnvironment()
    assert env.entered is False

    async with env:
        assert env.entered is True
        assert env.file_operator is not None
        assert env.shell is not None

    assert env.entered is False


async def test_environment_none_file_operator_and_shell() -> None:
    """Environment should work when file_operator and shell are None."""
    from ya_agent_environment import EnvironmentNotEnteredError

    class MinimalEnvironment(Environment):
        async def _setup(self) -> None:
            pass  # Leave file_operator and shell as None

        async def _teardown(self) -> None:
            pass

    async with MinimalEnvironment() as env:
        assert env.entered is True
        assert env.file_operator is None
        assert env.shell is None

        # get_context_instructions should still work with no file_operator/shell
        result = await env.get_context_instructions()
        assert result == ""

    # Before enter, get_context_instructions should raise
    env2 = MinimalEnvironment()
    with pytest.raises(EnvironmentNotEnteredError):
        await env2.get_context_instructions()


async def test_environment_get_toolsets_empty() -> None:
    """get_toolsets should return empty list by default."""
    async with MockEnvironment() as env:
        toolsets = env.get_toolsets()
        assert toolsets == []


async def test_environment_with_resource_factory_chaining() -> None:
    """with_resource_factory should return self for chaining."""

    async def factory1(env: Environment) -> SimpleResource:
        return SimpleResource()

    async def factory2(env: Environment) -> SimpleResource:
        return SimpleResource()

    env = MockEnvironment().with_resource_factory("a", factory1).with_resource_factory("b", factory2)

    async with env:
        assert "a" not in env.resources  # Not created yet
        await env.resources.get_or_create("a")
        await env.resources.get_or_create("b")
        assert "a" in env.resources
        assert "b" in env.resources


async def test_environment_with_resource_state_chaining() -> None:
    """with_resource_state should return self for chaining."""
    state = ResourceRegistryState(entries={})
    env = MockEnvironment().with_resource_state(state).with_resource_state(None)  # Clear state

    async with env:
        # Should not crash even with None state
        pass


async def test_environment_reenter_raises() -> None:
    """Environment should raise if entered twice."""
    env = MockEnvironment()
    async with env:
        with pytest.raises(RuntimeError, match="already been entered"):
            async with env:
                pass


async def test_environment_file_operator_before_enter() -> None:
    """Accessing file_operator before enter should raise."""
    from ya_agent_environment import EnvironmentNotEnteredError

    env = MockEnvironment()
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.file_operator


async def test_environment_shell_before_enter() -> None:
    """Accessing shell before enter should raise."""
    from ya_agent_environment import EnvironmentNotEnteredError

    env = MockEnvironment()
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.shell


async def test_environment_get_toolsets_combines_env_and_resources() -> None:
    """get_toolsets should combine environment and resource toolsets."""
    from ya_agent_environment.resources import BaseResource

    class ToolsetResource(BaseResource):
        def __init__(self, toolset: object) -> None:
            self._toolset = toolset

        async def close(self) -> None:
            pass

        def get_toolsets(self) -> list:
            return [self._toolset]

    toolset1 = object()
    toolset2 = object()
    env_toolset = object()

    async def factory1(env: Environment) -> ToolsetResource:
        return ToolsetResource(toolset1)

    async def factory2(env: Environment) -> ToolsetResource:
        return ToolsetResource(toolset2)

    env = MockEnvironment().with_resource_factory("r1", factory1).with_resource_factory("r2", factory2)

    async with env:
        # Add environment-level toolset
        env._toolsets.append(env_toolset)

        # Create resources
        await env.resources.get_or_create("r1")
        await env.resources.get_or_create("r2")

        # get_toolsets should combine all
        toolsets = env.get_toolsets()
        assert len(toolsets) == 3
        assert env_toolset in toolsets
        assert toolset1 in toolsets
        assert toolset2 in toolsets


async def test_environment_fork_raises_not_implemented() -> None:
    """Default fork() should raise NotImplementedError."""
    async with MockEnvironment() as env:
        with pytest.raises(NotImplementedError, match="MockEnvironment does not support fork"):
            env.fork()
