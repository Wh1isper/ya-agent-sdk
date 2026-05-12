"""Tests for resources module: ResourceRegistry, BaseResource, ResourceEntry, ResourceRegistryState."""

import pytest
from ya_agent_environment import (
    BaseResource,
    Environment,
    Resource,
    ResourceEntry,
    ResourceRegistry,
    ResourceRegistryState,
    ResumableResource,
)

from .conftest import (
    MinimalBaseResource,
    MockBaseResource,
    MockEnvironment,
    ResourceWithEnvAccess,
    ResourceWithInstructions,
    ResumableMockResource,
    SimpleResource,
)

# --- ResourceEntry and ResourceRegistryState Tests ---


def test_resource_entry_model() -> None:
    """Should create ResourceEntry with state."""
    entry = ResourceEntry(state={"key": "value", "count": 42})
    assert entry.state == {"key": "value", "count": 42}


def test_resource_registry_state_model() -> None:
    """Should create ResourceRegistryState with entries."""
    state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"cookies": []}),
            "cache": ResourceEntry(state={"entries": {}}),
        }
    )
    assert len(state.entries) == 2
    assert "browser" in state.entries
    assert state.entries["browser"].state == {"cookies": []}


def test_resource_registry_state_serialization() -> None:
    """Should serialize and deserialize ResourceRegistryState."""
    original = ResourceRegistryState(
        entries={
            "resource1": ResourceEntry(state={"data": "test"}),
        }
    )

    # Serialize to JSON
    json_str = original.model_dump_json()
    assert isinstance(json_str, str)

    # Deserialize back
    restored = ResourceRegistryState.model_validate_json(json_str)
    assert restored.entries["resource1"].state == {"data": "test"}


# --- ResourceRegistry Factory Tests ---


async def test_registry_register_factory() -> None:
    """Should register and use factory."""
    registry = ResourceRegistry()

    async def create_resource(env: Environment) -> SimpleResource:
        return SimpleResource()

    registry.register_factory("simple", create_resource)
    assert "simple" not in registry


async def test_registry_get_or_create_new() -> None:
    """Should create resource via factory when not exists."""
    async with MockEnvironment() as env:
        created_resources: list[SimpleResource] = []

        async def create_resource(env: Environment) -> SimpleResource:
            r = SimpleResource()
            created_resources.append(r)
            return r

        env.resources.register_factory("simple", create_resource)
        resource = await env.resources.get_or_create("simple")

        assert len(created_resources) == 1
        assert resource is created_resources[0]
        assert "simple" in env.resources


async def test_registry_get_or_create_existing() -> None:
    """Should return existing resource without calling factory."""
    async with MockEnvironment() as env:
        call_count = 0

        async def create_resource(env: Environment) -> SimpleResource:
            nonlocal call_count
            call_count += 1
            return SimpleResource()

        env.resources.register_factory("simple", create_resource)

        # First call creates
        r1 = await env.resources.get_or_create("simple")
        # Second call returns existing
        r2 = await env.resources.get_or_create("simple")

        assert call_count == 1
        assert r1 is r2


async def test_registry_get_or_create_no_factory() -> None:
    """Should raise KeyError when no factory registered."""
    async with MockEnvironment() as env:
        with pytest.raises(KeyError, match="No resource or factory registered"):
            await env.resources.get_or_create("missing")


async def test_registry_get_or_create_typed() -> None:
    """Should return typed resource."""
    async with MockEnvironment() as env:

        async def create_resource(env: Environment) -> SimpleResource:
            return SimpleResource()

        env.resources.register_factory("simple", create_resource)
        resource = await env.resources.get_or_create_typed("simple", SimpleResource)

        assert isinstance(resource, SimpleResource)


async def test_registry_get_or_create_typed_wrong_type() -> None:
    """Should raise TypeError for wrong resource type."""
    async with MockEnvironment() as env:

        async def create_resource(env: Environment) -> SimpleResource:
            return SimpleResource()

        env.resources.register_factory("simple", create_resource)
        await env.resources.get_or_create("simple")

        with pytest.raises(TypeError, match="expected ResumableMockResource"):
            await env.resources.get_or_create_typed("simple", ResumableMockResource)


# --- ResourceRegistry Export/Restore Tests ---


async def test_registry_export_state_resumable() -> None:
    """Should export state for resumable resources only."""
    async with MockEnvironment() as env:

        async def create_simple(env: Environment) -> SimpleResource:
            return SimpleResource()

        async def create_resumable(env: Environment) -> ResumableMockResource:
            r = ResumableMockResource()
            r.data = "test_data"
            return r

        env.resources.register_factory("simple", create_simple)
        env.resources.register_factory("resumable", create_resumable)

        await env.resources.get_or_create("simple")
        await env.resources.get_or_create("resumable")

        state = await env.resources.export_state()

        # Only resumable resource should be in state
        assert "simple" not in state.entries
        assert "resumable" in state.entries
        assert state.entries["resumable"].state == {"data": "test_data"}


async def test_registry_restore_all() -> None:
    """Should restore all resources from pending state."""
    # Create initial state
    pending_state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"data": "restored_data"}),
        }
    )

    async def create_browser(env: Environment) -> ResumableMockResource:
        return ResumableMockResource(initial_data="fresh")

    async with MockEnvironment(
        resource_state=pending_state,
        resource_factories={"browser": create_browser},
    ) as env:
        # Resource should be restored automatically on enter
        assert "browser" in env.resources

        browser = env.resources.get_typed("browser", ResumableMockResource)
        assert browser is not None
        assert browser.data == "restored_data"
        assert browser._restored_state == {"data": "restored_data"}


async def test_registry_restore_all_idempotent() -> None:
    """Should be idempotent - second call does nothing."""
    pending_state = ResourceRegistryState(
        entries={
            "resource": ResourceEntry(state={"data": "test"}),
        }
    )

    call_count = 0

    async def create_resource(env: Environment) -> ResumableMockResource:
        nonlocal call_count
        call_count += 1
        return ResumableMockResource()

    async with MockEnvironment(
        resource_state=pending_state,
        resource_factories={"resource": create_resource},
    ) as env:
        # First restore happens automatically on enter (call_count == 1)
        # Second restore should do nothing
        count2 = await env.resources.restore_all()

        assert count2 == 0
        assert call_count == 1


async def test_registry_restore_all_no_factory() -> None:
    """Should raise KeyError when factory missing for pending resource."""
    pending_state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"data": "test"}),
        }
    )

    registry = ResourceRegistry(state=pending_state)  # No factories

    with pytest.raises(KeyError, match="No factory registered for pending resource"):
        await registry.restore_all()


async def test_registry_restore_one() -> None:
    """Should restore single resource lazily."""
    pending_state = ResourceRegistryState(
        entries={
            "a": ResourceEntry(state={"data": "data_a"}),
            "b": ResourceEntry(state={"data": "data_b"}),
        }
    )

    async def create_resource(env: Environment) -> ResumableMockResource:
        return ResumableMockResource()

    # Use MockEnvironment without pending state, then manually add pending state
    async with MockEnvironment() as env:
        env.resources._pending_state = pending_state
        env.resources.register_factory("a", create_resource)
        env.resources.register_factory("b", create_resource)

        # Restore only "a"
        result = await env.resources.restore_one("a")

        assert result is True
        assert "a" in env.resources
        assert "b" not in env.resources

        resource_a = env.resources.get_typed("a", ResumableMockResource)
        assert resource_a is not None
        assert resource_a.data == "data_a"


async def test_registry_restore_one_not_in_pending() -> None:
    """Should return False when key not in pending state."""
    registry = ResourceRegistry()
    result = await registry.restore_one("missing")
    assert result is False


async def test_registry_close_all_clears_factories() -> None:
    """Should clear factories when closing all resources."""
    async with MockEnvironment() as env:

        async def create_resource(e: Environment) -> SimpleResource:
            return SimpleResource()

        env.resources.register_factory("simple", create_resource)
        await env.resources.get_or_create("simple")

        await env.resources.close_all()

        assert len(env.resources._factories) == 0
        assert len(env.resources) == 0


# --- BaseResource Tests ---


def test_base_resource_implements_protocols() -> None:
    """BaseResource subclasses should implement both Resource and ResumableResource."""
    resource = MockBaseResource()
    assert isinstance(resource, Resource)
    assert isinstance(resource, ResumableResource)


async def test_base_resource_close() -> None:
    """BaseResource.close() should be async."""
    resource = MockBaseResource()
    assert not resource.closed
    await resource.close()
    assert resource.closed


async def test_base_resource_export_state() -> None:
    """BaseResource subclass should export state."""
    resource = MockBaseResource(value="test_value")
    state = await resource.export_state()
    assert state == {"value": "test_value"}


async def test_base_resource_restore_state() -> None:
    """BaseResource subclass should restore state."""
    resource = MockBaseResource()
    await resource.restore_state({"value": "restored_value"})
    assert resource.value == "restored_value"


async def test_base_resource_default_export() -> None:
    """MinimalBaseResource should use default empty export."""
    resource = MinimalBaseResource()
    state = await resource.export_state()
    assert state == {}


async def test_base_resource_default_restore() -> None:
    """MinimalBaseResource should use default no-op restore."""
    resource = MinimalBaseResource()
    await resource.restore_state({"arbitrary": "data"})  # Should not raise


async def test_base_resource_with_registry() -> None:
    """BaseResource subclass should work with ResourceRegistry."""
    async with MockEnvironment() as env:

        async def create_mock(e: Environment) -> MockBaseResource:
            return MockBaseResource(value="initial")

        env.resources.register_factory("mock", create_mock)
        resource = await env.resources.get_or_create_typed("mock", MockBaseResource)
        resource.value = "modified"

        state = await env.resources.export_state()
        assert "mock" in state.entries
        assert state.entries["mock"].state == {"value": "modified"}


# --- Context Instructions Tests ---


async def test_base_resource_default_context_instructions() -> None:
    """BaseResource default get_context_instructions returns None."""
    resource = MinimalBaseResource()
    result = await resource.get_context_instructions()
    assert result is None


async def test_base_resource_custom_context_instructions() -> None:
    """BaseResource subclass can provide custom instructions."""
    resource = ResourceWithInstructions("Use browser for web tasks.")
    result = await resource.get_context_instructions()
    assert result == "Use browser for web tasks."


async def test_registry_get_context_instructions_empty() -> None:
    """ResourceRegistry returns None when no resources have instructions."""
    registry = ResourceRegistry()
    registry.set("simple", SimpleResource())
    result = await registry.get_context_instructions()
    assert result is None


async def test_registry_get_context_instructions_with_resources() -> None:
    """ResourceRegistry collects instructions from all resources."""
    async with MockEnvironment() as env:

        async def create_r1(e: Environment) -> ResourceWithInstructions:
            return ResourceWithInstructions("Instructions for R1")

        async def create_r2(e: Environment) -> ResourceWithInstructions:
            return ResourceWithInstructions("Instructions for R2")

        env.resources.register_factory("r1", create_r1)
        env.resources.register_factory("r2", create_r2)
        await env.resources.get_or_create("r1")
        await env.resources.get_or_create("r2")

        result = await env.resources.get_context_instructions()
        assert result is not None
        assert "Instructions for R1" in result
        assert "Instructions for R2" in result
        assert "r1" in result
        assert "r2" in result


async def test_registry_context_instructions_xml_format() -> None:
    """ResourceRegistry should return XML formatted instructions."""
    async with MockEnvironment() as env:

        async def create_r1(e: Environment) -> ResourceWithInstructions:
            return ResourceWithInstructions("Instruction 1")

        env.resources.register_factory("r1", create_r1)
        await env.resources.get_or_create("r1")

        result = await env.resources.get_context_instructions()
        assert result is not None
        assert "<resources>" in result
        assert "<resource" in result
        assert 'name="r1"' in result
        assert "Instruction 1" in result


# --- setup() Tests ---


async def test_base_resource_default_setup() -> None:
    """BaseResource should have no-op setup by default."""
    resource = MinimalBaseResource()
    await resource.setup()  # Should not raise


async def test_registry_get_or_create_calls_setup() -> None:
    """get_or_create should call setup() after factory creation."""
    async with MockEnvironment() as env:

        async def create_simple(e: Environment) -> SimpleResource:
            return SimpleResource()

        env.resources.register_factory("simple", create_simple)
        resource = await env.resources.get_or_create("simple")

        assert isinstance(resource, SimpleResource)
        assert resource.setup_called


async def test_registry_restore_all_calls_setup_before_restore() -> None:
    """restore_all should call setup() before restore_state()."""

    class SetupOrderResource(BaseResource):
        def __init__(self) -> None:
            self.call_order: list[str] = []

        async def setup(self) -> None:
            self.call_order.append("setup")

        async def restore_state(self, state: dict) -> None:
            self.call_order.append("restore")

        async def export_state(self) -> dict:
            return {}

        async def close(self) -> None:
            pass

    pending_state = ResourceRegistryState(entries={"ordered": ResourceEntry(state={})})

    async def create_ordered(e: Environment) -> SetupOrderResource:
        return SetupOrderResource()

    async with MockEnvironment(
        resource_state=pending_state,
        resource_factories={"ordered": create_ordered},
    ) as env:
        resource = env.resources.get_typed("ordered", SetupOrderResource)
        assert resource is not None
        assert resource.call_order == ["setup", "restore"]


async def test_registry_restore_one_calls_setup_before_restore() -> None:
    """restore_one should call setup() before restore_state()."""

    class SetupOrderResource(BaseResource):
        def __init__(self) -> None:
            self.call_order: list[str] = []

        async def setup(self) -> None:
            self.call_order.append("setup")

        async def restore_state(self, state: dict) -> None:
            self.call_order.append("restore")

        async def export_state(self) -> dict:
            return {}

        async def close(self) -> None:
            pass

    pending_state = ResourceRegistryState(entries={"ordered": ResourceEntry(state={})})

    async def create_ordered(e: Environment) -> SetupOrderResource:
        return SetupOrderResource()

    async with MockEnvironment() as env:
        env.resources._pending_state = pending_state
        env.resources.register_factory("ordered", create_ordered)

        await env.resources.restore_one("ordered")

        resource = env.resources.get_typed("ordered", SetupOrderResource)
        assert resource is not None
        assert resource.call_order == ["setup", "restore"]


# --- get_toolsets Tests ---


async def test_base_resource_default_get_toolsets() -> None:
    """BaseResource should return empty list by default."""
    resource = MinimalBaseResource()
    toolsets = resource.get_toolsets()
    assert toolsets == []


async def test_registry_get_toolsets_empty() -> None:
    """Registry should return empty list when no resources."""
    async with MockEnvironment() as env:
        toolsets = env.resources.get_toolsets()
        assert toolsets == []


async def test_registry_get_toolsets_collects_from_resources() -> None:
    """Registry should collect toolsets from all resources."""

    class ResourceWithToolset(BaseResource):
        def __init__(self, toolset: object) -> None:
            self._toolset = toolset

        async def close(self) -> None:
            pass

        def get_toolsets(self) -> list:
            return [self._toolset]

    async with MockEnvironment() as env:
        toolset1 = object()
        toolset2 = object()

        async def create_r1(e: Environment) -> ResourceWithToolset:
            return ResourceWithToolset(toolset1)

        async def create_r2(e: Environment) -> ResourceWithToolset:
            return ResourceWithToolset(toolset2)

        env.resources.register_factory("r1", create_r1)
        env.resources.register_factory("r2", create_r2)
        await env.resources.get_or_create("r1")
        await env.resources.get_or_create("r2")

        toolsets = env.resources.get_toolsets()
        assert len(toolsets) == 2
        assert toolset1 in toolsets
        assert toolset2 in toolsets


async def test_registry_get_toolsets_with_multiple_toolsets() -> None:
    """Resource can return multiple toolsets."""

    class MultiToolsetResource(BaseResource):
        def __init__(self, toolsets: list) -> None:
            self._toolsets = toolsets

        async def close(self) -> None:
            pass

        def get_toolsets(self) -> list:
            return self._toolsets

    async with MockEnvironment() as env:
        toolset_a = object()
        toolset_b = object()

        async def create_multi(e: Environment) -> MultiToolsetResource:
            return MultiToolsetResource([toolset_a, toolset_b])

        env.resources.register_factory("multi", create_multi)
        await env.resources.get_or_create("multi")

        toolsets = env.resources.get_toolsets()
        assert len(toolsets) == 2
        assert toolset_a in toolsets
        assert toolset_b in toolsets


# --- Factory Environment Access Tests ---


async def test_factory_can_access_file_operator() -> None:
    """Factory should be able to access env.file_operator."""

    async def create_resource(env: Environment) -> ResourceWithEnvAccess:
        return ResourceWithEnvAccess(
            file_operator=env.file_operator,
            shell=env.shell,
        )

    async with MockEnvironment().with_resource_factory("resource", create_resource) as env:
        resource = await env.resources.get_or_create_typed("resource", ResourceWithEnvAccess)

        # Resource should have captured the same file_operator and shell
        assert resource.file_operator is env.file_operator
        assert resource.shell is env.shell


async def test_factory_can_access_other_resources() -> None:
    """Factory should be able to access other registered resources via env.resources."""

    async def create_first(env: Environment) -> SimpleResource:
        return SimpleResource()

    async def create_second(env: Environment) -> SimpleResource:
        # Access first resource via env.resources
        first = env.resources.get_typed("first", SimpleResource)
        assert first is not None
        return SimpleResource()

    async with MockEnvironment() as env:
        env.resources.register_factory("first", create_first)
        env.resources.register_factory("second", create_second)

        # Create first resource
        await env.resources.get_or_create("first")
        # Create second resource - should be able to access first
        second = await env.resources.get_or_create("second")
        assert second is not None


async def test_registry_env_property_before_bind() -> None:
    """Accessing env before bind should raise RuntimeError."""
    registry = ResourceRegistry()

    with pytest.raises(RuntimeError, match="not bound to Environment"):
        _ = registry.env


async def test_registry_bind_returns_self() -> None:
    """bind() should return self for method chaining."""
    async with MockEnvironment() as env:
        registry = ResourceRegistry()
        result = registry.bind(env)
        assert result is registry
        assert registry.env is env


# --- ResourceRegistry Additional Methods Tests ---


async def test_registry_remove() -> None:
    """Should remove and return a resource."""
    async with MockEnvironment() as env:
        resource = SimpleResource()
        env.resources.set("test", resource)
        assert "test" in env.resources

        removed = env.resources.remove("test")
        assert removed is resource
        assert "test" not in env.resources


async def test_registry_remove_not_found() -> None:
    """Should return None when removing non-existent key."""
    async with MockEnvironment() as env:
        removed = env.resources.remove("nonexistent")
        assert removed is None


async def test_registry_keys() -> None:
    """Should return list of resource keys."""
    async with MockEnvironment() as env:
        env.resources.set("a", SimpleResource())
        env.resources.set("b", SimpleResource())
        env.resources.set("c", SimpleResource())

        keys = env.resources.keys()
        assert sorted(keys) == ["a", "b", "c"]


async def test_registry_len() -> None:
    """Should return number of resources."""
    async with MockEnvironment() as env:
        assert len(env.resources) == 0
        env.resources.set("a", SimpleResource())
        assert len(env.resources) == 1
        env.resources.set("b", SimpleResource())
        assert len(env.resources) == 2


async def test_registry_set_invalid_type() -> None:
    """Should raise TypeError when setting non-Resource object."""
    async with MockEnvironment() as env:
        with pytest.raises(TypeError, match="must implement Resource protocol"):
            env.resources.set("invalid", "not a resource")  # type: ignore[arg-type]


async def test_registry_get_typed_wrong_type_returns_none() -> None:
    """get_typed should return None if type doesn't match."""
    async with MockEnvironment() as env:
        env.resources.set("simple", SimpleResource())
        result = env.resources.get_typed("simple", ResumableMockResource)
        assert result is None


async def test_registry_close_all_with_async_close() -> None:
    """Should handle resources with async close() methods."""
    async with MockEnvironment() as env:
        resource = MockBaseResource()  # Has async close
        env.resources.set("async", resource)

        await env.resources.close_all()
        assert resource.closed


async def test_registry_close_all_with_exception() -> None:
    """Should continue cleanup even if a resource fails to close."""

    class FailingResource:
        def close(self) -> None:
            raise RuntimeError("Close failed")

        async def setup(self) -> None:
            pass

        def get_toolsets(self) -> list:
            return []

    async with MockEnvironment() as env:
        failing = FailingResource()
        good = SimpleResource()
        env.resources.set("failing", failing)
        env.resources.set("good", good)

        # Should not raise, even though one resource fails
        await env.resources.close_all()
        assert good.closed


async def test_registry_restore_one_no_factory() -> None:
    """restore_one should raise KeyError when factory is missing."""
    async with MockEnvironment() as env:
        env.resources._pending_state = ResourceRegistryState(entries={"missing": ResourceEntry(state={"data": "test"})})
        # No factory registered for "missing"

        with pytest.raises(KeyError, match="No factory registered for resource"):
            await env.resources.restore_one("missing")


async def test_registry_contains() -> None:
    """ResourceRegistry should support 'in' operator."""
    async with MockEnvironment() as env:
        assert "test" not in env.resources
        env.resources.set("test", SimpleResource())
        assert "test" in env.resources


async def test_registry_close_all_parallel() -> None:
    """close_all with parallel=True should close resources concurrently."""
    async with MockEnvironment() as env:
        resource1 = SimpleResource()
        resource2 = SimpleResource()
        resource3 = MinimalBaseResource()

        env.resources.set("r1", resource1)
        env.resources.set("r2", resource2)
        env.resources.set("r3", resource3)

        # Close with parallel=True
        await env.resources.close_all(parallel=True)

        assert resource1.closed
        assert resource2.closed
        assert resource3.closed
        assert len(env.resources) == 0


async def test_registry_close_all_parallel_with_exception() -> None:
    """Parallel close should continue even if a resource fails."""

    class FailingResource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            raise RuntimeError("Failed to close")

        async def setup(self) -> None:
            pass

        def get_toolsets(self) -> list:
            return []

    async with MockEnvironment() as env:
        good1 = SimpleResource()
        bad = FailingResource()
        good2 = SimpleResource()

        env.resources.set("good1", good1)
        env.resources.set("bad", bad)
        env.resources.set("good2", good2)

        # Should not raise
        await env.resources.close_all(parallel=True)

        assert good1.closed
        assert good2.closed
