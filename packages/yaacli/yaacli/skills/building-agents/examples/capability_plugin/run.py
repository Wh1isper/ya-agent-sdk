"""Discover, select, and execute the installed capability plugin."""

from __future__ import annotations

import asyncio
import sys

from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import build_default_capability_catalog, discover_capability_types

ENTRY_POINT_NAME = "example.text_metrics"
PLUGIN_MODULES = ("example_capability_plugin", "example_capability_plugin.capability")


async def run() -> None:
    references = [
        reference for reference in discover_capability_types() if reference.entry_point_name == ENTRY_POINT_NAME
    ]
    if len(references) != 1:
        raise RuntimeError(f"Expected one {ENTRY_POINT_NAME!r} entry point, found {len(references)}")
    imported_plugin_modules = [module_name for module_name in PLUGIN_MODULES if module_name in sys.modules]
    if imported_plugin_modules:
        raise RuntimeError(f"Metadata discovery imported plugin modules: {imported_plugin_modules!r}")

    reference = references[0]
    print(f"Discovered metadata: {reference.entry_point_name} -> {reference.import_target}")

    catalog = build_default_capability_catalog(selected_entry_points=[ENTRY_POINT_NAME])
    provenance = catalog.provenance(ENTRY_POINT_NAME)
    print(f"Selected plugin: {provenance.display_name}")

    spec = AgentSpec.from_dict({
        "name": "capability-plugin-example",
        "capabilities": [
            {
                ENTRY_POINT_NAME: {
                    "max_characters": 5_000,
                }
            }
        ],
    })
    runtime = create_agent(
        TestModel(call_tools=["text_metrics"]),
        spec=spec,
        custom_capability_types=catalog.custom_capability_types,
    )

    async with runtime:
        result = await runtime.agent.run(
            "Measure this text: capability plugins remain explicit.",
            deps=runtime.ctx,
        )

    print(f"Agent result: {result.output}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
