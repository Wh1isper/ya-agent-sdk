from __future__ import annotations

from typing import Any, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, Any]
JsonArray: TypeAlias = list[Any]
JsonValue: TypeAlias = JsonScalar | JsonObject | JsonArray
