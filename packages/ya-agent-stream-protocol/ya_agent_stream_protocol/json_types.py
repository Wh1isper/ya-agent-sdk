from __future__ import annotations

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray = list[JsonValue]
JsonObject = dict[str, JsonValue]
