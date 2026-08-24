"""Exact SQLite schema validation shared by durable YAACLI stores."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

SchemaObjectKey = tuple[str, str]
SESSION_SCHEMA_VERSION = 6


def user_schema_object_names(connection: sqlite3.Connection) -> frozenset[str]:
    """Return every non-internal object name in the current database."""
    rows = connection.execute("SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchall()
    return frozenset(str(row[0]) for row in rows)


def expected_schema_object_names(schema: str) -> frozenset[str]:
    """Return table and explicit-index names produced by a schema script."""
    return frozenset(name for (_kind, name), _sql in _expected_schema_objects(schema))


def validate_exact_schema_subset(
    connection: sqlite3.Connection,
    schema: str,
    *,
    error_prefix: str,
) -> None:
    """Require every expected table/index definition while allowing unrelated objects."""
    expected = dict(_expected_schema_objects(schema))
    current = _read_schema_objects(connection)
    missing = sorted(f"{kind}:{name}" for kind, name in expected if (kind, name) not in current)
    mismatched = sorted(
        f"{kind}:{name}"
        for (kind, name), sql in expected.items()
        if (kind, name) in current and current[(kind, name)] != sql
    )
    if not missing and not mismatched:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if mismatched:
        details.append(f"definition mismatch for {', '.join(mismatched)}")
    raise RuntimeError(
        f"{error_prefix}; {'; '.join(details)}. "
        "Migrate the database offline or recreate it; runtime schema compatibility is not supported."
    )


@lru_cache(maxsize=8)
def _expected_schema_objects(schema: str) -> tuple[tuple[SchemaObjectKey, str], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema)
        objects = _read_schema_objects(connection)
    finally:
        connection.close()
    return tuple(sorted(objects.items()))


def _read_schema_objects(connection: sqlite3.Connection) -> dict[SchemaObjectKey, str]:
    rows = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_schema
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        """
    ).fetchall()
    return {(str(row[0]), str(row[1])): _normalize_sql(str(row[2])) for row in rows}


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().removesuffix(";").split())
