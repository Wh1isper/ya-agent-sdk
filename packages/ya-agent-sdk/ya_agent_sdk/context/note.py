"""Note management for agent sessions.

This module provides a simple key-value note store that persists
across turns and sessions. Runtime instructions expose note keys and
agents can read note values on demand through the note tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NoteManager(BaseModel):
    """Manager for session-persistent note entries.

    Provides a simple key-value store that the agent can read/write.
    Runtime instructions expose keys on every user prompt, giving the
    agent persistent recall without injecting full note values.

    Root run attempts share one session manager. A subagent receives an independent
    snapshot so child mutation and durable continuation cannot alias parent state.

    Example:
        manager = NoteManager()
        manager.set("language", "User prefers Chinese communication")
        manager.set("os", "macOS")
        manager.delete("os")
        entries = manager.list_all()
    """

    entries: dict[str, str] = Field(default_factory=dict)
    """All note entries keyed by entry key."""

    def set(self, key: str, value: str) -> None:
        """Set or update a note entry.

        Args:
            key: Unique key for the note entry.
            value: Content to store.
        """
        self.entries[key] = value

    def get(self, key: str) -> str | None:
        """Get a note entry by key.

        Args:
            key: The key to look up.

        Returns:
            The stored value if found, None otherwise.
        """
        return self.entries.get(key)

    def delete(self, key: str) -> bool:
        """Delete a note entry by key.

        Args:
            key: The key to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    def list_all(self) -> list[tuple[str, str]]:
        """Get all entries sorted by key.

        Returns:
            List of (key, value) tuples sorted by key.
        """
        return sorted(self.entries.items())

    def list_keys(self) -> list[str]:
        """Get all entry keys sorted alphabetically.

        Returns:
            List of note entry keys.
        """
        return sorted(self.entries)

    def export_notes(self) -> dict[str, str]:
        """Export entries for serialization.

        Returns:
            Dict of key-value pairs.
        """
        return dict(self.entries)

    @classmethod
    def from_exported(cls, data: dict[str, Any]) -> NoteManager:
        """Restore NoteManager from exported data.

        Args:
            data: Exported note data from export_notes().

        Returns:
            Restored NoteManager instance.
        """
        return cls(entries={str(k): str(v) for k, v in data.items()})
