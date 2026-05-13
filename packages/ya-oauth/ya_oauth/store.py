from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ya_oauth.types import AuthFile, OAuthProviderRecord, TokenSnapshot

T = TypeVar("T")

DEFAULT_AUTH_DIR = Path.home() / ".yaai"
DEFAULT_AUTH_PATH = DEFAULT_AUTH_DIR / "auth.json"


class OAuthStore:
    """File-backed OAuth credential store with process-level locking."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else DEFAULT_AUTH_PATH
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> AuthFile:
        self._ensure_parent()
        with self._locked():
            return self._load_unlocked()

    def save(self, auth_file: AuthFile) -> None:
        self._ensure_parent()
        with self._locked():
            self._save_unlocked(auth_file)

    def get_provider(self, provider_name: str) -> OAuthProviderRecord | None:
        return self.load().providers.get(provider_name)

    def set_provider(self, provider_name: str, record: OAuthProviderRecord) -> None:
        def update(auth_file: AuthFile) -> None:
            auth_file.providers[provider_name] = record

        self.update(update)

    def delete_provider(self, provider_name: str) -> OAuthProviderRecord | None:
        deleted: OAuthProviderRecord | None = None

        def update(auth_file: AuthFile) -> None:
            nonlocal deleted
            deleted = auth_file.providers.pop(provider_name, None)

        self.update(update)
        return deleted

    def update(self, updater: Callable[[AuthFile], T]) -> T:
        self._ensure_parent()
        with self._locked():
            auth_file = self._load_unlocked()
            result = updater(auth_file)
            self._save_unlocked(auth_file)
            return result

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(PermissionError):
            os.chmod(self.path.parent, 0o700)

    @contextlib.contextmanager
    def _locked(self):  # type: ignore[no-untyped-def]
        self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> AuthFile:
        if not self.path.exists():
            return AuthFile()
        self._repair_file_mode_unlocked()
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return AuthFile.model_validate(data)

    def _save_unlocked(self, auth_file: AuthFile) -> None:
        payload = auth_file.model_dump(mode="json", exclude_none=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, text=True)
        tmp_path = Path(tmp_name)
        try:
            os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise
        with contextlib.suppress(PermissionError):
            os.chmod(self.path, 0o600)

    def _repair_file_mode_unlocked(self) -> None:
        mode = stat.S_IMODE(self.path.stat().st_mode)
        if mode != 0o600:
            os.chmod(self.path, 0o600)


class StoreBackedTokenSource:
    """OAuthTokenSource backed by OAuthStore and a provider-specific refresher."""

    def __init__(
        self,
        provider_name: str,
        *,
        store: OAuthStore | None = None,
        refresh_provider: Callable[[OAuthProviderRecord], OAuthProviderRecord],
    ) -> None:
        self.provider_name = provider_name
        self.store = store or OAuthStore()
        self._refresh_provider = refresh_provider

    async def get_token(self) -> TokenSnapshot:
        record = self.store.get_provider(self.provider_name)
        if record is None:
            raise RuntimeError(f"OAuth provider is not logged in: {self.provider_name}")
        return _snapshot(self.provider_name, record)

    async def refresh_token(self) -> TokenSnapshot:
        refreshed = await asyncio.to_thread(self._refresh_and_store)
        return _snapshot(self.provider_name, refreshed)

    def _refresh_and_store(self) -> OAuthProviderRecord:
        refreshed: OAuthProviderRecord | None = None

        def update(auth_file: AuthFile) -> None:
            nonlocal refreshed
            record = auth_file.providers.get(self.provider_name)
            if record is None:
                raise RuntimeError(f"OAuth provider is not logged in: {self.provider_name}")
            refreshed = self._refresh_provider(record)
            auth_file.providers[self.provider_name] = refreshed

        self.store.update(update)
        if refreshed is None:
            raise RuntimeError(f"OAuth provider refresh failed: {self.provider_name}")
        return refreshed


def _snapshot(provider_name: str, record: OAuthProviderRecord) -> TokenSnapshot:
    return TokenSnapshot(
        provider_name=provider_name,
        access_token=record.tokens.access_token,
        account=record.account,
        base_url=record.base_url,
    )
