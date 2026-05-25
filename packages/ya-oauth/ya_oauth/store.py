from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from types import TracebackType

from ya_oauth.types import AuthFile, OAuthProviderRecord, TokenSnapshot

T = TypeVar("T")

DEFAULT_AUTH_DIR = Path.home() / ".yaai"
DEFAULT_AUTH_PATH = DEFAULT_AUTH_DIR / "auth.json"


class FileLock:
    """Cross-platform advisory file lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._msvcrt = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._file = self.path.open("a+")
        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt
            if self._file.tell() == 0:
                self._file.write("\0")
                self._file.flush()
            self._file.seek(0)
            self._msvcrt.locking(self._file.fileno(), self._msvcrt.LK_LOCK, 1)
        else:
            _lock_posix_file(self._file.fileno())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is None:
            return
        try:
            if os.name == "nt" and self._msvcrt is not None:
                self._file.seek(0)
                self._msvcrt.locking(self._file.fileno(), self._msvcrt.LK_UNLCK, 1)
            else:
                _unlock_posix_file(self._file.fileno())
        finally:
            self._file.close()
            self._file = None


def _lock_posix_file(fd: int) -> None:
    import fcntl

    fcntl_module = cast(Any, fcntl)
    fcntl_module.flock(fd, fcntl_module.LOCK_EX)


def _unlock_posix_file(fd: int) -> None:
    import fcntl

    fcntl_module = cast(Any, fcntl)
    fcntl_module.flock(fd, fcntl_module.LOCK_UN)


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

    def _locked(self) -> FileLock:
        return FileLock(self.lock_path)

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
