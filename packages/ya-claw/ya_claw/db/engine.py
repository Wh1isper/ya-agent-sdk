from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


def create_engine(database_url: str, **kwargs: object) -> AsyncEngine:
    defaults: dict[str, Any] = {
        "echo": False,
        "pool_pre_ping": True,
    }

    connect_args: dict[str, Any] | None = None
    raw_connect_args = kwargs.pop("connect_args", None)

    if _is_sqlite_url(database_url):
        connect_args = {"check_same_thread": False}
    else:
        defaults.update({
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle": 3600,
        })

    if isinstance(raw_connect_args, Mapping):
        merged_connect_args = dict(connect_args or {})
        merged_connect_args.update(raw_connect_args)
        connect_args = merged_connect_args

    if connect_args is not None:
        defaults["connect_args"] = connect_args

    defaults.update(kwargs)
    return create_async_engine(database_url, **defaults)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def to_sync_database_url(database_url: str) -> str:
    return (
        database_url
        .replace("sqlite+aiosqlite://", "sqlite://")
        .replace("postgresql+asyncpg://", "postgresql+psycopg://")
        .replace("postgresql+psycopg_async://", "postgresql+psycopg://")
    )
