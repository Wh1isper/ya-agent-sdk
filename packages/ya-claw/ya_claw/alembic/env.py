from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from ya_claw.config import ClawSettings
from ya_claw.db.engine import to_sync_database_url
from ya_claw.orm import tables as _tables  # noqa: F401
from ya_claw.orm.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
settings = ClawSettings()


def get_url() -> str:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return to_sync_database_url(settings.resolved_database_url)


def include_object(obj: object, name: str | None, type_: str, reflected: bool, compare_to: object) -> bool:
    return not (type_ == "table" and reflected and compare_to is None)


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(get_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
