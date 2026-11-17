"""Alembic environment. The database URL always comes from app settings (never from alembic.ini)."""
from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.app import models  # noqa: F401  (registers every table on Base.metadata)
from backend.app.config import settings
from backend.app.database import Base

config = context.config
target_metadata = Base.metadata


def _configure(**kw) -> None:
    url = kw.get("url") or str(kw["connection"].engine.url)
    context.configure(
        target_metadata=target_metadata,
        render_as_batch=url.startswith("sqlite"),  # SQLite cannot ALTER most things in place
        compare_type=True,
        **kw,
    )


def run_migrations_offline() -> None:
    _configure(url=settings.database_url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:  # called from the app with an existing connection
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = engine_from_config({"sqlalchemy.url": settings.database_url}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as conn:
        _configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
