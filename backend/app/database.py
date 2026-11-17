from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.config import settings

if settings.database_url.startswith("sqlite"):
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _sqlite_foreign_keys(dbapi_conn, _):  # SQLite ignores ON DELETE CASCADE unless this is on
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
else:
    # Neon free tier: keep the pool small; pre_ping handles connections dropped by scale-to-zero.
    engine = create_engine(settings.database_url, pool_size=5, max_overflow=5, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency helper to yield database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
