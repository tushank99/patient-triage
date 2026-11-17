"""
Schema management through Alembic (replaces Base.metadata.create_all).

  run_migrations()  upgrade to head. A database created by the pre-Alembic code (tables exist but no
                    alembic_version) is stamped at the baseline first, then upgraded, so existing demo
                    databases keep their data.
  reset_database()  LOCAL/TEST ONLY: drop everything and rebuild from migrations.
"""
import logging
import os

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from backend.app.database import Base, engine

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "0001"


def alembic_config() -> Config:
    cfg = Config(os.path.join(BACKEND_DIR, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(BACKEND_DIR, "migrations"))
    return cfg


def run_migrations(target_engine=None) -> None:
    cfg = alembic_config()
    with (target_engine or engine).begin() as conn:
        cfg.attributes["connection"] = conn
        tables = set(inspect(conn).get_table_names())
        if "alembic_version" not in tables and "patients" in tables:
            logger.warning("Pre-Alembic database detected: stamping baseline %s before upgrading.", BASELINE)
            command.stamp(cfg, BASELINE)
        command.upgrade(cfg, "head")


def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    run_migrations()
