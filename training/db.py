"""Database helper for the training pipeline.

The pipeline stores the raw, clean and feature tables in a local SQLite file (data/perio.db) so
that every step can be inspected with SQL (see sql/views.sql). It needs no setup and costs
nothing. Set a DATABASE_URL environment variable to use another database (for example
PostgreSQL; install a driver such as psycopg2-binary first).
"""
import os
from pathlib import Path

from sqlalchemy import create_engine, text

from .config import ROOT

SQLITE_PATH = ROOT / "data" / "perio.db"


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        # SQLAlchemy wants "postgresql://", some providers show "postgres://"
        return url.replace("postgres://", "postgresql://", 1)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{SQLITE_PATH}"


def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


def run_sql_file(engine, path: Path) -> None:
    """Run every statement in a .sql file (statements separated by ';')."""
    sql = Path(path).read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
