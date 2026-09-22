# src/database/connection.py
"""
Database engine and session factory configuration.

This module centralizes:
    - Database URL resolution (from environment variables).
    - Engine creation with environment-aware options.
    - A configured `SessionLocal` factory for ORM sessions.

Environment variables:
    DATABASE_URL    Full SQLAlchemy database URL. Takes precedence over all
                    other settings if provided.
    TESTING         If set to a truthy value, uses an in-memory SQLite
                    database and disables SQL echo.
    SQL_ECHO        If set to a truthy value, logs all SQL statements.
                    Defaults to True in non-testing environments.

Typical usage:
    from src.database.connection import SessionLocal, engine

    with SessionLocal() as session:
        ...
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def is_testing() -> bool:
    """Return True when the TESTING environment variable is truthy."""
    return _env_flag("TESTING", default=False)


# ---------------------------------------------------------------------------
# Database URL resolution
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    """
    Resolve the database URL based on the current environment.

    Priority:
        1. DATABASE_URL environment variable, if set.
        2. In-memory SQLite when TESTING is truthy.
        3. Local file-based SQLite (default).

    Returns:
        A SQLAlchemy database URL string.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    if is_testing():
        return "sqlite:///:memory:"
    return "sqlite:///minerales_eds.db"


DATABASE_URL: str = get_database_url()
IS_TESTING: bool = is_testing()

# Enable SQL echo by default outside tests; allow explicit override.
ECHO_SQL: bool = _env_flag("SQL_ECHO", default=not IS_TESTING)


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def _create_engine() -> Engine:
    """
    Build the SQLAlchemy engine with environment-aware options.

    For in-memory SQLite we use `StaticPool` and
    `check_same_thread=False` so that all sessions share a single
    connection — otherwise each checkout would get a fresh, empty
    in-memory database and queries would fail with "no such table".
    """
    if DATABASE_URL.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

        # In-memory SQLite must use a shared connection pool.
        if ":memory:" in DATABASE_URL or IS_TESTING:
            return create_engine(
                DATABASE_URL,
                echo=ECHO_SQL,
                future=True,
                connect_args=connect_args,
                poolclass=StaticPool,
            )

        # File-based SQLite: also disable same-thread check for async use.
        return create_engine(
            DATABASE_URL,
            echo=ECHO_SQL,
            future=True,
            connect_args=connect_args,
        )

    # Non-SQLite backends (PostgreSQL, MySQL, etc.).
    return create_engine(
        DATABASE_URL,
        echo=ECHO_SQL,
        future=True,
        pool_pre_ping=True,   # Recover from stale connections.
        pool_recycle=1800,    # Recycle connections after 30 minutes.
    )


engine: Engine = _create_engine()


# ---------------------------------------------------------------------------
# SQLite tuning (foreign keys, WAL mode)
# ---------------------------------------------------------------------------

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_on_connect(dbapi_connection, _connection_record) -> None:
        """
        Enable foreign key enforcement and WAL mode on every new
        SQLite connection. Both are off by default in SQLite but are
        required for correct, concurrent behavior.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL mode is not supported for in-memory databases.
            if ":memory:" not in DATABASE_URL:
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
    class_=Session,
    future=True,
)


def get_session() -> Session:
    """
    Return a new ORM session bound to the configured engine.

    Callers are responsible for closing it, e.g.:

        with get_session() as session:
            ...
    """
    return SessionLocal()


def dispose_engine() -> None:
    """
    Dispose of the engine's connection pool.

    Call this on application shutdown (or after test teardown) to release
    all pooled connections cleanly.
    """
    try:
        engine.dispose()
        logger.debug("Database engine disposed.")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to dispose the database engine.")


# ---------------------------------------------------------------------------
# Startup logging (helpful when debugging environment issues)
# ---------------------------------------------------------------------------

logger.debug(
    "Database configured: url=%s testing=%s echo=%s",
    DATABASE_URL,
    IS_TESTING,
    ECHO_SQL,
)