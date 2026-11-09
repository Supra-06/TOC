"""Database engine, session management, and initialization.

Configures the SQLAlchemy engine and session factory, providing clean session
lifecycle management and dependency injection for service and API layers.
"""

from __future__ import annotations

from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import DATABASE_ECHO, DATABASE_URL
from backend.database.models import Base

# SQLite thread safety configuration
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine: Engine = create_engine(
    DATABASE_URL,
    echo=DATABASE_ECHO,
    connect_args=connect_args,
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all database tables using the target or default engine.

    Args:
        target_engine: Optional custom SQLAlchemy Engine. Defaults to global engine.
    """
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session context for dependency injection.

    Guarantees that the session is closed in a finally block upon completion.

    Yields:
        Active SQLAlchemy Session.
    """
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
