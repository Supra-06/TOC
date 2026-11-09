"""FastAPI dependencies for database sessions and anomaly service.

Provides clean dependency injection interfaces for FastAPI endpoints,
reusing connection and service lifecycles without duplicating logic.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.anomaly_service import AnomalyService, get_anomaly_service


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session from the connection pool.

    Reuses backend.database.connection.get_db to guarantee proper session closing.

    Yields:
        Active SQLAlchemy Session.
    """
    yield from get_db()


def get_service() -> AnomalyService:
    """Return the shared AnomalyService singleton.

    Maintains lazy model loading; does not trigger model artifact loading on call.

    Returns:
        AnomalyService instance.
    """
    return get_anomaly_service()

