"""Database package providing SQLAlchemy models, connection, and repository.

Exports the core ORM schema, engine/session lifecycle utilities, and repository
functions for persisting and querying workflow analysis records.
"""

from backend.database.connection import SessionLocal, engine, get_db, init_db
from backend.database.models import Base, WorkflowAnalysis
from backend.database.repository import (
    SUPPORTED_VERDICTS,
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_latest_analysis_by_order_id,
    get_summary_metrics,
    list_analyses,
    to_naive_utc,
    validate_and_classify_input,
)

__all__ = [
    "Base",
    "WorkflowAnalysis",
    "engine",
    "SessionLocal",
    "init_db",
    "get_db",
    "SUPPORTED_VERDICTS",
    "create_analysis_record",
    "create_analysis_batch",
    "get_analysis_by_id",
    "get_analyses_by_order_id",
    "get_latest_analysis_by_order_id",
    "list_analyses",
    "get_summary_metrics",
    "validate_and_classify_input",
    "to_naive_utc",
]
