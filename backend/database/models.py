"""SQLAlchemy ORM models for workflow trace analysis persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""
    pass


class WorkflowAnalysis(Base):
    """Immutable audit record representing a single workflow trace analysis execution."""

    __tablename__ = "workflow_analyses"

    # 1. Primary Key & Order Identifier
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)

    # 2. Raw Input Sequence Preservation
    input_format: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_input_payload: Mapped[Any] = mapped_column(JSON, nullable=False)

    # 3. High-Level Outcome & Decision Verdict
    verdict: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    anomaly_type: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # 4. Deterministic FSM Evaluation
    is_rule_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    final_state: Mapped[str] = mapped_column(String(64), nullable=False)
    total_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_transitions: Mapped[int] = mapped_column(Integer, nullable=False)
    fsm_error: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON(none_as_null=True), nullable=True)

    # 5. Statistical ML Evaluation (Strictly conditional)
    ml_evaluated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ml_prediction: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_ml_anomaly: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_decision_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ml_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON(none_as_null=True), nullable=True)
    ml_skip_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # 6. Operational Diagnostics
    diagnostic_severity: Mapped[str] = mapped_column(String(16), nullable=False)
    diagnostic_root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    diagnostic_key_factors: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    diagnostic_recommended_action: Mapped[str] = mapped_column(Text, nullable=False)

    # 7. Standardized UTC Timestamp (Naive UTC by contract)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    # Database Constraints & Composite Indexes
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('NORMAL', 'RULE_VIOLATION', 'STATISTICAL_ANOMALY', 'VALID_INCOMPLETE')",
            name="ck_verdict_valid",
        ),
        CheckConstraint(
            "anomaly_type IS NULL OR anomaly_type IN ('RULE_VIOLATION', 'STATISTICAL_ANOMALY')",
            name="ck_anomaly_type_valid",
        ),
        CheckConstraint(
            "input_format IN ('DELIMITED_STRING', 'EVENT_LIST')",
            name="ck_input_format_valid",
        ),
        CheckConstraint("total_steps >= 0", name="ck_total_steps_nonneg"),
        CheckConstraint("successful_transitions >= 0", name="ck_successful_transitions_nonneg"),
        CheckConstraint("successful_transitions <= total_steps", name="ck_transitions_lte_steps"),
        CheckConstraint(
            "diagnostic_severity IN ('CRITICAL', 'WARNING', 'INFO')",
            name="ck_diagnostic_severity_valid",
        ),
        CheckConstraint(
            """
            (
                ml_evaluated = 1 AND
                ml_prediction IS NOT NULL AND
                is_ml_anomaly IS NOT NULL AND
                anomaly_score IS NOT NULL AND
                raw_decision_score IS NOT NULL AND
                ml_features IS NOT NULL AND
                ml_features != 'null' AND
                ml_skip_reason IS NULL AND
                (
                    (ml_prediction = 'NORMAL' AND is_ml_anomaly = 0) OR
                    (ml_prediction = 'ANOMALY' AND is_ml_anomaly = 1)
                )
            )
            OR
            (
                ml_evaluated = 0 AND
                ml_prediction IS NULL AND
                is_ml_anomaly IS NULL AND
                anomaly_score IS NULL AND
                raw_decision_score IS NULL AND
                (ml_features IS NULL OR ml_features = 'null') AND
                ml_skip_reason IS NOT NULL
            )
            """,
            name="ck_ml_evaluation_consistency",
        ),
        Index("ix_analyses_order_created_id", "order_id", "created_at", "id"),
    )
