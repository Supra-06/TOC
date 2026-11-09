"""Database repository providing persistence, queries, and transactions.

Encapsulates all database operations for WorkflowAnalysis records, guaranteeing:
- Verbatim preservation of original string and list event inputs
- Strict UTC datetime handling and deterministic ordering
- ML prediction and verdict integrity constraints
- Atomic batch persistence with automatic rollback
- Validated query pagination, filtering, and summary metrics
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.database.models import WorkflowAnalysis
from backend.ml.feature_extraction.extractor import FEATURE_NAMES

SUPPORTED_VERDICTS: Tuple[str, ...] = (
    "NORMAL",
    "RULE_VIOLATION",
    "STATISTICAL_ANOMALY",
    "VALID_INCOMPLETE",
)


# ---------------------------------------------------------------------------
# Helper Functions: Input Classification & Timestamp Normalization
# ---------------------------------------------------------------------------

def validate_and_classify_input(raw_input: Any) -> Tuple[str, Any]:
    """Inspect and classify raw input without mutating its contents.

    Accepts:
      - Any str, including empty ("") and whitespace-only ("   ").
        Format: 'DELIMITED_STRING'
      - Any list[str], including empty lists ([]) and lists containing
        blank strings (["", "   "]).
        Format: 'EVENT_LIST'

    Rejects:
      - Tuples, sets, dictionaries, numbers, or other container types (TypeError).
      - Lists containing non-string elements (TypeError).

    Args:
        raw_input: Raw trace representation.

    Returns:
        Tuple of (input_format, raw_payload).

    Raises:
        TypeError: If raw_input has an unsupported type or contains non-string elements.
    """
    if isinstance(raw_input, str):
        return "DELIMITED_STRING", raw_input

    if isinstance(raw_input, list):
        if not all(isinstance(item, str) for item in raw_input):
            raise TypeError("All elements in raw input event list must be strings.")
        return "EVENT_LIST", raw_input

    if isinstance(raw_input, tuple):
        raise TypeError(
            f"Unsupported input type 'tuple'. Only 'str' (delimited) and 'list[str]' "
            f"are permitted to guarantee exact representation preservation."
        )

    raise TypeError(
        f"Unsupported raw input type '{type(raw_input).__name__}'. "
        f"Expected 'str' or 'list[str]'."
    )


def to_naive_utc(timestamp: Optional[datetime] = None) -> datetime:
    """Normalize a datetime to a UTC-naive representation for SQLite storage.

    Policy:
      - If timestamp is None: generates current UTC time as naive datetime.
      - If timestamp is timezone-aware: converts to UTC and drops tzinfo.
      - If timestamp is naive: treated as UTC by contract.

    Args:
        timestamp: Optional input datetime.

    Returns:
        Naive datetime object representing UTC.
    """
    if timestamp is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if timestamp.tzinfo is not None:
        return timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp


def _build_record(
    analysis_result: Dict[str, Any],
    raw_input: Union[str, List[str]],
    timestamp: Optional[datetime] = None,
) -> WorkflowAnalysis:
    """Construct and validate a WorkflowAnalysis ORM instance from an analysis result.

    Validates ML consistency and verdict domains prior to ORM creation.

    Args:
        analysis_result: Dictionary output from AnomalyService or hybrid analyzer.
        raw_input: Original raw input trace (str or list[str]).
        timestamp: Optional explicit timestamp.

    Returns:
        WorkflowAnalysis instance.

    Raises:
        TypeError: If raw_input type is unsupported.
        ValueError: If verdict or ML fields violate consistency rules.
    """
    input_format, raw_payload = validate_and_classify_input(raw_input)

    verdict = analysis_result.get("verdict")
    if verdict not in SUPPORTED_VERDICTS:
        raise ValueError(
            f"Unsupported verdict '{verdict}'. Expected one of: {SUPPORTED_VERDICTS}"
        )

    is_anomaly = bool(analysis_result.get("is_anomaly", False))
    anomaly_type = analysis_result.get("anomaly_type")

    # FSM Evaluation extraction
    fsm_eval = analysis_result.get("fsm_evaluation", {})
    is_rule_valid = bool(fsm_eval.get("is_rule_valid", False))
    is_complete = bool(fsm_eval.get("is_complete", False))
    final_state = str(fsm_eval.get("final_state", "UNKNOWN"))
    total_steps = int(fsm_eval.get("total_steps", 0))
    successful_transitions = int(fsm_eval.get("successful_transitions", 0))
    fsm_error = fsm_eval.get("error")

    # ML Evaluation & Consistency Validation
    ml_eval = analysis_result.get("ml_evaluation")
    ml_skip_reason = analysis_result.get("ml_skip_reason")

    if ml_eval is not None:
        if not isinstance(ml_eval, dict):
            raise ValueError(
                f"'ml_evaluation' must be a dictionary when present, got {type(ml_eval).__name__}."
            )

        ml_evaluated = True

        ml_prediction = ml_eval.get("prediction")
        if ml_prediction not in ("NORMAL", "ANOMALY"):
            raise ValueError(
                f"Invalid ml_prediction '{ml_prediction}'. Expected 'NORMAL' or 'ANOMALY'."
            )

        if "is_ml_anomaly" not in ml_eval:
            raise ValueError("ML-evaluated record must include 'is_ml_anomaly'.")
        is_ml_anomaly = ml_eval["is_ml_anomaly"]
        if type(is_ml_anomaly) is not bool:
            raise ValueError(
                f"'is_ml_anomaly' must be an actual boolean (True or False), got {type(is_ml_anomaly).__name__}: {is_ml_anomaly!r}"
            )

        if ml_prediction == "NORMAL" and is_ml_anomaly is not False:
            raise ValueError("Inconsistent ML evaluation: prediction 'NORMAL' requires is_ml_anomaly=False.")
        if ml_prediction == "ANOMALY" and is_ml_anomaly is not True:
            raise ValueError("Inconsistent ML evaluation: prediction 'ANOMALY' requires is_ml_anomaly=True.")

        if "anomaly_score" not in ml_eval or ml_eval["anomaly_score"] is None:
            raise ValueError("ML-evaluated record must include 'anomaly_score'.")
        score_val = ml_eval["anomaly_score"]
        if type(score_val) is bool or not isinstance(score_val, (int, float)):
            raise ValueError(
                f"'anomaly_score' must be numeric (int or float), got {type(score_val).__name__}: {score_val!r}"
            )
        if not math.isfinite(float(score_val)):
            raise ValueError(f"'anomaly_score' must be finite, got {score_val!r}")
        anomaly_score = float(score_val)

        if "raw_decision_score" not in ml_eval or ml_eval["raw_decision_score"] is None:
            raise ValueError("ML-evaluated record must include 'raw_decision_score'.")
        raw_val = ml_eval["raw_decision_score"]
        if type(raw_val) is bool or not isinstance(raw_val, (int, float)):
            raise ValueError(
                f"'raw_decision_score' must be numeric (int or float), got {type(raw_val).__name__}: {raw_val!r}"
            )
        if not math.isfinite(float(raw_val)):
            raise ValueError(f"'raw_decision_score' must be finite, got {raw_val!r}")
        raw_decision_score = float(raw_val)

        if "features" not in ml_eval:
            raise ValueError("ML-evaluated record must include 'features'.")
        ml_features = ml_eval["features"]
        if not isinstance(ml_features, dict):
            raise ValueError(f"'features' must be a dictionary, got {type(ml_features).__name__}")

        expected_feature_keys = set(FEATURE_NAMES)
        actual_feature_keys = set(ml_features.keys())

        missing_features = expected_feature_keys - actual_feature_keys
        if missing_features:
            raise ValueError(
                f"ML features dictionary is missing approved features: {sorted(missing_features)}."
            )

        unexpected_features = actual_feature_keys - expected_feature_keys
        if unexpected_features:
            raise ValueError(
                f"ML features dictionary contains unexpected features: {sorted(unexpected_features)}."
            )

        for feat_name in FEATURE_NAMES:
            feat_val = ml_features[feat_name]
            if type(feat_val) is bool or not isinstance(feat_val, (int, float)):
                raise ValueError(
                    f"Feature '{feat_name}' must be numeric (int or float), got {type(feat_val).__name__}: {feat_val!r}"
                )
            if not math.isfinite(float(feat_val)):
                raise ValueError(f"Feature '{feat_name}' has non-finite value: {feat_val!r}")

        if ml_skip_reason is not None:
            raise ValueError("ML-evaluated record must have ml_skip_reason=None.")
    else:
        ml_evaluated = False
        ml_prediction = None
        is_ml_anomaly = None
        anomaly_score = None
        raw_decision_score = None
        ml_features = None

        if not isinstance(ml_skip_reason, str) or not ml_skip_reason.strip():
            raise ValueError(
                "When ML evaluation is skipped (ml_evaluation is None), "
                "'ml_skip_reason' must be a non-empty string."
            )

    # Operational Diagnostics extraction
    diagnostics = analysis_result.get("diagnostics", {})
    diagnostic_severity = str(diagnostics.get("severity", "INFO"))
    diagnostic_root_cause = str(diagnostics.get("root_cause", ""))
    diagnostic_key_factors = list(diagnostics.get("key_factors", []))
    diagnostic_recommended_action = str(diagnostics.get("recommended_action", ""))

    summary = str(analysis_result.get("summary", ""))
    order_id = analysis_result.get("order_id")
    order_id_str = str(order_id) if order_id is not None else None

    utc_created_at = to_naive_utc(timestamp)

    return WorkflowAnalysis(
        order_id=order_id_str,
        input_format=input_format,
        raw_input_payload=raw_payload,
        verdict=verdict,
        is_anomaly=is_anomaly,
        anomaly_type=anomaly_type,
        summary=summary,
        is_rule_valid=is_rule_valid,
        is_complete=is_complete,
        final_state=final_state,
        total_steps=total_steps,
        successful_transitions=successful_transitions,
        fsm_error=fsm_error,
        ml_evaluated=ml_evaluated,
        ml_prediction=ml_prediction,
        is_ml_anomaly=is_ml_anomaly,
        anomaly_score=anomaly_score,
        raw_decision_score=raw_decision_score,
        ml_features=ml_features,
        ml_skip_reason=ml_skip_reason,
        diagnostic_severity=diagnostic_severity,
        diagnostic_root_cause=diagnostic_root_cause,
        diagnostic_key_factors=diagnostic_key_factors,
        diagnostic_recommended_action=diagnostic_recommended_action,
        created_at=utc_created_at,
    )


# ---------------------------------------------------------------------------
# Repository Write Operations
# ---------------------------------------------------------------------------

def create_analysis_record(
    db: Session,
    analysis_result: Dict[str, Any],
    raw_input: Union[str, List[str]],
    timestamp: Optional[datetime] = None,
) -> WorkflowAnalysis:
    """Validate, normalize, and persist a single workflow analysis record.

    Args:
        db: Active SQLAlchemy Session.
        analysis_result: Analysis result dictionary.
        raw_input: Original verbatim input trace (str or list[str]).
        timestamp: Optional explicit timestamp (normalized to UTC-naive).

    Returns:
        Refreshed WorkflowAnalysis instance.

    Raises:
        TypeError: If raw_input is of an unsupported type.
        ValueError: If analysis result violates consistency constraints.
        Exception: Re-raises any database-level exceptions after rollback.
    """
    record = _build_record(analysis_result, raw_input, timestamp)
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception:
        db.rollback()
        raise


def create_analysis_batch(
    db: Session,
    analysis_results: Sequence[Dict[str, Any]],
    raw_inputs: Sequence[Union[str, List[str]]],
    timestamp: Optional[datetime] = None,
) -> List[WorkflowAnalysis]:
    """Atomically persist a batch of analysis results.

    Guarantees:
      - Validates len(analysis_results) == len(raw_inputs) before database access.
      - Constructs all records using strict index alignment.
      - Uses exactly one database transaction.
      - Calls db.rollback() and re-raises on any insertion failure, ensuring
        zero partial writes.

    Args:
        db: Active SQLAlchemy Session.
        analysis_results: Sequence of analysis result dictionaries.
        raw_inputs: Sequence of verbatim input traces.
        timestamp: Optional timestamp for all records.

    Returns:
        List of refreshed WorkflowAnalysis instances.

    Raises:
        ValueError: If lengths of analysis_results and raw_inputs do not match.
        Exception: Any database or validation error, after rollback.
    """
    if len(analysis_results) != len(raw_inputs):
        raise ValueError(
            f"Batch count mismatch: received {len(analysis_results)} analysis results "
            f"but {len(raw_inputs)} raw inputs."
        )

    records: List[WorkflowAnalysis] = [
        _build_record(res, inp, timestamp)
        for res, inp in zip(analysis_results, raw_inputs, strict=True)
    ]

    try:
        db.add_all(records)
        db.commit()
        for rec in records:
            db.refresh(rec)
        return records
    except Exception:
        db.rollback()
        raise


# ---------------------------------------------------------------------------
# Repository Read Operations
# ---------------------------------------------------------------------------

def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[WorkflowAnalysis]:
    """Retrieve an analysis record by its primary key ID.

    Args:
        db: Active SQLAlchemy Session.
        analysis_id: Primary key integer.

    Returns:
        WorkflowAnalysis instance if found, or None.
    """
    return db.get(WorkflowAnalysis, analysis_id)


def get_analyses_by_order_id(
    db: Session,
    order_id: str,
    limit: int = 50,
) -> List[WorkflowAnalysis]:
    """Retrieve historical trace analyses for an order in deterministic order.

    Sorted by created_at DESC, id DESC.

    Args:
        db: Active SQLAlchemy Session.
        order_id: Order identifier.
        limit: Maximum number of records to return (must be > 0).

    Returns:
        List of WorkflowAnalysis instances.

    Raises:
        ValueError: If limit <= 0.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    stmt = (
        select(WorkflowAnalysis)
        .where(WorkflowAnalysis.order_id == str(order_id))
        .order_by(WorkflowAnalysis.created_at.desc(), WorkflowAnalysis.id.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def get_latest_analysis_by_order_id(
    db: Session,
    order_id: str,
) -> Optional[WorkflowAnalysis]:
    """Retrieve the most recent trace analysis for an order.

    Args:
        db: Active SQLAlchemy Session.
        order_id: Order identifier.

    Returns:
        Most recent WorkflowAnalysis instance, or None.
    """
    stmt = (
        select(WorkflowAnalysis)
        .where(WorkflowAnalysis.order_id == str(order_id))
        .order_by(WorkflowAnalysis.created_at.desc(), WorkflowAnalysis.id.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def list_analyses(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    verdict: Optional[str] = None,
    is_anomaly: Optional[bool] = None,
) -> Tuple[List[WorkflowAnalysis], int]:
    """Retrieve paginated analyses with optional filtering.

    Args:
        db: Active SQLAlchemy Session.
        skip: Non-negative offset (default 0).
        limit: Positive page size (default 50).
        verdict: Optional verdict filter.
        is_anomaly: Optional anomaly boolean filter.

    Returns:
        Tuple of (records_list, total_matching_count).

    Raises:
        ValueError: If skip < 0, limit <= 0, or verdict is invalid.
    """
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if verdict is not None and verdict not in SUPPORTED_VERDICTS:
        raise ValueError(
            f"Invalid verdict filter '{verdict}'. Supported: {SUPPORTED_VERDICTS}"
        )

    # Base query filters
    conditions = []
    if verdict is not None:
        conditions.append(WorkflowAnalysis.verdict == verdict)
    if is_anomaly is not None:
        conditions.append(WorkflowAnalysis.is_anomaly == is_anomaly)

    # Count query
    count_stmt = select(func.count(WorkflowAnalysis.id))
    if conditions:
        count_stmt = count_stmt.where(*conditions)
    total_count = int(db.scalar(count_stmt) or 0)

    # Data query
    data_stmt = (
        select(WorkflowAnalysis)
        .order_by(WorkflowAnalysis.created_at.desc(), WorkflowAnalysis.id.desc())
        .offset(skip)
        .limit(limit)
    )
    if conditions:
        data_stmt = data_stmt.where(*conditions)

    records = list(db.scalars(data_stmt).all())
    return records, total_count


def get_summary_metrics(db: Session) -> Dict[str, Any]:
    """Compute database-wide aggregated metrics across all stored analyses.

    Uses 'total_analyses' for execution counts, with 'unique_orders_count'
    calculated separately excluding NULL order IDs.

    Args:
        db: Active SQLAlchemy Session.

    Returns:
        Dictionary containing execution metrics and anomaly rates.
    """
    total_analyses = int(db.scalar(select(func.count(WorkflowAnalysis.id))) or 0)

    if total_analyses == 0:
        return {
            "total_analyses": 0,
            "normal_count": 0,
            "rule_violation_count": 0,
            "statistical_anomaly_count": 0,
            "valid_incomplete_count": 0,
            "total_anomalies": 0,
            "anomaly_rate": 0.0,
            "unique_orders_count": 0,
        }

    normal_count = int(
        db.scalar(select(func.count(WorkflowAnalysis.id)).where(WorkflowAnalysis.verdict == "NORMAL")) or 0
    )
    rule_violation_count = int(
        db.scalar(select(func.count(WorkflowAnalysis.id)).where(WorkflowAnalysis.verdict == "RULE_VIOLATION")) or 0
    )
    statistical_anomaly_count = int(
        db.scalar(select(func.count(WorkflowAnalysis.id)).where(WorkflowAnalysis.verdict == "STATISTICAL_ANOMALY")) or 0
    )
    valid_incomplete_count = int(
        db.scalar(select(func.count(WorkflowAnalysis.id)).where(WorkflowAnalysis.verdict == "VALID_INCOMPLETE")) or 0
    )

    total_anomalies = rule_violation_count + statistical_anomaly_count
    anomaly_rate = float(total_anomalies / total_analyses) if total_analyses > 0 else 0.0

    unique_orders_count = int(
        db.scalar(
            select(func.count(func.distinct(WorkflowAnalysis.order_id)))
            .where(WorkflowAnalysis.order_id.is_not(None))
        ) or 0
    )

    return {
        "total_analyses": total_analyses,
        "normal_count": normal_count,
        "rule_violation_count": rule_violation_count,
        "statistical_anomaly_count": statistical_anomaly_count,
        "valid_incomplete_count": valid_incomplete_count,
        "total_anomalies": total_anomalies,
        "anomaly_rate": anomaly_rate,
        "unique_orders_count": unique_orders_count,
    }

