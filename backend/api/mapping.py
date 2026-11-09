"""ORM to Pydantic response mapper for workflow analysis audit records.

Converts SQLAlchemy WorkflowAnalysis ORM instances into fully validated,
nested AnalysisRecordResponse Pydantic v2 schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from backend.database.models import WorkflowAnalysis
from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.schemas.anomaly import (
    AnalysisRecordResponse,
    DiagnosticsSchema,
    FsmErrorDetail,
    FsmEvaluationSchema,
    MlEvaluationSchema,
)


class RecordMappingError(ValueError):
    """Raised when a database record cannot be mapped to an API response."""
    pass


def record_to_response(record: WorkflowAnalysis) -> AnalysisRecordResponse:
    """Explicitly map a WorkflowAnalysis ORM record to an AnalysisRecordResponse.

    Reconstructs nested FSM, ML, and diagnostic structures from flat database columns,
    normalizes UTC-naive timestamps to UTC-aware datetimes, and strictly validates all
    sub-schemas.

    Args:
        record: SQLAlchemy WorkflowAnalysis instance retrieved from database.

    Returns:
        Fully populated, validated AnalysisRecordResponse schema.

    Raises:
        RecordMappingError: If the record is None, has an invalid type, is missing required
            fields, or contains inconsistent or unmappable data.
    """
    if record is None:
        raise RecordMappingError("Cannot map None record to AnalysisRecordResponse.")

    if not isinstance(record, WorkflowAnalysis):
        raise RecordMappingError(
            f"Expected WorkflowAnalysis ORM instance, got {type(record).__name__}."
        )

    # -----------------------------------------------------------------------
    # 1. Reconstruct FSM Evaluation & Error Detail
    # -----------------------------------------------------------------------
    fsm_error_obj: Optional[FsmErrorDetail] = None
    if record.fsm_error is not None:
        if not isinstance(record.fsm_error, dict):
            raise RecordMappingError(
                f"fsm_error must be a dictionary when present, got {type(record.fsm_error).__name__}."
            )
        required_err_keys = {
            "error_type",
            "failed_step",
            "failed_event",
            "state_at_failure",
            "allowed_events",
        }
        missing_err_keys = required_err_keys - set(record.fsm_error.keys())
        if missing_err_keys:
            raise RecordMappingError(
                f"fsm_error dictionary is missing required fields: {sorted(missing_err_keys)}."
            )
        if not isinstance(record.fsm_error["allowed_events"], list):
            raise RecordMappingError(
                f"fsm_error['allowed_events'] must be a list, got {type(record.fsm_error['allowed_events']).__name__}."
            )
        try:
            fsm_error_obj = FsmErrorDetail(
                error_type=record.fsm_error["error_type"],
                failed_step=record.fsm_error["failed_step"],
                failed_event=record.fsm_error["failed_event"],
                state_at_failure=record.fsm_error["state_at_failure"],
                allowed_events=record.fsm_error["allowed_events"],
            )
        except (ValidationError, TypeError, ValueError) as err:
            raise RecordMappingError(f"Failed to map FsmErrorDetail: {err}") from err

    try:
        fsm_eval_obj = FsmEvaluationSchema(
            is_rule_valid=record.is_rule_valid,
            is_complete=record.is_complete,
            final_state=record.final_state,
            total_steps=record.total_steps,
            successful_transitions=record.successful_transitions,
            error=fsm_error_obj,
        )
    except (ValidationError, TypeError, ValueError) as err:
        raise RecordMappingError(f"Failed to map FsmEvaluationSchema: {err}") from err

    # -----------------------------------------------------------------------
    # 2. Reconstruct ML Evaluation (Strictly conditional on ml_evaluated)
    # -----------------------------------------------------------------------
    ml_eval_obj: Optional[MlEvaluationSchema] = None
    ml_skip_reason: Optional[str] = record.ml_skip_reason

    if record.ml_evaluated:
        if record.ml_skip_reason is not None:
            raise RecordMappingError(
                f"ML was evaluated (ml_evaluated=True) but ml_skip_reason is not None: {record.ml_skip_reason!r}."
            )
        if record.ml_prediction is None:
            raise RecordMappingError("ML was evaluated but ml_prediction is missing (None).")
        if record.is_ml_anomaly is None:
            raise RecordMappingError("ML was evaluated but is_ml_anomaly is missing (None).")
        if record.anomaly_score is None:
            raise RecordMappingError("ML was evaluated but anomaly_score is missing (None).")
        if record.raw_decision_score is None:
            raise RecordMappingError("ML was evaluated but raw_decision_score is missing (None).")
        if record.ml_features is None:
            raise RecordMappingError("ML was evaluated but ml_features is missing (None).")
        if not isinstance(record.ml_features, dict):
            raise RecordMappingError(
                f"ML was evaluated but ml_features is not a dict: {type(record.ml_features).__name__}."
            )

        actual_features = set(record.ml_features.keys())
        expected_features = set(FEATURE_NAMES)
        missing_feats = expected_features - actual_features
        if missing_feats:
            raise RecordMappingError(
                f"ml_features dictionary is missing required features: {sorted(missing_feats)}."
            )
        extra_feats = actual_features - expected_features
        if extra_feats:
            raise RecordMappingError(
                f"ml_features dictionary contains unexpected features: {sorted(extra_feats)}."
            )

        # Build feature dictionary in canonical FEATURE_NAMES order
        ordered_features: Dict[str, float] = {}
        for feat_name in FEATURE_NAMES:
            feat_val = record.ml_features[feat_name]
            if type(feat_val) is bool or not isinstance(feat_val, (int, float)):
                raise RecordMappingError(
                    f"Feature '{feat_name}' must be numeric (int or float), got {type(feat_val).__name__}."
                )
            if not math.isfinite(float(feat_val)):
                raise RecordMappingError(
                    f"Feature '{feat_name}' has non-finite value: {feat_val!r}."
                )
            ordered_features[feat_name] = float(feat_val)

        try:
            ml_eval_obj = MlEvaluationSchema(
                prediction=record.ml_prediction,
                is_ml_anomaly=record.is_ml_anomaly,
                anomaly_score=record.anomaly_score,
                raw_decision_score=record.raw_decision_score,
                features=ordered_features,
            )
        except (ValidationError, TypeError, ValueError) as err:
            raise RecordMappingError(f"Failed to map MlEvaluationSchema: {err}") from err
        ml_skip_reason = None
    else:
        if not isinstance(record.ml_skip_reason, str) or not record.ml_skip_reason.strip():
            raise RecordMappingError(
                "When ML evaluation is skipped (ml_evaluated=False), "
                "'ml_skip_reason' must be a non-empty string."
            )
        ml_eval_obj = None
        ml_skip_reason = record.ml_skip_reason

    # -----------------------------------------------------------------------
    # 3. Reconstruct Operational Diagnostics
    # -----------------------------------------------------------------------
    try:
        key_factors = record.diagnostic_key_factors
        if key_factors is None:
            key_factors = []
        elif not isinstance(key_factors, list):
            raise RecordMappingError(
                f"diagnostic_key_factors must be a list, got {type(key_factors).__name__}."
            )

        diagnostics_obj = DiagnosticsSchema(
            severity=record.diagnostic_severity,
            root_cause=record.diagnostic_root_cause,
            key_factors=key_factors,
            recommended_action=record.diagnostic_recommended_action,
        )
    except (ValidationError, TypeError, ValueError) as err:
        raise RecordMappingError(f"Failed to map DiagnosticsSchema: {err}") from err

    # -----------------------------------------------------------------------
    # 4. Normalize Timestamp to UTC-Aware Datetime
    # -----------------------------------------------------------------------
    if record.created_at is None:
        raise RecordMappingError("Record 'created_at' timestamp cannot be None.")
    if not isinstance(record.created_at, datetime):
        raise RecordMappingError(
            f"Record 'created_at' must be a datetime, got {type(record.created_at).__name__}."
        )

    if record.created_at.tzinfo is None or record.created_at.utcoffset() is None:
        utc_created_at = record.created_at.replace(tzinfo=timezone.utc)
    else:
        utc_created_at = record.created_at.astimezone(timezone.utc)

    # -----------------------------------------------------------------------
    # 5. Top-Level Assembly & Validation
    # -----------------------------------------------------------------------
    try:
        return AnalysisRecordResponse(
            id=record.id,
            order_id=record.order_id,
            input_format=record.input_format,
            raw_input_payload=record.raw_input_payload,
            verdict=record.verdict,
            is_anomaly=record.is_anomaly,
            anomaly_type=record.anomaly_type,
            summary=record.summary,
            fsm_evaluation=fsm_eval_obj,
            ml_evaluation=ml_eval_obj,
            ml_skip_reason=ml_skip_reason,
            diagnostics=diagnostics_obj,
            created_at=utc_created_at,
        )
    except (ValidationError, TypeError, ValueError) as err:
        raise RecordMappingError(f"Failed to construct AnalysisRecordResponse: {err}") from err

