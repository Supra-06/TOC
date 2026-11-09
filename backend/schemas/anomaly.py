"""Pydantic v2 schemas for FSM, ML, and operational diagnostics results.

Defines schemas for deterministic automata validation, unsupervised Isolation
Forest evaluation, and combined workflow analysis audit records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.ml.feature_extraction.extractor import FEATURE_NAMES

# Supported domain taxonomies
VerdictType = Literal["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"]
AnomalyCategoryType = Literal["RULE_VIOLATION", "STATISTICAL_ANOMALY"]
DiagnosticSeverityType = Literal["CRITICAL", "WARNING", "INFO"]
InputFormatType = Literal["DELIMITED_STRING", "EVENT_LIST"]


class FsmErrorDetail(BaseModel):
    """Structured details of an FSM transition rule violation."""

    error_type: str = Field(..., description="Category of FSM error, e.g. INVALID_TRANSITION")
    failed_step: int = Field(..., ge=1, description="1-based step index where failure occurred")
    failed_event: str = Field(..., description="Event symbol that violated transition rules")
    state_at_failure: str = Field(..., description="Automaton state when failure occurred")
    allowed_events: List[str] = Field(default_factory=list, description="Permitted event symbols from failure state")


class FsmEvaluationSchema(BaseModel):
    """Deterministic FSM validation results."""

    is_rule_valid: bool = Field(..., description="True if all transitions conform to FSM rules")
    is_complete: bool = Field(..., description="True if trace terminated in an approved terminal state")
    final_state: str = Field(..., description="Automaton state upon trace termination")
    total_steps: int = Field(..., ge=0, description="Total events evaluated")
    successful_transitions: int = Field(..., ge=0, description="Number of successful state transitions")
    error: Optional[FsmErrorDetail] = Field(default=None, description="FSM violation details if invalid")


class MlEvaluationSchema(BaseModel):
    """Statistical Isolation Forest evaluation results."""

    prediction: Literal["NORMAL", "ANOMALY"] = Field(..., description="ML classification verdict")
    is_ml_anomaly: bool = Field(..., description="True if statistical anomaly, False if normal")
    anomaly_score: float = Field(..., description="Continuous numerical anomaly score (positive indicates outlier)")
    raw_decision_score: float = Field(..., description="Raw scikit-learn decision function score")
    features: Dict[str, float] = Field(..., description="Dictionary of all 14 engineered numerical features")

    @field_validator("anomaly_score", "raw_decision_score", mode="before")
    @classmethod
    def validate_finite_score(cls, v: Any, info) -> float:
        """Validate that ML scores are finite numeric values and not booleans or strings."""
        if type(v) is bool or not isinstance(v, (int, float)):
            raise ValueError(f"Field '{info.field_name}' must be numeric (int or float), got {type(v).__name__}")
        float_val = float(v)
        if not math.isfinite(float_val):
            raise ValueError(f"Field '{info.field_name}' must be finite, got {v!r}")
        return float_val

    @field_validator("features", mode="before")
    @classmethod
    def validate_features_dict(cls, v: Any) -> Dict[str, float]:
        """Validate that features contain exactly the 14 approved features with finite numeric values."""
        if not isinstance(v, dict):
            raise ValueError(f"'features' must be a dictionary, got {type(v).__name__}")

        expected_keys = set(FEATURE_NAMES)
        actual_keys = set(v.keys())

        missing = expected_keys - actual_keys
        if missing:
            raise ValueError(f"Features dictionary is missing approved features: {sorted(missing)}")

        unexpected = actual_keys - expected_keys
        if unexpected:
            raise ValueError(f"Features dictionary contains unexpected features: {sorted(unexpected)}")

        clean_features: Dict[str, float] = {}
        for feat_name in FEATURE_NAMES:
            feat_val = v[feat_name]
            if type(feat_val) is bool or not isinstance(feat_val, (int, float)):
                raise ValueError(
                    f"Feature '{feat_name}' must be numeric (int or float), got {type(feat_val).__name__}"
                )
            float_feat = float(feat_val)
            if not math.isfinite(float_feat):
                raise ValueError(f"Feature '{feat_name}' must be finite, got {feat_val!r}")
            clean_features[feat_name] = float_feat

        return clean_features

    @model_validator(mode="after")
    def validate_prediction_consistency(self) -> MlEvaluationSchema:
        """Validate consistency between prediction string and is_ml_anomaly flag."""
        if self.prediction == "NORMAL" and self.is_ml_anomaly is not False:
            raise ValueError("Inconsistent ML evaluation: prediction 'NORMAL' requires is_ml_anomaly=False.")
        if self.prediction == "ANOMALY" and self.is_ml_anomaly is not True:
            raise ValueError("Inconsistent ML evaluation: prediction 'ANOMALY' requires is_ml_anomaly=True.")
        return self


class DiagnosticsSchema(BaseModel):
    """Operational diagnostics and actionable remediation advice."""

    severity: DiagnosticSeverityType = Field(..., description="Operational severity: CRITICAL, WARNING, or INFO")
    root_cause: str = Field(..., description="Concise explanation of the underlying cause")
    key_factors: List[str] = Field(default_factory=list, description="Key factors contributing to diagnosis")
    recommended_action: str = Field(..., description="Remediation steps for operational staff")


class AnalysisRecordResponse(BaseModel):
    """Complete analysis and audit record response."""

    id: int = Field(..., ge=1, description="Database record primary key")
    order_id: Optional[str] = Field(default=None, description="Associated order identifier")
    input_format: InputFormatType = Field(..., description="Format classification: DELIMITED_STRING or EVENT_LIST")
    raw_input_payload: Any = Field(..., description="Verbatim original input trace")
    verdict: VerdictType = Field(..., description="Decision verdict in 4-state taxonomy")
    is_anomaly: bool = Field(..., description="True if trace is an FSM violation or ML anomaly")
    anomaly_type: Optional[AnomalyCategoryType] = Field(default=None, description="Category of anomaly if flagged")
    summary: str = Field(..., description="High-level human-readable verdict summary")
    fsm_evaluation: FsmEvaluationSchema
    ml_evaluation: Optional[MlEvaluationSchema] = Field(default=None, description="Null when ML evaluation is skipped")
    ml_skip_reason: Optional[str] = Field(default=None, description="Reason why ML evaluation was skipped")
    diagnostics: DiagnosticsSchema
    created_at: datetime = Field(..., description="Timezone-aware UTC timestamp of analysis record creation")

    @field_validator("created_at")
    @classmethod
    def validate_utc_datetime(cls, v: Any) -> datetime:
        """Validate that created_at is timezone-aware and has zero UTC offset."""
        if not isinstance(v, datetime):
            raise ValueError(f"'created_at' must be a datetime object, got {type(v).__name__}")
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("'created_at' must be a timezone-aware UTC datetime, not naive.")
        if v.utcoffset() != timedelta(0):
            raise ValueError(
                f"'created_at' must be in UTC with zero offset, got offset {v.utcoffset()}."
            )
        return v.astimezone(timezone.utc)
