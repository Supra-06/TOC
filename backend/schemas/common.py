"""Pydantic v2 schemas for common, pagination, metrics, health, and error responses."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from backend.schemas.anomaly import AnalysisRecordResponse


class LivenessResponse(BaseModel):
    """Response payload for container/application liveness probe."""

    status: str = Field(default="alive", description="Process liveness indicator")


class ReadinessResponse(BaseModel):
    """Response payload for backend readiness and service status probe."""

    service_status: str = Field(..., description="Operational status: 'OPERATIONAL' or 'DEGRADED'")
    dataset_spec_version: str = Field(..., description="Supported dataset specification version")
    model_artifact_path: str = Field(..., description="Filesystem path to trained model artifact")
    model_artifact_exists: bool = Field(..., description="Whether model artifact exists on disk")
    supported_verdicts: List[str] = Field(..., description="Approved four-verdict taxonomy")


class BatchSummaryMetrics(BaseModel):
    """Summary execution metrics across a batch submission."""

    total_orders: int = Field(..., ge=0, description="Total order traces evaluated in batch")
    normal_count: int = Field(..., ge=0, description="Count of normal executions")
    rule_violation_count: int = Field(..., ge=0, description="Count of FSM rule violations")
    statistical_anomaly_count: int = Field(..., ge=0, description="Count of ML statistical anomalies")
    valid_incomplete_count: int = Field(..., ge=0, description="Count of legitimate in-progress workflows")
    total_anomalies: int = Field(..., ge=0, description="Total flagged anomalies (rule violations + statistical)")
    anomaly_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of anomalous orders in batch")


class BatchAnalysisResponse(BaseModel):
    """Response payload for atomic batch trace analysis."""

    results: List[AnalysisRecordResponse] = Field(..., description="List of persisted individual analysis records")
    summary: BatchSummaryMetrics = Field(..., description="Aggregated summary metrics across the batch")


class OrderHistoryResponse(BaseModel):
    """Historical trace analysis timeline for a specific order."""

    order_id: str = Field(..., description="Queried order identifier")
    returned_count: int = Field(..., ge=0, description="Number of historical records returned in this response")
    analyses: List[AnalysisRecordResponse] = Field(..., description="Historical trace analyses ordered chronologically")


class PaginatedAnalysesResponse(BaseModel):
    """Paginated collection of analysis records."""

    items: List[AnalysisRecordResponse] = Field(..., description="Page of analysis records")
    total: int = Field(..., ge=0, description="Total matching records in database")
    skip: int = Field(..., ge=0, description="Pagination offset")
    limit: int = Field(..., gt=0, description="Page size limit")


class DatabaseSummaryMetricsResponse(BaseModel):
    """Database-wide aggregated execution metrics."""

    total_analyses: int = Field(..., ge=0, description="Total analysis records stored in database")
    normal_count: int = Field(..., ge=0, description="Total stored normal analyses")
    rule_violation_count: int = Field(..., ge=0, description="Total stored FSM rule violations")
    statistical_anomaly_count: int = Field(..., ge=0, description="Total stored ML statistical anomalies")
    valid_incomplete_count: int = Field(..., ge=0, description="Total stored valid incomplete analyses")
    total_anomalies: int = Field(..., ge=0, description="Total anomalies stored across database")
    anomaly_rate: float = Field(..., ge=0.0, le=1.0, description="Overall anomaly proportion")
    unique_orders_count: int = Field(..., ge=0, description="Count of distinct non-null order IDs")


class ErrorDetail(BaseModel):
    """Structured error detail representation."""

    code: str = Field(..., description="Machine-readable error classification code")
    message: str = Field(..., description="Client-safe error description")
    details: Optional[Any] = Field(default=None, description="Optional structured validation error metadata")


class ErrorResponse(BaseModel):
    """Standardized error envelope."""

    error: ErrorDetail = Field(..., description="Error details payload")

