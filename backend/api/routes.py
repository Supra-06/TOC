"""FastAPI route declarations for Process Workflow Anomaly Detection.

Exposes REST endpoints for liveness, readiness, workflow analysis ingestion,
batch analysis, historical trace inspection, and aggregated summary metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db_session, get_service
from backend.api.mapping import record_to_response
from backend.database.repository import (
    SUPPORTED_VERDICTS,
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_summary_metrics,
    list_analyses,
)
from backend.schemas.anomaly import AnalysisRecordResponse
from backend.schemas.common import (
    BatchAnalysisResponse,
    BatchSummaryMetrics,
    DatabaseSummaryMetricsResponse,
    LivenessResponse,
    OrderHistoryResponse,
    PaginatedAnalysesResponse,
    ReadinessResponse,
)
from backend.schemas.order import (
    BatchAnalysisRequest,
    TraceItemRequest,
    WorkflowAnalysisRequest,
)
from backend.services.anomaly_service import AnomalyService

# Routers
router = APIRouter(prefix="/api/v1", tags=["Workflow Analyses"])
root_router = APIRouter(tags=["Root & Operational"])


# ---------------------------------------------------------------------------
# Liveness Endpoints
# ---------------------------------------------------------------------------

@root_router.get(
    "/health",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Root process liveness probe",
)
@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="API v1 process liveness probe",
)
def get_liveness() -> LivenessResponse:
    """Return process liveness without checking database or ML model."""
    return LivenessResponse(status="alive")


# ---------------------------------------------------------------------------
# Readiness Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="API v1 readiness probe",
)
@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="API v1 readiness probe (alias)",
)
def get_readiness(
    service: AnomalyService = Depends(get_service),
    db: Session = Depends(get_db_session),
) -> ReadinessResponse:
    """Verify backend operational readiness without forced ML model loading.

    Checks:
      1. Model artifact exists on filesystem.
      2. Database connection can execute a lightweight query.

    Returns HTTP 200 when ready; raises HTTP 503 if dependencies are unavailable.
    """
    # 1. Check database connectivity
    db_healthy = False
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception:
        db_healthy = False

    # 2. Check service status
    svc_status = service.get_service_status()
    is_operational = svc_status.get("service_status") == "OPERATIONAL"
    artifact_exists = svc_status.get("model_artifact_exists", False)

    if not db_healthy or not is_operational or not artifact_exists:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend service dependencies unavailable (database unreachable or model artifact missing).",
        )

    return ReadinessResponse(
        service_status=svc_status["service_status"],
        dataset_spec_version=svc_status["dataset_spec_version"],
        model_artifact_path=svc_status["model_artifact_path"],
        model_artifact_exists=svc_status["model_artifact_exists"],
        supported_verdicts=svc_status["supported_verdicts"],
    )


# ---------------------------------------------------------------------------
# Single Analysis Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/analyses",
    response_model=AnalysisRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze and persist a single workflow trace",
)
def create_analysis(
    request: WorkflowAnalysisRequest,
    service: AnomalyService = Depends(get_service),
    db: Session = Depends(get_db_session),
) -> AnalysisRecordResponse:
    """Analyze a single workflow trace and persist the audit record."""
    analysis_result = service.analyze_order(
        event_sequence=request.event_sequence,
        order_id=request.order_id,
    )

    record = create_analysis_record(
        db=db,
        analysis_result=analysis_result,
        raw_input=request.event_sequence,
    )

    return record_to_response(record)


# ---------------------------------------------------------------------------
# Batch Analysis Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/analyses/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Atomically analyze and persist a batch of workflow traces",
)
def create_batch_analysis(
    request: BatchAnalysisRequest,
    service: AnomalyService = Depends(get_service),
    db: Session = Depends(get_db_session),
) -> BatchAnalysisResponse:
    """Analyze a batch of 1 to 500 workflow traces with atomic persistence."""
    service_items: List[Union[str, Sequence[str], Dict[str, Any]]] = []
    raw_inputs: List[Union[str, List[str]]] = []

    for item in request.traces:
        if isinstance(item, TraceItemRequest):
            service_items.append({
                "order_id": item.order_id,
                "event_sequence": item.event_sequence,
            })
            raw_inputs.append(item.event_sequence)
        else:
            service_items.append(item)
            raw_inputs.append(item)

    batch_res = service.analyze_order_batch(service_items)

    persisted_records = create_analysis_batch(
        db=db,
        analysis_results=batch_res["results"],
        raw_inputs=raw_inputs,
    )

    results = [record_to_response(rec) for rec in persisted_records]
    summary = BatchSummaryMetrics(**batch_res["summary"])

    return BatchAnalysisResponse(results=results, summary=summary)


# ---------------------------------------------------------------------------
# Query & Retrieval Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/analyses",
    response_model=PaginatedAnalysesResponse,
    status_code=status.HTTP_200_OK,
    summary="List paginated analyses with optional filtering",
)
def get_analyses_page(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, gt=0, le=1000, description="Page size limit"),
    verdict: Optional[str] = Query(None, description="Filter by verdict"),
    is_anomaly: Optional[bool] = Query(None, description="Filter by anomaly status"),
    db: Session = Depends(get_db_session),
) -> PaginatedAnalysesResponse:
    """Retrieve paginated analyses with optional verdict and anomaly filters."""
    if verdict is not None and verdict not in SUPPORTED_VERDICTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid verdict filter '{verdict}'. Supported: {SUPPORTED_VERDICTS}",
        )

    records, total = list_analyses(
        db=db,
        skip=skip,
        limit=limit,
        verdict=verdict,
        is_anomaly=is_anomaly,
    )

    items = [record_to_response(rec) for rec in records]
    return PaginatedAnalysesResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisRecordResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve an analysis record by ID",
)
def get_single_analysis(
    analysis_id: int = Path(..., gt=0, description="Positive integer analysis record ID"),
    db: Session = Depends(get_db_session),
) -> AnalysisRecordResponse:
    """Retrieve an existing analysis record by primary key."""
    record = get_analysis_by_id(db, analysis_id=analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record with ID {analysis_id} not found.",
        )
    return record_to_response(record)


@router.get(
    "/orders/{order_id}/analyses",
    response_model=OrderHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve chronological trace analyses for an order",
)
def get_order_history(
    order_id: str = Path(..., min_length=1, max_length=64, description="Target order identifier"),
    limit: int = Query(50, gt=0, le=1000, description="Maximum historical records"),
    db: Session = Depends(get_db_session),
) -> OrderHistoryResponse:
    """Retrieve historical trace analyses for an order in deterministic chronological order."""
    records = get_analyses_by_order_id(db, order_id=order_id, limit=limit)
    analyses = [record_to_response(r) for r in records]
    return OrderHistoryResponse(
        order_id=order_id,
        returned_count=len(analyses),
        analyses=analyses,
    )


@router.get(
    "/metrics",
    response_model=DatabaseSummaryMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve database-wide summary metrics",
)
def get_metrics(
    db: Session = Depends(get_db_session),
) -> DatabaseSummaryMetricsResponse:
    """Retrieve database-wide aggregated workflow execution metrics."""
    metrics_dict = get_summary_metrics(db)
    return DatabaseSummaryMetricsResponse(**metrics_dict)
