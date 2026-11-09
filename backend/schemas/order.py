"""Pydantic v2 schemas for order workflow trace requests.

Validates incoming single-order and batch trace requests while preserving
raw event representations (delimited strings or event token lists) exactly.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class WorkflowAnalysisRequest(BaseModel):
    """Request payload for analyzing and persisting a single workflow trace."""

    model_config = ConfigDict(strict=True)

    order_id: Optional[StrictStr] = Field(
        default=None,
        max_length=64,
        description="Optional unique order identifier (max 64 characters)",
    )
    event_sequence: Union[StrictStr, List[StrictStr]] = Field(
        ...,
        description="Verbatim workflow trace as semicolon-delimited string or list of event tokens",
    )


class TraceItemRequest(BaseModel):
    """Single workflow trace item within a batch request."""

    model_config = ConfigDict(strict=True)

    order_id: Optional[StrictStr] = Field(
        default=None,
        max_length=64,
        description="Optional unique order identifier (max 64 characters)",
    )
    event_sequence: Union[StrictStr, List[StrictStr]] = Field(
        ...,
        description="Verbatim workflow trace as semicolon-delimited string or list of event tokens",
    )


class BatchAnalysisRequest(BaseModel):
    """Request payload for analyzing a batch of workflow traces."""

    model_config = ConfigDict(strict=True)

    traces: List[Union[TraceItemRequest, StrictStr, List[StrictStr]]] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Batch collection of order trace items (1 to 500 items)",
    )
