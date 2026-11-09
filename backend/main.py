"""FastAPI application entry point for Process Workflow Anomaly Detection.

Instantiates the FastAPI application, registers route handlers, configures CORS
middleware, provides lifespan database initialization, and enforces standardized
error envelopes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.mapping import RecordMappingError
from backend.api.routes import root_router, router
from backend.core.config import CORS_ORIGINS, INIT_DB_ON_STARTUP
from backend.database.connection import init_db
from backend.schemas.common import ErrorDetail, ErrorResponse


def setup_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers enforcing standardized ErrorResponse schemas."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Request validation failed.",
                    details=exc.errors(),
                )
            ).model_dump(mode="json"),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            422: "UNPROCESSABLE_ENTITY",
            503: "SERVICE_UNAVAILABLE",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=code,
                    message=str(exc.detail),
                    details=None,
                )
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RecordMappingError)
    async def mapping_exception_handler(request: Request, exc: RecordMappingError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="MAPPING_ERROR",
                    message="Failed to format record response.",
                    details=None,
                )
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message="An internal server error occurred.",
                    details=None,
                )
            ).model_dump(mode="json"),
        )


def create_app(init_db_on_startup: Optional[bool] = None) -> FastAPI:
    """FastAPI application factory with configurable lifespan database initialization.

    Args:
        init_db_on_startup: When True, creates database tables on application lifespan
            startup. Defaults to INIT_DB_ON_STARTUP configuration value.

    Returns:
        Configured FastAPI application instance.
    """
    should_init_db = init_db_on_startup if init_db_on_startup is not None else INIT_DB_ON_STARTUP

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if should_init_db:
            init_db()
        yield

    app = FastAPI(
        title="Process Workflow Anomaly Detection API",
        version="1.0.0",
        description="REST API for Anomaly Detection in Process Workflows Using Automata and ML",
        lifespan=lifespan,
    )

    # 1. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # 2. Exception Handlers
    setup_exception_handlers(app)

    # 3. Mount Routers
    app.include_router(root_router)
    app.include_router(router)

    return app


# Default application instance (does not trigger database initialization on import)
app = create_app()
