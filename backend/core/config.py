"""Application configuration settings (paths, database URI, ML parameters).

Provides centralized, environment-aware configuration with deterministic path
resolution rooted at the backend filesystem directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base directory: .../backend
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Project root directory: .../TOC_PROJECT
PROJECT_ROOT = BACKEND_DIR.parent

# Database configuration
DEFAULT_DB_FILE = BACKEND_DIR / "toc_workflow.db"

# Ensure parent directory exists
DEFAULT_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# Canonical SQLite URL ensuring forward slashes for Windows compatibility
_DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_DB_FILE.as_posix()}"
DATABASE_URL = os.getenv("TOC_DATABASE_URL", _DEFAULT_SQLITE_URL)

# Database engine echo flag
DATABASE_ECHO = os.getenv("TOC_DATABASE_ECHO", "False").lower() in ("true", "1", "yes")

# ML Paths
ML_DIR = BACKEND_DIR / "ml"
ML_MODELS_DIR = ML_DIR / "models"
DEFAULT_MODEL_PATH = ML_MODELS_DIR / "isolation_forest.joblib"
DEFAULT_METADATA_PATH = ML_MODELS_DIR / "isolation_forest_metadata.json"

# API & Server Configuration
INIT_DB_ON_STARTUP = os.getenv("INIT_DB_ON_STARTUP", "False").lower() in ("true", "1", "yes")

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]
_ENV_CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()]
CORS_ORIGINS = _ENV_CORS_ORIGINS if _ENV_CORS_ORIGINS else _DEFAULT_CORS_ORIGINS
