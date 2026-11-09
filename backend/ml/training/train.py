"""Isolation Forest model training pipeline.

Trains an unsupervised Isolation Forest model on normal workflow profiles
to detect statistically anomalous event sequences.

Specification: backend/docs/DATASET_SPEC.md (v1.4, Section 5)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
from sklearn.ensemble import IsolationForest

from backend.ml.feature_extraction.extractor import FEATURE_NAMES, extract_feature_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset" / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "orders_workflow_dataset.csv"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"

EXPECTED_COLUMNS = [
    "trace_id",
    "event_sequence",
    "sequence_length",
    "is_rule_valid",
    "is_complete",
    "anomaly_category",
    "is_ml_anomaly",
]

TOTAL_EXPECTED_ROWS = 5000
N_TRAINING_SAMPLES = 2800
N_NORMAL_TEST = 700
N_ANOMALY_TEST = 500
N_RULE_VIOLATION = 1000

# Model Hyperparameters (DATASET_SPEC.md Section 5.2)
MODEL_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "contamination": "auto",
    "random_state": 42,
}

SPEC_VERSION = "1.4"


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_dataset_rows(rows: List[Dict[str, str]]) -> None:
    """Validate dataset row count, partition structure, and training cleanliness.

    Raises:
        ValueError: If row count, partition structure, or categories are invalid.
    """
    total = len(rows)
    if total != TOTAL_EXPECTED_ROWS:
        raise ValueError(
            f"Dataset row count mismatch: expected {TOTAL_EXPECTED_ROWS} rows, got {total}."
        )

    # 1. Training rows: [0 : 2800] must all be NORMAL
    for idx in range(N_TRAINING_SAMPLES):
        cat = rows[idx].get("anomaly_category")
        if cat != "NORMAL":
            raise ValueError(
                f"Training partition violation at row {idx + 1} ({rows[idx].get('trace_id')}): "
                f"expected category 'NORMAL', got '{cat}'."
            )

    # 2. Held-out normal test rows: [2800 : 3500] must all be NORMAL
    normal_test_end = N_TRAINING_SAMPLES + N_NORMAL_TEST
    for idx in range(N_TRAINING_SAMPLES, normal_test_end):
        cat = rows[idx].get("anomaly_category")
        if cat != "NORMAL":
            raise ValueError(
                f"Normal test partition violation at row {idx + 1} ({rows[idx].get('trace_id')}): "
                f"expected category 'NORMAL', got '{cat}'."
            )

    # 3. Held-out statistical anomaly test rows: [3500 : 4000] must all be STATISTICAL_ANOMALY
    anomaly_test_end = normal_test_end + N_ANOMALY_TEST
    for idx in range(normal_test_end, anomaly_test_end):
        cat = rows[idx].get("anomaly_category")
        if cat != "STATISTICAL_ANOMALY":
            raise ValueError(
                f"Statistical anomaly test partition violation at row {idx + 1} ({rows[idx].get('trace_id')}): "
                f"expected category 'STATISTICAL_ANOMALY', got '{cat}'."
            )

    # 4. FSM benchmark rows: [4000 : 5000] must all be RULE_VIOLATION
    for idx in range(anomaly_test_end, TOTAL_EXPECTED_ROWS):
        cat = rows[idx].get("anomaly_category")
        if cat != "RULE_VIOLATION":
            raise ValueError(
                f"Rule violation partition violation at row {idx + 1} ({rows[idx].get('trace_id')}): "
                f"expected category 'RULE_VIOLATION', got '{cat}'."
            )


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_model(
    dataset_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train the Isolation Forest model on the 2,800 clean NORMAL workflow traces.

    Args:
        dataset_path: Path to dataset CSV. Defaults to DEFAULT_DATASET_PATH.
        model_path: Destination path for joblib artifact. Defaults to DEFAULT_MODEL_PATH.

    Returns:
        Dictionary containing:
          - model: Fitted IsolationForest estimator
          - metadata: Metadata dictionary
          - training_sample_count: Number of training rows (2800)
          - feature_names: List of 14 feature names in approved order
          - model_path: String path to the saved artifact
    """
    dataset_file = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    model_target = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # 1. Load CSV and validate columns
    with open(dataset_file, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in fieldnames]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        rows = list(reader)

    # 2. Validate row count and partition structure
    validate_dataset_rows(rows)

    # 3. Extract training sequences: STRICTLY the first 2,800 rows (all NORMAL)
    training_rows = rows[:N_TRAINING_SAMPLES]
    training_sequences = [r["event_sequence"] for r in training_rows]

    # 4. Extract 14 numerical features (NO labels or identifiers used)
    X_train = extract_feature_matrix(training_sequences)
    if X_train.shape != (N_TRAINING_SAMPLES, len(FEATURE_NAMES)):
        raise ValueError(
            f"Unexpected training feature matrix shape: expected "
            f"({N_TRAINING_SAMPLES}, {len(FEATURE_NAMES)}), got {X_train.shape}."
        )

    # 5. Configure and fit IsolationForest
    model = IsolationForest(
        n_estimators=MODEL_CONFIG["n_estimators"],
        contamination=MODEL_CONFIG["contamination"],
        random_state=MODEL_CONFIG["random_state"],
    )
    model.fit(X_train)

    # 6. Build metadata dictionary
    metadata = {
        "model_name": "IsolationForest",
        "n_estimators": MODEL_CONFIG["n_estimators"],
        "contamination": MODEL_CONFIG["contamination"],
        "random_state": MODEL_CONFIG["random_state"],
        "training_sample_count": N_TRAINING_SAMPLES,
        "feature_names": list(FEATURE_NAMES),
        "dataset_spec_version": SPEC_VERSION,
    }

    # 7. Save model and metadata
    model_target.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(artifact, model_target)

    # Also save metadata as a readable JSON file alongside the joblib artifact
    metadata_json_path = model_target.parent / f"{model_target.stem}_metadata.json"
    with open(metadata_json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "model": model,
        "metadata": metadata,
        "training_sample_count": N_TRAINING_SAMPLES,
        "feature_names": list(FEATURE_NAMES),
        "model_path": str(model_target),
    }


def load_model(model_path: Optional[Union[str, Path]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Load the trained IsolationForest model and its metadata from joblib artifact."""
    target = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"Model artifact not found: {target}")

    data = joblib.load(target)
    if isinstance(data, dict) and "model" in data and "metadata" in data:
        return data["model"], data["metadata"]
    return data, getattr(data, "metadata_", {})


if __name__ == "__main__":
    result = train_model()
    print("Training successful!")
    print(f"Model: {result['metadata']['model_name']}")
    print(f"Estimators: {result['metadata']['n_estimators']}")
    print(f"Training samples: {result['training_sample_count']}")
    print(f"Artifact: {result['model_path']}")
