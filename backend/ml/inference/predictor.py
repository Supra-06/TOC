"""Isolation Forest inference and anomaly scoring module.

Performs online inference and anomaly scoring on workflow event sequences
using the trained Isolation Forest model artifact.

Specification: backend/docs/DATASET_SPEC.md (v1.4, Section 5.3 & Section 6)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np

from backend.ml.feature_extraction.extractor import (
    FEATURE_NAMES,
    extract_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"


# ---------------------------------------------------------------------------
# Metadata Validation
# ---------------------------------------------------------------------------

def validate_model_metadata(metadata: Dict[str, Any]) -> None:
    """Validate that the model metadata contains all required fields and approved values.

    Args:
        metadata: Metadata dictionary to validate.

    Raises:
        ValueError: If metadata is not a dict, is missing required fields, or has incorrect values.
    """
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Model artifact is missing required metadata dictionary.")

    required_fields = [
        "model_name",
        "n_estimators",
        "contamination",
        "random_state",
        "training_sample_count",
        "feature_names",
        "dataset_spec_version",
    ]
    missing = [f for f in required_fields if f not in metadata]
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

    if metadata["model_name"] != "IsolationForest":
        raise ValueError(
            f"Invalid model_name in metadata: expected 'IsolationForest', got '{metadata['model_name']}'."
        )

    if metadata["n_estimators"] != 100:
        raise ValueError(
            f"Invalid n_estimators in metadata: expected 100, got {metadata['n_estimators']}."
        )

    if metadata["contamination"] != "auto":
        raise ValueError(
            f"Invalid contamination in metadata: expected 'auto', got '{metadata['contamination']}'."
        )

    if metadata["random_state"] != 42:
        raise ValueError(
            f"Invalid random_state in metadata: expected 42, got {metadata['random_state']}."
        )

    if metadata["training_sample_count"] != 2800:
        raise ValueError(
            f"Invalid training_sample_count in metadata: expected 2800, got {metadata['training_sample_count']}."
        )

    feature_names = metadata["feature_names"]
    if not isinstance(feature_names, (list, tuple)) or list(feature_names) != list(FEATURE_NAMES):
        raise ValueError(
            f"Invalid feature_names in metadata: expected exact match with approved FEATURE_NAMES order.\n"
            f"Expected: {FEATURE_NAMES}\n"
            f"Got: {feature_names}"
        )

    if str(metadata["dataset_spec_version"]) != "1.4":
        raise ValueError(
            f"Invalid dataset_spec_version in metadata: expected '1.4', got '{metadata['dataset_spec_version']}'."
        )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_trained_model(
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the trained Isolation Forest model artifact and its metadata.

    Args:
        model_path: Path to the .joblib file. Defaults to DEFAULT_MODEL_PATH.

    Returns:
        Tuple of (model_estimator, metadata_dict).

    Raises:
        FileNotFoundError: If the model artifact file does not exist.
        ValueError: If the artifact is corrupted, malformed, or missing required components/metadata.
    """
    target = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"Model artifact not found at: {target}")

    try:
        data = joblib.load(target)
    except Exception as exc:
        raise ValueError(f"Failed to load model artifact from {target}: {exc}") from exc

    # Case 1: Artifact is a dictionary containing "model" and "metadata"
    if isinstance(data, dict):
        if "model" not in data:
            raise ValueError(
                f"Malformed model artifact at {target}: missing 'model' key in artifact dictionary."
            )
        model = data["model"]
        metadata = data.get("metadata", {})
    # Case 2: Artifact is the model object itself
    elif hasattr(data, "predict") and hasattr(data, "decision_function"):
        model = data
        metadata = getattr(data, "metadata_", {})
    else:
        raise ValueError(
            f"Malformed model artifact at {target}: expected IsolationForest estimator or dict, "
            f"got {type(data).__name__}."
        )

    # Validate that model estimator has necessary inference methods
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")) or \
       not hasattr(model, "decision_function") or not callable(getattr(model, "decision_function")):
        raise ValueError(
            f"Loaded model in {target} lacks callable 'predict' or 'decision_function' methods."
        )

    # If metadata was not stored inside the joblib dictionary, check for adjacent metadata JSON
    if not metadata:
        json_meta_path = target.parent / f"{target.stem}_metadata.json"
        if json_meta_path.exists():
            try:
                with open(json_meta_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read adjacent metadata JSON from {json_meta_path}: {exc}"
                ) from exc

    # Validate metadata content and approved values
    validate_model_metadata(metadata)

    return model, metadata


# ---------------------------------------------------------------------------
# Inference & Anomaly Scoring
# ---------------------------------------------------------------------------

def predict_anomaly(
    event_sequence: Union[str, Sequence[str]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Score and classify a single workflow event sequence using Isolation Forest.

    Scoring convention (DATASET_SPEC.md Section 5.3):
      Scikit-learn's decision_function(X) returns:
        > 0 for inliers (normal)
        < 0 for outliers (anomalies)

      Our project defines:
        anomaly_score = -decision_function(X)
        raw_decision_score = decision_function(X)

      Classification Agreement Requirement:
        - model.predict(X) == -1 means ANOMALY.
        - model.predict(X) == 1 means NORMAL.
        - anomaly_score = -decision_function(X).
        - anomaly_score > 0 means ANOMALY.
        - anomaly_score <= 0 means NORMAL.

        Both classifications must agree. If they disagree, a RuntimeError is raised.

    Args:
        event_sequence: Semicolon-delimited string (e.g. "CONFIRM_PAYMENT;PACK_ITEMS;...")
            or a list of event strings (e.g. ["CONFIRM_PAYMENT", "PACK_ITEMS"]).
        model: An optional pre-loaded estimator or (model, metadata) tuple. If None,
            the model will be loaded from model_path.
        model_path: Optional path to the model artifact if model is not supplied.

    Returns:
        A dictionary containing:
          - prediction (str): "ANOMALY" or "NORMAL"
          - is_ml_anomaly (bool): True if classified as anomaly, False if normal
          - anomaly_score (float): -raw_decision_score (higher = more anomalous)
          - raw_decision_score (float): native decision_function score
          - features (Dict[str, float]): All 14 extracted numerical features in approved order

    Raises:
        RuntimeError: If model.predict() and anomaly_score disagree on classification.
        ValueError: If input format or extracted feature dimensions are invalid.
    """
    # 1. Resolve estimator (reusing supplied model if available)
    if model is not None:
        if isinstance(model, tuple) and len(model) >= 1:
            estimator = model[0]
        elif isinstance(model, dict) and "model" in model:
            estimator = model["model"]
        else:
            estimator = model
    else:
        estimator, _ = load_trained_model(model_path=model_path)

    if not hasattr(estimator, "predict") or not hasattr(estimator, "decision_function"):
        raise ValueError("Supplied model object lacks required 'predict' or 'decision_function' methods.")

    # 2. Extract features using the approved feature extraction function
    feature_vector = extract_features(event_sequence)
    if len(feature_vector) != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature extraction returned {len(feature_vector)} features; expected exactly {len(FEATURE_NAMES)}."
        )

    # Returned feature dictionary preserving approved order
    features_dict: Dict[str, float] = {
        name: float(val) for name, val in zip(FEATURE_NAMES, feature_vector)
    }

    # Convert to 2D numpy array of shape (1, 14)
    X = np.asarray([feature_vector], dtype=np.float64)
    if X.shape != (1, len(FEATURE_NAMES)):
        raise ValueError(
            f"Feature matrix shape mismatch: expected (1, {len(FEATURE_NAMES)}), got {X.shape}."
        )

    # 3. Model inference and scoring
    raw_decision = float(estimator.decision_function(X)[0])
    raw_prediction = int(estimator.predict(X)[0])

    # Compute inverted anomaly score: higher = more anomalous
    anomaly_score = float(-raw_decision)

    # Determine classification from predict()
    if raw_prediction == -1:
        pred_is_anomaly = True
    elif raw_prediction == 1:
        pred_is_anomaly = False
    else:
        raise RuntimeError(
            f"Unexpected estimator predict() output: {raw_prediction}. Expected 1 (NORMAL) or -1 (ANOMALY)."
        )

    # Determine classification from decision_function / anomaly_score
    score_is_anomaly = (anomaly_score > 0.0)

    # Explicitly verify agreement between predict() and anomaly_score
    if pred_is_anomaly != score_is_anomaly:
        raise RuntimeError(
            f"Prediction and score disagreement: predict() returned {raw_prediction} "
            f"(interpreted as is_anomaly={pred_is_anomaly}) but anomaly_score is {anomaly_score} "
            f"(raw_decision_score={raw_decision}, interpreted as is_anomaly={score_is_anomaly})."
        )

    # Use the estimator's prediction as the returned classification after verifying agreement
    if raw_prediction == -1:
        prediction_label = "ANOMALY"
        is_anomaly = True
    else:
        prediction_label = "NORMAL"
        is_anomaly = False

    return {
        "prediction": prediction_label,
        "is_ml_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "raw_decision_score": raw_decision,
        "features": features_dict,
    }


def predict_anomaly_batch(
    event_sequences: Sequence[Union[str, Sequence[str]]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Score and classify a batch of workflow event sequences efficiently.

    Args:
        event_sequences: Sequence of event sequences (strings or lists).
        model: Optional pre-loaded estimator or (model, metadata) tuple.
        model_path: Optional path to the model artifact.

    Returns:
        List of prediction dictionaries corresponding to each input sequence.
    """
    if model is not None:
        if isinstance(model, tuple) and len(model) >= 1:
            estimator = model[0]
        elif isinstance(model, dict) and "model" in model:
            estimator = model["model"]
        else:
            estimator = model
    else:
        estimator, _ = load_trained_model(model_path=model_path)

    return [predict_anomaly(seq, model=estimator) for seq in event_sequences]
