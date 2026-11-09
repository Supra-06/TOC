"""Isolation Forest and FSM workflow anomaly evaluation module.

Evaluates the trained Isolation Forest model on the 1,200 primary test traces
(700 NORMAL, 500 STATISTICAL_ANOMALY) and evaluates the 1,000 RULE_VIOLATION
traces separately against the deterministic FSM validator.

Specification: backend/docs/DATASET_SPEC.md (v1.4, Section 5 & Section 7)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.fsm.validator import validate_trace
from backend.ml.feature_extraction.extractor import (
    extract_feature_matrix,
    parse_event_sequence,
)
from backend.ml.inference.predictor import DEFAULT_MODEL_PATH, load_trained_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset" / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "orders_workflow_dataset.csv"

EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"

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
N_PRIMARY_TEST = N_NORMAL_TEST + N_ANOMALY_TEST  # 1,200
N_RULE_VIOLATION = 1000

SPEC_VERSION = "1.4"


# ---------------------------------------------------------------------------
# Dataset Loading & Partition Validation
# ---------------------------------------------------------------------------

def load_and_validate_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and validate dataset partitions strictly according to DATASET_SPEC.md.

    Args:
        dataset_path: Path to dataset CSV. Defaults to DEFAULT_DATASET_PATH.

    Returns:
        Tuple of (training_rows, primary_test_rows, rule_violation_rows).

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If row counts, headers, or category partition boundaries are violated.
    """
    target = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not target.exists():
        raise FileNotFoundError(f"Dataset file not found at: {target}")

    with open(target, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in fieldnames]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        rows = list(reader)

    total_rows = len(rows)
    if total_rows != TOTAL_EXPECTED_ROWS:
        raise ValueError(
            f"Dataset total row count mismatch: expected {TOTAL_EXPECTED_ROWS}, got {total_rows}."
        )

    # Partition 1: Training rows [0 : 2800] - 100% NORMAL
    training_rows = rows[:N_TRAINING_SAMPLES]
    for idx, r in enumerate(training_rows):
        if r.get("anomaly_category") != "NORMAL":
            raise ValueError(
                f"Training row {idx} ({r.get('trace_id')}): expected category 'NORMAL', "
                f"got '{r.get('anomaly_category')}'."
            )

    # Partition 2: Normal test rows [2800 : 3500] - 100% NORMAL
    normal_test_start = N_TRAINING_SAMPLES
    normal_test_end = normal_test_start + N_NORMAL_TEST
    normal_test_rows = rows[normal_test_start:normal_test_end]
    for idx, r in enumerate(normal_test_rows, start=normal_test_start):
        if r.get("anomaly_category") != "NORMAL":
            raise ValueError(
                f"Normal test row {idx} ({r.get('trace_id')}): expected category 'NORMAL', "
                f"got '{r.get('anomaly_category')}'."
            )

    # Partition 3: Statistical anomaly test rows [3500 : 4000] - 100% STATISTICAL_ANOMALY
    anomaly_test_start = normal_test_end
    anomaly_test_end = anomaly_test_start + N_ANOMALY_TEST
    anomaly_test_rows = rows[anomaly_test_start:anomaly_test_end]
    for idx, r in enumerate(anomaly_test_rows, start=anomaly_test_start):
        if r.get("anomaly_category") != "STATISTICAL_ANOMALY":
            raise ValueError(
                f"Anomaly test row {idx} ({r.get('trace_id')}): expected category 'STATISTICAL_ANOMALY', "
                f"got '{r.get('anomaly_category')}'."
            )

    # Partition 4: Rule violation benchmark rows [4000 : 5000] - 100% RULE_VIOLATION
    rule_violation_start = anomaly_test_end
    rule_violation_end = rule_violation_start + N_RULE_VIOLATION
    rule_violation_rows = rows[rule_violation_start:rule_violation_end]
    for idx, r in enumerate(rule_violation_rows, start=rule_violation_start):
        if r.get("anomaly_category") != "RULE_VIOLATION":
            raise ValueError(
                f"Rule violation row {idx} ({r.get('trace_id')}): expected category 'RULE_VIOLATION', "
                f"got '{r.get('anomaly_category')}'."
            )

    # Primary ML test set is strictly the 700 normal test + 500 statistical anomaly test traces
    primary_test_rows = normal_test_rows + anomaly_test_rows

    return training_rows, primary_test_rows, rule_violation_rows


# ---------------------------------------------------------------------------
# Primary ML Evaluation
# ---------------------------------------------------------------------------

def evaluate_primary_ml(model: Any, test_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate Isolation Forest on the primary 1,200 test traces.

    Ground Truth Construction:
      - Ground truth labels are constructed strictly from the `is_ml_anomaly` column:
        * is_ml_anomaly == False (or "False") -> NORMAL (0, Negative class)
        * is_ml_anomaly == True  (or "True")  -> ANOMALY (1, Positive class)
      - Validates category consistency:
        * anomaly_category == "NORMAL" must match is_ml_anomaly == False
        * anomaly_category == "STATISTICAL_ANOMALY" must match is_ml_anomaly == True
        * RULE_VIOLATION must never appear in primary ML evaluation
      - Missing, blank, or unrecognized is_ml_anomaly raises ValueError.

    Scoring and Validation:
      - Continuous anomaly score: anomaly_score = -model.decision_function(X)
      - Binary prediction: predict == -1 => 1 (ANOMALY); predict == 1 => 0 (NORMAL)
      - Validates output shapes, finite numerical values, and strictly rejects unexpected predictions.
      - Verifies that predict() and anomaly_score classifications strictly agree.

    Args:
        model: Trained Isolation Forest estimator.
        test_rows: List of 1,200 test row dictionaries.

    Returns:
        Structured dictionary of ML performance metrics.

    Raises:
        ValueError: If test_rows structure, counts, categories, labels, shapes, or scores are invalid.
        RuntimeError: If predict() returns unexpected values or disagrees with anomaly_score.
    """
    if len(test_rows) != N_PRIMARY_TEST:
        raise ValueError(
            f"Primary ML evaluation requires exactly {N_PRIMARY_TEST} traces, got {len(test_rows)}."
        )

    # 1. Construct ground truth labels strictly from is_ml_anomaly
    # Never use label columns as model features (data leakage prevention)
    y_true_list: List[int] = []
    sequences: List[str] = []

    actual_normal_count = 0
    actual_anomaly_count = 0

    for idx, row in enumerate(test_rows):
        trace_id = row.get("trace_id", f"index_{idx}")
        cat = row.get("anomaly_category")
        raw_ml_anomaly = row.get("is_ml_anomaly")

        # RULE_VIOLATION must never appear in the primary ML test set
        if cat == "RULE_VIOLATION":
            raise ValueError(
                f"RULE_VIOLATION trace {trace_id} must never appear in the primary ML test set."
            )

        # Missing or blank is_ml_anomaly is rejected
        if raw_ml_anomaly is None or (isinstance(raw_ml_anomaly, str) and not raw_ml_anomaly.strip()):
            raise ValueError(
                f"Missing or blank 'is_ml_anomaly' label in primary test set at trace {trace_id}."
            )

        # Parse is_ml_anomaly into boolean
        if isinstance(raw_ml_anomaly, bool):
            is_anomaly_val = raw_ml_anomaly
        elif isinstance(raw_ml_anomaly, str):
            val_clean = raw_ml_anomaly.strip().lower()
            if val_clean == "true":
                is_anomaly_val = True
            elif val_clean == "false":
                is_anomaly_val = False
            else:
                raise ValueError(
                    f"Unrecognized 'is_ml_anomaly' value '{raw_ml_anomaly}' at trace {trace_id}. "
                    f"Expected 'True' or 'False'."
                )
        else:
            raise ValueError(
                f"Invalid 'is_ml_anomaly' type {type(raw_ml_anomaly).__name__} at trace {trace_id}."
            )

        # Ground truth label: 0 for NORMAL, 1 for ANOMALY
        ground_truth_label = 1 if is_anomaly_val else 0

        # Validate category and label consistency
        if cat == "NORMAL":
            if is_anomaly_val:
                raise ValueError(
                    f"Inconsistent labels at trace {trace_id}: category is 'NORMAL' "
                    f"but is_ml_anomaly is {raw_ml_anomaly}."
                )
            actual_normal_count += 1
        elif cat == "STATISTICAL_ANOMALY":
            if not is_anomaly_val:
                raise ValueError(
                    f"Inconsistent labels at trace {trace_id}: category is 'STATISTICAL_ANOMALY' "
                    f"but is_ml_anomaly is {raw_ml_anomaly}."
                )
            actual_anomaly_count += 1
        else:
            raise ValueError(
                f"Invalid anomaly_category '{cat}' at trace {trace_id}. "
                f"Only 'NORMAL' and 'STATISTICAL_ANOMALY' are permitted in primary ML evaluation."
            )

        y_true_list.append(ground_truth_label)
        sequences.append(row["event_sequence"])

    if actual_normal_count != N_NORMAL_TEST or actual_anomaly_count != N_ANOMALY_TEST:
        raise ValueError(
            f"Ground truth category mismatch: expected {N_NORMAL_TEST} NORMAL and "
            f"{N_ANOMALY_TEST} STATISTICAL_ANOMALY; got {actual_normal_count} and {actual_anomaly_count}."
        )

    y_true = np.array(y_true_list, dtype=np.int64)

    # 2. Extract features using approved feature extraction implementation
    X_test = extract_feature_matrix(sequences)

    # 3. Model inference and scoring
    try:
        raw_decisions_out = model.decision_function(X_test)
    except Exception as exc:
        raise RuntimeError(f"Error executing model.decision_function(): {exc}") from exc

    try:
        raw_predictions_out = model.predict(X_test)
    except Exception as exc:
        raise RuntimeError(f"Error executing model.predict(): {exc}") from exc

    # Validate output dimensions and shapes
    raw_decisions = np.asarray(raw_decisions_out, dtype=np.float64)
    raw_predictions = np.asarray(raw_predictions_out)

    if raw_decisions.ndim == 2 and raw_decisions.shape == (N_PRIMARY_TEST, 1):
        raw_decisions = raw_decisions.ravel()
    elif raw_decisions.ndim != 1 or raw_decisions.shape != (N_PRIMARY_TEST,):
        raise ValueError(
            f"Expected decision_function to return exactly {N_PRIMARY_TEST} scores, "
            f"got shape {raw_decisions.shape}."
        )

    if raw_predictions.ndim == 2 and raw_predictions.shape == (N_PRIMARY_TEST, 1):
        raw_predictions = raw_predictions.ravel()
    elif raw_predictions.ndim != 1 or raw_predictions.shape != (N_PRIMARY_TEST,):
        raise ValueError(
            f"Expected predict to return exactly {N_PRIMARY_TEST} predictions, "
            f"got shape {raw_predictions.shape}."
        )

    # Validate finite numerical values
    if not np.all(np.isfinite(raw_decisions)):
        raise ValueError("decision_function output contains non-finite values (NaN or Inf).")

    if not np.all(np.isfinite(raw_predictions)):
        raise ValueError("predict output contains non-finite values (NaN or Inf).")

    # Reject unexpected predictions: must be strictly 1 (NORMAL) or -1 (ANOMALY)
    invalid_preds = [int(p) for p in raw_predictions if p not in (1, -1)]
    if invalid_preds:
        raise RuntimeError(
            f"Unexpected model prediction value(s) encountered: {invalid_preds[:5]}. "
            f"Expected only 1 (NORMAL) or -1 (ANOMALY)."
        )

    raw_predictions = raw_predictions.astype(np.int64)

    # Negate decision function so higher score = greater anomaly likelihood
    anomaly_scores = -raw_decisions

    # 4. Enforce strict agreement between predict and score
    # predict == -1 <=> anomaly_score > 0
    # predict == 1  <=> anomaly_score <= 0
    pred_is_anomaly = (raw_predictions == -1)
    score_is_anomaly = (anomaly_scores > 0.0)
    disagreements = np.where(pred_is_anomaly != score_is_anomaly)[0]

    if len(disagreements) > 0:
        first_bad = disagreements[0]
        raise RuntimeError(
            f"Prediction and score disagreement at sample index {first_bad}: "
            f"predict()={raw_predictions[first_bad]}, "
            f"anomaly_score={anomaly_scores[first_bad]}, "
            f"raw_decision={raw_decisions[first_bad]}."
        )

    # 5. Convert predictions to binary labels: 1 = ANOMALY, 0 = NORMAL
    y_pred = np.where(raw_predictions == -1, 1, 0)

    predicted_normal_count = int(np.sum(y_pred == 0))
    predicted_anomaly_count = int(np.sum(y_pred == 1))

    # 6. Calculate metrics
    # Confusion matrix layout for labels=[0, 1]:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_list: List[List[int]] = [[int(val) for val in row] for row in cm]

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))

    return {
        "evaluation_sample_count": len(test_rows),
        "actual_normal_count": actual_normal_count,
        "actual_anomaly_count": actual_anomaly_count,
        "predicted_normal_count": predicted_normal_count,
        "predicted_anomaly_count": predicted_anomaly_count,
        "confusion_matrix": cm_list,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Separate FSM Benchmark
# ---------------------------------------------------------------------------

def evaluate_fsm_benchmark(rule_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate 1,000 RULE_VIOLATION traces separately against deterministic FSM.

    Args:
        rule_rows: List of 1,000 rule violation trace dictionaries.

    Returns:
        Structured dictionary of FSM benchmark metrics.
    """
    total = len(rule_rows)
    if total != N_RULE_VIOLATION:
        raise ValueError(
            f"FSM benchmark requires exactly {N_RULE_VIOLATION} traces, got {total}."
        )

    correctly_rejected = 0
    incorrectly_accepted = 0

    for r in rule_rows:
        events = parse_event_sequence(r["event_sequence"])
        validation_result = validate_trace(events)
        if not validation_result["is_valid"]:
            correctly_rejected += 1
        else:
            incorrectly_accepted += 1

    detection_rate = float(correctly_rejected / total) if total > 0 else 0.0

    return {
        "total_rule_violation_traces": total,
        "correctly_rejected": correctly_rejected,
        "incorrectly_accepted": incorrectly_accepted,
        "detection_rate": detection_rate,
    }


# ---------------------------------------------------------------------------
# Master Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    results_path: Optional[Union[str, Path]] = None,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Execute complete evaluation workflow: Primary ML evaluation + FSM benchmark.

    Args:
        dataset_path: Path to dataset CSV file. Defaults to DEFAULT_DATASET_PATH.
        model_path: Path to model joblib artifact. Defaults to DEFAULT_MODEL_PATH.
        results_path: Path to save evaluation JSON. Defaults to DEFAULT_RESULTS_PATH.
        save_results: Whether to save results to disk.

    Returns:
        Complete evaluation results dictionary.
    """
    # 1. Load and validate dataset
    _, primary_test_rows, rule_violation_rows = load_and_validate_dataset(dataset_path)

    # 2. Load trained model artifact and metadata
    model, metadata = load_trained_model(model_path=model_path)

    # 3. Primary ML evaluation (1,200 traces: 700 NORMAL, 500 STATISTICAL_ANOMALY)
    ml_metrics = evaluate_primary_ml(model, primary_test_rows)

    # 4. Separate FSM benchmark (1,000 RULE_VIOLATION traces)
    fsm_metrics = evaluate_fsm_benchmark(rule_violation_rows)

    # 5. Assemble structured evaluation report
    results: Dict[str, Any] = {
        "dataset_spec_version": SPEC_VERSION,
        "model_metadata": metadata,
        "primary_ml_evaluation": ml_metrics,
        "fsm_benchmark": fsm_metrics,
    }

    # 6. Save JSON results if requested
    if save_results:
        target_json = Path(results_path) if results_path else DEFAULT_RESULTS_PATH
        target_json.parent.mkdir(parents=True, exist_ok=True)
        with open(target_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return results


# ---------------------------------------------------------------------------
# Formatted Report Generator
# ---------------------------------------------------------------------------

def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Format evaluation results into a human-readable ASCII report."""
    ml = results["primary_ml_evaluation"]
    fsm = results["fsm_benchmark"]
    meta = results.get("model_metadata", {})

    cm = ml["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    lines = [
        "=" * 78,
        "  WORKFLOW ANOMALY DETECTION EVALUATION REPORT",
        "  Dual-Engine Architecture: Deterministic FSM + Isolation Forest ML",
        f"  Dataset Spec Version: {results.get('dataset_spec_version', 'N/A')}",
        "=" * 78,
        "",
        "1. MODEL CONFIGURATION & METADATA",
        f"   - Model Name:              {meta.get('model_name', 'N/A')}",
        f"   - Estimators:              {meta.get('n_estimators', 'N/A')}",
        f"   - Contamination:           {meta.get('contamination', 'N/A')}",
        f"   - Random State:            {meta.get('random_state', 'N/A')}",
        f"   - Training Sample Count:   {meta.get('training_sample_count', 'N/A')} (100% clean NORMAL)",
        f"   - Feature Count:           {len(meta.get('feature_names', []))} engineered features",
        "",
        "2. PRIMARY ML EVALUATION (1,200 Held-Out FSM-Valid Traces)",
        f"   - Total ML Test Samples:   {ml['evaluation_sample_count']}",
        f"   - Actual Normal Traces:    {ml['actual_normal_count']}",
        f"   - Actual Anomaly Traces:   {ml['actual_anomaly_count']}",
        f"   - Predicted Normal:        {ml['predicted_normal_count']}",
        f"   - Predicted Anomaly:       {ml['predicted_anomaly_count']}",
        "",
        "   Confusion Matrix:",
        f"     Actual Normal:   TN = {tn:<5}  FP = {fp:<5}",
        f"     Actual Anomaly:  FN = {fn:<5}  TP = {tp:<5}",
        "",
        "   Performance Metrics:",
        f"     - Accuracy:              {ml['accuracy']:.4f}  ({ml['accuracy'] * 100:.2f}%)",
        f"     - Precision:             {ml['precision']:.4f}  ({ml['precision'] * 100:.2f}%)",
        f"     - Recall:                {ml['recall']:.4f}  ({ml['recall'] * 100:.2f}%)",
        f"     - F1-Score:              {ml['f1_score']:.4f}",
        f"     - ROC-AUC:               {ml['roc_auc']:.4f}",
        "",
        "3. SEPARATE FSM BENCHMARK (1,000 Rule Violation Traces)",
        f"   - Total Rule Violations:   {fsm['total_rule_violation_traces']}",
        f"   - Correctly Rejected:      {fsm['correctly_rejected']}",
        f"   - Incorrectly Accepted:    {fsm['incorrectly_accepted']}",
        f"   - FSM Detection Rate:      {fsm['detection_rate']:.4f}  ({fsm['detection_rate'] * 100:.2f}%)",
        "",
        "=" * 78,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    eval_results = run_evaluation()
    print(format_evaluation_report(eval_results))
    print(f"\nResults saved to: {DEFAULT_RESULTS_PATH}")
