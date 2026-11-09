"""Unit tests for the Isolation Forest ML evaluation and FSM benchmarking module.

Covers:
1. Successful evaluation using the existing trained model artifact.
2. Correct primary evaluation sample count (1,200).
3. Correct ground-truth category counts (700 NORMAL, 500 STATISTICAL_ANOMALY).
4. Exclusion of RULE_VIOLATION traces from primary ML metrics.
5. Conversion of Isolation Forest predictions to binary labels (1 -> 0, -1 -> 1).
6. Anomaly-score direction for ROC-AUC (-decision_function).
7. Confusion matrix dimensions and counts.
8. Separate FSM benchmark results.
9. Correct JSON serialization of evaluation results.
10. Missing dataset handling.
11. Missing model artifact handling.
12. Invalid dataset category and count handling.
13. Ground-truth labels constructed strictly from `is_ml_anomaly`.
14. Rejection of NORMAL category with `is_ml_anomaly=True`.
15. Rejection of STATISTICAL_ANOMALY category with `is_ml_anomaly=False`.
16. Rejection of missing, blank, or unrecognized `is_ml_anomaly` values.
17. Rejection of unexpected model predictions (e.g. 0, 2) with RuntimeError.
18. Rejection of incorrect prediction array dimensions.
19. Rejection of incorrect decision score dimensions.
20. Rejection of non-finite decision scores (NaN or Inf).

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_ml_evaluation.py -v
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import unittest

import numpy as np

from backend.ml.evaluation.evaluate import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_PATH,
    N_ANOMALY_TEST,
    N_NORMAL_TEST,
    N_PRIMARY_TEST,
    N_RULE_VIOLATION,
    evaluate_fsm_benchmark,
    evaluate_primary_ml,
    load_and_validate_dataset,
    run_evaluation,
)
from backend.ml.inference.predictor import DEFAULT_MODEL_PATH, load_trained_model


class DeterministicBinaryFakeModel:
    """Fake model returning deterministic predictions and decision scores for testing."""

    def __init__(self, predictions: Any, decision_scores: Any) -> None:
        self.predictions = np.asarray(predictions)
        self.decision_scores = np.asarray(decision_scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.decision_scores


class TestMLEvaluation(unittest.TestCase):
    """Test suite for ML evaluation and FSM benchmarking."""

    @classmethod
    def setUpClass(cls) -> None:
        # Load the production model once for read-only evaluation tests
        cls.model, cls.metadata = load_trained_model(DEFAULT_MODEL_PATH)
        # Load production dataset rows once
        cls.train_rows, cls.test_rows, cls.rule_rows = load_and_validate_dataset(
            DEFAULT_DATASET_PATH
        )

    # -----------------------------------------------------------------------
    # 1. Successful Evaluation Using Existing Trained Model
    # -----------------------------------------------------------------------

    def test_01_successful_evaluation_using_existing_model(self) -> None:
        results = run_evaluation(save_results=False)

        # Check top-level structure
        self.assertIn("dataset_spec_version", results)
        self.assertEqual(results["dataset_spec_version"], "1.4")
        self.assertIn("model_metadata", results)
        self.assertIn("primary_ml_evaluation", results)
        self.assertIn("fsm_benchmark", results)

        ml = results["primary_ml_evaluation"]
        for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            self.assertIn(metric_name, ml)
            val = ml[metric_name]
            self.assertIsInstance(val, float)
            self.assertTrue(np.isfinite(val))
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    # -----------------------------------------------------------------------
    # 2. Correct Primary Evaluation Sample Count of 1,200
    # -----------------------------------------------------------------------

    def test_02_correct_primary_evaluation_sample_count(self) -> None:
        results = run_evaluation(save_results=False)
        ml = results["primary_ml_evaluation"]
        self.assertEqual(ml["evaluation_sample_count"], N_PRIMARY_TEST)
        self.assertEqual(len(self.test_rows), N_PRIMARY_TEST)

    # -----------------------------------------------------------------------
    # 3. Correct Normal / Anomaly Ground-Truth Counts
    # -----------------------------------------------------------------------

    def test_03_correct_normal_and_anomaly_ground_truth_counts(self) -> None:
        results = run_evaluation(save_results=False)
        ml = results["primary_ml_evaluation"]

        self.assertEqual(ml["actual_normal_count"], N_NORMAL_TEST)
        self.assertEqual(ml["actual_anomaly_count"], N_ANOMALY_TEST)
        self.assertEqual(
            ml["actual_normal_count"] + ml["actual_anomaly_count"],
            N_PRIMARY_TEST,
        )

    # -----------------------------------------------------------------------
    # 4. Correct Exclusion of RULE_VIOLATION Traces from Primary ML Metrics
    # -----------------------------------------------------------------------

    def test_04_correct_exclusion_of_rule_violation_traces_from_primary_ml(self) -> None:
        # None of the primary test rows should be RULE_VIOLATION
        for r in self.test_rows:
            self.assertIn(r["anomaly_category"], ["NORMAL", "STATISTICAL_ANOMALY"])
            self.assertNotEqual(r["anomaly_category"], "RULE_VIOLATION")

        # If a RULE_VIOLATION row is mistakenly included, evaluate_primary_ml must raise ValueError
        bad_test_rows = [copy.deepcopy(r) for r in self.test_rows]
        bad_test_rows[0] = {
            "trace_id": "TRACE_BAD",
            "event_sequence": "CONFIRM_PAYMENT",
            "anomaly_category": "RULE_VIOLATION",
            "is_ml_anomaly": "False",
        }
        with self.assertRaises(ValueError) as ctx:
            evaluate_primary_ml(self.model, bad_test_rows)
        self.assertIn("rule_violation", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 5. Conversion of Isolation Forest Predictions into Binary Labels
    # -----------------------------------------------------------------------

    def test_05_conversion_of_predictions_to_binary_labels(self) -> None:
        # Construct 1,200 fake predictions where all are predicted normal (1)
        # raw_pred = 1 => y_pred = 0 (NORMAL)
        preds_all_normal = [1] * N_PRIMARY_TEST
        scores_all_normal = [0.1] * N_PRIMARY_TEST  # positive raw_decision => anomaly_score <= 0

        fake_model = DeterministicBinaryFakeModel(preds_all_normal, scores_all_normal)
        res = evaluate_primary_ml(fake_model, self.test_rows)

        self.assertEqual(res["predicted_normal_count"], N_PRIMARY_TEST)
        self.assertEqual(res["predicted_anomaly_count"], 0)
        # Confusion matrix: TN = 700, FP = 0, FN = 500, TP = 0
        self.assertEqual(res["confusion_matrix"], [[700, 0], [500, 0]])

        # Now test where all are predicted anomaly (-1)
        # raw_pred = -1 => y_pred = 1 (ANOMALY)
        preds_all_anomaly = [-1] * N_PRIMARY_TEST
        scores_all_anomaly = [-0.1] * N_PRIMARY_TEST  # negative raw_decision => anomaly_score > 0

        fake_model_anomaly = DeterministicBinaryFakeModel(preds_all_anomaly, scores_all_anomaly)
        res_anomaly = evaluate_primary_ml(fake_model_anomaly, self.test_rows)

        self.assertEqual(res_anomaly["predicted_normal_count"], 0)
        self.assertEqual(res_anomaly["predicted_anomaly_count"], N_PRIMARY_TEST)
        # Confusion matrix: TN = 0, FP = 700, FN = 0, TP = 500
        self.assertEqual(res_anomaly["confusion_matrix"], [[0, 700], [0, 500]])

    # -----------------------------------------------------------------------
    # 6. Correct Anomaly-Score Direction for ROC-AUC
    # -----------------------------------------------------------------------

    def test_06_correct_anomaly_score_direction_for_roc_auc(self) -> None:
        # Construct fake model where statistical anomalies have high anomaly scores
        # (negative raw_decisions => positive anomaly_scores = -raw_decisions)
        # and normal traces have low anomaly scores (positive raw_decisions => negative anomaly_scores)
        preds: List[int] = []
        raw_decisions: List[float] = []

        for r in self.test_rows:
            if r["anomaly_category"] == "STATISTICAL_ANOMALY":
                preds.append(-1)
                raw_decisions.append(-0.5)  # anomaly_score = +0.5 (high)
            else:
                preds.append(1)
                raw_decisions.append(0.5)   # anomaly_score = -0.5 (low)

        perfect_model = DeterministicBinaryFakeModel(preds, raw_decisions)
        res = evaluate_primary_ml(perfect_model, self.test_rows)

        # Because anomalies have strictly higher scores than normals, ROC-AUC must be mathematically 1.0
        self.assertAlmostEqual(res["roc_auc"], 1.0, places=6)

    # -----------------------------------------------------------------------
    # 7. Confusion Matrix Dimensions and Counts
    # -----------------------------------------------------------------------

    def test_07_confusion_matrix_dimensions_and_counts(self) -> None:
        results = run_evaluation(save_results=False)
        ml = results["primary_ml_evaluation"]
        cm = ml["confusion_matrix"]

        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)
        self.assertEqual(len(cm[1]), 2)

        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        # Row sums must equal ground-truth counts
        self.assertEqual(tn + fp, N_NORMAL_TEST)
        self.assertEqual(fn + tp, N_ANOMALY_TEST)

        # Column sums must equal predicted counts
        self.assertEqual(tn + fn, ml["predicted_normal_count"])
        self.assertEqual(fp + tp, ml["predicted_anomaly_count"])

        # Total sum must equal 1,200
        self.assertEqual(tn + fp + fn + tp, N_PRIMARY_TEST)

    # -----------------------------------------------------------------------
    # 8. Separate FSM Benchmark Results
    # -----------------------------------------------------------------------

    def test_08_separate_fsm_benchmark_results(self) -> None:
        results = run_evaluation(save_results=False)
        fsm = results["fsm_benchmark"]

        self.assertEqual(fsm["total_rule_violation_traces"], N_RULE_VIOLATION)
        self.assertEqual(fsm["correctly_rejected"], N_RULE_VIOLATION)
        self.assertEqual(fsm["incorrectly_accepted"], 0)
        self.assertEqual(fsm["detection_rate"], 1.0)

        # Direct test on synthetic rule violation rows
        synthetic_violations = [
            {"event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS;DELIVER_ITEMS"}  # skips pack
        ] * N_RULE_VIOLATION
        direct_bench = evaluate_fsm_benchmark(synthetic_violations)
        self.assertEqual(direct_bench["correctly_rejected"], N_RULE_VIOLATION)
        self.assertEqual(direct_bench["detection_rate"], 1.0)

    # -----------------------------------------------------------------------
    # 9. Correct JSON Serialization
    # -----------------------------------------------------------------------

    def test_09_correct_json_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_json_path = Path(tmp_dir) / "test_evaluation_results.json"
            results = run_evaluation(results_path=temp_json_path, save_results=True)

            self.assertTrue(temp_json_path.exists())

            # Read back from disk and verify clean JSON parse and structure
            with open(temp_json_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)

            self.assertEqual(loaded["dataset_spec_version"], "1.4")
            self.assertEqual(
                loaded["primary_ml_evaluation"]["evaluation_sample_count"],
                1200,
            )
            self.assertEqual(
                loaded["fsm_benchmark"]["total_rule_violation_traces"],
                1000,
            )
            self.assertEqual(
                loaded["primary_ml_evaluation"]["confusion_matrix"],
                results["primary_ml_evaluation"]["confusion_matrix"],
            )

    # -----------------------------------------------------------------------
    # 10. Missing Dataset Handling
    # -----------------------------------------------------------------------

    def test_10_missing_dataset_handling(self) -> None:
        missing_path = Path("backend/ml/dataset/data/non_existent.csv")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_and_validate_dataset(missing_path)
        self.assertIn("not found", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 11. Missing Model Artifact Handling
    # -----------------------------------------------------------------------

    def test_11_missing_model_artifact_handling(self) -> None:
        missing_model = Path("backend/ml/models/non_existent.joblib")
        with self.assertRaises(FileNotFoundError) as ctx:
            run_evaluation(model_path=missing_model, save_results=False)
        self.assertIn("not found", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 12. Invalid Dataset Category / Count Handling
    # -----------------------------------------------------------------------

    def test_12_invalid_dataset_category_and_count_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Case A: Incorrect total row count (e.g. 10 rows instead of 5,000)
            short_csv = Path(tmp_dir) / "short.csv"
            with open(short_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "trace_id", "event_sequence", "sequence_length",
                    "is_rule_valid", "is_complete", "anomaly_category", "is_ml_anomaly"
                ])
                for i in range(10):
                    writer.writerow([f"TRACE_{i}", "CANCEL_ORDER", "1", "True", "True", "NORMAL", "False"])

            with self.assertRaises(ValueError) as ctx_a:
                load_and_validate_dataset(short_csv)
            self.assertIn("row count mismatch", str(ctx_a.exception).lower())

            # Case B: Missing required column
            bad_header_csv = Path(tmp_dir) / "bad_header.csv"
            with open(bad_header_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["trace_id", "event_sequence"])

            with self.assertRaises(ValueError) as ctx_b:
                load_and_validate_dataset(bad_header_csv)
            self.assertIn("missing required columns", str(ctx_b.exception).lower())

    # -----------------------------------------------------------------------
    # 13. Ground-Truth Labels Constructed Strictly from is_ml_anomaly
    # -----------------------------------------------------------------------

    def test_13_ground_truth_labels_constructed_from_is_ml_anomaly(self) -> None:
        # Construct deterministic fake test rows with verified is_ml_anomaly
        # Verify that evaluate_primary_ml correctly maps "False" -> 0 and "True" -> 1
        fake_preds = [1] * N_NORMAL_TEST + [-1] * N_ANOMALY_TEST
        fake_scores = [0.2] * N_NORMAL_TEST + [-0.2] * N_ANOMALY_TEST
        fake_model = DeterministicBinaryFakeModel(fake_preds, fake_scores)

        res = evaluate_primary_ml(fake_model, self.test_rows)
        # If ground-truth was correctly mapped to 700 normals (0) and 500 anomalies (1),
        # this model achieves 100% precision, recall, and accuracy
        self.assertEqual(res["actual_normal_count"], 700)
        self.assertEqual(res["actual_anomaly_count"], 500)
        self.assertEqual(res["confusion_matrix"], [[700, 0], [0, 500]])
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)

    # -----------------------------------------------------------------------
    # 14. NORMAL Category with is_ml_anomaly=True is Rejected
    # -----------------------------------------------------------------------

    def test_14_normal_category_with_is_ml_anomaly_true_is_rejected(self) -> None:
        bad_rows = [copy.deepcopy(r) for r in self.test_rows]
        # Invert the is_ml_anomaly label for the first NORMAL row
        bad_rows[0]["is_ml_anomaly"] = "True"

        with self.assertRaises(ValueError) as ctx:
            evaluate_primary_ml(self.model, bad_rows)
        self.assertIn("inconsistent", str(ctx.exception).lower())
        self.assertIn("normal", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 15. STATISTICAL_ANOMALY Category with is_ml_anomaly=False is Rejected
    # -----------------------------------------------------------------------

    def test_15_statistical_anomaly_category_with_is_ml_anomaly_false_is_rejected(self) -> None:
        bad_rows = [copy.deepcopy(r) for r in self.test_rows]
        # Invert the is_ml_anomaly label for the first STATISTICAL_ANOMALY row (index 700)
        bad_rows[N_NORMAL_TEST]["is_ml_anomaly"] = "False"

        with self.assertRaises(ValueError) as ctx:
            evaluate_primary_ml(self.model, bad_rows)
        self.assertIn("inconsistent", str(ctx.exception).lower())
        self.assertIn("statistical_anomaly", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 16. Missing, Blank, or Unrecognized is_ml_anomaly Rejected
    # -----------------------------------------------------------------------

    def test_16_missing_or_blank_is_ml_anomaly_rejected(self) -> None:
        invalid_values = ["", "   ", None, "invalid_token", "2"]

        for bad_val in invalid_values:
            with self.subTest(bad_val=bad_val):
                bad_rows = [copy.deepcopy(r) for r in self.test_rows]
                bad_rows[0]["is_ml_anomaly"] = bad_val
                with self.assertRaises(ValueError) as ctx:
                    evaluate_primary_ml(self.model, bad_rows)
                self.assertTrue(
                    "missing" in str(ctx.exception).lower()
                    or "blank" in str(ctx.exception).lower()
                    or "unrecognized" in str(ctx.exception).lower()
                    or "invalid" in str(ctx.exception).lower()
                )

    # -----------------------------------------------------------------------
    # 17. Unexpected Predictions (e.g. 0, 2) Raise RuntimeError
    # -----------------------------------------------------------------------

    def test_17_unexpected_predictions_raise_runtime_error(self) -> None:
        # Fake model returning a prediction of 0 (or 2) instead of 1 or -1
        bad_preds = [1] * (N_PRIMARY_TEST - 1) + [0]
        bad_scores = [0.1] * (N_PRIMARY_TEST - 1) + [0.1]
        fake_model = DeterministicBinaryFakeModel(bad_preds, bad_scores)

        with self.assertRaises(RuntimeError) as ctx:
            evaluate_primary_ml(fake_model, self.test_rows)
        self.assertIn("unexpected model prediction", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 18. Incorrect Prediction Array Dimensions Rejected
    # -----------------------------------------------------------------------

    def test_18_incorrect_prediction_dimensions_rejected(self) -> None:
        # Shape mismatch: returns 500 predictions instead of 1,200
        bad_preds = [1] * 500
        bad_scores = [0.1] * N_PRIMARY_TEST
        fake_model = DeterministicBinaryFakeModel(bad_preds, bad_scores)

        with self.assertRaises(ValueError) as ctx:
            evaluate_primary_ml(fake_model, self.test_rows)
        self.assertIn("predict to return exactly 1200", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 19. Incorrect Decision Score Dimensions Rejected
    # -----------------------------------------------------------------------

    def test_19_incorrect_decision_score_dimensions_rejected(self) -> None:
        # Shape mismatch: returns 500 decision scores instead of 1,200
        bad_preds = [1] * N_PRIMARY_TEST
        bad_scores = [0.1] * 500
        fake_model = DeterministicBinaryFakeModel(bad_preds, bad_scores)

        with self.assertRaises(ValueError) as ctx:
            evaluate_primary_ml(fake_model, self.test_rows)
        self.assertIn("decision_function to return exactly 1200", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 20. Non-Finite Decision Scores (NaN or Inf) Rejected
    # -----------------------------------------------------------------------

    def test_20_non_finite_decision_scores_rejected(self) -> None:
        for non_finite_val in [np.nan, np.inf, -np.inf]:
            with self.subTest(non_finite_val=non_finite_val):
                bad_preds = [1] * N_PRIMARY_TEST
                bad_scores = [0.1] * (N_PRIMARY_TEST - 1) + [non_finite_val]
                fake_model = DeterministicBinaryFakeModel(bad_preds, bad_scores)

                with self.assertRaises(ValueError) as ctx:
                    evaluate_primary_ml(fake_model, self.test_rows)
                self.assertIn("non-finite", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
