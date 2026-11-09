"""Unit tests for the Isolation Forest inference and anomaly scoring module.

Verifies:
1. Model loading and metadata validation.
2. Rejection of missing, corrupted, or incorrect metadata.
3. Feature extraction alignment (14 approved numerical features in exact order).
4. Prediction and score agreement enforcement (RuntimeError on discrepancy).
5. Robustness across inputs (string vs list, empty, whitespace, unknown tokens).
6. Reproducibility and model reuse without disk reload.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_ml_inference.py -v
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from backend.ml.feature_extraction.extractor import FEATURE_NAMES, extract_features
from backend.ml.inference.predictor import (
    DEFAULT_MODEL_PATH,
    load_trained_model,
    predict_anomaly,
    predict_anomaly_batch,
    validate_model_metadata,
)


class FakeModelDisagreementPredictNormal:
    """Deterministic fake model where predict says NORMAL (1) but decision_function is negative."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([-0.15])  # raw < 0 => anomaly_score > 0 (ANOMALY)


class FakeModelDisagreementPredictAnomaly:
    """Deterministic fake model where predict says ANOMALY (-1) but decision_function is positive."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([-1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.25])  # raw > 0 => anomaly_score <= 0 (NORMAL)


class FakeModelAgreeingNormal:
    """Deterministic fake model where predict and decision_function both indicate NORMAL."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.15])  # anomaly_score = -0.15 <= 0 (NORMAL)


class FakeModelAgreeingAnomaly:
    """Deterministic fake model where predict and decision_function both indicate ANOMALY."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([-1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([-0.20])  # anomaly_score = 0.20 > 0 (ANOMALY)


class TestMLInference(unittest.TestCase):
    """Test suite for Isolation Forest inference and anomaly prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        # Load the production model once for reuse across read-only tests
        cls.model, cls.metadata = load_trained_model(DEFAULT_MODEL_PATH)

    # -----------------------------------------------------------------------
    # 1. Model Loading & Metadata Validation
    # -----------------------------------------------------------------------

    def test_01_successfully_loads_saved_trained_model(self) -> None:
        model, metadata = load_trained_model(DEFAULT_MODEL_PATH)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "decision_function"))
        self.assertIsInstance(model, IsolationForest)

    def test_02_correct_model_metadata_loading(self) -> None:
        self.assertIsInstance(self.metadata, dict)
        self.assertEqual(self.metadata.get("model_name"), "IsolationForest")
        self.assertEqual(self.metadata.get("n_estimators"), 100)
        self.assertEqual(self.metadata.get("contamination"), "auto")
        self.assertEqual(self.metadata.get("random_state"), 42)
        self.assertEqual(self.metadata.get("training_sample_count"), 2800)
        self.assertEqual(self.metadata.get("feature_names"), FEATURE_NAMES)
        self.assertEqual(self.metadata.get("dataset_spec_version"), "1.4")

    def test_03_failure_when_model_artifact_is_missing(self) -> None:
        missing_path = Path("backend/ml/models/non_existent_model.joblib")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_trained_model(missing_path)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_04_failure_when_model_artifact_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Case A: Object is not an estimator or expected dict
            corrupt_path = Path(tmp_dir) / "corrupt.joblib"
            joblib.dump("not_a_model_object", corrupt_path)
            with self.assertRaises(ValueError):
                load_trained_model(corrupt_path)

            # Case B: Dict missing "model" key
            bad_dict_path = Path(tmp_dir) / "bad_dict.joblib"
            joblib.dump({"wrong_key": 123}, bad_dict_path)
            with self.assertRaises(ValueError):
                load_trained_model(bad_dict_path)

            # Case C: Model estimator lacks predict/decision_function
            no_methods_path = Path(tmp_dir) / "no_methods.joblib"
            joblib.dump({"model": [1, 2, 3], "metadata": self.metadata}, no_methods_path)
            with self.assertRaises(ValueError):
                load_trained_model(no_methods_path)

    def test_05_missing_metadata_is_rejected(self) -> None:
        # Artifact with missing or empty metadata
        with tempfile.TemporaryDirectory() as tmp_dir:
            no_meta_path = Path(tmp_dir) / "no_meta.joblib"
            joblib.dump({"model": self.model}, no_meta_path)
            with self.assertRaises(ValueError) as ctx:
                load_trained_model(no_meta_path)
            self.assertIn("metadata", str(ctx.exception).lower())

        # Test validate_model_metadata directly with missing fields
        valid_meta = copy.deepcopy(self.metadata)
        for key in valid_meta.keys():
            incomplete_meta = copy.deepcopy(valid_meta)
            del incomplete_meta[key]
            with self.assertRaises(ValueError) as ctx:
                validate_model_metadata(incomplete_meta)
            self.assertIn("missing", str(ctx.exception).lower())

    def test_06_incorrect_feature_names_or_order_rejected(self) -> None:
        # Reversed feature order
        reversed_meta = copy.deepcopy(self.metadata)
        reversed_meta["feature_names"] = list(reversed(FEATURE_NAMES))
        with self.assertRaises(ValueError) as ctx:
            validate_model_metadata(reversed_meta)
        self.assertIn("feature_names", str(ctx.exception).lower())

        # Altered feature name
        altered_meta = copy.deepcopy(self.metadata)
        altered_meta["feature_names"][0] = "invalid_feature_name"
        with self.assertRaises(ValueError) as ctx:
            validate_model_metadata(altered_meta)
        self.assertIn("feature_names", str(ctx.exception).lower())

        # Incomplete feature list (e.g. 13 features)
        truncated_meta = copy.deepcopy(self.metadata)
        truncated_meta["feature_names"] = FEATURE_NAMES[:-1]
        with self.assertRaises(ValueError) as ctx:
            validate_model_metadata(truncated_meta)
        self.assertIn("feature_names", str(ctx.exception).lower())

    def test_07_incorrect_model_configuration_metadata_rejected(self) -> None:
        test_cases = [
            ("model_name", "RandomForest"),
            ("n_estimators", 50),
            ("contamination", 0.05),
            ("random_state", 99),
            ("training_sample_count", 2500),
            ("dataset_spec_version", "1.3"),
        ]
        for field, invalid_val in test_cases:
            with self.subTest(field=field, invalid_val=invalid_val):
                bad_meta = copy.deepcopy(self.metadata)
                bad_meta[field] = invalid_val
                with self.assertRaises(ValueError) as ctx:
                    validate_model_metadata(bad_meta)
                self.assertIn(field, str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 2. Prediction and Score Agreement
    # -----------------------------------------------------------------------

    def test_08_prediction_score_disagreement_raises_runtime_error(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"

        # Case A: predict says NORMAL (1), but score says ANOMALY (> 0)
        fake_model_a = FakeModelDisagreementPredictNormal()
        with self.assertRaises(RuntimeError) as ctx_a:
            predict_anomaly(seq, model=fake_model_a)
        self.assertIn("disagreement", str(ctx_a.exception).lower())

        # Case B: predict says ANOMALY (-1), but score says NORMAL (<= 0)
        fake_model_b = FakeModelDisagreementPredictAnomaly()
        with self.assertRaises(RuntimeError) as ctx_b:
            predict_anomaly(seq, model=fake_model_b)
        self.assertIn("disagreement", str(ctx_b.exception).lower())

    def test_09_agreeing_fake_models_return_expected_classifications(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"

        # Agreeing Normal Fake Model
        normal_res = predict_anomaly(seq, model=FakeModelAgreeingNormal())
        self.assertEqual(normal_res["prediction"], "NORMAL")
        self.assertFalse(normal_res["is_ml_anomaly"])
        self.assertLessEqual(normal_res["anomaly_score"], 0.0)

        # Agreeing Anomaly Fake Model
        anomaly_res = predict_anomaly(seq, model=FakeModelAgreeingAnomaly())
        self.assertEqual(anomaly_res["prediction"], "ANOMALY")
        self.assertTrue(anomaly_res["is_ml_anomaly"])
        self.assertGreater(anomaly_res["anomaly_score"], 0.0)

    def test_10_prediction_agrees_with_actual_estimator_predict_output(self) -> None:
        test_sequences = [
            # Standard fulfillment
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            # Immediate cancel
            "CANCEL_ORDER",
            # Single retry then fulfillment
            "FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            # Refund path
            "CONFIRM_PAYMENT;PACK_ITEMS;REFUND_ORDER",
            # Multi-retry anomaly
            "FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CANCEL_ORDER",
        ]

        for seq in test_sequences:
            with self.subTest(seq=seq):
                # Extract features directly using approved extractor
                features = extract_features(seq)
                X = np.asarray([features], dtype=np.float64)

                actual_pred = int(self.model.predict(X)[0])
                actual_dec = float(self.model.decision_function(X)[0])

                result = predict_anomaly(seq, model=self.model)

                # Verify returned values match actual estimator outputs
                self.assertEqual(result["raw_decision_score"], actual_dec)
                self.assertEqual(result["anomaly_score"], -actual_dec)

                if actual_pred == 1:
                    self.assertEqual(result["prediction"], "NORMAL")
                    self.assertFalse(result["is_ml_anomaly"])
                    self.assertLessEqual(result["anomaly_score"], 0.0)
                elif actual_pred == -1:
                    self.assertEqual(result["prediction"], "ANOMALY")
                    self.assertTrue(result["is_ml_anomaly"])
                    self.assertGreater(result["anomaly_score"], 0.0)
                else:
                    self.fail(f"Unexpected predict() output: {actual_pred}")

    # -----------------------------------------------------------------------
    # 3. Prediction Result Structure & 14 Features
    # -----------------------------------------------------------------------

    def test_11_correct_prediction_result_structure(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = predict_anomaly(seq, model=self.model)

        expected_keys = {
            "prediction",
            "is_ml_anomaly",
            "anomaly_score",
            "raw_decision_score",
            "features",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIn(result["prediction"], ["NORMAL", "ANOMALY"])
        self.assertIsInstance(result["is_ml_anomaly"], bool)
        self.assertIsInstance(result["anomaly_score"], float)
        self.assertIsInstance(result["raw_decision_score"], float)
        self.assertIsInstance(result["features"], dict)

    def test_12_exactly_14_features_returned(self) -> None:
        seq = "CANCEL_ORDER"
        result = predict_anomaly(seq, model=self.model)
        self.assertEqual(len(result["features"]), 14)

    def test_13_feature_names_and_order_match_approved_list(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = predict_anomaly(seq, model=self.model)
        self.assertEqual(list(result["features"].keys()), FEATURE_NAMES)

    # -----------------------------------------------------------------------
    # 4. Input Robustness & Formatting
    # -----------------------------------------------------------------------

    def test_14_correct_conversion_of_string_event_sequences(self) -> None:
        seq_str = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = predict_anomaly(seq_str, model=self.model)
        self.assertEqual(result["features"]["seq_length"], 4)
        self.assertEqual(result["features"]["count_CONFIRM_PAYMENT"], 1)
        self.assertEqual(result["features"]["count_PACK_ITEMS"], 1)
        self.assertEqual(result["features"]["count_SHIP_ITEMS"], 1)
        self.assertEqual(result["features"]["count_DELIVER_ITEMS"], 1)

    def test_15_correct_handling_of_list_event_sequences(self) -> None:
        seq_list = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
        result_from_list = predict_anomaly(seq_list, model=self.model)
        result_from_str = predict_anomaly(";".join(seq_list), model=self.model)

        self.assertEqual(result_from_list["prediction"], result_from_str["prediction"])
        self.assertEqual(result_from_list["anomaly_score"], result_from_str["anomaly_score"])
        self.assertEqual(result_from_list["features"], result_from_str["features"])

    def test_16_empty_and_whitespace_sequence_handling(self) -> None:
        for empty_val in ["", "   ", []]:
            with self.subTest(empty_val=empty_val):
                result = predict_anomaly(empty_val, model=self.model)
                self.assertIn("prediction", result)
                self.assertEqual(result["features"]["seq_length"], 0)
                self.assertTrue(np.isfinite(result["anomaly_score"]))
                self.assertTrue(np.isfinite(result["raw_decision_score"]))

    def test_17_unknown_event_handling_without_silent_removal(self) -> None:
        seq = "CONFIRM_PAYMENT;HACK_CHECKOUT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = predict_anomaly(seq, model=self.model)
        self.assertEqual(result["features"]["seq_length"], 5)
        self.assertEqual(result["features"]["unique_event_count"], 5)
        self.assertEqual(result["features"]["count_CONFIRM_PAYMENT"], 1)

    # -----------------------------------------------------------------------
    # 5. Scoring Math & Types
    # -----------------------------------------------------------------------

    def test_18_anomaly_score_equals_negative_decision_function(self) -> None:
        test_sequences = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
            "FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CANCEL_ORDER",
        ]
        for seq in test_sequences:
            result = predict_anomaly(seq, model=self.model)
            expected_anomaly_score = -result["raw_decision_score"]
            self.assertAlmostEqual(result["anomaly_score"], expected_anomaly_score, places=9)

    def test_19_scores_are_ordinary_python_floats(self) -> None:
        result = predict_anomaly("CANCEL_ORDER", model=self.model)
        self.assertIs(type(result["anomaly_score"]), float)
        self.assertIs(type(result["raw_decision_score"]), float)
        self.assertNotIsInstance(result["anomaly_score"], np.generic)
        self.assertNotIsInstance(result["raw_decision_score"], np.generic)

    # -----------------------------------------------------------------------
    # 6. Reproducibility, Model Reuse & Batch Prediction
    # -----------------------------------------------------------------------

    def test_20_supplied_model_is_reused_without_disk_reload(self) -> None:
        with patch("backend.ml.inference.predictor.load_trained_model") as mock_load:
            predict_anomaly("CANCEL_ORDER", model=self.model)
            mock_load.assert_not_called()

        with patch("backend.ml.inference.predictor.load_trained_model", return_value=(self.model, self.metadata)) as mock_load:
            predict_anomaly("CANCEL_ORDER", model=None)
            mock_load.assert_called_once()

    def test_21_repeated_predictions_are_reproducible(self) -> None:
        seq = "FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        res1 = predict_anomaly(seq, model=self.model)
        res2 = predict_anomaly(seq, model=self.model)

        self.assertEqual(res1["prediction"], res2["prediction"])
        self.assertEqual(res1["is_ml_anomaly"], res2["is_ml_anomaly"])
        self.assertEqual(res1["anomaly_score"], res2["anomaly_score"])
        self.assertEqual(res1["raw_decision_score"], res2["raw_decision_score"])
        self.assertEqual(res1["features"], res2["features"])

    def test_22_batch_prediction(self) -> None:
        sequences = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
        ]
        batch_results = predict_anomaly_batch(sequences, model=self.model)
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results[0]["features"]["seq_length"], 4)
        self.assertEqual(batch_results[1]["features"]["seq_length"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
