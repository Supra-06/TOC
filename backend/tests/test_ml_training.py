"""Unit tests for the Isolation Forest training module.

Verifies the 12 training requirements in Phase 4, Step 3.
All tests use temporary directories and temporary CSV copies to ensure
no project dataset or artifact files are overwritten during testing.

Run with:
    python -m unittest backend/tests/test_ml_training.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from typing import Dict, List
import unittest

import joblib
from sklearn.ensemble import IsolationForest

from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.ml.training.train import (
    DEFAULT_DATASET_PATH,
    MODEL_CONFIG,
    N_TRAINING_SAMPLES,
    SPEC_VERSION,
    TOTAL_EXPECTED_ROWS,
    load_model,
    train_model,
)


def _copy_dataset_rows(source_path: Path) -> List[Dict[str, str]]:
    with open(source_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_temp_dataset(dest_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(dest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestMLTraining(unittest.TestCase):
    """Test suite for Isolation Forest training pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.mkdtemp(prefix="toc_train_test_")
        cls._tmp_model_path = Path(cls._tmp_dir) / "test_isolation_forest.joblib"
        cls.train_result = train_model(
            dataset_path=DEFAULT_DATASET_PATH,
            model_path=cls._tmp_model_path,
        )

    # 1. Training uses exactly 2,800 NORMAL rows.
    def test_01_training_uses_exactly_2800_normal_rows(self) -> None:
        self.assertEqual(self.train_result["training_sample_count"], N_TRAINING_SAMPLES)
        self.assertEqual(N_TRAINING_SAMPLES, 2800)

        # Inspect the fitted model's sample count attribute
        model = self.train_result["model"]
        # In scikit-learn IsolationForest, n_samples_ or estimators_ samples matches training count
        if hasattr(model, "n_samples_"):
            self.assertEqual(model.n_samples_, 2800)

    # 2. The model configuration matches the approved values.
    def test_02_model_configuration_matches_approved_values(self) -> None:
        model: IsolationForest = self.train_result["model"]
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.contamination, "auto")
        self.assertEqual(model.random_state, 42)

        meta = self.train_result["metadata"]
        self.assertEqual(meta["model_name"], "IsolationForest")
        self.assertEqual(meta["n_estimators"], 100)
        self.assertEqual(meta["contamination"], "auto")
        self.assertEqual(meta["random_state"], 42)

    # 3. The feature names and order match FEATURE_NAMES.
    def test_03_feature_names_and_order_match_approved_list(self) -> None:
        meta = self.train_result["metadata"]
        self.assertEqual(meta["feature_names"], FEATURE_NAMES)
        self.assertEqual(self.train_result["feature_names"], FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 14)

    # 4. The model artifact is saved.
    def test_04_model_artifact_is_saved(self) -> None:
        self.assertTrue(self._tmp_model_path.exists())
        self.assertGreater(self._tmp_model_path.stat().st_size, 0)

        # Metadata JSON file must also be saved alongside
        metadata_json = self._tmp_model_path.parent / f"{self._tmp_model_path.stem}_metadata.json"
        self.assertTrue(metadata_json.exists())

    # 5. The saved model can be loaded using joblib.
    def test_05_saved_model_can_be_loaded_using_joblib(self) -> None:
        loaded_raw = joblib.load(self._tmp_model_path)
        self.assertIsInstance(loaded_raw, dict)
        self.assertIn("model", loaded_raw)
        self.assertIn("metadata", loaded_raw)
        self.assertIsInstance(loaded_raw["model"], IsolationForest)

        # Also test via helper load_model
        loaded_model, loaded_meta = load_model(self._tmp_model_path)
        self.assertIsInstance(loaded_model, IsolationForest)
        self.assertIsInstance(loaded_meta, dict)

    # 6. Metadata contains the required fields and values.
    def test_06_metadata_contains_required_fields_and_values(self) -> None:
        meta = self.train_result["metadata"]
        required_keys = [
            "model_name",
            "n_estimators",
            "contamination",
            "random_state",
            "training_sample_count",
            "feature_names",
            "dataset_spec_version",
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Missing required metadata key: '{key}'")

        self.assertEqual(meta["model_name"], "IsolationForest")
        self.assertEqual(meta["n_estimators"], 100)
        self.assertEqual(meta["contamination"], "auto")
        self.assertEqual(meta["random_state"], 42)
        self.assertEqual(meta["training_sample_count"], 2800)
        self.assertEqual(meta["dataset_spec_version"], SPEC_VERSION)

    # 7. Training rejects a dataset with an incorrect total row count.
    def test_07_rejects_incorrect_total_row_count(self) -> None:
        rows = _copy_dataset_rows(DEFAULT_DATASET_PATH)
        # Truncate to 4,999 rows
        short_rows = rows[:-1]
        temp_csv = Path(self._tmp_dir) / "short_dataset.csv"
        _write_temp_dataset(temp_csv, short_rows)

        with self.assertRaises(ValueError) as ctx:
            train_model(dataset_path=temp_csv, model_path=Path(self._tmp_dir) / "dummy.joblib")
        self.assertIn("row count mismatch", str(ctx.exception).lower())

    # 8. Training rejects a dataset with an incorrect category partition.
    def test_08_rejects_incorrect_category_partition(self) -> None:
        rows = _copy_dataset_rows(DEFAULT_DATASET_PATH)
        # Corrupt partition: change an anomaly test row (row 3501) to RULE_VIOLATION
        rows[3500]["anomaly_category"] = "RULE_VIOLATION"
        temp_csv = Path(self._tmp_dir) / "corrupt_partition.csv"
        _write_temp_dataset(temp_csv, rows)

        with self.assertRaises(ValueError) as ctx:
            train_model(dataset_path=temp_csv, model_path=Path(self._tmp_dir) / "dummy.joblib")
        self.assertIn("partition violation", str(ctx.exception).lower())

    # 9. Training rejects a dataset where the first 2,800 rows are not all NORMAL.
    def test_09_rejects_non_normal_in_training_partition(self) -> None:
        rows = _copy_dataset_rows(DEFAULT_DATASET_PATH)
        # Invert row 500 in training partition to STATISTICAL_ANOMALY
        rows[499]["anomaly_category"] = "STATISTICAL_ANOMALY"
        temp_csv = Path(self._tmp_dir) / "non_normal_train.csv"
        _write_temp_dataset(temp_csv, rows)

        with self.assertRaises(ValueError) as ctx:
            train_model(dataset_path=temp_csv, model_path=Path(self._tmp_dir) / "dummy.joblib")
        self.assertIn("training partition violation", str(ctx.exception).lower())

    # 10. Training does not use any label or identifier columns as features.
    def test_10_no_labels_or_identifiers_used_as_features(self) -> None:
        meta = self.train_result["metadata"]
        features = meta["feature_names"]
        forbidden = [
            "trace_id",
            "is_rule_valid",
            "is_complete",
            "anomaly_category",
            "is_ml_anomaly",
        ]
        for f in features:
            self.assertNotIn(
                f,
                forbidden,
                f"Forbidden column '{f}' found in model feature set!"
            )

    # 11. Training is reproducible with the fixed random seed.
    def test_11_reproducible_training_with_fixed_seed(self) -> None:
        model_path_1 = Path(self._tmp_dir) / "repro_model_1.joblib"
        model_path_2 = Path(self._tmp_dir) / "repro_model_2.joblib"

        res1 = train_model(dataset_path=DEFAULT_DATASET_PATH, model_path=model_path_1)
        res2 = train_model(dataset_path=DEFAULT_DATASET_PATH, model_path=model_path_2)

        # Generate sample test input (14 features)
        import numpy as np
        test_X = np.ones((10, 14), dtype=np.float64)

        scores1 = res1["model"].decision_function(test_X)
        scores2 = res2["model"].decision_function(test_X)

        np.testing.assert_array_equal(
            scores1,
            scores2,
            err_msg="Models trained with same seed produced divergent decision scores!"
        )

    # 12. Training fails clearly if the dataset is missing or malformed.
    def test_12_fails_clearly_on_missing_or_malformed_dataset(self) -> None:
        # Non-existent file
        non_existent = Path(self._tmp_dir) / "does_not_exist.csv"
        with self.assertRaises(FileNotFoundError):
            train_model(dataset_path=non_existent, model_path=Path(self._tmp_dir) / "dummy.joblib")

        # Missing required columns
        malformed_csv = Path(self._tmp_dir) / "malformed.csv"
        with open(malformed_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["trace_id", "wrong_column"])
            writer.writerow(["TRACE_00001", "foo"])

        with self.assertRaises(ValueError) as ctx:
            train_model(dataset_path=malformed_csv, model_path=Path(self._tmp_dir) / "dummy.joblib")
        self.assertIn("missing required columns", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

