"""Unit tests for the backend anomaly detection service (Phase 5).

Verifies:
1. AnomalyService instantiation and factory function get_anomaly_service().
2. get_service_status() returns OPERATIONAL (or DEGRADED) without triggering model loading.
3. Single order analysis with and without order_id.
4. Diagnostic enrichment across all four verdicts:
   - RULE_VIOLATION -> CRITICAL
   - STATISTICAL_ANOMALY -> WARNING (strictly derived from ML features)
   - NORMAL -> INFO
   - VALID_INCOMPLETE -> INFO
   - Unsupported verdict -> raises ValueError
5. Batch order analysis accepting strings, event lists, and structured dictionaries.
6. Batch summary metrics and anomaly rate calculation including incomplete orders.
7. Empty batch handling (0.0 anomaly rate).
8. Preservation of lazy model loading in batch analysis.
9. Full JSON-serializability of single and batch service responses.
10. Real model smoke test against production model artifact.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_anomaly_service.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

import numpy as np

from backend.services.anomaly_service import (
    AnomalyService,
    _build_diagnostics,
    get_anomaly_service,
)


# ---------------------------------------------------------------------------
# Controlled Deterministic Mock Models
# ---------------------------------------------------------------------------

class MockNormalModel:
    """Deterministic mock model that always predicts NORMAL (1)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.15])  # raw > 0 => anomaly_score <= 0 (NORMAL)


class MockAnomalyModel:
    """Deterministic mock model that always predicts ANOMALY (-1)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([-1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([-0.20])  # raw < 0 => anomaly_score > 0 (ANOMALY)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestAnomalyService(unittest.TestCase):
    """Test suite for AnomalyService and service orchestration layer."""

    def setUp(self) -> None:
        self.mock_normal = MockNormalModel()
        self.mock_anomaly = MockAnomalyModel()
        self.service_normal = AnomalyService(model=self.mock_normal)
        self.service_anomaly = AnomalyService(model=self.mock_anomaly)

    # -----------------------------------------------------------------------
    # 1. Service Factory and Instantiation
    # -----------------------------------------------------------------------

    def test_01_service_factory_and_initialization(self) -> None:
        # Default factory returns an AnomalyService instance
        service_default = get_anomaly_service()
        self.assertIsInstance(service_default, AnomalyService)

        # Factory with custom parameters returns new instance
        service_custom = get_anomaly_service(model=self.mock_normal)
        self.assertIsInstance(service_custom, AnomalyService)
        self.assertIsNot(service_default, service_custom)

    # -----------------------------------------------------------------------
    # 2. Service Status (Zero Model Loading)
    # -----------------------------------------------------------------------

    def test_02_get_service_status_does_not_trigger_model_loading(self) -> None:
        service = AnomalyService()

        with patch("backend.ml.inference.predictor.load_trained_model") as mock_load:
            status = service.get_service_status()
            # Must NOT load the model into memory
            mock_load.assert_not_called()

        self.assertEqual(status["service_status"], "OPERATIONAL")
        self.assertEqual(status["dataset_spec_version"], "1.4")
        self.assertTrue(status["model_artifact_exists"])
        self.assertFalse(status["model_loaded_in_memory"])
        self.assertIn("NORMAL", status["supported_verdicts"])
        self.assertIn("RULE_VIOLATION", status["supported_verdicts"])
        self.assertIn("STATISTICAL_ANOMALY", status["supported_verdicts"])
        self.assertIn("VALID_INCOMPLETE", status["supported_verdicts"])

        # Test DEGRADED status when model artifact does not exist and no model is injected
        service_missing = AnomalyService(model_path="non_existent_artifact.joblib")
        with patch("backend.ml.inference.predictor.load_trained_model") as mock_load_missing:
            status_missing = service_missing.get_service_status()
            mock_load_missing.assert_not_called()

        self.assertEqual(status_missing["service_status"], "DEGRADED")
        self.assertFalse(status_missing["model_artifact_exists"])
        self.assertFalse(status_missing["model_loaded_in_memory"])

    # -----------------------------------------------------------------------
    # 3. Single Order Analysis — NORMAL
    # -----------------------------------------------------------------------

    def test_03_single_order_analysis_normal_with_order_id(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        res = self.service_normal.analyze_order(seq, order_id="ORD-1001")

        self.assertEqual(res["order_id"], "ORD-1001")
        self.assertEqual(res["verdict"], "NORMAL")
        self.assertFalse(res["is_anomaly"])
        self.assertIsNone(res["anomaly_type"])

        # Diagnostic assertions
        diag = res["diagnostics"]
        self.assertEqual(diag["severity"], "INFO")
        self.assertIn("normally", diag["root_cause"].lower())
        self.assertIn("DELIVERED", diag["root_cause"])
        self.assertIn("no operational intervention", diag["recommended_action"].lower())
        self.assertIsNotNone(res["ml_evaluation"])
        self.assertIsNone(res["ml_skip_reason"])

    def test_04_single_order_analysis_without_order_id(self) -> None:
        seq = "CANCEL_ORDER"
        res = self.service_normal.analyze_order(seq)

        self.assertIsNone(res["order_id"])
        self.assertEqual(res["verdict"], "NORMAL")
        self.assertFalse(res["is_anomaly"])
        self.assertEqual(res["diagnostics"]["severity"], "INFO")

    # -----------------------------------------------------------------------
    # 4. Single Order Analysis — RULE_VIOLATION (CRITICAL)
    # -----------------------------------------------------------------------

    def test_05_rule_violation_diagnostic_enrichment_critical(self) -> None:
        # Illegal transition: skip packing (PAID -> SHIPPED)
        seq = "CONFIRM_PAYMENT;SHIP_ITEMS;DELIVER_ITEMS"
        res = self.service_normal.analyze_order(seq, order_id="ORD-ERR-01")

        self.assertEqual(res["order_id"], "ORD-ERR-01")
        self.assertEqual(res["verdict"], "RULE_VIOLATION")
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["anomaly_type"], "RULE_VIOLATION")

        # Diagnostics: CRITICAL
        diag = res["diagnostics"]
        self.assertEqual(diag["severity"], "CRITICAL")
        self.assertIn("INVALID_TRANSITION", diag["root_cause"])
        self.assertIn("SHIP_ITEMS", diag["root_cause"])
        self.assertIn("PAID", diag["root_cause"])
        self.assertIn("cannot be executed", diag["recommended_action"].lower())

        # ML must be skipped
        self.assertIsNone(res["ml_evaluation"])
        self.assertIsNotNone(res["ml_skip_reason"])

    # -----------------------------------------------------------------------
    # 5. Single Order Analysis — STATISTICAL_ANOMALY (WARNING)
    # -----------------------------------------------------------------------

    def test_06_statistical_anomaly_diagnostic_enrichment_warning(self) -> None:
        # Multi-retry sequence (3 cycles per DATASET_SPEC.md v1.4)
        seq = (
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        )
        res = self.service_anomaly.analyze_order(seq, order_id="ORD-STAT-01")

        self.assertEqual(res["order_id"], "ORD-STAT-01")
        self.assertEqual(res["verdict"], "STATISTICAL_ANOMALY")
        self.assertTrue(res["is_anomaly"])
        self.assertEqual(res["anomaly_type"], "STATISTICAL_ANOMALY")

        # Diagnostics: WARNING
        diag = res["diagnostics"]
        self.assertEqual(diag["severity"], "WARNING")
        self.assertIn("anomaly score", diag["root_cause"].lower())
        self.assertIn("checkout friction", diag["recommended_action"].lower())

        # Key factors must strictly reflect engineered ML feature values
        key_factors_text = " ".join(diag["key_factors"])
        self.assertIn("count_RETRY_PAYMENT", key_factors_text)
        self.assertIn("count_FAIL_PAYMENT", key_factors_text)
        self.assertIn("payment_failure_rate", key_factors_text)
        self.assertIn("of events represented by payment retries", key_factors_text)

        # ML must be evaluated
        self.assertIsNotNone(res["ml_evaluation"])
        self.assertIsNone(res["ml_skip_reason"])

    # -----------------------------------------------------------------------
    # 6. Single Order Analysis — VALID_INCOMPLETE (INFO) & Unsupported Verdict
    # -----------------------------------------------------------------------

    def test_07_valid_incomplete_diagnostic_enrichment_info(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS"
        res = self.service_normal.analyze_order(seq, order_id="ORD-INC-01")

        self.assertEqual(res["order_id"], "ORD-INC-01")
        self.assertEqual(res["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(res["is_anomaly"])
        self.assertIsNone(res["anomaly_type"])

        # Diagnostics: INFO
        diag = res["diagnostics"]
        self.assertEqual(diag["severity"], "INFO")
        self.assertIn("PACKED", diag["root_cause"])
        self.assertIn("await subsequent lifecycle", diag["recommended_action"].lower())

        # ML must be skipped
        self.assertIsNone(res["ml_evaluation"])
        self.assertIsNotNone(res["ml_skip_reason"])

    def test_08_unsupported_verdict_raises_value_error(self) -> None:
        # Verify that _build_diagnostics explicitly raises ValueError for unsupported verdicts
        invalid_analysis = {"verdict": "UNSUPPORTED_TEST_VERDICT"}
        with self.assertRaises(ValueError) as ctx:
            _build_diagnostics(invalid_analysis)
        self.assertIn("unsupported verdict", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 7. Batch Analysis — Empty Input
    # -----------------------------------------------------------------------

    def test_09_empty_batch_analysis(self) -> None:
        res = self.service_normal.analyze_order_batch([])

        self.assertEqual(res["results"], [])
        summary = res["summary"]
        self.assertEqual(summary["total_orders"], 0)
        self.assertEqual(summary["normal_count"], 0)
        self.assertEqual(summary["rule_violation_count"], 0)
        self.assertEqual(summary["statistical_anomaly_count"], 0)
        self.assertEqual(summary["valid_incomplete_count"], 0)
        self.assertEqual(summary["total_anomalies"], 0)
        self.assertEqual(summary["anomaly_rate"], 0.0)

    # -----------------------------------------------------------------------
    # 8. Batch Analysis — String Lists and Structured Dictionaries
    # -----------------------------------------------------------------------

    def test_10_batch_analysis_with_homogeneous_string_list(self) -> None:
        batch_strings = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
        ]
        res = self.service_normal.analyze_order_batch(batch_strings)

        self.assertEqual(len(res["results"]), 2)
        self.assertEqual(res["summary"]["total_orders"], 2)
        self.assertEqual(res["summary"]["normal_count"], 2)
        self.assertEqual(res["summary"]["anomaly_rate"], 0.0)

    def test_11_batch_analysis_with_structured_dictionaries(self) -> None:
        batch_dicts = [
            {"order_id": "ORD-1", "event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS"},  # Rule violation
            {"order_id": "ORD-2", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},  # Valid incomplete
            {"order_id": "ORD-3", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},  # Normal
        ]
        res = self.service_normal.analyze_order_batch(batch_dicts)

        self.assertEqual(len(res["results"]), 3)
        self.assertEqual(res["results"][0]["order_id"], "ORD-1")
        self.assertEqual(res["results"][0]["verdict"], "RULE_VIOLATION")

        self.assertEqual(res["results"][1]["order_id"], "ORD-2")
        self.assertEqual(res["results"][1]["verdict"], "VALID_INCOMPLETE")

        self.assertEqual(res["results"][2]["order_id"], "ORD-3")
        self.assertEqual(res["results"][2]["verdict"], "NORMAL")

    # -----------------------------------------------------------------------
    # 9. Batch Summary Metrics & Anomaly Rate
    # -----------------------------------------------------------------------

    def test_12_batch_summary_anomaly_rate_calculation(self) -> None:
        # Batch of 4 orders evaluated with service_anomaly (MockAnomalyModel):
        # - ORD-A: Completed valid fulfillment -> flagged as STATISTICAL_ANOMALY by MockAnomalyModel
        # - ORD-B: Invalid transition (skips PACK) -> flagged as RULE_VIOLATION (ML skipped)
        # - ORD-C: In-progress prefix (PACKED) -> flagged as VALID_INCOMPLETE (ML skipped, non-anomaly)
        # - ORD-D: Completed cancellation -> flagged as STATISTICAL_ANOMALY by MockAnomalyModel
        # Summary metrics:
        # - total_orders = 4 (denominator includes the incomplete order ORD-C)
        # - rule_violation_count = 1
        # - statistical_anomaly_count = 2
        # - valid_incomplete_count = 1
        # - normal_count = 0
        # - total_anomalies = 1 + 2 = 3
        # - anomaly_rate = 3 / 4 = 0.75 (75.0%)
        mixed_batch = [
            {"order_id": "ORD-A", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},
            {"order_id": "ORD-B", "event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS"},
            {"order_id": "ORD-C", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
            {"order_id": "ORD-D", "event_sequence": "CANCEL_ORDER"},
        ]

        res = self.service_anomaly.analyze_order_batch(mixed_batch)
        summary = res["summary"]

        self.assertEqual(summary["total_orders"], 4)
        self.assertEqual(summary["rule_violation_count"], 1)
        self.assertEqual(summary["valid_incomplete_count"], 1)
        self.assertEqual(summary["statistical_anomaly_count"], 2)
        self.assertEqual(summary["total_anomalies"], 3)
        self.assertAlmostEqual(summary["anomaly_rate"], 0.75)

    # -----------------------------------------------------------------------
    # 10. Lazy Model Loading in Batch
    # -----------------------------------------------------------------------

    def test_13_batch_analysis_preserves_lazy_model_loading(self) -> None:
        # Batch of only invalid and incomplete traces
        non_eligible_batch = [
            {"order_id": "ORD-1", "event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS"},
            {"order_id": "ORD-2", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
        ]
        service_uninitialized = AnomalyService()

        with patch("backend.pipeline.analyzer.load_trained_model") as mock_load:
            res = service_uninitialized.analyze_order_batch(non_eligible_batch)
            # Model loading must NOT be triggered
            mock_load.assert_not_called()

        self.assertEqual(res["summary"]["total_orders"], 2)
        self.assertEqual(res["summary"]["rule_violation_count"], 1)
        self.assertEqual(res["summary"]["valid_incomplete_count"], 1)

    # -----------------------------------------------------------------------
    # 11. Full JSON Serializability
    # -----------------------------------------------------------------------

    def test_14_all_responses_are_json_serializable(self) -> None:
        test_traces = [
            {"order_id": "ORD-1", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},
            {"order_id": "ORD-2", "event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS"},
            {"order_id": "ORD-3", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
            {"order_id": "ORD-4", "event_sequence": "FAIL_PAYMENT;RETRY_PAYMENT;CANCEL_ORDER"},
        ]

        batch_result = self.service_normal.analyze_order_batch(test_traces)
        # Must serialize cleanly without TypeError
        serialized = json.dumps(batch_result)
        self.assertIsInstance(serialized, str)

        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["summary"]["total_orders"], 4)

    # -----------------------------------------------------------------------
    # 12. Real Model Smoke Test
    # -----------------------------------------------------------------------

    def test_15_real_trained_model_smoke_test(self) -> None:
        # Smoke test using production service instance and production model artifact
        service = get_anomaly_service()

        res = service.analyze_order(
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            order_id="REAL-SMOKE-01",
        )
        self.assertEqual(res["order_id"], "REAL-SMOKE-01")
        self.assertIn(res["verdict"], ["NORMAL", "STATISTICAL_ANOMALY"])
        self.assertIn(res["diagnostics"]["severity"], ["INFO", "WARNING"])
        self.assertIsNotNone(res["ml_evaluation"])
        self.assertIsInstance(res["ml_evaluation"]["anomaly_score"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
