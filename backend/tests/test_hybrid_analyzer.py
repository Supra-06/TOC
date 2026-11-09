"""Unit tests for the hybrid workflow analyzer (Phase 4, Step 6).

Verifies:
1. Strict FSM gating: FSM-invalid traces yield RULE_VIOLATION and skip ML.
2. Completeness gating: Incomplete valid traces yield VALID_INCOMPLETE and skip ML.
3. ML evaluation on eligible traces: Completed valid traces yield NORMAL or STATISTICAL_ANOMALY.
4. Statistical anomaly traces adhere to Dataset Specification v1.4 (>= 3 retry cycles).
5. Deterministic integration tests using controlled mock models.
6. Lazy model loading in batch analysis (model not loaded if no trace is eligible; loaded once if eligible).
7. JSON-serializability of all returned outputs.
8. Real-model smoke test independent of assumed classifications.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_hybrid_analyzer.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import patch

import numpy as np

from backend.pipeline.analyzer import analyze_trace, analyze_trace_batch


# ---------------------------------------------------------------------------
# Controlled Deterministic Mock Models
# ---------------------------------------------------------------------------

class MockNormalModel:
    """Deterministic mock model that always predicts NORMAL (1) with positive decision score."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.15])  # raw > 0 => anomaly_score <= 0 (NORMAL)


class MockAnomalyModel:
    """Deterministic mock model that always predicts ANOMALY (-1) with negative decision score."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([-1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([-0.20])  # raw < 0 => anomaly_score > 0 (ANOMALY)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestHybridAnalyzer(unittest.TestCase):
    """Test suite for the hybrid workflow analyzer."""

    def setUp(self) -> None:
        self.mock_normal = MockNormalModel()
        self.mock_anomaly = MockAnomalyModel()

    # -----------------------------------------------------------------------
    # 1. Standard Normal Completed Traces
    # -----------------------------------------------------------------------

    def test_01_standard_normal_completed_trace(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "NORMAL")
        self.assertFalse(result["is_anomaly"])
        self.assertIsNone(result["anomaly_type"])
        self.assertIn("normally", result["summary"].lower())

        # FSM evaluation asserts
        fsm = result["fsm_evaluation"]
        self.assertTrue(fsm["is_rule_valid"])
        self.assertTrue(fsm["is_complete"])
        self.assertEqual(fsm["final_state"], "DELIVERED")
        self.assertIsNone(fsm["error"])

        # ML evaluation asserts
        ml = result["ml_evaluation"]
        self.assertIsNotNone(ml)
        self.assertEqual(ml["prediction"], "NORMAL")
        self.assertFalse(ml["is_ml_anomaly"])
        self.assertLessEqual(ml["anomaly_score"], 0.0)
        self.assertEqual(len(ml["features"]), 14)
        self.assertIsNone(result["ml_skip_reason"])

    def test_02_normal_cancellation_completed_trace(self) -> None:
        seq = "CANCEL_ORDER"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "NORMAL")
        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["fsm_evaluation"]["final_state"], "CANCELLED")
        self.assertIsNotNone(result["ml_evaluation"])
        self.assertIsNone(result["ml_skip_reason"])

    # -----------------------------------------------------------------------
    # 2. Statistical Anomaly Traces (Aligned with DATASET_SPEC.md v1.4)
    # -----------------------------------------------------------------------

    def test_03_statistical_anomaly_completed_trace(self) -> None:
        # DATASET_SPEC.md v1.4 Section 4.1.3:
        # Statistical anomalies contain >= 3 consecutive FAIL_PAYMENT -> RETRY_PAYMENT cycles
        # followed by valid fulfillment or cancellation.
        # Test Case A: 3 retry cycles followed by fulfillment
        seq_fulfillment = (
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        )
        res_fulfillment = analyze_trace(seq_fulfillment, model=self.mock_anomaly)

        self.assertEqual(res_fulfillment["verdict"], "STATISTICAL_ANOMALY")
        self.assertTrue(res_fulfillment["is_anomaly"])
        self.assertEqual(res_fulfillment["anomaly_type"], "STATISTICAL_ANOMALY")
        self.assertIn("statistical anomaly", res_fulfillment["summary"].lower())
        self.assertTrue(res_fulfillment["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(res_fulfillment["fsm_evaluation"]["is_complete"])
        self.assertEqual(res_fulfillment["fsm_evaluation"]["final_state"], "DELIVERED")
        self.assertIsNotNone(res_fulfillment["ml_evaluation"])
        self.assertTrue(res_fulfillment["ml_evaluation"]["is_ml_anomaly"])
        self.assertGreater(res_fulfillment["ml_evaluation"]["anomaly_score"], 0.0)

        # Test Case B: 3 retry cycles followed by cancellation
        seq_cancellation = (
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "CANCEL_ORDER"
        )
        res_cancellation = analyze_trace(seq_cancellation, model=self.mock_anomaly)

        self.assertEqual(res_cancellation["verdict"], "STATISTICAL_ANOMALY")
        self.assertTrue(res_cancellation["is_anomaly"])
        self.assertEqual(res_cancellation["anomaly_type"], "STATISTICAL_ANOMALY")
        self.assertTrue(res_cancellation["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(res_cancellation["fsm_evaluation"]["is_complete"])
        self.assertEqual(res_cancellation["fsm_evaluation"]["final_state"], "CANCELLED")

    # -----------------------------------------------------------------------
    # 3. Rule Violations (FSM Priority & ML Skipped)
    # -----------------------------------------------------------------------

    def test_04_rule_violation_invalid_transition_skips_ml(self) -> None:
        # Illegal transition: skip packing (PAID -> SHIPPED)
        seq = "CONFIRM_PAYMENT;SHIP_ITEMS;DELIVER_ITEMS"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "RULE_VIOLATION")
        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], "RULE_VIOLATION")
        self.assertIn("rule violation", result["summary"].lower())

        # FSM evaluation asserts
        fsm = result["fsm_evaluation"]
        self.assertFalse(fsm["is_rule_valid"])
        self.assertFalse(fsm["is_complete"])
        self.assertIsNotNone(fsm["error"])
        self.assertEqual(fsm["error"]["error_type"], "INVALID_TRANSITION")
        self.assertEqual(fsm["error"]["failed_event"], "SHIP_ITEMS")

        # ML evaluation must be null when skipped
        self.assertIsNone(result["ml_evaluation"])
        self.assertIsNotNone(result["ml_skip_reason"])
        self.assertIn("failed fsm rule validation", result["ml_skip_reason"].lower())

    def test_05_rule_violation_unknown_event_skips_ml(self) -> None:
        seq = "CONFIRM_PAYMENT;HACK_CHECKOUT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "RULE_VIOLATION")
        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], "RULE_VIOLATION")
        self.assertEqual(result["fsm_evaluation"]["error"]["error_type"], "UNKNOWN_EVENT")
        self.assertIsNone(result["ml_evaluation"])
        self.assertIsNotNone(result["ml_skip_reason"])

    def test_06_rule_violation_post_terminal_action_skips_ml(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS;CANCEL_ORDER"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "RULE_VIOLATION")
        self.assertEqual(result["fsm_evaluation"]["error"]["error_type"], "TERMINAL_STATE_VIOLATION")
        self.assertIsNone(result["ml_evaluation"])
        self.assertIsNotNone(result["ml_skip_reason"])

    def test_07_fsm_priority_over_ml_prediction(self) -> None:
        # Even if a model would predict NORMAL or ANOMALY, FSM invalidity strictly governs
        seq = "CONFIRM_PAYMENT;SHIP_ITEMS"

        # Passing MockNormalModel
        res_with_normal = analyze_trace(seq, model=self.mock_normal)
        self.assertEqual(res_with_normal["verdict"], "RULE_VIOLATION")
        self.assertIsNone(res_with_normal["ml_evaluation"])

        # Passing MockAnomalyModel
        res_with_anomaly = analyze_trace(seq, model=self.mock_anomaly)
        self.assertEqual(res_with_anomaly["verdict"], "RULE_VIOLATION")
        self.assertIsNone(res_with_anomaly["ml_evaluation"])

    # -----------------------------------------------------------------------
    # 4. Valid Incomplete Traces (ML Skipped)
    # -----------------------------------------------------------------------

    def test_08_valid_incomplete_prefix_skips_ml(self) -> None:
        # Valid workflow prefix in progress (at PACKED)
        seq = "CONFIRM_PAYMENT;PACK_ITEMS"
        result = analyze_trace(seq, model=self.mock_normal)

        self.assertEqual(result["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(result["is_anomaly"])
        self.assertIsNone(result["anomaly_type"])
        self.assertIn("in-progress", result["summary"].lower())

        # FSM evaluation asserts
        fsm = result["fsm_evaluation"]
        self.assertTrue(fsm["is_rule_valid"])
        self.assertFalse(fsm["is_complete"])
        self.assertEqual(fsm["final_state"], "PACKED")
        self.assertIsNone(fsm["error"])

        # ML evaluation must be null when skipped
        self.assertIsNone(result["ml_evaluation"])
        self.assertIsNotNone(result["ml_skip_reason"])
        self.assertIn("in-progress", result["ml_skip_reason"].lower())

    def test_09_incomplete_trace_with_retries_is_valid_incomplete(self) -> None:
        # Valid sequence with payment retries, but not complete (currently at ORDER_CREATED)
        seq = "FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT"
        result = analyze_trace(seq, model=self.mock_anomaly)

        # Must NOT be classified as STATISTICAL_ANOMALY because it is incomplete
        self.assertEqual(result["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(result["is_anomaly"])
        self.assertIsNone(result["anomaly_type"])
        self.assertIsNone(result["ml_evaluation"])
        self.assertIsNotNone(result["ml_skip_reason"])

    # -----------------------------------------------------------------------
    # 5. Input Formatting & Single-Trace Model Reuse
    # -----------------------------------------------------------------------

    def test_10_string_and_list_inputs_yield_identical_results(self) -> None:
        seq_str = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        seq_list = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]

        res_str = analyze_trace(seq_str, model=self.mock_normal)
        res_list = analyze_trace(seq_list, model=self.mock_normal)

        self.assertEqual(res_str["verdict"], res_list["verdict"])
        self.assertEqual(res_str["is_anomaly"], res_list["is_anomaly"])
        self.assertEqual(res_str["summary"], res_list["summary"])
        self.assertEqual(res_str["fsm_evaluation"], res_list["fsm_evaluation"])
        self.assertEqual(res_str["ml_evaluation"], res_list["ml_evaluation"])

    def test_11_supplied_model_is_reused_without_disk_reload(self) -> None:
        with patch("backend.ml.inference.predictor.load_trained_model") as mock_load:
            analyze_trace("CONFIRM_PAYMENT;CANCEL_ORDER", model=self.mock_normal)
            mock_load.assert_not_called()

    # -----------------------------------------------------------------------
    # 6. Lazy Model Loading in Batch Analysis
    # -----------------------------------------------------------------------

    def test_12_batch_with_only_rule_violations_does_not_load_model(self) -> None:
        # Batch of only invalid traces: model must NOT be loaded
        invalid_batch = [
            "CONFIRM_PAYMENT;SHIP_ITEMS",
            "PACK_ITEMS;CONFIRM_PAYMENT",
            "CONFIRM_PAYMENT;HACK_CHECKOUT",
        ]
        with patch("backend.pipeline.analyzer.load_trained_model") as mock_load:
            results = analyze_trace_batch(invalid_batch, model=None)
            mock_load.assert_not_called()

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["verdict"], "RULE_VIOLATION")
            self.assertIsNone(r["ml_evaluation"])

    def test_13_batch_with_only_valid_incomplete_traces_does_not_load_model(self) -> None:
        # Batch of only incomplete traces: model must NOT be loaded
        incomplete_batch = [
            "CONFIRM_PAYMENT",
            "CONFIRM_PAYMENT;PACK_ITEMS",
            "FAIL_PAYMENT;RETRY_PAYMENT",
        ]
        with patch("backend.pipeline.analyzer.load_trained_model") as mock_load:
            results = analyze_trace_batch(incomplete_batch, model=None)
            mock_load.assert_not_called()

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["verdict"], "VALID_INCOMPLETE")
            self.assertIsNone(r["ml_evaluation"])

    def test_14_mixed_batch_loads_model_exactly_once_and_reuses_it(self) -> None:
        # Mixed batch: rule violations, valid incomplete, and eligible completed traces
        mixed_batch = [
            "CONFIRM_PAYMENT;SHIP_ITEMS",                          # Rule violation (skips ML)
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",  # Eligible for ML #1
            "CONFIRM_PAYMENT;PACK_ITEMS",                          # Valid incomplete (skips ML)
            "CANCEL_ORDER",                                        # Eligible for ML #2
            "CONFIRM_PAYMENT;HACK_CHECKOUT",                       # Rule violation (skips ML)
        ]

        with patch("backend.pipeline.analyzer.load_trained_model", return_value=(self.mock_normal, {})) as mock_load:
            results = analyze_trace_batch(mixed_batch, model=None)
            # Model should be loaded exactly once for the batch
            mock_load.assert_called_once()

        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["verdict"], "RULE_VIOLATION")
        self.assertIsNone(results[0]["ml_evaluation"])

        self.assertEqual(results[1]["verdict"], "NORMAL")
        self.assertIsNotNone(results[1]["ml_evaluation"])

        self.assertEqual(results[2]["verdict"], "VALID_INCOMPLETE")
        self.assertIsNone(results[2]["ml_evaluation"])

        self.assertEqual(results[3]["verdict"], "NORMAL")
        self.assertIsNotNone(results[3]["ml_evaluation"])

        self.assertEqual(results[4]["verdict"], "RULE_VIOLATION")
        self.assertIsNone(results[4]["ml_evaluation"])

    # -----------------------------------------------------------------------
    # 7. JSON-Serializability Guarantee
    # -----------------------------------------------------------------------

    def test_15_all_results_are_json_serializable(self) -> None:
        test_sequences = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
            "CONFIRM_PAYMENT;SHIP_ITEMS;DELIVER_ITEMS",
            "CONFIRM_PAYMENT;PACK_ITEMS",
            "CONFIRM_PAYMENT;HACK_CHECKOUT;PACK_ITEMS",
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS;CANCEL_ORDER",
        ]

        for seq in test_sequences:
            with self.subTest(seq=seq):
                result = analyze_trace(seq, model=self.mock_normal)
                # Ensure json.dumps succeeds without TypeError or OverflowError
                serialized = json.dumps(result)
                self.assertIsInstance(serialized, str)
                deserialized = json.loads(serialized)
                self.assertEqual(deserialized["verdict"], result["verdict"])

    # -----------------------------------------------------------------------
    # 8. Real Model Smoke Test (Independent of Assumed Classifications)
    # -----------------------------------------------------------------------

    def test_16_real_trained_model_smoke_test(self) -> None:
        # Smoke test using the actual saved model artifact from Phase 4 Step 3
        # Verifies:
        # - The model loads successfully.
        # - Eligible completed traces receive ML evaluations with finite numerical scores.
        # - The returned verdict agrees with the actual is_ml_anomaly prediction.
        # - Invalid and incomplete traces skip ML.

        # 1. Eligible completed trace
        completed_seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        res_completed = analyze_trace(completed_seq)

        self.assertTrue(res_completed["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(res_completed["fsm_evaluation"]["is_complete"])
        self.assertIsNotNone(res_completed["ml_evaluation"])
        self.assertIsNone(res_completed["ml_skip_reason"])

        ml_eval = res_completed["ml_evaluation"]
        self.assertTrue(np.isfinite(ml_eval["anomaly_score"]))
        self.assertTrue(np.isfinite(ml_eval["raw_decision_score"]))
        self.assertIsInstance(ml_eval["is_ml_anomaly"], bool)
        self.assertIn(ml_eval["prediction"], ["NORMAL", "ANOMALY"])

        # Verify agreement between verdict and actual model prediction
        if ml_eval["is_ml_anomaly"]:
            self.assertEqual(res_completed["verdict"], "STATISTICAL_ANOMALY")
            self.assertTrue(res_completed["is_anomaly"])
            self.assertEqual(res_completed["anomaly_type"], "STATISTICAL_ANOMALY")
            self.assertGreater(ml_eval["anomaly_score"], 0.0)
        else:
            self.assertEqual(res_completed["verdict"], "NORMAL")
            self.assertFalse(res_completed["is_anomaly"])
            self.assertIsNone(res_completed["anomaly_type"])
            self.assertLessEqual(ml_eval["anomaly_score"], 0.0)

        # 2. Invalid trace: must skip ML
        res_invalid = analyze_trace("CONFIRM_PAYMENT;SHIP_ITEMS")
        self.assertEqual(res_invalid["verdict"], "RULE_VIOLATION")
        self.assertTrue(res_invalid["is_anomaly"])
        self.assertEqual(res_invalid["anomaly_type"], "RULE_VIOLATION")
        self.assertIsNone(res_invalid["ml_evaluation"])
        self.assertIsNotNone(res_invalid["ml_skip_reason"])

        # 3. Incomplete trace: must skip ML
        res_incomplete = analyze_trace("CONFIRM_PAYMENT;PACK_ITEMS")
        self.assertEqual(res_incomplete["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(res_incomplete["is_anomaly"])
        self.assertIsNone(res_incomplete["anomaly_type"])
        self.assertIsNone(res_incomplete["ml_evaluation"])
        self.assertIsNotNone(res_incomplete["ml_skip_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
