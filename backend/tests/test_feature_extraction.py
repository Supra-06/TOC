"""Unit tests for the workflow feature extraction module.

Verifies the 14 approved numerical features and their extraction logic against
all criteria in Phase 4, Step 2.

Run with:
    python -m unittest backend/tests/test_feature_extraction.py -v
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.ml.feature_extraction.extractor import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_features,
    extract_features_dict,
    extract_feature_matrix,
    parse_event_sequence,
)


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for feature extraction pipeline."""

    # 1. Exactly 14 features are extracted.
    def test_01_exactly_14_features_extracted(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        f_list = extract_features(seq)
        f_dict = extract_features_dict(seq)

        self.assertEqual(len(f_list), 14)
        self.assertEqual(len(f_dict), 14)
        self.assertEqual(NUM_FEATURES, 14)

    # 2. Feature names and order match the approved list.
    def test_02_feature_names_and_order_match_approved_list(self) -> None:
        expected_names = [
            "seq_length",
            "count_CONFIRM_PAYMENT",
            "count_FAIL_PAYMENT",
            "count_RETRY_PAYMENT",
            "count_CANCEL_ORDER",
            "count_PACK_ITEMS",
            "count_SHIP_ITEMS",
            "count_DELIVER_ITEMS",
            "count_REFUND_ORDER",
            "retry_ratio",
            "payment_failure_rate",
            "bigram_FAIL_to_RETRY",
            "bigram_PACK_to_SHIP",
            "unique_event_count",
        ]
        self.assertEqual(FEATURE_NAMES, expected_names)

        # Check that dictionary keys strictly follow the expected order
        f_dict = extract_features_dict("CANCEL_ORDER")
        self.assertEqual(list(f_dict.keys()), expected_names)

    # 3. Standard successful fulfillment sequence produces the correct feature values.
    def test_03_standard_fulfillment_sequence(self) -> None:
        # Happy Path: CONFIRM_PAYMENT -> PACK_ITEMS -> SHIP_ITEMS -> DELIVER_ITEMS
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"

        expected_dict = {
            "seq_length": 4,
            "count_CONFIRM_PAYMENT": 1,
            "count_FAIL_PAYMENT": 0,
            "count_RETRY_PAYMENT": 0,
            "count_CANCEL_ORDER": 0,
            "count_PACK_ITEMS": 1,
            "count_SHIP_ITEMS": 1,
            "count_DELIVER_ITEMS": 1,
            "count_REFUND_ORDER": 0,
            "retry_ratio": 0.0,
            "payment_failure_rate": 0.0,
            "bigram_FAIL_to_RETRY": 0,
            "bigram_PACK_to_SHIP": 1,
            "unique_event_count": 4,
        }
        expected_vector = [4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0]

        actual_dict = extract_features_dict(seq)
        actual_vector = extract_features(seq)

        self.assertEqual(actual_dict, expected_dict)
        self.assertEqual(actual_vector, expected_vector)

    # 4. Simple cancellation sequence produces the correct feature values.
    def test_04_simple_cancellation_sequence(self) -> None:
        seq = "CANCEL_ORDER"

        expected_dict = {
            "seq_length": 1,
            "count_CONFIRM_PAYMENT": 0,
            "count_FAIL_PAYMENT": 0,
            "count_RETRY_PAYMENT": 0,
            "count_CANCEL_ORDER": 1,
            "count_PACK_ITEMS": 0,
            "count_SHIP_ITEMS": 0,
            "count_DELIVER_ITEMS": 0,
            "count_REFUND_ORDER": 0,
            "retry_ratio": 0.0,
            "payment_failure_rate": 0.0,
            "bigram_FAIL_to_RETRY": 0,
            "bigram_PACK_to_SHIP": 0,
            "unique_event_count": 1,
        }
        expected_vector = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        actual_dict = extract_features_dict(seq)
        actual_vector = extract_features(seq)

        self.assertEqual(actual_dict, expected_dict)
        self.assertEqual(actual_vector, expected_vector)

    # 5. Retry sequence produces the correct retry count and ratio.
    def test_05_retry_sequence_counts_and_ratios(self) -> None:
        # 3 Retries + Fulfillment: N = 10
        seq = (
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "FAIL_PAYMENT;RETRY_PAYMENT;"
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        )

        expected_dict = {
            "seq_length": 10,
            "count_CONFIRM_PAYMENT": 1,
            "count_FAIL_PAYMENT": 3,
            "count_RETRY_PAYMENT": 3,
            "count_CANCEL_ORDER": 0,
            "count_PACK_ITEMS": 1,
            "count_SHIP_ITEMS": 1,
            "count_DELIVER_ITEMS": 1,
            "count_REFUND_ORDER": 0,
            "retry_ratio": 3 / 10,           # 0.3
            "payment_failure_rate": 3 / 4,   # 0.75
            "bigram_FAIL_to_RETRY": 3,
            "bigram_PACK_to_SHIP": 1,
            "unique_event_count": 6,
        }
        expected_vector = [10.0, 1.0, 3.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.3, 0.75, 3.0, 1.0, 6.0]

        actual_dict = extract_features_dict(seq)
        actual_vector = extract_features(seq)

        self.assertEqual(actual_dict, expected_dict)
        self.assertEqual(actual_vector, expected_vector)

    # 6. FAIL_PAYMENT -> RETRY_PAYMENT bigram is counted correctly.
    def test_06_fail_to_retry_bigram_counting(self) -> None:
        # 4 consecutive retry cycles
        seq = [
            "FAIL_PAYMENT", "RETRY_PAYMENT",
            "FAIL_PAYMENT", "RETRY_PAYMENT",
            "FAIL_PAYMENT", "RETRY_PAYMENT",
            "FAIL_PAYMENT", "RETRY_PAYMENT",
            "CANCEL_ORDER",
        ]
        f_dict = extract_features_dict(seq)
        self.assertEqual(f_dict["bigram_FAIL_to_RETRY"], 4)
        self.assertEqual(f_dict["bigram_PACK_to_SHIP"], 0)

    # 7. PACK_ITEMS -> SHIP_ITEMS bigram is counted correctly.
    def test_07_pack_to_ship_bigram_counting(self) -> None:
        seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
        f_dict = extract_features_dict(seq)
        self.assertEqual(f_dict["bigram_PACK_to_SHIP"], 1)

        # Duplicate PACK_ITEMS -> SHIP_ITEMS pattern (e.g. from an invalid trace)
        seq_repeated = "PACK_ITEMS;SHIP_ITEMS;PACK_ITEMS;SHIP_ITEMS"
        f_dict_rep = extract_features_dict(seq_repeated)
        self.assertEqual(f_dict_rep["bigram_PACK_to_SHIP"], 2)

    # 8. Non-adjacent events are not incorrectly counted as bigrams.
    def test_08_non_adjacent_events_not_counted_as_bigrams(self) -> None:
        # FAIL_PAYMENT and RETRY_PAYMENT separated by CONFIRM_PAYMENT
        seq1 = "FAIL_PAYMENT;CONFIRM_PAYMENT;RETRY_PAYMENT"
        f_dict1 = extract_features_dict(seq1)
        self.assertEqual(f_dict1["count_FAIL_PAYMENT"], 1)
        self.assertEqual(f_dict1["count_RETRY_PAYMENT"], 1)
        self.assertEqual(f_dict1["bigram_FAIL_to_RETRY"], 0)

        # Reverse order: RETRY_PAYMENT -> FAIL_PAYMENT
        seq2 = "RETRY_PAYMENT;FAIL_PAYMENT"
        f_dict2 = extract_features_dict(seq2)
        self.assertEqual(f_dict2["bigram_FAIL_to_RETRY"], 0)

        # PACK_ITEMS and SHIP_ITEMS separated by REFUND_ORDER
        seq3 = "PACK_ITEMS;REFUND_ORDER;SHIP_ITEMS"
        f_dict3 = extract_features_dict(seq3)
        self.assertEqual(f_dict3["bigram_PACK_to_SHIP"], 0)

    # 9. Empty sequence is handled safely.
    def test_09_empty_sequence_handled_safely(self) -> None:
        empty_inputs = ["", "   ", []]
        for empty_in in empty_inputs:
            with self.subTest(empty_input=empty_in):
                f_dict = extract_features_dict(empty_in)
                f_list = extract_features(empty_in)

                expected_zeros = {
                    "seq_length": 0,
                    "count_CONFIRM_PAYMENT": 0,
                    "count_FAIL_PAYMENT": 0,
                    "count_RETRY_PAYMENT": 0,
                    "count_CANCEL_ORDER": 0,
                    "count_PACK_ITEMS": 0,
                    "count_SHIP_ITEMS": 0,
                    "count_DELIVER_ITEMS": 0,
                    "count_REFUND_ORDER": 0,
                    "retry_ratio": 0.0,
                    "payment_failure_rate": 0.0,
                    "bigram_FAIL_to_RETRY": 0,
                    "bigram_PACK_to_SHIP": 0,
                    "unique_event_count": 0,
                }
                self.assertEqual(f_dict, expected_zeros)
                self.assertEqual(f_list, [0.0] * 14)
                # Verify all values are finite and non-NaN
                self.assertTrue(all(np.isfinite(f_list)))

    # 10. Multiple sequences produce a matrix with the correct dimensions.
    def test_10_feature_matrix_dimensions(self) -> None:
        sequences = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
            "CONFIRM_PAYMENT;REFUND_ORDER",
            "FAIL_PAYMENT;RETRY_PAYMENT;CANCEL_ORDER",
            "",
        ]
        matrix = extract_feature_matrix(sequences)

        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (5, 14))

        # Check empty batch
        empty_matrix = extract_feature_matrix([])
        self.assertEqual(empty_matrix.shape, (0, 14))

    # 11. The matrix contains only finite numerical values.
    def test_11_matrix_contains_only_finite_numerical_values(self) -> None:
        sequences = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
            "",
            "FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
        ]
        matrix = extract_feature_matrix(sequences)

        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertFalse(np.any(np.isnan(matrix)))
        self.assertFalse(np.any(np.isinf(matrix)))
        self.assertEqual(matrix.dtype, np.float64)

    # 12. No identifiers or labels are included in the extracted features.
    def test_12_no_identifiers_or_labels_in_features(self) -> None:
        forbidden_substrings = [
            "trace_id",
            "is_rule_valid",
            "is_complete",
            "anomaly_category",
            "is_ml_anomaly",
            "label",
            "target",
        ]
        for name in FEATURE_NAMES:
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden,
                    name.lower(),
                    f"Forbidden label or metadata key '{forbidden}' found in feature name '{name}'."
                )

    # 13. extract_feature_matrix accepts NumPy array and optional Pandas Series.
    def test_13_extract_feature_matrix_accepts_numpy_array_and_series(self) -> None:
        raw_seqs = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            "CANCEL_ORDER",
            "CONFIRM_PAYMENT;REFUND_ORDER",
        ]
        # Test with NumPy 1D array of strings
        np_seqs = np.array(raw_seqs, dtype=object)
        matrix_from_np = extract_feature_matrix(np_seqs)

        self.assertIsInstance(matrix_from_np, np.ndarray)
        self.assertEqual(matrix_from_np.shape, (3, 14))
        self.assertTrue(np.all(np.isfinite(matrix_from_np)))

        # Verify values match normal list input
        expected_matrix = extract_feature_matrix(raw_seqs)
        np.testing.assert_array_equal(matrix_from_np, expected_matrix)

        # Test empty NumPy array
        empty_np = np.array([], dtype=object)
        empty_matrix_from_np = extract_feature_matrix(empty_np)
        self.assertEqual(empty_matrix_from_np.shape, (0, 14))

        # If pandas is available, also test with pd.Series
        try:
            import pandas as pd
            series_seqs = pd.Series(raw_seqs)
            matrix_from_series = extract_feature_matrix(series_seqs)
            self.assertIsInstance(matrix_from_series, np.ndarray)
            self.assertEqual(matrix_from_series.shape, (3, 14))
            np.testing.assert_array_equal(matrix_from_series, expected_matrix)

            empty_series = pd.Series([], dtype=object)
            empty_matrix_from_series = extract_feature_matrix(empty_series)
            self.assertEqual(empty_matrix_from_series.shape, (0, 14))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

