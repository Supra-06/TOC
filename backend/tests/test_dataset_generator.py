"""Unit tests for the synthetic dataset generator.

Validates the dataset generation against all 12 criteria specified in Phase 4, Step 1.
Run with:
    python -m unittest backend/tests/test_dataset_generator.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
from typing import Dict, List, Set
import unittest

from backend.fsm.validator import validate_trace
from backend.ml.dataset.generator import (
    CSV_COLUMNS,
    N_NORMAL_TEST,
    N_NORMAL_TOTAL,
    N_NORMAL_TRAIN,
    N_RULE_VIOLATION,
    N_STATISTICAL_ANOMALY,
    TOTAL_TRACES,
    generate_dataset,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _events_from_row(row: Dict[str, str]) -> List[str]:
    seq = row["event_sequence"]
    return seq.split(";") if seq else []


class TestDatasetGenerator(unittest.TestCase):
    """Automated tests for synthetic dataset generator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.mkdtemp(prefix="toc_gen_test_")
        cls._csv_path = Path(cls._tmp_dir) / "orders_workflow_dataset.csv"
        generate_dataset(output_path=cls._csv_path)
        cls.rows: List[Dict[str, str]] = _read_csv(cls._csv_path)

    # 1. Exactly 5,000 rows are generated.
    def test_01_total_row_count_is_5000(self) -> None:
        self.assertEqual(len(self.rows), TOTAL_TRACES)

    # 2. All category and partition counts are correct.
    def test_02_category_and_partition_counts(self) -> None:
        normal_rows = [r for r in self.rows if r["anomaly_category"] == "NORMAL"]
        anomaly_rows = [r for r in self.rows if r["anomaly_category"] == "STATISTICAL_ANOMALY"]
        violation_rows = [r for r in self.rows if r["anomaly_category"] == "RULE_VIOLATION"]

        self.assertEqual(len(normal_rows), N_NORMAL_TOTAL)
        self.assertEqual(len(anomaly_rows), N_STATISTICAL_ANOMALY)
        self.assertEqual(len(violation_rows), N_RULE_VIOLATION)

        # Training partition (first 2,800 rows)
        train_partition = self.rows[:N_NORMAL_TRAIN]
        self.assertTrue(all(r["anomaly_category"] == "NORMAL" for r in train_partition))

        # Testing partition (next 700 normal rows)
        test_normal_partition = self.rows[N_NORMAL_TRAIN:N_NORMAL_TOTAL]
        self.assertTrue(all(r["anomaly_category"] == "NORMAL" for r in test_normal_partition))

    # 3. All trace IDs are unique.
    def test_03_all_trace_ids_are_unique(self) -> None:
        ids = [r["trace_id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    # 4. Every NORMAL trace is valid and complete.
    def test_04_every_normal_trace_is_valid_and_complete(self) -> None:
        for r in self.rows:
            if r["anomaly_category"] == "NORMAL":
                self.assertEqual(r["is_rule_valid"], "True")
                self.assertEqual(r["is_complete"], "True")
                # Explicitly validate every NORMAL trace using FSM validator
                res = validate_trace(_events_from_row(r))
                self.assertTrue(
                    res["is_valid"],
                    f"NORMAL trace {r['trace_id']} failed FSM validation: {res.get('error')}"
                )
                self.assertTrue(
                    res["is_complete"],
                    f"NORMAL trace {r['trace_id']} is incomplete (final_state={res.get('final_state')})"
                )

    # 5. Every NORMAL trace has at most one retry.
    def test_05_every_normal_trace_has_at_most_one_retry(self) -> None:
        for r in self.rows:
            if r["anomaly_category"] == "NORMAL":
                events = _events_from_row(r)
                retries = events.count("RETRY_PAYMENT")
                self.assertLessEqual(retries, 1, f"Trace {r['trace_id']} exceeded 1 retry.")

    # 6. Every STATISTICAL_ANOMALY trace is valid and complete.
    def test_06_every_statistical_anomaly_is_valid_and_complete(self) -> None:
        for r in self.rows:
            if r["anomaly_category"] == "STATISTICAL_ANOMALY":
                self.assertEqual(r["is_rule_valid"], "True")
                self.assertEqual(r["is_complete"], "True")
                # Explicitly validate every STATISTICAL_ANOMALY trace using FSM validator
                res = validate_trace(_events_from_row(r))
                self.assertTrue(
                    res["is_valid"],
                    f"STATISTICAL_ANOMALY trace {r['trace_id']} failed FSM validation: {res.get('error')}"
                )
                self.assertTrue(
                    res["is_complete"],
                    f"STATISTICAL_ANOMALY trace {r['trace_id']} is incomplete (final_state={res.get('final_state')})"
                )

    # 7. Every statistical anomaly has between 3 and 6 consecutive failure-retry cycles.
    def test_07_every_statistical_anomaly_has_3_to_6_cycles(self) -> None:
        valid_fulfillment_tail = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
        valid_cancellation_tail = ["CANCEL_ORDER"]

        for r in self.rows:
            if r["anomaly_category"] == "STATISTICAL_ANOMALY":
                events = _events_from_row(r)
                fail_count = events.count("FAIL_PAYMENT")
                retry_count = events.count("RETRY_PAYMENT")

                # Verify cycle bounds
                self.assertGreaterEqual(fail_count, 3, f"Trace {r['trace_id']} has < 3 cycles.")
                self.assertLessEqual(fail_count, 6, f"Trace {r['trace_id']} has > 6 cycles.")
                self.assertEqual(fail_count, retry_count, f"Trace {r['trace_id']} failure and retry counts mismatch.")

                k = fail_count
                prefix_len = 2 * k

                # Verify consecutive FAIL_PAYMENT -> RETRY_PAYMENT loop pattern
                for i in range(k):
                    self.assertEqual(
                        events[2 * i],
                        "FAIL_PAYMENT",
                        f"Trace {r['trace_id']} expected FAIL_PAYMENT at index {2 * i}, got {events[2 * i]}."
                    )
                    self.assertEqual(
                        events[2 * i + 1],
                        "RETRY_PAYMENT",
                        f"Trace {r['trace_id']} expected RETRY_PAYMENT at index {2 * i + 1}, got {events[2 * i + 1]}."
                    )

                # Verify valid termination tail after the 2k prefix
                tail = events[prefix_len:]
                is_fulfillment = (tail == valid_fulfillment_tail)
                is_cancellation = (tail == valid_cancellation_tail)
                self.assertTrue(
                    is_fulfillment or is_cancellation,
                    f"Trace {r['trace_id']} has invalid resolution tail: {tail}"
                )

    # 8. Every RULE_VIOLATION trace is invalid according to validate_trace().
    def test_08_every_rule_violation_trace_is_invalid(self) -> None:
        for r in self.rows:
            if r["anomaly_category"] == "RULE_VIOLATION":
                self.assertEqual(r["is_rule_valid"], "False")
                self.assertEqual(r["is_complete"], "False")
                # Explicitly verify with validate_trace()
                events = _events_from_row(r)
                res = validate_trace(events)
                self.assertFalse(res["is_valid"], f"Trace {r['trace_id']} unexpectedly valid: {events}")

    # 9. No identical event sequence receives contradictory category labels.
    def test_09_no_identical_sequence_has_contradictory_labels(self) -> None:
        seq_to_categories: Dict[str, Set[str]] = {}
        for r in self.rows:
            seq = r["event_sequence"]
            cat = r["anomaly_category"]
            seq_to_categories.setdefault(seq, set()).add(cat)

        conflicts = {seq: cats for seq, cats in seq_to_categories.items() if len(cats) > 1}
        self.assertEqual(len(conflicts), 0, f"Found conflicting labels: {conflicts}")

    # 10. The CSV has exactly the seven required columns.
    def test_10_csv_has_exactly_seven_required_columns(self) -> None:
        with open(self._csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
        self.assertEqual(fieldnames, CSV_COLUMNS)

    # 11. RULE_VIOLATION rows have empty/null is_ml_anomaly.
    def test_11_rule_violation_rows_have_empty_is_ml_anomaly(self) -> None:
        for r in self.rows:
            if r["anomaly_category"] == "RULE_VIOLATION":
                self.assertEqual(r["is_ml_anomaly"], "")

    # 12. Two generations using seed 42 produce identical CSV output.
    def test_12_reproducibility_with_seed_42(self) -> None:
        path_1 = Path(self._tmp_dir) / "repro_1.csv"
        path_2 = Path(self._tmp_dir) / "repro_2.csv"
        generate_dataset(output_path=path_1)
        generate_dataset(output_path=path_2)

        content_1 = path_1.read_bytes()
        content_2 = path_2.read_bytes()
        self.assertEqual(content_1, content_2, "Outputs with seed 42 were not identical.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

