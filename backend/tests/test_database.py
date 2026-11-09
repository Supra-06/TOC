"""Unit and integration tests for SQLite and SQLAlchemy persistence (Phase 6).

Verifies:
 1. Table creation and metadata initialization.
 2. String input preservation (delimited format).
 3. List input preservation (event list format).
 4. Empty string preservation ("").
 5. Empty list preservation ([]).
 6. Whitespace-only string preservation ("   ").
 7. Lists containing blank strings (["", "   "]).
 8. Rejection of tuples and unsupported types.
 9. Rejection of lists containing non-string elements.
10. UTC-aware timestamp normalization.
11. Non-UTC timezone conversion to UTC-naive.
12. Naive timestamp treatment as UTC by contract.
13. All four verdict types (NORMAL, RULE_VIOLATION, STATISTICAL_ANOMALY, VALID_INCOMPLETE).
14. ML prediction and anomaly-flag consistency.
15. Conditional nullability of ML fields when skipped vs. evaluated.
16. Database-level CheckConstraint enforcement.
17. Single-record persistence and retrieval.
18. Successful mixed-verdict atomic batch persistence.
19. Batch length mismatch rejection prior to database access.
20. Genuine SQLite trigger-based mid-batch transaction rollback.
21. Deterministic historical ordering with identical timestamps.
22. Latest analysis retrieval by order ID.
23. Pagination with skip and limit.
24. Filtering by verdict and is_anomaly.
25. Repository argument validation (negative skip, non-positive limit, invalid verdict).
26. Summary metric aggregations across analyses and unique orders.
27. Empty-database summary metric handling.
28. Strict ML skip-reason validation (non-empty string required when skipped, None when evaluated).
29. Real AnomalyService integration: valid incomplete trace ([CONFIRM_PAYMENT]).
30. Real AnomalyService integration: rule violation trace ([PACK_ITEMS]).
31. Strict ML-evaluated field integrity validation (missing/non-boolean/non-finite values rejected).

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_database.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import unittest

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import init_db
from backend.database.models import Base, WorkflowAnalysis
from backend.database.repository import (
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_latest_analysis_by_order_id,
    get_summary_metrics,
    list_analyses,
    to_naive_utc,
    validate_and_classify_input,
)
from backend.fsm.states import OrderEvent, OrderState
from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.services.anomaly_service import AnomalyService


# ---------------------------------------------------------------------------
# Test Fixture Helpers Using Authoritative Project States and Events
# ---------------------------------------------------------------------------

def _make_14_features(seq_length: int = 4, count_retry: int = 0) -> Dict[str, Union[int, float]]:
    return {
        "seq_length": seq_length,
        "count_CONFIRM_PAYMENT": 1,
        "count_FAIL_PAYMENT": 0,
        "count_RETRY_PAYMENT": count_retry,
        "count_CANCEL_ORDER": 0,
        "count_PACK_ITEMS": 1,
        "count_SHIP_ITEMS": 1,
        "count_DELIVER_ITEMS": 1,
        "count_REFUND_ORDER": 0,
        "retry_ratio": float(count_retry / seq_length) if seq_length > 0 else 0.0,
        "payment_failure_rate": 0.0,
        "bigram_FAIL_to_RETRY": 0,
        "bigram_PACK_to_SHIP": 1,
        "unique_event_count": 4,
    }


def _make_normal_result(order_id: str = "ORD-NORMAL") -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "verdict": "NORMAL",
        "is_anomaly": False,
        "anomaly_type": None,
        "summary": "Completed trace with typical profile.",
        "fsm_evaluation": {
            "is_rule_valid": True,
            "is_complete": True,
            "final_state": OrderState.DELIVERED.value,
            "total_steps": 4,
            "successful_transitions": 4,
            "error": None,
        },
        "ml_evaluation": {
            "prediction": "NORMAL",
            "is_ml_anomaly": False,
            "anomaly_score": -0.1542,
            "raw_decision_score": 0.1542,
            "features": _make_14_features(4, count_retry=0),
        },
        "ml_skip_reason": None,
        "diagnostics": {
            "severity": "INFO",
            "root_cause": f"Order completed normally terminating in terminal state '{OrderState.DELIVERED.value}'.",
            "key_factors": [f"Terminal state: {OrderState.DELIVERED.value}"],
            "recommended_action": "Standard order execution following expected baseline paths. No operational intervention required.",
        },
    }


def _make_rule_violation_result(order_id: str = "ORD-VIOLATION") -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "verdict": "RULE_VIOLATION",
        "is_anomaly": True,
        "anomaly_type": "RULE_VIOLATION",
        "summary": f"FSM rule violation (INVALID_TRANSITION) at step 1: event '{OrderEvent.PACK_ITEMS.value}' is not permitted in state '{OrderState.ORDER_CREATED.value}'.",
        "fsm_evaluation": {
            "is_rule_valid": False,
            "is_complete": False,
            "final_state": OrderState.ORDER_CREATED.value,
            "total_steps": 1,
            "successful_transitions": 0,
            "error": {
                "error_type": "INVALID_TRANSITION",
                "failed_step": 1,
                "failed_event": OrderEvent.PACK_ITEMS.value,
                "state_at_failure": OrderState.ORDER_CREATED.value,
                "allowed_events": [
                    OrderEvent.CANCEL_ORDER.value,
                    OrderEvent.CONFIRM_PAYMENT.value,
                    OrderEvent.FAIL_PAYMENT.value,
                ],
            },
        },
        "ml_evaluation": None,
        "ml_skip_reason": "ML evaluation skipped: trace failed FSM rule validation (RULE_VIOLATION).",
        "diagnostics": {
            "severity": "CRITICAL",
            "root_cause": f"FSM rule violation (INVALID_TRANSITION) at step 1: event '{OrderEvent.PACK_ITEMS.value}' is not permitted in state '{OrderState.ORDER_CREATED.value}'.",
            "key_factors": [
                "Error type: INVALID_TRANSITION",
                "Failed step: 1",
                f"Offending event: {OrderEvent.PACK_ITEMS.value}",
                f"State at failure: {OrderState.ORDER_CREATED.value}",
            ],
            "recommended_action": "Review order workflow.",
        },
    }


def _make_statistical_anomaly_result(order_id: str = "ORD-STAT-ANOM") -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "verdict": "STATISTICAL_ANOMALY",
        "is_anomaly": True,
        "anomaly_type": "STATISTICAL_ANOMALY",
        "summary": "Completed trace with statistically unusual volume.",
        "fsm_evaluation": {
            "is_rule_valid": True,
            "is_complete": True,
            "final_state": OrderState.DELIVERED.value,
            "total_steps": 8,
            "successful_transitions": 8,
            "error": None,
        },
        "ml_evaluation": {
            "prediction": "ANOMALY",
            "is_ml_anomaly": True,
            "anomaly_score": 0.2450,
            "raw_decision_score": -0.2450,
            "features": _make_14_features(8, count_retry=4),
        },
        "ml_skip_reason": None,
        "diagnostics": {
            "severity": "WARNING",
            "root_cause": "Excessive payment retry attempts.",
            "key_factors": ["Anomaly score: 0.2450", "count_RETRY_PAYMENT: 4"],
            "recommended_action": "Investigate payment processor friction.",
        },
    }


def _make_valid_incomplete_result(order_id: str = "ORD-INCOMPLETE") -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "verdict": "VALID_INCOMPLETE",
        "is_anomaly": False,
        "anomaly_type": None,
        "summary": "Order workflow prefix is structurally valid but incomplete; currently in active state 'PAID'.",
        "fsm_evaluation": {
            "is_rule_valid": True,
            "is_complete": False,
            "final_state": OrderState.PAID.value,
            "total_steps": 1,
            "successful_transitions": 1,
            "error": None,
        },
        "ml_evaluation": None,
        "ml_skip_reason": "ML evaluation skipped: trace is in-progress and has not reached a terminal state (VALID_INCOMPLETE).",
        "diagnostics": {
            "severity": "INFO",
            "root_cause": "Active in-progress order.",
            "key_factors": [f"Current active state: {OrderState.PAID.value}"],
            "recommended_action": "Await subsequent events.",
        },
    }


# ---------------------------------------------------------------------------
# Test Suite Class
# ---------------------------------------------------------------------------

class TestDatabaseIntegration(unittest.TestCase):
    """Test suite for SQLite and SQLAlchemy persistence layer."""

    def setUp(self) -> None:
        """Create fresh isolated in-memory SQLite database for each test."""
        self.engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session = self.SessionLocal()

    def tearDown(self) -> None:
        """Clean up session and drop all tables."""
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    # -----------------------------------------------------------------------
    # 1. Table Creation and Initialization
    # -----------------------------------------------------------------------

    def test_01_init_db_creates_tables(self) -> None:
        """Verify init_db creates all expected tables."""
        temp_engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        init_db(target_engine=temp_engine)
        with temp_engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_analyses'")
            ).scalar()
            self.assertEqual(result, "workflow_analyses")
        temp_engine.dispose()

    # -----------------------------------------------------------------------
    # 2. Input Sequence Preservation
    # -----------------------------------------------------------------------

    def test_02_preserve_delimited_string_input(self) -> None:
        """Verify string inputs are stored verbatim as DELIMITED_STRING."""
        raw_str = f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value};{OrderEvent.SHIP_ITEMS.value};{OrderEvent.DELIVER_ITEMS.value}"
        rec = create_analysis_record(self.db, _make_normal_result(), raw_str)
        self.assertEqual(rec.input_format, "DELIMITED_STRING")
        self.assertEqual(rec.raw_input_payload, raw_str)

        retrieved = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.raw_input_payload, raw_str)

    def test_03_preserve_event_list_input(self) -> None:
        """Verify list inputs are stored verbatim as EVENT_LIST."""
        raw_list = [
            OrderEvent.CONFIRM_PAYMENT.value,
            OrderEvent.PACK_ITEMS.value,
            OrderEvent.SHIP_ITEMS.value,
            OrderEvent.DELIVER_ITEMS.value,
        ]
        rec = create_analysis_record(self.db, _make_normal_result(), raw_list)
        self.assertEqual(rec.input_format, "EVENT_LIST")
        self.assertEqual(rec.raw_input_payload, raw_list)

        retrieved = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.raw_input_payload, raw_list)

    def test_04_preserve_empty_string(self) -> None:
        """Verify empty string is preserved and associated with VALID_INCOMPLETE."""
        rec = create_analysis_record(self.db, _make_valid_incomplete_result(), "")
        self.assertEqual(rec.input_format, "DELIMITED_STRING")
        self.assertEqual(rec.raw_input_payload, "")

        retrieved = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.raw_input_payload, "")

    def test_05_preserve_empty_list(self) -> None:
        """Verify empty list is preserved and associated with VALID_INCOMPLETE."""
        rec = create_analysis_record(self.db, _make_valid_incomplete_result(), [])
        self.assertEqual(rec.input_format, "EVENT_LIST")
        self.assertEqual(rec.raw_input_payload, [])

        retrieved = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.raw_input_payload, [])

    def test_06_preserve_whitespace_only_string(self) -> None:
        """Verify whitespace string is preserved without stripping."""
        ws = "   \t  "
        rec = create_analysis_record(self.db, _make_valid_incomplete_result(), ws)
        self.assertEqual(rec.raw_input_payload, ws)

    def test_07_preserve_list_containing_blank_strings(self) -> None:
        """Verify list containing empty and whitespace strings is preserved."""
        raw = ["", "  ", OrderEvent.CONFIRM_PAYMENT.value]
        rec = create_analysis_record(self.db, _make_valid_incomplete_result(), raw)
        self.assertEqual(rec.raw_input_payload, raw)

    def test_08_rejection_of_unsupported_types(self) -> None:
        """Verify tuples, sets, and numbers are rejected with TypeError."""
        with self.assertRaises(TypeError):
            validate_and_classify_input(("A", "B"))
        with self.assertRaises(TypeError):
            validate_and_classify_input({"A", "B"})
        with self.assertRaises(TypeError):
            validate_and_classify_input(12345)
        with self.assertRaises(TypeError):
            create_analysis_record(self.db, _make_normal_result(), ("A", "B"))

    def test_09_rejection_of_lists_with_non_strings(self) -> None:
        """Verify lists containing non-string items raise TypeError."""
        with self.assertRaises(TypeError):
            validate_and_classify_input([OrderEvent.CONFIRM_PAYMENT.value, 123])
        with self.assertRaises(TypeError):
            create_analysis_record(self.db, _make_normal_result(), [None])

    # -----------------------------------------------------------------------
    # 3. UTC Timestamp Normalization and Retrieval Policy
    # -----------------------------------------------------------------------

    def test_10_utc_aware_timestamp_normalized_to_naive(self) -> None:
        """Verify timezone-aware UTC datetime is stored as UTC-naive."""
        aware_now = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
        naive = to_naive_utc(aware_now)
        self.assertIsNone(naive.tzinfo)
        self.assertEqual(naive, datetime(2026, 11, 9, 12, 0, 0))

    def test_11_non_utc_timezone_converted_to_utc_naive(self) -> None:
        """Verify non-UTC timezone (+5:30) is correctly converted to UTC-naive."""
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        ist_time = datetime(2026, 11, 9, 17, 30, 0, tzinfo=ist_tz)
        naive = to_naive_utc(ist_time)
        self.assertIsNone(naive.tzinfo)
        self.assertEqual(naive, datetime(2026, 11, 9, 12, 0, 0))

    def test_12_naive_timestamp_treated_as_utc(self) -> None:
        """Verify naive timestamp is preserved and treated as UTC by contract."""
        naive_time = datetime(2026, 11, 9, 14, 0, 0)
        result = to_naive_utc(naive_time)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, naive_time)

    # -----------------------------------------------------------------------
    # 4. Verdicts and Conditional ML Fields
    # -----------------------------------------------------------------------

    def test_13_all_four_verdict_types_stored_successfully(self) -> None:
        """Verify persistence of all 4 approved verdicts."""
        r_norm = create_analysis_record(self.db, _make_normal_result("ORD-1"), f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value};{OrderEvent.SHIP_ITEMS.value};{OrderEvent.DELIVER_ITEMS.value}")
        r_viol = create_analysis_record(self.db, _make_rule_violation_result("ORD-2"), OrderEvent.PACK_ITEMS.value)
        r_stat = create_analysis_record(self.db, _make_statistical_anomaly_result("ORD-3"), f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value};{OrderEvent.SHIP_ITEMS.value};{OrderEvent.DELIVER_ITEMS.value}")
        r_inco = create_analysis_record(self.db, _make_valid_incomplete_result("ORD-4"), OrderEvent.CONFIRM_PAYMENT.value)

        self.assertEqual(r_norm.verdict, "NORMAL")
        self.assertEqual(r_viol.verdict, "RULE_VIOLATION")
        self.assertEqual(r_stat.verdict, "STATISTICAL_ANOMALY")
        self.assertEqual(r_inco.verdict, "VALID_INCOMPLETE")

    def test_14_ml_prediction_and_anomaly_flag_consistency(self) -> None:
        """Verify NORMAL implies is_ml_anomaly=False and ANOMALY implies is_ml_anomaly=True."""
        # NORMAL with is_ml_anomaly=True must fail
        bad_normal = _make_normal_result()
        bad_normal["ml_evaluation"]["is_ml_anomaly"] = True
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, bad_normal, "CONFIRM_PAYMENT")

        # ANOMALY with is_ml_anomaly=False must fail
        bad_anomaly = _make_statistical_anomaly_result()
        bad_anomaly["ml_evaluation"]["is_ml_anomaly"] = False
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, bad_anomaly, "CONFIRM_PAYMENT")

    def test_15_ml_evaluated_and_skipped_conditional_fields(self) -> None:
        """Verify ML-evaluated fields are populated while ML-skipped fields are null."""
        # Evaluated record
        rec_eval = create_analysis_record(self.db, _make_normal_result(), "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS")
        self.assertTrue(rec_eval.ml_evaluated)
        self.assertEqual(rec_eval.ml_prediction, "NORMAL")
        self.assertFalse(rec_eval.is_ml_anomaly)
        self.assertIsNotNone(rec_eval.anomaly_score)
        self.assertIsNotNone(rec_eval.raw_decision_score)
        self.assertIsNotNone(rec_eval.ml_features)
        self.assertIsNone(rec_eval.ml_skip_reason)

        # Skipped record
        rec_skip = create_analysis_record(self.db, _make_rule_violation_result(), "PACK_ITEMS")
        self.assertFalse(rec_skip.ml_evaluated)
        self.assertIsNone(rec_skip.ml_prediction)
        self.assertIsNone(rec_skip.is_ml_anomaly)
        self.assertIsNone(rec_skip.anomaly_score)
        self.assertIsNone(rec_skip.raw_decision_score)
        self.assertIsNone(rec_skip.ml_features)
        self.assertIsNotNone(rec_skip.ml_skip_reason)

    def test_16_database_level_check_constraints(self) -> None:
        """Verify SQLite CheckConstraints reject invalid verdicts, step counts, and ML combinations."""
        # 1. Invalid verdict constraint
        with self.assertRaises(IntegrityError):
            raw_rec = WorkflowAnalysis(
                order_id="ORD-X",
                input_format="DELIMITED_STRING",
                raw_input_payload="A;B",
                verdict="UNSUPPORTED_VERDICT",
                is_anomaly=False,
                summary="summary",
                is_rule_valid=True,
                is_complete=True,
                final_state=OrderState.DELIVERED.value,
                total_steps=1,
                successful_transitions=1,
                diagnostic_severity="INFO",
                diagnostic_root_cause="cause",
                diagnostic_key_factors=[],
                diagnostic_recommended_action="action",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                ml_evaluated=False,
                ml_skip_reason="skipped",
            )
            self.db.add(raw_rec)
            self.db.commit()
        self.db.rollback()

        # 2. Negative total_steps constraint
        with self.assertRaises(IntegrityError):
            raw_rec2 = WorkflowAnalysis(
                order_id="ORD-Y",
                input_format="DELIMITED_STRING",
                raw_input_payload="A;B",
                verdict="NORMAL",
                is_anomaly=False,
                summary="summary",
                is_rule_valid=True,
                is_complete=True,
                final_state=OrderState.DELIVERED.value,
                total_steps=-1,
                successful_transitions=0,
                diagnostic_severity="INFO",
                diagnostic_root_cause="cause",
                diagnostic_key_factors=[],
                diagnostic_recommended_action="action",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                ml_evaluated=False,
                ml_skip_reason="skipped",
            )
            self.db.add(raw_rec2)
            self.db.commit()
        self.db.rollback()

        # 3. Transitions > total_steps constraint
        with self.assertRaises(IntegrityError):
            raw_rec3 = WorkflowAnalysis(
                order_id="ORD-Z",
                input_format="DELIMITED_STRING",
                raw_input_payload="A;B",
                verdict="NORMAL",
                is_anomaly=False,
                summary="summary",
                is_rule_valid=True,
                is_complete=True,
                final_state=OrderState.DELIVERED.value,
                total_steps=2,
                successful_transitions=5,
                diagnostic_severity="INFO",
                diagnostic_root_cause="cause",
                diagnostic_key_factors=[],
                diagnostic_recommended_action="action",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                ml_evaluated=False,
                ml_skip_reason="skipped",
            )
            self.db.add(raw_rec3)
            self.db.commit()
        self.db.rollback()

    # -----------------------------------------------------------------------
    # 5. Atomic Batch Persistence & Rollback
    # -----------------------------------------------------------------------

    def test_17_single_record_persistence_and_retrieval(self) -> None:
        """Verify single record persistence and complete retrieval by ID."""
        res = _make_normal_result("ORD-SINGLE")
        rec = create_analysis_record(self.db, res, "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS")
        self.assertIsNotNone(rec.id)

        fetched = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.order_id, "ORD-SINGLE")
        self.assertEqual(fetched.verdict, "NORMAL")
        self.assertEqual(fetched.diagnostic_severity, "INFO")

    def test_18_successful_mixed_verdict_batch_persistence(self) -> None:
        """Verify atomic batch persistence with all four verdict types."""
        results = [
            _make_normal_result("BATCH-1"),
            _make_rule_violation_result("BATCH-2"),
            _make_statistical_anomaly_result("BATCH-3"),
            _make_valid_incomplete_result("BATCH-4"),
        ]
        inputs = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            [OrderEvent.PACK_ITEMS.value],
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            [],
        ]
        records = create_analysis_batch(self.db, results, inputs)

        self.assertEqual(len(records), 4)
        for rec in records:
            self.assertIsNotNone(rec.id)

        count = self.db.scalar(select(func.count(WorkflowAnalysis.id)))
        self.assertEqual(count, 4)

    def test_19_batch_count_mismatch_rejected_prior_to_db(self) -> None:
        """Verify batch with mismatched results and inputs raises ValueError before DB access."""
        results = [_make_normal_result("B-1"), _make_normal_result("B-2")]
        inputs = ["CONFIRM_PAYMENT"]
        with self.assertRaises(ValueError):
            create_analysis_batch(self.db, results, inputs)

        count = self.db.scalar(select(func.count(WorkflowAnalysis.id)))
        self.assertEqual(count, 0)

    def test_20_genuine_sqlite_trigger_mid_batch_rollback(self) -> None:
        """Verify genuine database engine failure mid-flush triggers rollback with 0 records remaining.

        Uses an SQLite BEFORE INSERT trigger on the table that fires when order_id
        is 'TRIGGER_ABORT_ROLLBACK'. All records pass repository-level validation.
        """
        # 1. Install native SQLite trigger
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                CREATE TRIGGER trigger_fail_mid_batch
                BEFORE INSERT ON workflow_analyses
                FOR EACH ROW
                WHEN NEW.order_id = 'TRIGGER_ABORT_ROLLBACK'
                BEGIN
                    SELECT RAISE(ABORT, 'Simulated mid-batch database engine failure');
                END;
                """)
            )
            conn.commit()

        # 2. Construct 3 valid records passing repository validation
        res1 = _make_normal_result("BATCH-VALID-1")
        res2 = _make_valid_incomplete_result("BATCH-VALID-2")
        res3 = _make_statistical_anomaly_result("TRIGGER_ABORT_ROLLBACK")

        inputs = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            [OrderEvent.CONFIRM_PAYMENT.value],
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
        ]

        # 3. Attempt batch persistence; assert database engine exception is raised
        with self.assertRaises(Exception) as ctx:
            create_analysis_batch(self.db, [res1, res2, res3], inputs)

        self.assertIn("Simulated mid-batch database engine failure", str(ctx.exception))

        # 4. Verify 0 records remain in the table (records 1 and 2 were rolled back)
        count = self.db.scalar(select(func.count(WorkflowAnalysis.id)))
        self.assertEqual(count, 0)

    # -----------------------------------------------------------------------
    # 6. Queries, Ordering, and Pagination
    # -----------------------------------------------------------------------

    def test_21_deterministic_historical_ordering_identical_timestamps(self) -> None:
        """Verify ordering uses created_at DESC, id DESC when timestamps are identical."""
        fixed_time = datetime(2026, 11, 9, 15, 0, 0)
        # Create 3 analyses for the same order at the exact same timestamp
        rec1 = create_analysis_record(self.db, _make_normal_result("ORD-SAME-TIME"), OrderEvent.CONFIRM_PAYMENT.value, fixed_time)
        rec2 = create_analysis_record(self.db, _make_normal_result("ORD-SAME-TIME"), f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value}", fixed_time)
        rec3 = create_analysis_record(self.db, _make_normal_result("ORD-SAME-TIME"), f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value};{OrderEvent.SHIP_ITEMS.value}", fixed_time)

        analyses = get_analyses_by_order_id(self.db, "ORD-SAME-TIME")
        self.assertEqual(len(analyses), 3)
        # IDs must be strictly descending: rec3 (highest ID) first, rec1 last
        self.assertEqual([a.id for a in analyses], [rec3.id, rec2.id, rec1.id])

    def test_22_get_latest_analysis_by_order_id(self) -> None:
        """Verify get_latest_analysis_by_order_id retrieves the most recent record."""
        t1 = datetime(2026, 11, 9, 10, 0, 0)
        t2 = datetime(2026, 11, 9, 11, 0, 0)

        create_analysis_record(self.db, _make_valid_incomplete_result("ORD-EVOLVE"), OrderEvent.CONFIRM_PAYMENT.value, t1)
        rec2 = create_analysis_record(self.db, _make_normal_result("ORD-EVOLVE"), f"{OrderEvent.CONFIRM_PAYMENT.value};{OrderEvent.PACK_ITEMS.value};{OrderEvent.SHIP_ITEMS.value};{OrderEvent.DELIVER_ITEMS.value}", t2)

        latest = get_latest_analysis_by_order_id(self.db, "ORD-EVOLVE")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, rec2.id)
        self.assertEqual(latest.verdict, "NORMAL")

    def test_23_pagination_skip_and_limit(self) -> None:
        """Verify pagination with skip and limit."""
        for i in range(10):
            create_analysis_record(self.db, _make_normal_result(f"ORD-PAG-{i}"), OrderEvent.CONFIRM_PAYMENT.value)

        page1, total = list_analyses(self.db, skip=0, limit=4)
        self.assertEqual(len(page1), 4)
        self.assertEqual(total, 10)

        page2, _ = list_analyses(self.db, skip=4, limit=4)
        self.assertEqual(len(page2), 4)

        page3, _ = list_analyses(self.db, skip=8, limit=4)
        self.assertEqual(len(page3), 2)

    def test_24_filtering_by_verdict_and_anomaly(self) -> None:
        """Verify filtering by verdict and is_anomaly."""
        create_analysis_record(self.db, _make_normal_result("O-1"), OrderEvent.CONFIRM_PAYMENT.value)
        create_analysis_record(self.db, _make_normal_result("O-2"), OrderEvent.CONFIRM_PAYMENT.value)
        create_analysis_record(self.db, _make_rule_violation_result("O-3"), OrderEvent.PACK_ITEMS.value)
        create_analysis_record(self.db, _make_statistical_anomaly_result("O-4"), OrderEvent.CONFIRM_PAYMENT.value)

        # Filter verdict = RULE_VIOLATION
        viol_list, count_viol = list_analyses(self.db, verdict="RULE_VIOLATION")
        self.assertEqual(len(viol_list), 1)
        self.assertEqual(count_viol, 1)

        # Filter is_anomaly = True (RULE_VIOLATION + STATISTICAL_ANOMALY)
        anom_list, count_anom = list_analyses(self.db, is_anomaly=True)
        self.assertEqual(len(anom_list), 2)
        self.assertEqual(count_anom, 2)

    def test_25_repository_argument_validation(self) -> None:
        """Verify repository raises ValueError for invalid pagination and filter arguments."""
        with self.assertRaises(ValueError):
            list_analyses(self.db, skip=-1)
        with self.assertRaises(ValueError):
            list_analyses(self.db, limit=0)
        with self.assertRaises(ValueError):
            list_analyses(self.db, verdict="INVALID_VERDICT")
        with self.assertRaises(ValueError):
            get_analyses_by_order_id(self.db, "ORD-1", limit=0)

    # -----------------------------------------------------------------------
    # 7. Summary Metrics
    # -----------------------------------------------------------------------

    def test_26_summary_metric_calculations(self) -> None:
        """Verify database-wide summary metrics calculations across analyses and orders."""
        # 2 NORMAL (order O-1, order O-2)
        create_analysis_record(self.db, _make_normal_result("O-1"), OrderEvent.CONFIRM_PAYMENT.value)
        create_analysis_record(self.db, _make_normal_result("O-2"), OrderEvent.CONFIRM_PAYMENT.value)
        # 1 RULE_VIOLATION (order O-3)
        create_analysis_record(self.db, _make_rule_violation_result("O-3"), OrderEvent.PACK_ITEMS.value)
        # 1 STATISTICAL_ANOMALY (order O-1 again, historical progression!)
        create_analysis_record(self.db, _make_statistical_anomaly_result("O-1"), OrderEvent.CONFIRM_PAYMENT.value)
        # 1 VALID_INCOMPLETE without order_id (anonymous trace)
        res_anon = _make_valid_incomplete_result()
        res_anon["order_id"] = None
        create_analysis_record(self.db, res_anon, OrderEvent.CONFIRM_PAYMENT.value)

        metrics = get_summary_metrics(self.db)

        self.assertEqual(metrics["total_analyses"], 5)
        self.assertEqual(metrics["normal_count"], 2)
        self.assertEqual(metrics["rule_violation_count"], 1)
        self.assertEqual(metrics["statistical_anomaly_count"], 1)
        self.assertEqual(metrics["valid_incomplete_count"], 1)
        self.assertEqual(metrics["total_anomalies"], 2)
        self.assertAlmostEqual(metrics["anomaly_rate"], 2 / 5, places=4)
        # Unique non-null order IDs: O-1, O-2, O-3 (total 3 unique orders)
        self.assertEqual(metrics["unique_orders_count"], 3)

    def test_27_empty_database_summary_metrics(self) -> None:
        """Verify summary metrics on an empty database return 0 counts and 0.0 rate."""
        metrics = get_summary_metrics(self.db)
        self.assertEqual(metrics["total_analyses"], 0)
        self.assertEqual(metrics["normal_count"], 0)
        self.assertEqual(metrics["rule_violation_count"], 0)
        self.assertEqual(metrics["statistical_anomaly_count"], 0)
        self.assertEqual(metrics["valid_incomplete_count"], 0)
        self.assertEqual(metrics["total_anomalies"], 0)
        self.assertEqual(metrics["anomaly_rate"], 0.0)
        self.assertEqual(metrics["unique_orders_count"], 0)

    # -----------------------------------------------------------------------
    # 8. Strict ML Skip-Reason Validation
    # -----------------------------------------------------------------------

    def test_28_ml_skip_reason_validation(self) -> None:
        """Verify strict handling of ml_skip_reason when ML is skipped vs. evaluated."""
        # 1. Valid ML-skipped result with non-empty reason succeeds
        valid_skip = _make_rule_violation_result("SKIP-VALID")
        valid_skip["ml_skip_reason"] = "Valid skip explanation"
        rec = create_analysis_record(self.db, valid_skip, OrderEvent.PACK_ITEMS.value)
        self.assertEqual(rec.ml_skip_reason, "Valid skip explanation")

        # 2. Missing (None) skip reason on ML-skipped result raises ValueError
        missing_skip = _make_rule_violation_result("SKIP-MISSING")
        missing_skip["ml_skip_reason"] = None
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, missing_skip, OrderEvent.PACK_ITEMS.value)

        # 3. Empty or whitespace skip reason on ML-skipped result raises ValueError
        empty_skip = _make_rule_violation_result("SKIP-EMPTY")
        empty_skip["ml_skip_reason"] = "   "
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, empty_skip, OrderEvent.PACK_ITEMS.value)

        # 4. ML-evaluated result with non-None skip reason raises ValueError
        bad_evaluated = _make_normal_result("EVAL-BAD-SKIP")
        bad_evaluated["ml_skip_reason"] = "Unexpected skip reason on evaluated result"
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, bad_evaluated, OrderEvent.CONFIRM_PAYMENT.value)

    # -----------------------------------------------------------------------
    # 9. Real AnomalyService Integration Tests
    # -----------------------------------------------------------------------

    def test_29_real_anomaly_service_valid_incomplete_integration(self) -> None:
        """Verify persistence of actual AnomalyService output for a valid incomplete trace."""
        service = AnomalyService()
        raw_events = [OrderEvent.CONFIRM_PAYMENT.value]
        order_id = "ORD-REAL-INCOMPLETE-1"

        # 1. Obtain real analysis result from AnomalyService
        service_result = service.analyze_order(raw_events, order_id=order_id)

        # 2. Persist via repository
        rec = create_analysis_record(self.db, service_result, raw_events)
        self.assertIsNotNone(rec.id)

        # 3. Retrieve and verify complete field parity
        fetched = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.order_id, order_id)
        self.assertEqual(fetched.verdict, "VALID_INCOMPLETE")
        self.assertFalse(fetched.is_anomaly)
        self.assertIsNone(fetched.anomaly_type)
        self.assertTrue(fetched.is_rule_valid)
        self.assertFalse(fetched.is_complete)
        self.assertEqual(fetched.final_state, OrderState.PAID.value)
        self.assertEqual(fetched.total_steps, 1)
        self.assertEqual(fetched.successful_transitions, 1)
        self.assertIsNone(fetched.fsm_error)
        self.assertFalse(fetched.ml_evaluated)
        self.assertIsNone(fetched.ml_prediction)
        self.assertIsNone(fetched.ml_features)
        self.assertEqual(fetched.ml_skip_reason, service_result["ml_skip_reason"])
        self.assertEqual(fetched.diagnostic_severity, "INFO")
        self.assertEqual(fetched.input_format, "EVENT_LIST")
        self.assertEqual(fetched.raw_input_payload, raw_events)

    def test_30_real_anomaly_service_rule_violation_integration(self) -> None:
        """Verify persistence of actual AnomalyService output for an invalid trace."""
        service = AnomalyService()
        raw_events = [OrderEvent.PACK_ITEMS.value]  # Invalid from ORDER_CREATED
        order_id = "ORD-REAL-VIOLATION-1"

        # 1. Obtain real analysis result from AnomalyService
        service_result = service.analyze_order(raw_events, order_id=order_id)

        # 2. Persist via repository
        rec = create_analysis_record(self.db, service_result, raw_events)
        self.assertIsNotNone(rec.id)

        # 3. Retrieve and verify complete field parity
        fetched = get_analysis_by_id(self.db, rec.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.order_id, order_id)
        self.assertEqual(fetched.verdict, "RULE_VIOLATION")
        self.assertTrue(fetched.is_anomaly)
        self.assertEqual(fetched.anomaly_type, "RULE_VIOLATION")
        self.assertFalse(fetched.is_rule_valid)
        self.assertFalse(fetched.is_complete)
        self.assertEqual(fetched.final_state, OrderState.ORDER_CREATED.value)
        self.assertEqual(fetched.total_steps, 1)
        self.assertEqual(fetched.successful_transitions, 0)
        self.assertIsNotNone(fetched.fsm_error)
        self.assertEqual(fetched.fsm_error.get("error_type"), "INVALID_TRANSITION")
        self.assertEqual(fetched.fsm_error.get("failed_event"), OrderEvent.PACK_ITEMS.value)
        self.assertFalse(fetched.ml_evaluated)
        self.assertIsNone(fetched.ml_prediction)
        self.assertIsNone(fetched.ml_features)
        self.assertEqual(fetched.ml_skip_reason, service_result["ml_skip_reason"])
        self.assertEqual(fetched.diagnostic_severity, "CRITICAL")
        self.assertEqual(fetched.input_format, "EVENT_LIST")
        self.assertEqual(fetched.raw_input_payload, raw_events)

    # -----------------------------------------------------------------------
    # 10. Strict ML-Evaluated Data Integrity Tests
    # -----------------------------------------------------------------------

    def test_31_ml_evaluated_strict_field_validation(self) -> None:
        """Verify repository persistence rejects missing, non-boolean, non-numeric, or non-finite ML fields."""
        # 1. Missing is_ml_anomaly raises ValueError
        res_missing_flag = _make_normal_result("ORD-TEST-1")
        del res_missing_flag["ml_evaluation"]["is_ml_anomaly"]
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_missing_flag, "CONFIRM_PAYMENT")

        # 2. Non-boolean is_ml_anomaly ("False", 0, 1, "True") raises ValueError
        for non_bool in ("False", "True", 0, 1, None, []):
            res_non_bool = _make_normal_result("ORD-TEST-2")
            res_non_bool["ml_evaluation"]["is_ml_anomaly"] = non_bool
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_non_bool, "CONFIRM_PAYMENT")

        # 3. Missing anomaly_score raises ValueError
        res_missing_score = _make_normal_result("ORD-TEST-3")
        del res_missing_score["ml_evaluation"]["anomaly_score"]
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_missing_score, "CONFIRM_PAYMENT")

        # 4. Non-numeric or boolean anomaly_score raises ValueError
        for bad_score in ("0.1", "invalid", True, False, [0.1]):
            res_bad_score = _make_normal_result("ORD-TEST-4")
            res_bad_score["ml_evaluation"]["anomaly_score"] = bad_score
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_bad_score, "CONFIRM_PAYMENT")

        # 5. Non-finite anomaly_score (NaN, +inf, -inf) raises ValueError
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            res_non_finite = _make_normal_result("ORD-TEST-5")
            res_non_finite["ml_evaluation"]["anomaly_score"] = non_finite
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_non_finite, "CONFIRM_PAYMENT")

        # 6. Missing raw_decision_score raises ValueError
        res_missing_raw = _make_normal_result("ORD-TEST-6")
        del res_missing_raw["ml_evaluation"]["raw_decision_score"]
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_missing_raw, "CONFIRM_PAYMENT")

        # 7. Non-numeric or boolean raw_decision_score raises ValueError
        for bad_raw in ("-0.1", "invalid", True, False, [-0.1]):
            res_bad_raw = _make_normal_result("ORD-TEST-7")
            res_bad_raw["ml_evaluation"]["raw_decision_score"] = bad_raw
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_bad_raw, "CONFIRM_PAYMENT")

        # 8. Non-finite raw_decision_score (NaN, +inf, -inf) raises ValueError
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            res_non_finite_raw = _make_normal_result("ORD-TEST-8")
            res_non_finite_raw["ml_evaluation"]["raw_decision_score"] = non_finite
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_non_finite_raw, "CONFIRM_PAYMENT")

        # 9. Missing features raises ValueError
        res_missing_feat = _make_normal_result("ORD-TEST-9")
        del res_missing_feat["ml_evaluation"]["features"]
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_missing_feat, "CONFIRM_PAYMENT")

        # 10. Non-dictionary features raises ValueError
        for bad_feat in (["feat1", "feat2"], "not_a_dict", 123, None):
            res_bad_feat = _make_normal_result("ORD-TEST-10")
            res_bad_feat["ml_evaluation"]["features"] = bad_feat
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_bad_feat, "CONFIRM_PAYMENT")

        # 11. Empty feature dictionary raises ValueError
        res_empty_feat = _make_normal_result("ORD-TEST-11")
        res_empty_feat["ml_evaluation"]["features"] = {}
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_empty_feat, "CONFIRM_PAYMENT")

        # 12. Missing one approved feature raises ValueError
        res_missing_one = _make_normal_result("ORD-TEST-12")
        del res_missing_one["ml_evaluation"]["features"]["unique_event_count"]
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_missing_one, "CONFIRM_PAYMENT")

        # 13. Unexpected feature name raises ValueError
        res_unexpected_feat = _make_normal_result("ORD-TEST-13")
        res_unexpected_feat["ml_evaluation"]["features"]["unexpected_feature"] = 1.0
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_unexpected_feat, "CONFIRM_PAYMENT")

        # 14. Non-numeric feature value (e.g. "invalid") raises ValueError
        res_non_num_feat = _make_normal_result("ORD-TEST-14")
        res_non_num_feat["ml_evaluation"]["features"]["seq_length"] = "invalid"
        with self.assertRaises(ValueError):
            create_analysis_record(self.db, res_non_num_feat, "CONFIRM_PAYMENT")

        # 15. Boolean feature value raises ValueError
        for bool_val in (True, False):
            res_bool_feat = _make_normal_result("ORD-TEST-15")
            res_bool_feat["ml_evaluation"]["features"]["count_CONFIRM_PAYMENT"] = bool_val
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_bool_feat, "CONFIRM_PAYMENT")

        # 16. Non-finite numerical feature values raise ValueError
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            res_non_finite_feat = _make_normal_result("ORD-TEST-16")
            res_non_finite_feat["ml_evaluation"]["features"]["seq_length"] = non_finite
            with self.assertRaises(ValueError):
                create_analysis_record(self.db, res_non_finite_feat, "CONFIRM_PAYMENT")

        # 17. Confirm valid complete dictionary containing all 14 approved features is accepted
        valid_normal = _make_normal_result("ORD-VALID-NORM")
        self.assertEqual(len(valid_normal["ml_evaluation"]["features"]), 14)
        self.assertEqual(set(valid_normal["ml_evaluation"]["features"].keys()), set(FEATURE_NAMES))
        rec_norm = create_analysis_record(self.db, valid_normal, "CONFIRM_PAYMENT")
        self.assertEqual(rec_norm.verdict, "NORMAL")
        self.assertFalse(rec_norm.is_ml_anomaly)
        self.assertEqual(rec_norm.ml_features, valid_normal["ml_evaluation"]["features"])

        valid_anomaly = _make_statistical_anomaly_result("ORD-VALID-ANOM")
        self.assertEqual(len(valid_anomaly["ml_evaluation"]["features"]), 14)
        self.assertEqual(set(valid_anomaly["ml_evaluation"]["features"].keys()), set(FEATURE_NAMES))
        rec_anom = create_analysis_record(self.db, valid_anomaly, "CONFIRM_PAYMENT")
        self.assertEqual(rec_anom.verdict, "STATISTICAL_ANOMALY")
        self.assertTrue(rec_anom.is_ml_anomaly)
        self.assertEqual(rec_anom.ml_features, valid_anomaly["ml_evaluation"]["features"])


if __name__ == "__main__":
    unittest.main()
