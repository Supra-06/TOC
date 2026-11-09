"""Unit tests for ORM-to-Pydantic response mapper (Phase 7 Step 3).

Verifies that record_to_response correctly reconstructs nested Pydantic
AnalysisRecordResponse instances from SQLAlchemy WorkflowAnalysis ORM records,
normalizes timestamps to UTC, and raises RecordMappingError on inconsistent data.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_api_mapping -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.mapping import RecordMappingError, record_to_response
from backend.database.models import Base, WorkflowAnalysis
from backend.database.repository import create_analysis_record
from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.schemas.anomaly import AnalysisRecordResponse


def _make_valid_14_features() -> dict[str, float]:
    """Helper creating a full 14-feature dictionary."""
    return {
        "seq_length": 4.0,
        "count_CONFIRM_PAYMENT": 1.0,
        "count_FAIL_PAYMENT": 0.0,
        "count_RETRY_PAYMENT": 0.0,
        "count_CANCEL_ORDER": 0.0,
        "count_PACK_ITEMS": 1.0,
        "count_SHIP_ITEMS": 1.0,
        "count_DELIVER_ITEMS": 1.0,
        "count_REFUND_ORDER": 0.0,
        "retry_ratio": 0.0,
        "payment_failure_rate": 0.0,
        "bigram_FAIL_to_RETRY": 0.0,
        "bigram_PACK_to_SHIP": 1.0,
        "unique_event_count": 4.0,
    }


class TestApiMapping(unittest.TestCase):
    """Test suite covering record_to_response mapper function."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize in-memory SQLite engine and session factory."""
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionFactory = sessionmaker(bind=cls.engine, autocommit=False, autoflush=False)

    def setUp(self) -> None:
        """Create fresh session for each test."""
        self.session: Session = self.SessionFactory()

    def tearDown(self) -> None:
        """Rollback and close session after each test."""
        self.session.rollback()
        self.session.close()

    def test_01_map_normal_record(self) -> None:
        """Verify mapping of a NORMAL analysis record with ML evaluated."""
        analysis_result = {
            "order_id": "ORD-101",
            "verdict": "NORMAL",
            "is_anomaly": False,
            "anomaly_type": None,
            "summary": "Workflow executed normally.",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 4,
                "successful_transitions": 4,
                "error": None,
            },
            "ml_evaluation": {
                "prediction": "NORMAL",
                "is_ml_anomaly": False,
                "anomaly_score": -0.18,
                "raw_decision_score": 0.18,
                "features": _make_valid_14_features(),
            },
            "ml_skip_reason": None,
            "diagnostics": {
                "severity": "INFO",
                "root_cause": "Normal process completion.",
                "key_factors": ["Standard progression"],
                "recommended_action": "No remediation required.",
            },
        }

        record = create_analysis_record(
            db=self.session,
            analysis_result=analysis_result,
            raw_input="CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            timestamp=datetime(2026, 11, 9, 12, 0, 0),
        )

        resp = record_to_response(record)
        self.assertIsInstance(resp, AnalysisRecordResponse)
        self.assertEqual(resp.id, record.id)
        self.assertEqual(resp.order_id, "ORD-101")
        self.assertEqual(resp.verdict, "NORMAL")
        self.assertFalse(resp.is_anomaly)
        self.assertIsNone(resp.anomaly_type)
        self.assertIsNone(resp.ml_skip_reason)
        self.assertIsNotNone(resp.ml_evaluation)
        self.assertEqual(resp.ml_evaluation.prediction, "NORMAL")
        self.assertFalse(resp.ml_evaluation.is_ml_anomaly)
        self.assertAlmostEqual(resp.ml_evaluation.anomaly_score, -0.18)
        self.assertEqual(resp.created_at.tzinfo, timezone.utc)

    def test_02_map_rule_violation_record_with_fsm_error(self) -> None:
        """Verify mapping of a RULE_VIOLATION record including structured FSM error details."""
        analysis_result = {
            "order_id": "ORD-ERR-1",
            "verdict": "RULE_VIOLATION",
            "is_anomaly": True,
            "anomaly_type": "RULE_VIOLATION",
            "summary": "FSM transition rule breach.",
            "fsm_evaluation": {
                "is_rule_valid": False,
                "is_complete": False,
                "final_state": "ORDER_CREATED",
                "total_steps": 2,
                "successful_transitions": 1,
                "error": {
                    "error_type": "INVALID_TRANSITION",
                    "failed_step": 2,
                    "failed_event": "DELIVER_ITEMS",
                    "state_at_failure": "PAID",
                    "allowed_events": ["PACK_ITEMS"],
                },
            },
            "ml_evaluation": None,
            "ml_skip_reason": "ML evaluation skipped: trace failed FSM rule validation (RULE_VIOLATION).",
            "diagnostics": {
                "severity": "CRITICAL",
                "root_cause": "Invalid transition attempted.",
                "key_factors": ["Event out of sequence"],
                "recommended_action": "Audit order state machine.",
            },
        }

        record = create_analysis_record(
            db=self.session,
            analysis_result=analysis_result,
            raw_input="CONFIRM_PAYMENT;DELIVER_ITEMS",
        )

        resp = record_to_response(record)
        self.assertEqual(resp.verdict, "RULE_VIOLATION")
        self.assertTrue(resp.is_anomaly)
        self.assertEqual(resp.anomaly_type, "RULE_VIOLATION")
        self.assertIsNone(resp.ml_evaluation)
        self.assertEqual(
            resp.ml_skip_reason,
            "ML evaluation skipped: trace failed FSM rule validation (RULE_VIOLATION).",
        )
        self.assertIsNotNone(resp.fsm_evaluation.error)
        self.assertEqual(resp.fsm_evaluation.error.error_type, "INVALID_TRANSITION")
        self.assertEqual(resp.fsm_evaluation.error.failed_step, 2)
        self.assertEqual(resp.fsm_evaluation.error.failed_event, "DELIVER_ITEMS")
        self.assertEqual(resp.fsm_evaluation.error.state_at_failure, "PAID")
        self.assertEqual(resp.fsm_evaluation.error.allowed_events, ["PACK_ITEMS"])
        self.assertEqual(resp.diagnostics.severity, "CRITICAL")

    def test_03_map_statistical_anomaly_record_with_14_features(self) -> None:
        """Verify mapping of a STATISTICAL_ANOMALY record with all 14 ML features."""
        features = _make_valid_14_features()
        features["count_RETRY_PAYMENT"] = 5.0
        features["retry_ratio"] = 5.0 / 4.0

        analysis_result = {
            "order_id": "ORD-ANOM-1",
            "verdict": "STATISTICAL_ANOMALY",
            "is_anomaly": True,
            "anomaly_type": "STATISTICAL_ANOMALY",
            "summary": "Statistical anomaly detected by Isolation Forest.",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 9,
                "successful_transitions": 9,
                "error": None,
            },
            "ml_evaluation": {
                "prediction": "ANOMALY",
                "is_ml_anomaly": True,
                "anomaly_score": 0.42,
                "raw_decision_score": -0.42,
                "features": features,
            },
            "ml_skip_reason": None,
            "diagnostics": {
                "severity": "WARNING",
                "root_cause": "High retry volume detected.",
                "key_factors": ["Excessive payment retries"],
                "recommended_action": "Inspect payment gateway logs.",
            },
        }

        record = create_analysis_record(
            db=self.session,
            analysis_result=analysis_result,
            raw_input=["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
        )

        resp = record_to_response(record)
        self.assertEqual(resp.verdict, "STATISTICAL_ANOMALY")
        self.assertTrue(resp.is_anomaly)
        self.assertEqual(resp.anomaly_type, "STATISTICAL_ANOMALY")
        self.assertIsNotNone(resp.ml_evaluation)
        self.assertEqual(resp.ml_evaluation.prediction, "ANOMALY")
        self.assertTrue(resp.ml_evaluation.is_ml_anomaly)
        self.assertAlmostEqual(resp.ml_evaluation.anomaly_score, 0.42)
        self.assertEqual(len(resp.ml_evaluation.features), 14)
        self.assertEqual(list(resp.ml_evaluation.features.keys()), list(FEATURE_NAMES))
        self.assertEqual(resp.diagnostics.severity, "WARNING")

    def test_04_map_valid_incomplete_record_ml_skipped(self) -> None:
        """Verify mapping of a VALID_INCOMPLETE record with ML skipped."""
        analysis_result = {
            "order_id": "ORD-INC-1",
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "anomaly_type": None,
            "summary": "Valid in-progress workflow trace.",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1,
                "successful_transitions": 1,
                "error": None,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "ML evaluation skipped: trace is valid but incomplete (VALID_INCOMPLETE).",
            "diagnostics": {
                "severity": "INFO",
                "root_cause": "Incomplete workflow trace in progress.",
                "key_factors": ["Awaiting next transition"],
                "recommended_action": "Wait for subsequent events.",
            },
        }

        record = create_analysis_record(
            db=self.session,
            analysis_result=analysis_result,
            raw_input="CONFIRM_PAYMENT",
        )

        resp = record_to_response(record)
        self.assertEqual(resp.verdict, "VALID_INCOMPLETE")
        self.assertFalse(resp.is_anomaly)
        self.assertIsNone(resp.anomaly_type)
        self.assertIsNone(resp.ml_evaluation)
        self.assertEqual(
            resp.ml_skip_reason,
            "ML evaluation skipped: trace is valid but incomplete (VALID_INCOMPLETE).",
        )
        self.assertEqual(resp.diagnostics.severity, "INFO")

    def test_05_correct_reconstruction_of_nested_fsm_fields(self) -> None:
        """Verify exact field preservation in FsmEvaluationSchema."""
        analysis_result = {
            "verdict": "NORMAL",
            "is_anomaly": False,
            "summary": "OK",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 4,
                "successful_transitions": 4,
                "error": None,
            },
            "ml_evaluation": {
                "prediction": "NORMAL",
                "is_ml_anomaly": False,
                "anomaly_score": -0.1,
                "raw_decision_score": 0.1,
                "features": _make_valid_14_features(),
            },
            "diagnostics": {
                "severity": "INFO",
                "root_cause": "Normal",
                "key_factors": [],
                "recommended_action": "None",
            },
        }
        record = create_analysis_record(self.session, analysis_result, "CONFIRM_PAYMENT")
        resp = record_to_response(record)

        self.assertTrue(resp.fsm_evaluation.is_rule_valid)
        self.assertTrue(resp.fsm_evaluation.is_complete)
        self.assertEqual(resp.fsm_evaluation.final_state, "DELIVERED")
        self.assertEqual(resp.fsm_evaluation.total_steps, 4)
        self.assertEqual(resp.fsm_evaluation.successful_transitions, 4)
        self.assertIsNone(resp.fsm_evaluation.error)

    def test_06_correct_reconstruction_of_nested_diagnostics_fields(self) -> None:
        """Verify exact field preservation in DiagnosticsSchema."""
        analysis_result = {
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "summary": "Incomplete order",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1,
                "successful_transitions": 1,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "Incomplete trace",
            "diagnostics": {
                "severity": "INFO",
                "root_cause": "Workflow in progress",
                "key_factors": ["Awaiting shipment", "Payment verified"],
                "recommended_action": "Monitor progress",
            },
        }
        record = create_analysis_record(self.session, analysis_result, "CONFIRM_PAYMENT")
        resp = record_to_response(record)

        self.assertEqual(resp.diagnostics.severity, "INFO")
        self.assertEqual(resp.diagnostics.root_cause, "Workflow in progress")
        self.assertEqual(resp.diagnostics.key_factors, ["Awaiting shipment", "Payment verified"])
        self.assertEqual(resp.diagnostics.recommended_action, "Monitor progress")

    def test_07_preservation_of_delimited_string_input(self) -> None:
        """Verify preservation of delimited string format and payload."""
        analysis_result = {
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "summary": "OK",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1,
                "successful_transitions": 1,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "Incomplete",
            "diagnostics": {"severity": "INFO", "root_cause": "", "key_factors": [], "recommended_action": ""},
        }
        raw_str = "CONFIRM_PAYMENT;PACK_ITEMS"
        record = create_analysis_record(self.session, analysis_result, raw_str)
        resp = record_to_response(record)

        self.assertEqual(resp.input_format, "DELIMITED_STRING")
        self.assertEqual(resp.raw_input_payload, raw_str)

    def test_08_preservation_of_event_list_input(self) -> None:
        """Verify preservation of event list format and payload."""
        analysis_result = {
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "summary": "OK",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1,
                "successful_transitions": 1,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "Incomplete",
            "diagnostics": {"severity": "INFO", "root_cause": "", "key_factors": [], "recommended_action": ""},
        }
        raw_list = ["CONFIRM_PAYMENT", "PACK_ITEMS"]
        record = create_analysis_record(self.session, analysis_result, raw_list)
        resp = record_to_response(record)

        self.assertEqual(resp.input_format, "EVENT_LIST")
        self.assertEqual(resp.raw_input_payload, raw_list)

    def test_09_preservation_of_empty_and_whitespace_input(self) -> None:
        """Verify preservation of empty strings, whitespace strings, empty lists, and lists with blanks."""
        empty_analysis = {
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "summary": "Empty trace",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "ORDER_CREATED",
                "total_steps": 0,
                "successful_transitions": 0,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "Trace empty",
            "diagnostics": {"severity": "INFO", "root_cause": "", "key_factors": [], "recommended_action": ""},
        }

        # 1. Empty string
        rec_empty_str = create_analysis_record(self.session, empty_analysis, "")
        resp_empty_str = record_to_response(rec_empty_str)
        self.assertEqual(resp_empty_str.input_format, "DELIMITED_STRING")
        self.assertEqual(resp_empty_str.raw_input_payload, "")

        # 2. Whitespace string
        rec_ws_str = create_analysis_record(self.session, empty_analysis, "   \t  ")
        resp_ws_str = record_to_response(rec_ws_str)
        self.assertEqual(resp_ws_str.input_format, "DELIMITED_STRING")
        self.assertEqual(resp_ws_str.raw_input_payload, "   \t  ")

        # 3. Empty list
        rec_empty_list = create_analysis_record(self.session, empty_analysis, [])
        resp_empty_list = record_to_response(rec_empty_list)
        self.assertEqual(resp_empty_list.input_format, "EVENT_LIST")
        self.assertEqual(resp_empty_list.raw_input_payload, [])

        # 4. List containing blank strings
        rec_blank_list = create_analysis_record(self.session, empty_analysis, ["", "   "])
        resp_blank_list = record_to_response(rec_blank_list)
        self.assertEqual(resp_blank_list.input_format, "EVENT_LIST")
        self.assertEqual(resp_blank_list.raw_input_payload, ["", "   "])

    def test_10_correct_ml_prediction_and_score_mapping(self) -> None:
        """Verify accurate numeric score and prediction mapping."""
        analysis_result = {
            "verdict": "NORMAL",
            "is_anomaly": False,
            "summary": "OK",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 4,
                "successful_transitions": 4,
            },
            "ml_evaluation": {
                "prediction": "NORMAL",
                "is_ml_anomaly": False,
                "anomaly_score": -0.25,
                "raw_decision_score": 0.25,
                "features": _make_valid_14_features(),
            },
            "diagnostics": {"severity": "INFO", "root_cause": "", "key_factors": [], "recommended_action": ""},
        }
        record = create_analysis_record(self.session, analysis_result, "CONFIRM_PAYMENT")
        resp = record_to_response(record)

        self.assertIsNotNone(resp.ml_evaluation)
        self.assertEqual(resp.ml_evaluation.prediction, "NORMAL")
        self.assertFalse(resp.ml_evaluation.is_ml_anomaly)
        self.assertEqual(resp.ml_evaluation.anomaly_score, -0.25)
        self.assertEqual(resp.ml_evaluation.raw_decision_score, 0.25)

    def test_11_missing_required_ml_data_raises_mapping_error(self) -> None:
        """Verify RecordMappingError is raised when ml_evaluated=True but required fields are None."""
        base_record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="NORMAL",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=True,
            final_state="DELIVERED",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=True,
            ml_prediction="NORMAL",
            is_ml_anomaly=False,
            anomaly_score=-0.1,
            raw_decision_score=0.1,
            ml_features=_make_valid_14_features(),
            ml_skip_reason=None,
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        # 1. Missing ml_prediction
        base_record.ml_prediction = None
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)
        base_record.ml_prediction = "NORMAL"

        # 2. Missing is_ml_anomaly
        base_record.is_ml_anomaly = None
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)
        base_record.is_ml_anomaly = False

        # 3. Missing anomaly_score
        base_record.anomaly_score = None
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)
        base_record.anomaly_score = -0.1

        # 4. Missing raw_decision_score
        base_record.raw_decision_score = None
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)
        base_record.raw_decision_score = 0.1

        # 5. Missing ml_features
        base_record.ml_features = None
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)

    def test_12_invalid_fsm_error_raises_mapping_error(self) -> None:
        """Verify RecordMappingError is raised when fsm_error has invalid structure or failed_step=0."""
        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="RULE_VIOLATION",
            is_anomaly=True,
            anomaly_type="RULE_VIOLATION",
            summary="Rule violation",
            is_rule_valid=False,
            is_complete=False,
            final_state="ORDER_CREATED",
            total_steps=1,
            successful_transitions=0,
            fsm_error={"error_type": "INVALID_TRANSITION"},  # Missing failed_step, etc.
            ml_evaluated=False,
            ml_skip_reason="Rule violation",
            diagnostic_severity="CRITICAL",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        # Missing required keys
        with self.assertRaises(RecordMappingError):
            record_to_response(record)

        # failed_step=0 violates ge=1
        record.fsm_error = {
            "error_type": "INVALID_TRANSITION",
            "failed_step": 0,
            "failed_event": "PACK_ITEMS",
            "state_at_failure": "ORDER_CREATED",
            "allowed_events": [],
        }
        with self.assertRaises(RecordMappingError):
            record_to_response(record)

    def test_13_utc_naive_timestamp_becomes_utc_aware(self) -> None:
        """Verify UTC-naive database timestamp is converted to UTC-aware datetime."""
        naive_dt = datetime(2026, 11, 9, 12, 0, 0)  # No tzinfo
        analysis_result = {
            "verdict": "VALID_INCOMPLETE",
            "is_anomaly": False,
            "summary": "OK",
            "fsm_evaluation": {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1,
                "successful_transitions": 1,
            },
            "ml_evaluation": None,
            "ml_skip_reason": "Incomplete",
            "diagnostics": {"severity": "INFO", "root_cause": "", "key_factors": [], "recommended_action": ""},
        }
        record = create_analysis_record(self.session, analysis_result, "CONFIRM_PAYMENT", timestamp=naive_dt)
        self.assertIsNone(record.created_at.tzinfo)

        resp = record_to_response(record)
        self.assertIsNotNone(resp.created_at.tzinfo)
        self.assertEqual(resp.created_at.tzinfo, timezone.utc)
        self.assertEqual(resp.created_at.year, 2026)
        self.assertEqual(resp.created_at.hour, 12)

    def test_14_already_aware_timestamp_normalized_to_utc(self) -> None:
        """Verify timezone-aware datetime with non-zero offset is normalized to UTC."""
        offset_tz = timezone(timedelta(hours=5, minutes=30))
        aware_dt = datetime(2026, 11, 9, 17, 30, 0, tzinfo=offset_tz)

        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="VALID_INCOMPLETE",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=False,
            final_state="PAID",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=False,
            ml_skip_reason="Incomplete",
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=aware_dt,
        )

        resp = record_to_response(record)
        self.assertEqual(resp.created_at.tzinfo, timezone.utc)
        self.assertEqual(resp.created_at.hour, 12)  # 17:30 - 05:30 = 12:00 UTC

    def test_15_invalid_persisted_values_not_silently_replaced(self) -> None:
        """Verify invalid values (None record, wrong class, unapproved verdict) raise RecordMappingError."""
        # 1. None record
        with self.assertRaises(RecordMappingError):
            record_to_response(None)  # type: ignore

        # 2. Non-ORM object
        with self.assertRaises(RecordMappingError):
            record_to_response({"id": 1})  # type: ignore

        # 3. Unapproved verdict
        bad_verdict_rec = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="NOT_A_REAL_VERDICT",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=False,
            final_state="PAID",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=False,
            ml_skip_reason="Incomplete",
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )
        with self.assertRaises(RecordMappingError):
            record_to_response(bad_verdict_rec)

    def test_16_corrupted_ml_features_raise_mapping_error(self) -> None:
        """Verify corrupted ml_features (missing, extra, non-numeric, or boolean) raise RecordMappingError."""
        base_record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="NORMAL",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=True,
            final_state="DELIVERED",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=True,
            ml_prediction="NORMAL",
            is_ml_anomaly=False,
            anomaly_score=-0.1,
            raw_decision_score=0.1,
            ml_features=_make_valid_14_features(),
            ml_skip_reason=None,
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        # 1. Missing feature
        bad_feats_missing = _make_valid_14_features()
        del bad_feats_missing["seq_length"]
        base_record.ml_features = bad_feats_missing
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)

        # 2. Unexpected extra feature
        bad_feats_extra = _make_valid_14_features()
        bad_feats_extra["unapproved_extra_feature"] = 1.0
        base_record.ml_features = bad_feats_extra
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)

        # 3. Boolean feature value (must not be treated as int)
        bad_feats_bool = _make_valid_14_features()
        bad_feats_bool["count_CONFIRM_PAYMENT"] = True
        base_record.ml_features = bad_feats_bool
        with self.assertRaises(RecordMappingError):
            record_to_response(base_record)

    def test_17_fsm_error_missing_allowed_events_raises_mapping_error(self) -> None:
        """Verify record_to_response raises RecordMappingError when allowed_events is omitted."""
        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="RULE_VIOLATION",
            is_anomaly=True,
            anomaly_type="RULE_VIOLATION",
            summary="Rule violation",
            is_rule_valid=False,
            is_complete=False,
            final_state="ORDER_CREATED",
            total_steps=1,
            successful_transitions=0,
            fsm_error={
                "error_type": "INVALID_TRANSITION",
                "failed_step": 1,
                "failed_event": "DELIVER_ITEMS",
                "state_at_failure": "PAID",
                # "allowed_events" is omitted
            },
            ml_evaluated=False,
            ml_skip_reason="Rule violation",
            diagnostic_severity="CRITICAL",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        with self.assertRaises(RecordMappingError) as ctx:
            record_to_response(record)

        self.assertIn("allowed_events", str(ctx.exception))

    def test_18_ml_evaluated_with_skip_reason_raises_mapping_error(self) -> None:
        """Verify RecordMappingError is raised when ml_evaluated=True but ml_skip_reason is not None."""
        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="NORMAL",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=True,
            final_state="DELIVERED",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=True,
            ml_prediction="NORMAL",
            is_ml_anomaly=False,
            anomaly_score=-0.1,
            raw_decision_score=0.1,
            ml_features=_make_valid_14_features(),
            ml_skip_reason="Should be None when ml_evaluated=True",
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        with self.assertRaises(RecordMappingError) as ctx:
            record_to_response(record)

        self.assertIn("ml_skip_reason is not None", str(ctx.exception))

    def test_19_ml_skipped_with_missing_or_empty_skip_reason_raises_mapping_error(self) -> None:
        """Verify RecordMappingError is raised when ml_evaluated=False but ml_skip_reason is missing or empty."""
        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="VALID_INCOMPLETE",
            is_anomaly=False,
            summary="OK",
            is_rule_valid=True,
            is_complete=False,
            final_state="PAID",
            total_steps=1,
            successful_transitions=1,
            fsm_error=None,
            ml_evaluated=False,
            ml_skip_reason=None,  # Missing
            diagnostic_severity="INFO",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        with self.assertRaises(RecordMappingError) as ctx:
            record_to_response(record)
        self.assertIn("ml_skip_reason", str(ctx.exception))

        # Blank/whitespace string
        record.ml_skip_reason = "   "
        with self.assertRaises(RecordMappingError) as ctx:
            record_to_response(record)
        self.assertIn("ml_skip_reason", str(ctx.exception))

    def test_20_fsm_error_invalid_allowed_events_type_raises_mapping_error(self) -> None:
        """Verify record_to_response raises RecordMappingError when allowed_events is not a list (e.g. string)."""
        record = WorkflowAnalysis(
            id=1,
            input_format="EVENT_LIST",
            raw_input_payload=["CONFIRM_PAYMENT"],
            verdict="RULE_VIOLATION",
            is_anomaly=True,
            anomaly_type="RULE_VIOLATION",
            summary="Rule violation",
            is_rule_valid=False,
            is_complete=False,
            final_state="ORDER_CREATED",
            total_steps=1,
            successful_transitions=0,
            fsm_error={
                "error_type": "INVALID_TRANSITION",
                "failed_step": 1,
                "failed_event": "DELIVER_ITEMS",
                "state_at_failure": "PAID",
                "allowed_events": "PACK_ITEMS",  # Invalid non-list value (string)
            },
            ml_evaluated=False,
            ml_skip_reason="Rule violation",
            diagnostic_severity="CRITICAL",
            diagnostic_root_cause="",
            diagnostic_key_factors=[],
            diagnostic_recommended_action="",
            created_at=datetime(2026, 11, 9, 12, 0, 0),
        )

        with self.assertRaises(RecordMappingError) as ctx:
            record_to_response(record)

        self.assertIn("allowed_events", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

