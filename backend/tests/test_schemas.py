"""Unit tests for Pydantic v2 API request and response schemas (Phase 7 Step 2).

Verifies:
 1. Request validation: delimited string input.
 2. Request validation: event list input.
 3. Request validation: preservation of empty string.
 4. Request validation: preservation of empty list.
 5. Request validation: preservation of whitespace-only strings and list items.
 6. Request validation: rejection of non-string event elements in list.
 7. Request validation: rejection of integer and boolean event_sequence.
 8. Request validation: rejection of unsupported containers (tuples, sets, dicts).
 9. Request validation: rejection of missing event_sequence.
10. Request validation: optional order_id (accepted when None and valid string).
11. Request validation: rejection of order_id exceeding 64 characters.
12. Batch validation: rejection of empty batch (min_length=1).
13. Batch validation: rejection of batch exceeding 500 entries (max_length=500).
14. Batch validation: acceptance of batch with delimited string items.
15. Batch validation: acceptance of batch with event-list items.
16. Batch validation: acceptance of batch with structured TraceItemRequest objects.
17. Batch validation: acceptance of batch mixing all three supported formats.
18. Batch validation: preservation of empty and whitespace-only strings/lists.
19. Batch validation: rejection of tuple as event sequence.
20. Batch validation: rejection of set as event sequence.
21. Batch validation: rejection of dictionary as event sequence.
22. Batch validation: rejection of lists containing integers, booleans, or None.
23. Response validation: all four approved verdicts (NORMAL, RULE_VIOLATION, STATISTICAL_ANOMALY, VALID_INCOMPLETE).
24. FSM validation: failed_step lower bound (accepts 1, rejects 0 and negative).
25. Response validation: FSM error detail present vs. None.
26. Response validation: ML evaluation present vs. None.
27. Response validation: ML skip reason present vs. None.
28. ML evaluation: complete valid 14-feature dictionary acceptance.
29. ML evaluation: rejection of missing feature in features dictionary.
30. ML evaluation: rejection of unexpected feature in features dictionary.
31. ML evaluation: rejection of non-numeric feature values.
32. ML evaluation: rejection of boolean feature values.
33. ML evaluation: rejection of NaN, +inf, -inf in scores and feature values.
34. Response validation: rejection of invalid verdict strings.
35. Response validation: rejection of invalid anomaly_type strings.
36. Response validation: rejection of invalid diagnostic severity strings.
37. Timestamp validation: acceptance of timezone-aware UTC datetime.
38. Timestamp validation: rejection of naive datetime (objects and strings).
39. Timestamp validation: rejection of datetime with non-zero offset (+05:30).
40. Timestamp validation: acceptance of ISO-8601 UTC string ending in 'Z'.
41. Timestamp validation: rejection of ISO-8601 string with non-zero offset.
42. Common schemas: BatchAnalysisResponse and OrderHistoryResponse with returned_count.
43. Common schemas: LivenessResponse, ReadinessResponse (without model_loaded_in_memory), DatabaseSummaryMetricsResponse, and ErrorResponse.
44. Batch validation: acceptance of raw JSON-style dictionaries as structured batch items.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend/tests/test_schemas.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.schemas import (
    AnalysisRecordResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BatchSummaryMetrics,
    DatabaseSummaryMetricsResponse,
    DiagnosticsSchema,
    ErrorDetail,
    ErrorResponse,
    FsmErrorDetail,
    FsmEvaluationSchema,
    LivenessResponse,
    MlEvaluationSchema,
    OrderHistoryResponse,
    PaginatedAnalysesResponse,
    ReadinessResponse,
    TraceItemRequest,
    WorkflowAnalysisRequest,
)


def _make_valid_14_features(seq_length: float = 4.0, count_retry: float = 0.0) -> dict[str, float]:
    """Helper creating a complete 14-feature dictionary with valid finite floats."""
    return {
        "seq_length": seq_length,
        "count_CONFIRM_PAYMENT": 1.0,
        "count_FAIL_PAYMENT": 0.0,
        "count_RETRY_PAYMENT": count_retry,
        "count_CANCEL_ORDER": 0.0,
        "count_PACK_ITEMS": 1.0,
        "count_SHIP_ITEMS": 1.0,
        "count_DELIVER_ITEMS": 1.0,
        "count_REFUND_ORDER": 0.0,
        "retry_ratio": float(count_retry / seq_length) if seq_length > 0 else 0.0,
        "payment_failure_rate": 0.0,
        "bigram_FAIL_to_RETRY": 0.0,
        "bigram_PACK_to_SHIP": 1.0,
        "unique_event_count": 4.0,
    }


def _make_valid_analysis_record_dict(
    id: int = 1,
    order_id: str | None = "ORD-001",
    verdict: str = "NORMAL",
    is_anomaly: bool = False,
    anomaly_type: str | None = None,
    ml_eval: dict | None = None,
    ml_skip_reason: str | None = None,
    fsm_error: dict | None = None,
    created_at: datetime | str = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc),
) -> dict:
    """Helper building a complete raw dictionary matching AnalysisRecordResponse."""
    return {
        "id": id,
        "order_id": order_id,
        "input_format": "EVENT_LIST",
        "raw_input_payload": ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
        "verdict": verdict,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "summary": "Sample summary text",
        "fsm_evaluation": {
            "is_rule_valid": True,
            "is_complete": True,
            "final_state": "DELIVERED",
            "total_steps": 4,
            "successful_transitions": 4,
            "error": fsm_error,
        },
        "ml_evaluation": ml_eval,
        "ml_skip_reason": ml_skip_reason,
        "diagnostics": {
            "severity": "INFO",
            "root_cause": "Sample root cause explanation",
            "key_factors": ["Factor 1"],
            "recommended_action": "Sample recommended action",
        },
        "created_at": created_at,
    }


class TestPydanticSchemas(unittest.TestCase):
    """Test suite covering Pydantic v2 request and response schemas."""

    # -----------------------------------------------------------------------
    # 1. Single Request Validation Tests
    # -----------------------------------------------------------------------

    def test_01_request_delimited_string_input(self) -> None:
        """Verify request accepts a delimited string without modification."""
        req = WorkflowAnalysisRequest(event_sequence="CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS")
        self.assertEqual(req.event_sequence, "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS")
        self.assertIsNone(req.order_id)

    def test_02_request_event_list_input(self) -> None:
        """Verify request accepts a list of strings without modification."""
        events = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"]
        req = WorkflowAnalysisRequest(event_sequence=events)
        self.assertEqual(req.event_sequence, events)

    def test_03_request_empty_string_preserved(self) -> None:
        """Verify request preserves an empty string verbatim."""
        req = WorkflowAnalysisRequest(event_sequence="")
        self.assertEqual(req.event_sequence, "")

    def test_04_request_empty_list_preserved(self) -> None:
        """Verify request preserves an empty list verbatim."""
        req = WorkflowAnalysisRequest(event_sequence=[])
        self.assertEqual(req.event_sequence, [])

    def test_05_request_whitespace_preserved(self) -> None:
        """Verify whitespace strings and blank list items are preserved without stripping."""
        ws_str = "   \t  "
        req_str = WorkflowAnalysisRequest(event_sequence=ws_str)
        self.assertEqual(req_str.event_sequence, ws_str)

        ws_list = ["", "   ", "CONFIRM_PAYMENT"]
        req_list = WorkflowAnalysisRequest(event_sequence=ws_list)
        self.assertEqual(req_list.event_sequence, ws_list)

    def test_06_request_rejects_non_string_list_elements(self) -> None:
        """Verify list containing non-string items (ints, booleans, None) is rejected."""
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=["CONFIRM_PAYMENT", 123])
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=["CONFIRM_PAYMENT", True])
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=["CONFIRM_PAYMENT", None])

    def test_07_request_rejects_integer_and_boolean_event_sequence(self) -> None:
        """Verify integer or boolean event_sequence is rejected without implicit coercion."""
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=123)
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=True)
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=False)

    def test_08_request_rejects_unsupported_containers(self) -> None:
        """Verify tuples, sets, and dictionaries are rejected for event_sequence."""
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence=("CONFIRM_PAYMENT", "PACK_ITEMS"))
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence={"CONFIRM_PAYMENT", "PACK_ITEMS"})
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(event_sequence={"event": "CONFIRM_PAYMENT"})

    def test_09_request_rejects_missing_event_sequence(self) -> None:
        """Verify missing event_sequence raises ValidationError."""
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest.model_validate({})

    def test_10_request_optional_order_id(self) -> None:
        """Verify order_id accepts None or a valid string."""
        req1 = WorkflowAnalysisRequest(order_id=None, event_sequence="CONFIRM_PAYMENT")
        self.assertIsNone(req1.order_id)
        req2 = WorkflowAnalysisRequest(order_id="ORD-101", event_sequence="CONFIRM_PAYMENT")
        self.assertEqual(req2.order_id, "ORD-101")

    def test_11_request_rejects_order_id_exceeding_max_length(self) -> None:
        """Verify order_id longer than 64 characters is rejected."""
        long_id = "A" * 65
        with self.assertRaises(ValidationError):
            WorkflowAnalysisRequest(order_id=long_id, event_sequence="CONFIRM_PAYMENT")
        # Exact boundary 64 is accepted
        req_boundary = WorkflowAnalysisRequest(order_id="A" * 64, event_sequence="CONFIRM_PAYMENT")
        self.assertEqual(len(req_boundary.order_id or ""), 64)

    # -----------------------------------------------------------------------
    # 2. Batch Request Validation Tests
    # -----------------------------------------------------------------------

    def test_12_batch_rejects_empty_list(self) -> None:
        """Verify BatchAnalysisRequest rejects an empty traces list."""
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[])

    def test_13_batch_rejects_exceeding_500_traces(self) -> None:
        """Verify BatchAnalysisRequest rejects traces exceeding 500 items."""
        traces_501 = ["CONFIRM_PAYMENT" for _ in range(501)]
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=traces_501)

    def test_14_batch_with_delimited_string_items(self) -> None:
        """Verify BatchAnalysisRequest accepts a batch composed of delimited string traces."""
        traces = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS",
            "CONFIRM_PAYMENT;FAIL_PAYMENT;CANCEL_ORDER",
        ]
        batch = BatchAnalysisRequest(traces=traces)
        self.assertEqual(len(batch.traces), 2)
        self.assertEqual(batch.traces[0], "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS")
        self.assertEqual(batch.traces[1], "CONFIRM_PAYMENT;FAIL_PAYMENT;CANCEL_ORDER")

    def test_15_batch_with_event_list_items(self) -> None:
        """Verify BatchAnalysisRequest accepts a batch composed of event-list traces."""
        traces = [
            ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"],
            ["CONFIRM_PAYMENT", "CANCEL_ORDER"],
        ]
        batch = BatchAnalysisRequest(traces=traces)
        self.assertEqual(len(batch.traces), 2)
        self.assertEqual(batch.traces[0], ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"])
        self.assertEqual(batch.traces[1], ["CONFIRM_PAYMENT", "CANCEL_ORDER"])

    def test_16_batch_with_structured_trace_items(self) -> None:
        """Verify BatchAnalysisRequest accepts structured TraceItemRequest items with optional order_id."""
        traces = [
            TraceItemRequest(order_id="ORD-101", event_sequence="CONFIRM_PAYMENT;PACK_ITEMS"),
            TraceItemRequest(order_id=None, event_sequence=["CONFIRM_PAYMENT", "PACK_ITEMS"]),
            TraceItemRequest(order_id="A" * 64, event_sequence="CONFIRM_PAYMENT"),
        ]
        batch = BatchAnalysisRequest(traces=traces)
        self.assertEqual(len(batch.traces), 3)
        self.assertEqual(batch.traces[0].order_id, "ORD-101")
        self.assertIsNone(batch.traces[1].order_id)
        self.assertEqual(len(batch.traces[2].order_id or ""), 64)

        # Reject structured item with order_id > 64 chars
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(
                traces=[TraceItemRequest(order_id="A" * 65, event_sequence="CONFIRM_PAYMENT")]
            )

    def test_17_batch_mixing_all_three_formats(self) -> None:
        """Verify BatchAnalysisRequest accepts a batch mixing delimited strings, event lists, and structured items."""
        traces = [
            "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS",
            ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
            TraceItemRequest(order_id="ORD-MIXED-1", event_sequence="CONFIRM_PAYMENT;PACK_ITEMS"),
        ]
        batch = BatchAnalysisRequest(traces=traces)
        self.assertEqual(len(batch.traces), 3)
        self.assertIsInstance(batch.traces[0], str)
        self.assertIsInstance(batch.traces[1], list)
        self.assertIsInstance(batch.traces[2], TraceItemRequest)

    def test_18_batch_preserves_empty_and_whitespace_traces(self) -> None:
        """Verify BatchAnalysisRequest preserves empty and whitespace-only strings and lists."""
        traces = [
            "",
            "   \t  ",
            [],
            ["", "   "],
            TraceItemRequest(order_id=None, event_sequence=""),
            TraceItemRequest(order_id="ORD-WS", event_sequence=[]),
        ]
        batch = BatchAnalysisRequest(traces=traces)
        self.assertEqual(len(batch.traces), 6)
        self.assertEqual(batch.traces[0], "")
        self.assertEqual(batch.traces[1], "   \t  ")
        self.assertEqual(batch.traces[2], [])
        self.assertEqual(batch.traces[3], ["", "   "])
        self.assertEqual(batch.traces[4].event_sequence, "")
        self.assertEqual(batch.traces[5].event_sequence, [])

    def test_19_batch_rejects_tuple_as_event_sequence(self) -> None:
        """Verify BatchAnalysisRequest rejects tuple directly and in structured items."""
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[("CONFIRM_PAYMENT", "PACK_ITEMS")])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(
                traces=[{"order_id": "ORD-1", "event_sequence": ("CONFIRM_PAYMENT", "PACK_ITEMS")}]
            )

    def test_20_batch_rejects_set_as_event_sequence(self) -> None:
        """Verify BatchAnalysisRequest rejects set directly and in structured items."""
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[{"CONFIRM_PAYMENT", "PACK_ITEMS"}])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(
                traces=[{"order_id": "ORD-1", "event_sequence": {"CONFIRM_PAYMENT", "PACK_ITEMS"}}]
            )

    def test_21_batch_rejects_dictionary_as_event_sequence(self) -> None:
        """Verify BatchAnalysisRequest rejects arbitrary dictionaries and dictionaries as event sequences."""
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[{"unrecognized_key": "val"}])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(
                traces=[{"order_id": "ORD-1", "event_sequence": {"event": "CONFIRM_PAYMENT"}}]
            )

    def test_22_batch_rejects_lists_containing_integers_booleans_or_none(self) -> None:
        """Verify BatchAnalysisRequest rejects list items containing ints, bools, or None."""
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[["CONFIRM_PAYMENT", 123]])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[["CONFIRM_PAYMENT", True]])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(traces=[["CONFIRM_PAYMENT", None]])
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest(
                traces=[TraceItemRequest(order_id="ORD-1", event_sequence=["CONFIRM_PAYMENT", 123])]
            )

    # -----------------------------------------------------------------------
    # 3. Response and Sub-Schema Validation Tests
    # -----------------------------------------------------------------------

    def test_23_response_accepts_all_four_verdicts(self) -> None:
        """Verify AnalysisRecordResponse accepts exactly the four approved verdicts."""
        for verdict in ("NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"):
            data = _make_valid_analysis_record_dict(
                verdict=verdict,
                ml_skip_reason="Skipped" if verdict in ("RULE_VIOLATION", "VALID_INCOMPLETE") else None,
                ml_eval=(
                    {
                        "prediction": "NORMAL" if verdict == "NORMAL" else "ANOMALY",
                        "is_ml_anomaly": verdict != "NORMAL",
                        "anomaly_score": 0.1 if verdict != "NORMAL" else -0.1,
                        "raw_decision_score": -0.1 if verdict != "NORMAL" else 0.1,
                        "features": _make_valid_14_features(),
                    }
                    if verdict in ("NORMAL", "STATISTICAL_ANOMALY")
                    else None
                ),
            )
            record = AnalysisRecordResponse.model_validate(data)
            self.assertEqual(record.verdict, verdict)

    def test_24_fsm_failed_step_validation(self) -> None:
        """Verify FsmErrorDetail.failed_step accepts 1, rejects 0, and rejects negative values."""
        # 1. failed_step=1 is accepted
        detail_valid = FsmErrorDetail(
            error_type="INVALID_TRANSITION",
            failed_step=1,
            failed_event="PACK_ITEMS",
            state_at_failure="ORDER_CREATED",
            allowed_events=["CONFIRM_PAYMENT"],
        )
        self.assertEqual(detail_valid.failed_step, 1)

        # 2. failed_step=0 is rejected (ge=1 constraint)
        with self.assertRaises(ValidationError):
            FsmErrorDetail(
                error_type="INVALID_TRANSITION",
                failed_step=0,
                failed_event="PACK_ITEMS",
                state_at_failure="ORDER_CREATED",
                allowed_events=["CONFIRM_PAYMENT"],
            )

        # 3. negative failed_step is rejected
        with self.assertRaises(ValidationError):
            FsmErrorDetail(
                error_type="INVALID_TRANSITION",
                failed_step=-1,
                failed_event="PACK_ITEMS",
                state_at_failure="ORDER_CREATED",
                allowed_events=["CONFIRM_PAYMENT"],
            )

    def test_25_response_fsm_error_present_and_absent(self) -> None:
        """Verify FSM error detail is correctly validated when present and supports None."""
        # 1. Absent (None)
        data_no_err = _make_valid_analysis_record_dict(fsm_error=None)
        rec_no_err = AnalysisRecordResponse.model_validate(data_no_err)
        self.assertIsNone(rec_no_err.fsm_evaluation.error)

        # 2. Present
        err_detail = {
            "error_type": "INVALID_TRANSITION",
            "failed_step": 1,
            "failed_event": "PACK_ITEMS",
            "state_at_failure": "ORDER_CREATED",
            "allowed_events": ["CONFIRM_PAYMENT", "FAIL_PAYMENT"],
        }
        data_with_err = _make_valid_analysis_record_dict(fsm_error=err_detail)
        rec_with_err = AnalysisRecordResponse.model_validate(data_with_err)
        self.assertIsNotNone(rec_with_err.fsm_evaluation.error)
        self.assertEqual(rec_with_err.fsm_evaluation.error.error_type, "INVALID_TRANSITION")

    def test_26_response_ml_eval_present_and_absent(self) -> None:
        """Verify ml_evaluation supports None (when skipped) and complete schema when present."""
        # 1. Skipped (None)
        data_skip = _make_valid_analysis_record_dict(ml_eval=None, ml_skip_reason="FSM violation")
        rec_skip = AnalysisRecordResponse.model_validate(data_skip)
        self.assertIsNone(rec_skip.ml_evaluation)
        self.assertEqual(rec_skip.ml_skip_reason, "FSM violation")

        # 2. Present
        ml_eval = {
            "prediction": "NORMAL",
            "is_ml_anomaly": False,
            "anomaly_score": -0.15,
            "raw_decision_score": 0.15,
            "features": _make_valid_14_features(),
        }
        data_eval = _make_valid_analysis_record_dict(ml_eval=ml_eval, ml_skip_reason=None)
        rec_eval = AnalysisRecordResponse.model_validate(data_eval)
        self.assertIsNotNone(rec_eval.ml_evaluation)
        self.assertEqual(rec_eval.ml_evaluation.prediction, "NORMAL")

    def test_27_response_ml_skip_reason_present_and_absent(self) -> None:
        """Verify ml_skip_reason supports None and string values."""
        data_none = _make_valid_analysis_record_dict(ml_skip_reason=None)
        self.assertIsNone(AnalysisRecordResponse.model_validate(data_none).ml_skip_reason)

        data_str = _make_valid_analysis_record_dict(ml_skip_reason="Trace incomplete")
        self.assertEqual(
            AnalysisRecordResponse.model_validate(data_str).ml_skip_reason,
            "Trace incomplete",
        )

    def test_28_ml_eval_accepts_complete_14_features(self) -> None:
        """Verify MlEvaluationSchema accepts the exact 14 approved features."""
        features = _make_valid_14_features()
        self.assertEqual(len(features), 14)
        self.assertEqual(set(features.keys()), set(FEATURE_NAMES))

        ml_eval = MlEvaluationSchema(
            prediction="NORMAL",
            is_ml_anomaly=False,
            anomaly_score=-0.2,
            raw_decision_score=0.2,
            features=features,
        )
        self.assertEqual(len(ml_eval.features), 14)

    def test_29_ml_eval_rejects_missing_feature(self) -> None:
        """Verify MlEvaluationSchema rejects features dictionary missing an approved feature."""
        features = _make_valid_14_features()
        del features["unique_event_count"]
        with self.assertRaises(ValidationError):
            MlEvaluationSchema(
                prediction="NORMAL",
                is_ml_anomaly=False,
                anomaly_score=-0.2,
                raw_decision_score=0.2,
                features=features,
            )

    def test_30_ml_eval_rejects_unexpected_feature(self) -> None:
        """Verify MlEvaluationSchema rejects features dictionary with unexpected keys."""
        features = _make_valid_14_features()
        features["extra_unapproved_feature"] = 1.0
        with self.assertRaises(ValidationError):
            MlEvaluationSchema(
                prediction="NORMAL",
                is_ml_anomaly=False,
                anomaly_score=-0.2,
                raw_decision_score=0.2,
                features=features,
            )

    def test_31_ml_eval_rejects_non_numeric_feature_values(self) -> None:
        """Verify MlEvaluationSchema rejects string or object feature values."""
        features = _make_valid_14_features()
        features["seq_length"] = "four"  # string
        with self.assertRaises(ValidationError):
            MlEvaluationSchema(
                prediction="NORMAL",
                is_ml_anomaly=False,
                anomaly_score=-0.2,
                raw_decision_score=0.2,
                features=features,
            )

    def test_32_ml_eval_rejects_boolean_feature_values(self) -> None:
        """Verify MlEvaluationSchema explicitly rejects boolean feature values."""
        features = _make_valid_14_features()
        features["count_CONFIRM_PAYMENT"] = True
        with self.assertRaises(ValidationError):
            MlEvaluationSchema(
                prediction="NORMAL",
                is_ml_anomaly=False,
                anomaly_score=-0.2,
                raw_decision_score=0.2,
                features=features,
            )

    def test_33_ml_eval_rejects_nan_and_infinity_in_scores_and_features(self) -> None:
        """Verify NaN, +inf, and -inf are rejected in scores and features."""
        valid_feats = _make_valid_14_features()

        # Score rejection
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError):
                MlEvaluationSchema(
                    prediction="NORMAL",
                    is_ml_anomaly=False,
                    anomaly_score=non_finite,
                    raw_decision_score=0.2,
                    features=valid_feats,
                )
            with self.assertRaises(ValidationError):
                MlEvaluationSchema(
                    prediction="NORMAL",
                    is_ml_anomaly=False,
                    anomaly_score=-0.2,
                    raw_decision_score=non_finite,
                    features=valid_feats,
                )

        # Feature rejection
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            bad_feats = _make_valid_14_features()
            bad_feats["retry_ratio"] = non_finite
            with self.assertRaises(ValidationError):
                MlEvaluationSchema(
                    prediction="NORMAL",
                    is_ml_anomaly=False,
                    anomaly_score=-0.2,
                    raw_decision_score=0.2,
                    features=bad_feats,
                )

    def test_34_response_rejects_invalid_verdict(self) -> None:
        """Verify AnalysisRecordResponse rejects verdicts outside the approved 4-state taxonomy."""
        data = _make_valid_analysis_record_dict(verdict="UNAPPROVED_VERDICT")
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data)

    def test_35_response_rejects_invalid_anomaly_type(self) -> None:
        """Verify AnalysisRecordResponse rejects invalid anomaly_type strings."""
        data = _make_valid_analysis_record_dict(anomaly_type="UNKNOWN_ANOMALY")
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data)

    def test_36_diagnostics_rejects_invalid_severity(self) -> None:
        """Verify DiagnosticsSchema rejects severity values other than CRITICAL, WARNING, INFO."""
        with self.assertRaises(ValidationError):
            DiagnosticsSchema(
                severity="EXTREME",
                root_cause="cause",
                key_factors=[],
                recommended_action="action",
            )
        # Valid severities succeed
        for sev in ("CRITICAL", "WARNING", "INFO"):
            diag = DiagnosticsSchema(
                severity=sev,
                root_cause="cause",
                key_factors=[],
                recommended_action="action",
            )
            self.assertEqual(diag.severity, sev)

    # -----------------------------------------------------------------------
    # 4. UTC Timestamp Contract Validation Tests
    # -----------------------------------------------------------------------

    def test_37_timestamp_accepts_utc_aware_datetime(self) -> None:
        """Verify AnalysisRecordResponse accepts a timezone-aware UTC datetime."""
        aware_dt = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
        data = _make_valid_analysis_record_dict(created_at=aware_dt)
        rec = AnalysisRecordResponse.model_validate(data)
        self.assertEqual(rec.created_at.tzinfo, timezone.utc)
        self.assertEqual(rec.created_at, aware_dt)

    def test_38_timestamp_rejects_naive_datetime(self) -> None:
        """Verify AnalysisRecordResponse rejects naive datetime objects and naive strings."""
        naive_dt = datetime(2026, 11, 9, 12, 0, 0)  # No tzinfo
        data_dt = _make_valid_analysis_record_dict(created_at=naive_dt)
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data_dt)

        data_str = _make_valid_analysis_record_dict(created_at="2026-11-09T12:00:00")
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data_str)

    def test_39_timestamp_rejects_non_zero_offset_datetime(self) -> None:
        """Verify AnalysisRecordResponse rejects datetime objects with non-zero UTC offsets (+05:30)."""
        offset_tz = timezone(timedelta(hours=5, minutes=30))
        offset_dt = datetime(2026, 11, 9, 17, 30, 0, tzinfo=offset_tz)
        data = _make_valid_analysis_record_dict(created_at=offset_dt)
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data)

    def test_40_timestamp_accepts_iso_utc_string_ending_in_z(self) -> None:
        """Verify AnalysisRecordResponse accepts an ISO-8601 UTC string ending in 'Z'."""
        data = _make_valid_analysis_record_dict(created_at="2026-11-09T12:00:00Z")
        rec = AnalysisRecordResponse.model_validate(data)
        self.assertEqual(rec.created_at.tzinfo, timezone.utc)
        self.assertEqual(rec.created_at.year, 2026)
        self.assertEqual(rec.created_at.hour, 12)

    def test_41_timestamp_rejects_iso_string_with_non_zero_offset(self) -> None:
        """Verify AnalysisRecordResponse rejects an ISO-8601 string with a non-zero offset (+05:30)."""
        data = _make_valid_analysis_record_dict(created_at="2026-11-09T17:30:00+05:30")
        with self.assertRaises(ValidationError):
            AnalysisRecordResponse.model_validate(data)

    # -----------------------------------------------------------------------
    # 5. Common and Operational Schemas Validation Tests
    # -----------------------------------------------------------------------

    def test_42_common_batch_and_order_history_responses(self) -> None:
        """Verify BatchAnalysisResponse and OrderHistoryResponse with returned_count."""
        rec_dict = _make_valid_analysis_record_dict()
        rec = AnalysisRecordResponse.model_validate(rec_dict)

        # BatchAnalysisResponse
        batch_resp = BatchAnalysisResponse(
            results=[rec],
            summary=BatchSummaryMetrics(
                total_orders=1,
                normal_count=1,
                rule_violation_count=0,
                statistical_anomaly_count=0,
                valid_incomplete_count=0,
                total_anomalies=0,
                anomaly_rate=0.0,
            ),
        )
        self.assertEqual(len(batch_resp.results), 1)
        self.assertEqual(batch_resp.summary.total_orders, 1)

        # OrderHistoryResponse with returned_count (not total_records)
        hist_resp = OrderHistoryResponse(
            order_id="ORD-001",
            returned_count=1,
            analyses=[rec],
        )
        self.assertEqual(hist_resp.order_id, "ORD-001")
        self.assertEqual(hist_resp.returned_count, 1)

    def test_43_common_operational_and_error_schemas(self) -> None:
        """Verify LivenessResponse, ReadinessResponse (without model_loaded_in_memory), DatabaseSummaryMetricsResponse, and ErrorResponse."""
        # Liveness
        live = LivenessResponse()
        self.assertEqual(live.status, "alive")

        # Readiness (note: model_loaded_in_memory was removed per Phase 5 contract alignment)
        ready = ReadinessResponse(
            service_status="OPERATIONAL",
            dataset_spec_version="1.4",
            model_artifact_path="backend/ml/models/isolation_forest.joblib",
            model_artifact_exists=True,
            supported_verdicts=["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"],
        )
        self.assertEqual(ready.service_status, "OPERATIONAL")
        self.assertEqual(ready.dataset_spec_version, "1.4")
        self.assertTrue(ready.model_artifact_exists)

        # DatabaseSummaryMetricsResponse
        metrics = DatabaseSummaryMetricsResponse(
            total_analyses=10,
            normal_count=6,
            rule_violation_count=2,
            statistical_anomaly_count=1,
            valid_incomplete_count=1,
            total_anomalies=3,
            anomaly_rate=0.3,
            unique_orders_count=8,
        )
        self.assertEqual(metrics.total_analyses, 10)
        self.assertEqual(metrics.anomaly_rate, 0.3)

        # ErrorResponse
        err = ErrorResponse(
            error=ErrorDetail(
                code="NOT_FOUND",
                message="Analysis record with ID 999 not found",
                details={"analysis_id": 999},
            )
        )
        self.assertEqual(err.error.code, "NOT_FOUND")

    def test_44_batch_accepts_raw_structured_dictionaries(self) -> None:
        """Verify JSON-style dictionaries are accepted as structured batch items."""
        payload = {
            "traces": [
                {
                    "order_id": "ORD-101",
                    "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS",
                },
                {
                    "order_id": None,
                    "event_sequence": ["CONFIRM_PAYMENT", "PACK_ITEMS"],
                },
            ]
        }

        batch = BatchAnalysisRequest.model_validate(payload)

        self.assertEqual(len(batch.traces), 2)
        self.assertIsInstance(batch.traces[0], TraceItemRequest)
        self.assertIsInstance(batch.traces[1], TraceItemRequest)
        self.assertEqual(batch.traces[0].order_id, "ORD-101")
        self.assertEqual(
            batch.traces[1].event_sequence,
            ["CONFIRM_PAYMENT", "PACK_ITEMS"],
        )


if __name__ == "__main__":
    unittest.main()
