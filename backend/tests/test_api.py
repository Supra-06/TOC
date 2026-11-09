"""Comprehensive integration test suite for REST API endpoints (Phase 7 Step 4).

Verifies:
 - Liveness & Readiness endpoints
 - Single workflow analysis ingestion & persistence
 - Batch workflow analysis ingestion & atomic persistence
 - Analysis retrieval, order history, pagination, and filtering
 - Aggregated database summary metrics
 - Standardized error envelopes and HTTP status codes
 - Complete database isolation with zero physical writes

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_api -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.dependencies import get_db_session, get_service
from backend.api.mapping import RecordMappingError
from backend.core.config import DEFAULT_MODEL_PATH
from backend.database.models import Base
from backend.database.repository import create_analysis_record
from backend.main import create_app
from backend.ml.feature_extraction.extractor import FEATURE_NAMES


def _make_14_features() -> Dict[str, float]:
    """Helper producing full 14-feature dictionary."""
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


class DeterministicAnomalyServiceDouble:
    """Test double implementing AnomalyService contract with predictable classifications."""

    def __init__(self, operational: bool = True, model_exists: bool = True):
        self.operational = operational
        self.model_exists = model_exists
        self._model_path = DEFAULT_MODEL_PATH
        self._model = None

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "service_status": "OPERATIONAL" if self.operational else "DEGRADED",
            "dataset_spec_version": "1.4",
            "model_artifact_path": str(self._model_path),
            "model_artifact_exists": self.model_exists,
            "supported_verdicts": ["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"],
        }

    def analyze_order(
        self,
        event_sequence: Union[str, Sequence[str]],
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        seq_str = ";".join(event_sequence) if isinstance(event_sequence, (list, tuple)) else str(event_sequence)

        if "RULE_VIOLATION" in seq_str or "INVALID" in seq_str:
            verdict = "RULE_VIOLATION"
            is_anomaly = True
            anomaly_type = "RULE_VIOLATION"
            fsm_eval = {
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
            }
            ml_eval = None
            ml_skip_reason = "ML evaluation skipped: trace failed FSM rule validation (RULE_VIOLATION)."
            diag = {
                "severity": "CRITICAL",
                "root_cause": "FSM transition rule breach.",
                "key_factors": ["Invalid event sequence"],
                "recommended_action": "Audit order flow.",
            }
        elif "STATISTICAL_ANOMALY" in seq_str:
            verdict = "STATISTICAL_ANOMALY"
            is_anomaly = True
            anomaly_type = "STATISTICAL_ANOMALY"
            fsm_eval = {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 4,
                "successful_transitions": 4,
                "error": None,
            }
            ml_eval = {
                "prediction": "ANOMALY",
                "is_ml_anomaly": True,
                "anomaly_score": 0.42,
                "raw_decision_score": -0.42,
                "features": _make_14_features(),
            }
            ml_skip_reason = None
            diag = {
                "severity": "WARNING",
                "root_cause": "Statistical anomaly detected.",
                "key_factors": ["High volume"],
                "recommended_action": "Investigate logs.",
            }
        elif seq_str == "" or seq_str == "[]" or "INCOMPLETE" in seq_str or seq_str == "CONFIRM_PAYMENT":
            verdict = "VALID_INCOMPLETE"
            is_anomaly = False
            anomaly_type = None
            fsm_eval = {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID",
                "total_steps": 1 if seq_str else 0,
                "successful_transitions": 1 if seq_str else 0,
                "error": None,
            }
            ml_eval = None
            ml_skip_reason = "ML evaluation skipped: trace is valid but incomplete (VALID_INCOMPLETE)."
            diag = {
                "severity": "INFO",
                "root_cause": "Order in progress.",
                "key_factors": ["Awaiting completion"],
                "recommended_action": "Wait for next event.",
            }
        else:
            verdict = "NORMAL"
            is_anomaly = False
            anomaly_type = None
            fsm_eval = {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 4,
                "successful_transitions": 4,
                "error": None,
            }
            ml_eval = {
                "prediction": "NORMAL",
                "is_ml_anomaly": False,
                "anomaly_score": -0.2,
                "raw_decision_score": 0.2,
                "features": _make_14_features(),
            }
            ml_skip_reason = None
            diag = {
                "severity": "INFO",
                "root_cause": "Normal order completion.",
                "key_factors": ["Standard progression"],
                "recommended_action": "No remediation required.",
            }

        return {
            "order_id": str(order_id) if order_id is not None else None,
            "verdict": verdict,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "summary": f"Workflow result: {verdict}",
            "diagnostics": diag,
            "fsm_evaluation": fsm_eval,
            "ml_evaluation": ml_eval,
            "ml_skip_reason": ml_skip_reason,
        }

    def analyze_order_batch(
        self,
        traces: Sequence[Union[str, Sequence[str], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        normal_count = 0
        rule_violation_count = 0
        statistical_anomaly_count = 0
        valid_incomplete_count = 0

        for item in traces:
            if isinstance(item, dict):
                order_id = item.get("order_id")
                seq = item.get("event_sequence", "")
            else:
                order_id = None
                seq = item

            res = self.analyze_order(seq, order_id=order_id)
            results.append(res)

            v = res["verdict"]
            if v == "NORMAL":
                normal_count += 1
            elif v == "RULE_VIOLATION":
                rule_violation_count += 1
            elif v == "STATISTICAL_ANOMALY":
                statistical_anomaly_count += 1
            elif v == "VALID_INCOMPLETE":
                valid_incomplete_count += 1

        total = len(results)
        anomalies = rule_violation_count + statistical_anomaly_count
        rate = float(anomalies / total) if total > 0 else 0.0

        return {
            "results": results,
            "summary": {
                "total_orders": total,
                "normal_count": normal_count,
                "rule_violation_count": rule_violation_count,
                "statistical_anomaly_count": statistical_anomaly_count,
                "valid_incomplete_count": valid_incomplete_count,
                "total_anomalies": anomalies,
                "anomaly_rate": rate,
            },
        }


class TestApiIntegration(unittest.TestCase):
    """Integration test suite covering FastAPI routes, persistence, and error handling."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create an isolated in-memory SQLite database for testing."""
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionFactory = sessionmaker(bind=cls.engine, autocommit=False, autoflush=False)

    def setUp(self) -> None:
        """Set up test application, clean tables, dependency overrides, and TestClient."""
        # Clean tables in memory
        with self.engine.begin() as connection:
            connection.execute(Base.metadata.tables["workflow_analyses"].delete())

        self.service_double = DeterministicAnomalyServiceDouble()
        self.app = create_app(init_db_on_startup=False)

        def override_db():
            session = self.SessionFactory()
            try:
                yield session
            finally:
                session.close()

        def override_svc():
            return self.service_double

        self.app.dependency_overrides[get_db_session] = override_db
        self.app.dependency_overrides[get_service] = override_svc
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Clear dependency overrides."""
        self.app.dependency_overrides.clear()

    # -----------------------------------------------------------------------
    # Liveness & Readiness Tests (7 - 13)
    # -----------------------------------------------------------------------

    def test_07_root_health_returns_200(self) -> None:
        """Verify GET /health returns 200 with LivenessResponse."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "alive"})

    def test_08_api_health_live_returns_200(self) -> None:
        """Verify GET /api/v1/health/live returns 200 with LivenessResponse."""
        res = self.client.get("/api/v1/health/live")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "alive"})

    def test_09_liveness_does_not_access_database_or_load_model(self) -> None:
        """Verify liveness probe succeeds even if database dependency fails."""
        def broken_db():
            raise RuntimeError("Database unavailable")
            yield  # pragma: no cover

        self.app.dependency_overrides[get_db_session] = broken_db
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "alive"})

    def test_10_readiness_returns_200_when_dependencies_healthy(self) -> None:
        """Verify GET /api/v1/health/ready and GET /api/v1/health return 200 when healthy."""
        for path in ("/api/v1/health", "/api/v1/health/ready"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["service_status"], "OPERATIONAL")
            self.assertEqual(data["dataset_spec_version"], "1.4")
            self.assertTrue(data["model_artifact_exists"])
            self.assertIn("NORMAL", data["supported_verdicts"])

    def test_11_readiness_returns_503_when_model_artifact_unavailable(self) -> None:
        """Verify readiness returns 503 when model artifact does not exist on disk."""
        self.service_double.model_exists = False
        res = self.client.get("/api/v1/health/ready")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["error"]["code"], "SERVICE_UNAVAILABLE")

    def test_12_readiness_returns_503_when_database_unavailable(self) -> None:
        """Verify readiness returns 503 when database query fails."""
        def failing_db():
            mock_session = MagicMock()
            mock_session.execute.side_effect = RuntimeError("DB connection dropped")
            yield mock_session

        self.app.dependency_overrides[get_db_session] = failing_db
        res = self.client.get("/api/v1/health/ready")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["error"]["code"], "SERVICE_UNAVAILABLE")

    def test_13_readiness_does_not_force_model_loading(self) -> None:
        """Verify readiness probe does not load the ML model into memory."""
        res = self.client.get("/api/v1/health/ready")
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(self.service_double._model)

    # -----------------------------------------------------------------------
    # Single-Analysis Tests (14 - 22)
    # -----------------------------------------------------------------------

    def test_14_valid_normal_trace_returns_201(self) -> None:
        """Verify valid normal trace returns 201 with AnalysisRecordResponse."""
        payload = {
            "order_id": "ORD-NORM-1",
            "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["order_id"], "ORD-NORM-1")
        self.assertEqual(data["verdict"], "NORMAL")
        self.assertFalse(data["is_anomaly"])
        self.assertIsNone(data["anomaly_type"])
        self.assertIsNotNone(data["ml_evaluation"])
        self.assertEqual(data["ml_evaluation"]["prediction"], "NORMAL")

    def test_15_rule_violation_returns_201(self) -> None:
        """Verify rule violation returns 201 with RULE_VIOLATION verdict and FSM error."""
        payload = {
            "order_id": "ORD-RULE-1",
            "event_sequence": "CONFIRM_PAYMENT;DELIVER_ITEMS;RULE_VIOLATION",
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["verdict"], "RULE_VIOLATION")
        self.assertTrue(data["is_anomaly"])
        self.assertEqual(data["anomaly_type"], "RULE_VIOLATION")
        self.assertIsNone(data["ml_evaluation"])
        self.assertIsNotNone(data["ml_skip_reason"])
        self.assertIsNotNone(data["fsm_evaluation"]["error"])
        self.assertEqual(data["diagnostics"]["severity"], "CRITICAL")

    def test_16_statistical_anomaly_returns_201(self) -> None:
        """Verify statistical anomaly returns 201 with STATISTICAL_ANOMALY verdict."""
        payload = {
            "order_id": "ORD-STAT-1",
            "event_sequence": "CONFIRM_PAYMENT;STATISTICAL_ANOMALY",
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["verdict"], "STATISTICAL_ANOMALY")
        self.assertTrue(data["is_anomaly"])
        self.assertEqual(data["anomaly_type"], "STATISTICAL_ANOMALY")
        self.assertIsNotNone(data["ml_evaluation"])
        self.assertEqual(data["ml_evaluation"]["prediction"], "ANOMALY")
        self.assertEqual(data["diagnostics"]["severity"], "WARNING")

    def test_17_valid_incomplete_trace_returns_201(self) -> None:
        """Verify valid incomplete trace returns 201 with VALID_INCOMPLETE verdict."""
        payload = {
            "order_id": "ORD-INC-1",
            "event_sequence": "CONFIRM_PAYMENT",
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(data["is_anomaly"])
        self.assertIsNone(data["ml_evaluation"])
        self.assertIsNotNone(data["ml_skip_reason"])

    def test_18_original_string_input_is_preserved(self) -> None:
        """Verify delimited string input format and raw payload are preserved."""
        raw_seq = "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS"
        res = self.client.post("/api/v1/analyses", json={"event_sequence": raw_seq})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["input_format"], "DELIMITED_STRING")
        self.assertEqual(data["raw_input_payload"], raw_seq)

    def test_19_original_list_input_is_preserved(self) -> None:
        """Verify list input format and raw payload are preserved."""
        raw_list = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"]
        res = self.client.post("/api/v1/analyses", json={"event_sequence": raw_list})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["input_format"], "EVENT_LIST")
        self.assertEqual(data["raw_input_payload"], raw_list)

    def test_20_optional_order_id_is_preserved(self) -> None:
        """Verify optional order_id is preserved when provided and None when omitted."""
        res_with_id = self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-CUSTOM-99", "event_sequence": "CONFIRM_PAYMENT"},
        )
        self.assertEqual(res_with_id.status_code, 201)
        self.assertEqual(res_with_id.json()["order_id"], "ORD-CUSTOM-99")

        res_no_id = self.client.post(
            "/api/v1/analyses",
            json={"event_sequence": "CONFIRM_PAYMENT"},
        )
        self.assertEqual(res_no_id.status_code, 201)
        self.assertIsNone(res_no_id.json()["order_id"])

    def test_21_result_is_persisted_and_retrievable(self) -> None:
        """Verify single analysis is persisted and can be retrieved by primary key."""
        post_res = self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-RETRIEVE-1", "event_sequence": "CONFIRM_PAYMENT"},
        )
        self.assertEqual(post_res.status_code, 201)
        analysis_id = post_res.json()["id"]

        get_res = self.client.get(f"/api/v1/analyses/{analysis_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["id"], analysis_id)
        self.assertEqual(get_res.json()["order_id"], "ORD-RETRIEVE-1")

    def test_22_invalid_request_types_return_422(self) -> None:
        """Verify invalid payload types (missing event_sequence, integer sequence) return 422."""
        res_missing = self.client.post("/api/v1/analyses", json={"order_id": "ORD-1"})
        self.assertEqual(res_missing.status_code, 422)
        self.assertEqual(res_missing.json()["error"]["code"], "VALIDATION_ERROR")

        res_int = self.client.post("/api/v1/analyses", json={"event_sequence": 12345})
        self.assertEqual(res_int.status_code, 422)

        res_tuple = self.client.post(
            "/api/v1/analyses",
            json={"event_sequence": ["CONFIRM_PAYMENT", 123]},
        )
        self.assertEqual(res_tuple.status_code, 422)

    # -----------------------------------------------------------------------
    # Batch-Analysis Tests (23 - 32)
    # -----------------------------------------------------------------------

    def test_23_normal_batch_returns_201(self) -> None:
        """Verify normal batch analysis submission returns 201 with summary."""
        payload = {
            "traces": [
                {"order_id": "BATCH-1", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
                {"order_id": "BATCH-2", "event_sequence": ["CONFIRM_PAYMENT", "PACK_ITEMS"]},
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["summary"]["total_orders"], 2)
        self.assertEqual(data["summary"]["normal_count"], 2)
        self.assertEqual(data["summary"]["anomaly_rate"], 0.0)

    def test_24_mixed_input_formats_are_accepted(self) -> None:
        """Verify batch accepts string, list, and structured TraceItemRequest objects."""
        payload = {
            "traces": [
                "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS",
                ["CONFIRM_PAYMENT", "PACK_ITEMS"],
                {"order_id": "ORD-STRUCTURED", "event_sequence": "CONFIRM_PAYMENT"},
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        results = res.json()["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["input_format"], "DELIMITED_STRING")
        self.assertEqual(results[1]["input_format"], "EVENT_LIST")
        self.assertEqual(results[2]["order_id"], "ORD-STRUCTURED")

    def test_25_different_verdicts_are_preserved(self) -> None:
        """Verify batch preserves different verdicts across traces."""
        payload = {
            "traces": [
                "CONFIRM_PAYMENT;PACK_ITEMS",  # NORMAL
                "CONFIRM_PAYMENT;RULE_VIOLATION",  # RULE_VIOLATION
                "CONFIRM_PAYMENT;STATISTICAL_ANOMALY",  # STATISTICAL_ANOMALY
                "",  # VALID_INCOMPLETE
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        results = res.json()["results"]
        verdicts = [r["verdict"] for r in results]
        self.assertEqual(
            verdicts,
            ["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"],
        )

    def test_26_original_input_values_preserved_for_every_trace(self) -> None:
        """Verify each trace in batch preserves its original input exactly."""
        payload = {
            "traces": [
                "CONFIRM_PAYMENT;PACK_ITEMS",
                ["CONFIRM_PAYMENT", "PACK_ITEMS"],
                "",
                [],
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        results = res.json()["results"]
        self.assertEqual(results[0]["raw_input_payload"], "CONFIRM_PAYMENT;PACK_ITEMS")
        self.assertEqual(results[1]["raw_input_payload"], ["CONFIRM_PAYMENT", "PACK_ITEMS"])
        self.assertEqual(results[2]["raw_input_payload"], "")
        self.assertEqual(results[3]["raw_input_payload"], [])

    def test_27_batch_summary_counts_are_correct(self) -> None:
        """Verify summary metrics calculation in batch response."""
        payload = {
            "traces": [
                "NORMAL_TRACE",
                "RULE_VIOLATION_TRACE",
                "STATISTICAL_ANOMALY_TRACE",
                "INCOMPLETE_TRACE",
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        summary = res.json()["summary"]
        self.assertEqual(summary["total_orders"], 4)
        self.assertEqual(summary["normal_count"], 1)
        self.assertEqual(summary["rule_violation_count"], 1)
        self.assertEqual(summary["statistical_anomaly_count"], 1)
        self.assertEqual(summary["valid_incomplete_count"], 1)
        self.assertEqual(summary["total_anomalies"], 2)
        self.assertEqual(summary["anomaly_rate"], 0.5)

    def test_28_batch_persistence_is_atomic(self) -> None:
        """Verify database transaction rollback ensures zero writes when batch fails."""
        with patch("backend.api.routes.create_analysis_batch") as mock_batch:
            mock_batch.side_effect = RuntimeError("Database write failure")
            res = self.client.post("/api/v1/analyses/batch", json={"traces": ["CONFIRM_PAYMENT"]})
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.json()["error"]["code"], "INTERNAL_SERVER_ERROR")

        # Verify nothing persisted in database
        with self.SessionFactory() as session:
            count = session.query(Base.metadata.tables["workflow_analyses"]).count()
            self.assertEqual(count, 0)

    def test_29_empty_batch_is_rejected(self) -> None:
        """Verify BatchAnalysisRequest rejects an empty traces list with 422."""
        res = self.client.post("/api/v1/analyses/batch", json={"traces": []})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["error"]["code"], "VALIDATION_ERROR")

    def test_30_batch_exceeding_500_traces_is_rejected(self) -> None:
        """Verify BatchAnalysisRequest rejects traces exceeding 500 items with 422."""
        payload = {"traces": ["CONFIRM_PAYMENT" for _ in range(501)]}
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_31_batch_size_500_accepted_by_schema(self) -> None:
        """Verify exactly 500 traces are accepted at the upper boundary."""
        payload = {"traces": ["CONFIRM_PAYMENT" for _ in range(500)]}
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(res.json()["results"]), 500)

    def test_32_invalid_trace_structure_returns_422(self) -> None:
        """Verify invalid trace structure inside batch (integer, invalid dict) returns 422."""
        res_int = self.client.post("/api/v1/analyses/batch", json={"traces": [12345]})
        self.assertEqual(res_int.status_code, 422)

        res_dict = self.client.post("/api/v1/analyses/batch", json={"traces": [{"unknown_key": "val"}]})
        self.assertEqual(res_dict.status_code, 422)

    # -----------------------------------------------------------------------
    # Retrieval and Metrics Tests (33 - 43)
    # -----------------------------------------------------------------------

    def test_33_existing_analysis_retrieval_returns_200(self) -> None:
        """Verify GET /api/v1/analyses/{id} returns 200 for existing record."""
        post_res = self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-EXISTING-1", "event_sequence": "CONFIRM_PAYMENT"},
        )
        rec_id = post_res.json()["id"]

        get_res = self.client.get(f"/api/v1/analyses/{rec_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["id"], rec_id)

    def test_34_missing_analysis_returns_404(self) -> None:
        """Verify GET /api/v1/analyses/{id} returns 404 when ID does not exist."""
        res = self.client.get("/api/v1/analyses/999999")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["error"]["code"], "NOT_FOUND")

    def test_35_order_history_returns_correct_records(self) -> None:
        """Verify GET /api/v1/orders/{order_id}/analyses returns historical records."""
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-HIST-1", "event_sequence": "CONFIRM_PAYMENT"},
        )
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-HIST-1", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
        )

        res = self.client.get("/api/v1/orders/ORD-HIST-1/analyses")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["order_id"], "ORD-HIST-1")
        self.assertEqual(data["returned_count"], 2)
        self.assertEqual(len(data["analyses"]), 2)

    def test_36_empty_order_history_returns_empty_result(self) -> None:
        """Verify GET /api/v1/orders/{order_id}/analyses returns returned_count=0 when none exist."""
        res = self.client.get("/api/v1/orders/ORD-NONEXISTENT/analyses")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["order_id"], "ORD-NONEXISTENT")
        self.assertEqual(data["returned_count"], 0)
        self.assertEqual(data["analyses"], [])

    def test_37_pagination_skip_and_limit(self) -> None:
        """Verify pagination with skip and limit parameters."""
        for i in range(5):
            self.client.post(
                "/api/v1/analyses",
                json={"order_id": f"ORD-PAG-{i}", "event_sequence": "CONFIRM_PAYMENT"},
            )

        res = self.client.get("/api/v1/analyses?skip=2&limit=2")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["skip"], 2)
        self.assertEqual(data["limit"], 2)
        self.assertEqual(len(data["items"]), 2)

    def test_38_verdict_filtering_works(self) -> None:
        """Verify filtering by verdict parameter."""
        self.client.post("/api/v1/analyses", json={"event_sequence": "CONFIRM_PAYMENT"})  # VALID_INCOMPLETE
        self.client.post("/api/v1/analyses", json={"event_sequence": "RULE_VIOLATION"})  # RULE_VIOLATION

        res_rule = self.client.get("/api/v1/analyses?verdict=RULE_VIOLATION")
        self.assertEqual(res_rule.status_code, 200)
        data = res_rule.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["verdict"], "RULE_VIOLATION")

    def test_39_anomaly_filtering_works(self) -> None:
        """Verify filtering by is_anomaly boolean flag."""
        self.client.post("/api/v1/analyses", json={"event_sequence": "CONFIRM_PAYMENT"})  # is_anomaly=False
        self.client.post("/api/v1/analyses", json={"event_sequence": "RULE_VIOLATION"})  # is_anomaly=True

        res_anom = self.client.get("/api/v1/analyses?is_anomaly=true")
        self.assertEqual(res_anom.status_code, 200)
        self.assertEqual(res_anom.json()["total"], 1)
        self.assertTrue(res_anom.json()["items"][0]["is_anomaly"])

    def test_40_invalid_pagination_returns_422(self) -> None:
        """Verify negative skip or non-positive limit returns 422."""
        res_neg_skip = self.client.get("/api/v1/analyses?skip=-1")
        self.assertEqual(res_neg_skip.status_code, 422)

        res_zero_limit = self.client.get("/api/v1/analyses?limit=0")
        self.assertEqual(res_zero_limit.status_code, 422)

    def test_41_invalid_verdict_filter_returns_422(self) -> None:
        """Verify unsupported verdict filter returns 422."""
        res = self.client.get("/api/v1/analyses?verdict=INVALID_VERDICT_NAME")
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["error"]["code"], "UNPROCESSABLE_ENTITY")

    def test_42_metrics_match_repository_results(self) -> None:
        """Verify GET /api/v1/metrics reflects persisted database counts."""
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-M-1", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"},
        )  # NORMAL
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-M-2", "event_sequence": "RULE_VIOLATION"},
        )  # RULE_VIOLATION

        res = self.client.get("/api/v1/metrics")
        self.assertEqual(res.status_code, 200)
        metrics = res.json()
        self.assertEqual(metrics["total_analyses"], 2)
        self.assertEqual(metrics["normal_count"], 1)
        self.assertEqual(metrics["rule_violation_count"], 1)
        self.assertEqual(metrics["total_anomalies"], 1)
        self.assertEqual(metrics["anomaly_rate"], 0.5)
        self.assertEqual(metrics["unique_orders_count"], 2)

    def test_43_empty_database_metrics_handled_correctly(self) -> None:
        """Verify metrics endpoint returns zeroed metrics on empty database."""
        res = self.client.get("/api/v1/metrics")
        self.assertEqual(res.status_code, 200)
        metrics = res.json()
        self.assertEqual(metrics["total_analyses"], 0)
        self.assertEqual(metrics["total_anomalies"], 0)
        self.assertEqual(metrics["anomaly_rate"], 0.0)
        self.assertEqual(metrics["unique_orders_count"], 0)

    # -----------------------------------------------------------------------
    # Error Handling Tests (44 - 47)
    # -----------------------------------------------------------------------

    def test_44_database_failures_return_safe_500_response(self) -> None:
        """Verify unhandled database failures return safe 500 without leaking credentials."""
        with patch("backend.api.routes.create_analysis_record") as mock_create:
            mock_create.side_effect = RuntimeError("sqlite3.OperationalError: disk I/O error")
            res = self.client.post("/api/v1/analyses", json={"event_sequence": "CONFIRM_PAYMENT"})
            self.assertEqual(res.status_code, 500)
            data = res.json()
            self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")
            self.assertEqual(data["error"]["message"], "An internal server error occurred.")
            self.assertNotIn("sqlite3", str(data))

    def test_45_mapping_failures_do_not_return_successful_response(self) -> None:
        """Verify mapping errors return 500 MAPPING_ERROR instead of misleading 200."""
        with patch("backend.api.routes.record_to_response") as mock_map:
            mock_map.side_effect = RecordMappingError("Corrupted record payload")
            res = self.client.post("/api/v1/analyses", json={"event_sequence": "CONFIRM_PAYMENT"})
            self.assertEqual(res.status_code, 500)
            data = res.json()
            self.assertEqual(data["error"]["code"], "MAPPING_ERROR")

    def test_46_readiness_failures_return_safe_503(self) -> None:
        """Verify readiness failures return HTTP 503 SERVICE_UNAVAILABLE error envelope."""
        self.service_double.operational = False
        res = self.client.get("/api/v1/health/ready")
        self.assertEqual(res.status_code, 503)
        data = res.json()
        self.assertEqual(data["error"]["code"], "SERVICE_UNAVAILABLE")

    def test_47_error_response_bodies_match_approved_schema(self) -> None:
        """Verify error responses conform to ErrorResponse structure across status codes."""
        # 1. 404 Not Found
        res_404 = self.client.get("/api/v1/analyses/999999")
        self.assertIn("error", res_404.json())
        self.assertIn("code", res_404.json()["error"])
        self.assertIn("message", res_404.json()["error"])

        # 2. 422 Validation Error
        res_422 = self.client.post("/api/v1/analyses", json={})
        self.assertIn("error", res_422.json())
        self.assertEqual(res_422.json()["error"]["code"], "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
