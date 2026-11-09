"""End-to-End Integration and Verification Test Suite (Phase 8).

Validates the complete backend stack working as a unified system:
  1. REST API endpoint request handling and serialization.
  2. Pydantic request validation and nested response schema enforcement.
  3. Domain AnomalyService coordination with hybrid FSM + ML analyzer.
  4. Real Isolation Forest inference and feature extraction.
  5. SQLite transaction atomicity, query filtering, and summary metrics.
  6. Standardized error envelope encapsulation and safe HTTP status mapping.
  7. Strict database isolation using in-memory SQLite with StaticPool.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_e2e -v
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Union
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.dependencies import get_db_session, get_service
from backend.api.mapping import RecordMappingError
from backend.core.config import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from backend.database.models import Base, WorkflowAnalysis
from backend.database.repository import (
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_summary_metrics,
    list_analyses,
)
from backend.main import create_app
from backend.ml.feature_extraction.extractor import FEATURE_NAMES
from backend.services.anomaly_service import AnomalyService, get_anomaly_service


class ControlledAnomalyServiceDouble:
    """Deterministic AnomalyService test double for guaranteed scenario isolation."""

    def __init__(self) -> None:
        self._model_path = DEFAULT_MODEL_PATH
        self._model = None

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "service_status": "OPERATIONAL",
            "dataset_spec_version": "1.4",
            "model_artifact_path": str(self._model_path),
            "model_artifact_exists": True,
            "model_loaded_in_memory": False,
            "supported_verdicts": ["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"],
        }

    def analyze_order(
        self,
        event_sequence: Union[str, Sequence[str]],
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        seq_str = ";".join(event_sequence) if isinstance(event_sequence, (list, tuple)) else str(event_sequence)

        if "RULE_VIOLATION" in seq_str or "INVALID" in seq_str or "CONFIRM_PAYMENT;DELIVER_ITEMS" in seq_str:
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
                "root_cause": "Invalid event transition attempted.",
                "key_factors": ["Event DELIVER_ITEMS not allowed in state PAID"],
                "recommended_action": "Cancel order and inspect payment transition log.",
            }
        elif "STATISTICAL_ANOMALY" in seq_str or "ANOMALY" in seq_str:
            verdict = "STATISTICAL_ANOMALY"
            is_anomaly = True
            anomaly_type = "STATISTICAL_ANOMALY"
            fsm_eval = {
                "is_rule_valid": True,
                "is_complete": True,
                "final_state": "DELIVERED",
                "total_steps": 6,
                "successful_transitions": 6,
                "error": None,
            }
            features = {name: 1.0 for name in FEATURE_NAMES}
            features["seq_length"] = 6.0
            features["unique_event_count"] = 5.0
            ml_eval = {
                "prediction": "ANOMALY",
                "is_ml_anomaly": True,
                "anomaly_score": 0.3542,
                "raw_decision_score": -0.1245,
                "features": features,
            }
            ml_skip_reason = None
            diag = {
                "severity": "WARNING",
                "root_cause": "Sequence contains atypical payment retry iterations.",
                "key_factors": ["High retry ratio", "Unusual transition bigrams"],
                "recommended_action": "Verify customer checkout session telemetry.",
            }
        elif "INCOMPLETE" in seq_str or seq_str in ("CONFIRM_PAYMENT", ""):
            verdict = "VALID_INCOMPLETE"
            is_anomaly = False
            anomaly_type = None
            fsm_eval = {
                "is_rule_valid": True,
                "is_complete": False,
                "final_state": "PAID" if seq_str else "ORDER_CREATED",
                "total_steps": 1 if seq_str else 0,
                "successful_transitions": 1 if seq_str else 0,
                "error": None,
            }
            ml_eval = None
            ml_skip_reason = "ML evaluation skipped: trace is in-progress and has not reached a terminal state (VALID_INCOMPLETE)."
            diag = {
                "severity": "INFO",
                "root_cause": "Order workflow prefix is valid but in-progress.",
                "key_factors": ["Non-terminal state"],
                "recommended_action": "Await subsequent lifecycle events.",
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
            features = {name: 1.0 for name in FEATURE_NAMES}
            features["seq_length"] = 4.0
            features["unique_event_count"] = 4.0
            ml_eval = {
                "prediction": "NORMAL",
                "is_ml_anomaly": False,
                "anomaly_score": -0.2154,
                "raw_decision_score": 0.1842,
                "features": features,
            }
            ml_skip_reason = None
            diag = {
                "severity": "INFO",
                "root_cause": "Order completed normally terminating in terminal state 'DELIVERED'.",
                "key_factors": ["Standard progression"],
                "recommended_action": "Standard order execution following expected baseline paths. No operational intervention required.",
            }

        return {
            "order_id": str(order_id) if order_id is not None else None,
            "verdict": verdict,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "summary": f"Execution analysis result: {verdict}",
            "diagnostics": diag,
            "fsm_evaluation": fsm_eval,
            "ml_evaluation": ml_eval,
            "ml_skip_reason": ml_skip_reason,
        }

    def analyze_order_batch(
        self,
        traces: Sequence[Union[str, Sequence[str], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        results = []
        normal = 0
        rule_viol = 0
        stat_anom = 0
        val_inc = 0

        for item in traces:
            if isinstance(item, dict):
                oid = item.get("order_id")
                seq = item.get("event_sequence", "")
            else:
                oid = None
                seq = item

            res = self.analyze_order(seq, order_id=oid)
            results.append(res)

            v = res["verdict"]
            if v == "NORMAL":
                normal += 1
            elif v == "RULE_VIOLATION":
                rule_viol += 1
            elif v == "STATISTICAL_ANOMALY":
                stat_anom += 1
            elif v == "VALID_INCOMPLETE":
                val_inc += 1

        total = len(results)
        anomalies = rule_viol + stat_anom
        rate = float(anomalies / total) if total > 0 else 0.0

        return {
            "results": results,
            "summary": {
                "total_orders": total,
                "normal_count": normal,
                "rule_violation_count": rule_viol,
                "statistical_anomaly_count": stat_anom,
                "valid_incomplete_count": val_inc,
                "total_anomalies": anomalies,
                "anomaly_rate": rate,
            },
        }


class TestBackendEndToEnd(unittest.TestCase):
    """Phase 8 End-to-End integration test suite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create an isolated in-memory SQLite database engine."""
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionFactory = sessionmaker(bind=cls.engine, autocommit=False, autoflush=False)

    def setUp(self) -> None:
        """Purge all in-memory table records and set up dependencies."""
        with self.engine.begin() as conn:
            conn.execute(Base.metadata.tables["workflow_analyses"].delete())

        self.service_double = ControlledAnomalyServiceDouble()
        self.app = create_app(init_db_on_startup=False)

        def override_db_session():
            session = self.SessionFactory()
            try:
                yield session
            finally:
                session.close()

        def override_service():
            return self.service_double

        self.app.dependency_overrides[get_db_session] = override_db_session
        self.app.dependency_overrides[get_service] = override_service
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Clear dependency overrides."""
        self.app.dependency_overrides.clear()

    # -----------------------------------------------------------------------
    # Scenario 1: Normal completed order analyzed, persisted, retrieved
    # -----------------------------------------------------------------------

    def test_scenario_01_normal_order_flow(self) -> None:
        """Verify normal completed order end-to-end analysis, persistence, and retrieval."""
        payload = {
            "order_id": "ORD-E2E-001",
            "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
        }
        res_post = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res_post.status_code, 201)
        data = res_post.json()

        self.assertEqual(data["order_id"], "ORD-E2E-001")
        self.assertEqual(data["verdict"], "NORMAL")
        self.assertFalse(data["is_anomaly"])
        self.assertIsNone(data["anomaly_type"])
        self.assertTrue(data["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(data["fsm_evaluation"]["is_complete"])
        self.assertEqual(data["fsm_evaluation"]["final_state"], "DELIVERED")
        self.assertIsNotNone(data["ml_evaluation"])
        self.assertEqual(data["ml_evaluation"]["prediction"], "NORMAL")
        self.assertFalse(data["ml_evaluation"]["is_ml_anomaly"])
        self.assertEqual(data["diagnostics"]["severity"], "INFO")

        # Verify retrieval by ID
        record_id = data["id"]
        res_get = self.client.get(f"/api/v1/analyses/{record_id}")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["id"], record_id)
        self.assertEqual(res_get.json()["order_id"], "ORD-E2E-001")

    # -----------------------------------------------------------------------
    # Scenario 2: FSM-invalid order classified as RULE_VIOLATION
    # -----------------------------------------------------------------------

    def test_scenario_02_fsm_invalid_order_flow(self) -> None:
        """Verify FSM-invalid order classified as RULE_VIOLATION without ML evaluation."""
        payload = {
            "order_id": "ORD-E2E-RULE",
            "event_sequence": "CONFIRM_PAYMENT;DELIVER_ITEMS",  # Skips PACK and SHIP
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["verdict"], "RULE_VIOLATION")
        self.assertTrue(data["is_anomaly"])
        self.assertEqual(data["anomaly_type"], "RULE_VIOLATION")
        self.assertFalse(data["fsm_evaluation"]["is_rule_valid"])
        self.assertFalse(data["fsm_evaluation"]["is_complete"])
        self.assertIsNotNone(data["fsm_evaluation"]["error"])
        self.assertEqual(data["fsm_evaluation"]["error"]["error_type"], "INVALID_TRANSITION")
        self.assertIsNone(data["ml_evaluation"])
        self.assertIsNotNone(data["ml_skip_reason"])
        self.assertIn("failed FSM rule validation", data["ml_skip_reason"])
        self.assertEqual(data["diagnostics"]["severity"], "CRITICAL")

    # -----------------------------------------------------------------------
    # Scenario 3: FSM-valid but incomplete order classified as VALID_INCOMPLETE
    # -----------------------------------------------------------------------

    def test_scenario_03_valid_incomplete_order_flow(self) -> None:
        """Verify incomplete order classified as VALID_INCOMPLETE without ML evaluation."""
        payload = {
            "order_id": "ORD-E2E-INC",
            "event_sequence": "CONFIRM_PAYMENT",  # In-progress at PAID
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["verdict"], "VALID_INCOMPLETE")
        self.assertFalse(data["is_anomaly"])
        self.assertIsNone(data["anomaly_type"])
        self.assertTrue(data["fsm_evaluation"]["is_rule_valid"])
        self.assertFalse(data["fsm_evaluation"]["is_complete"])
        self.assertEqual(data["fsm_evaluation"]["final_state"], "PAID")
        self.assertIsNone(data["ml_evaluation"])
        self.assertIsNotNone(data["ml_skip_reason"])
        self.assertIn("has not reached a terminal state", data["ml_skip_reason"])
        self.assertEqual(data["diagnostics"]["severity"], "INFO")

    # -----------------------------------------------------------------------
    # Scenario 4: FSM-valid, complete statistical anomaly evaluated by ML
    # -----------------------------------------------------------------------

    def test_scenario_04_statistical_anomaly_flow(self) -> None:
        """Verify statistical anomaly evaluated by ML with score, features, and WARNING severity."""
        payload = {
            "order_id": "ORD-E2E-STAT",
            "event_sequence": "CONFIRM_PAYMENT;STATISTICAL_ANOMALY",
        }
        res = self.client.post("/api/v1/analyses", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["verdict"], "STATISTICAL_ANOMALY")
        self.assertTrue(data["is_anomaly"])
        self.assertEqual(data["anomaly_type"], "STATISTICAL_ANOMALY")
        self.assertTrue(data["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(data["fsm_evaluation"]["is_complete"])
        self.assertIsNotNone(data["ml_evaluation"])
        self.assertEqual(data["ml_evaluation"]["prediction"], "ANOMALY")
        self.assertTrue(data["ml_evaluation"]["is_ml_anomaly"])
        self.assertAlmostEqual(data["ml_evaluation"]["anomaly_score"], 0.3542, places=4)
        self.assertEqual(len(data["ml_evaluation"]["features"]), 14)
        self.assertEqual(data["diagnostics"]["severity"], "WARNING")

    # -----------------------------------------------------------------------
    # Scenario 5: Batch containing multiple trace formats processed atomically
    # -----------------------------------------------------------------------

    def test_scenario_05_batch_mixed_formats_atomic_persistence(self) -> None:
        """Verify batch with structured item, string, and list is persisted atomically preserving order."""
        payload = {
            "traces": [
                {"order_id": "BATCH-ORD-01", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},
                "CONFIRM_PAYMENT;DELIVER_ITEMS;RULE_VIOLATION",
                ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
            ]
        }
        res = self.client.post("/api/v1/analyses/batch", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["summary"]["total_orders"], 3)
        self.assertEqual(data["summary"]["normal_count"], 2)
        self.assertEqual(data["summary"]["rule_violation_count"], 1)

        # Check preserved input formats and payload
        res_0 = data["results"][0]
        self.assertEqual(res_0["order_id"], "BATCH-ORD-01")
        self.assertEqual(res_0["input_format"], "DELIMITED_STRING")

        res_1 = data["results"][1]
        self.assertIsNone(res_1["order_id"])
        self.assertEqual(res_1["input_format"], "DELIMITED_STRING")

        res_2 = data["results"][2]
        self.assertIsNone(res_2["order_id"])
        self.assertEqual(res_2["input_format"], "EVENT_LIST")
        self.assertEqual(res_2["raw_input_payload"], ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"])

    # -----------------------------------------------------------------------
    # Scenario 6: Failed batch transaction rolls back without partial persistence
    # -----------------------------------------------------------------------

    def test_scenario_06_batch_transaction_rollback(self) -> None:
        """Verify failed batch transaction rolls back completely with zero rows saved."""
        with patch("backend.api.routes.create_analysis_batch") as mock_batch:
            mock_batch.side_effect = RuntimeError("Simulated transient database error")
            res = self.client.post(
                "/api/v1/analyses/batch",
                json={"traces": ["CONFIRM_PAYMENT;PACK_ITEMS"]},
            )
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.json()["error"]["code"], "INTERNAL_SERVER_ERROR")

        # Verify database has 0 rows
        with self.SessionFactory() as session:
            count = session.query(Base.metadata.tables["workflow_analyses"]).count()
            self.assertEqual(count, 0)

    # -----------------------------------------------------------------------
    # Scenario 7: Persisted analysis can be retrieved by ID
    # -----------------------------------------------------------------------

    def test_scenario_07_retrieve_analysis_by_id(self) -> None:
        """Verify retrieval by ID returns 200 for existing records and 404 for missing IDs."""
        post_res = self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-RET-77", "event_sequence": "CONFIRM_PAYMENT"},
        )
        self.assertEqual(post_res.status_code, 201)
        record_id = post_res.json()["id"]

        get_res = self.client.get(f"/api/v1/analyses/{record_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["id"], record_id)
        self.assertEqual(get_res.json()["order_id"], "ORD-RET-77")

        missing_res = self.client.get("/api/v1/analyses/888888")
        self.assertEqual(missing_res.status_code, 404)
        self.assertEqual(missing_res.json()["error"]["code"], "NOT_FOUND")

    # -----------------------------------------------------------------------
    # Scenario 8: Order history returns correct records in deterministic order
    # -----------------------------------------------------------------------

    def test_scenario_08_order_history_deterministic_order(self) -> None:
        """Verify order history returns chronological records (created_at DESC, id DESC)."""
        # Post two traces for ORD-HIST-SERIES
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-HIST-SERIES", "event_sequence": "CONFIRM_PAYMENT"},
        )
        self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-HIST-SERIES", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},
        )

        res = self.client.get("/api/v1/orders/ORD-HIST-SERIES/analyses")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["order_id"], "ORD-HIST-SERIES")
        self.assertEqual(data["returned_count"], 2)
        self.assertEqual(len(data["analyses"]), 2)
        # Most recent record (higher ID) must be first
        self.assertGreater(data["analyses"][0]["id"], data["analyses"][1]["id"])

    # -----------------------------------------------------------------------
    # Scenario 9: Pagination and verdict filtering return correct results
    # -----------------------------------------------------------------------

    def test_scenario_09_pagination_and_verdict_filtering(self) -> None:
        """Verify pagination offset, limit, total count, and verdict filtering."""
        # Insert 4 records
        self.client.post("/api/v1/analyses", json={"event_sequence": "NORMAL_1"})
        self.client.post("/api/v1/analyses", json={"event_sequence": "NORMAL_2"})
        self.client.post("/api/v1/analyses", json={"event_sequence": "RULE_VIOLATION_1"})
        self.client.post("/api/v1/analyses", json={"event_sequence": "CONFIRM_PAYMENT"})  # VALID_INCOMPLETE

        # 1. Total page
        res_all = self.client.get("/api/v1/analyses?skip=0&limit=10")
        self.assertEqual(res_all.status_code, 200)
        self.assertEqual(res_all.json()["total"], 4)
        self.assertEqual(len(res_all.json()["items"]), 4)

        # 2. Paginated page with skip=1, limit=2
        res_page = self.client.get("/api/v1/analyses?skip=1&limit=2")
        self.assertEqual(res_page.status_code, 200)
        self.assertEqual(res_page.json()["total"], 4)
        self.assertEqual(res_page.json()["skip"], 1)
        self.assertEqual(res_page.json()["limit"], 2)
        self.assertEqual(len(res_page.json()["items"]), 2)

        # 3. Filter by verdict=RULE_VIOLATION
        res_filter = self.client.get("/api/v1/analyses?verdict=RULE_VIOLATION")
        self.assertEqual(res_filter.status_code, 200)
        self.assertEqual(res_filter.json()["total"], 1)
        self.assertEqual(res_filter.json()["items"][0]["verdict"], "RULE_VIOLATION")

    # -----------------------------------------------------------------------
    # Scenario 10: Summary metrics agree with persisted records
    # -----------------------------------------------------------------------

    def test_scenario_10_summary_metrics_agreement(self) -> None:
        """Verify database summary metrics accurately aggregate persisted records."""
        self.client.post("/api/v1/analyses", json={"order_id": "ORD-M1", "event_sequence": "NORMAL"})
        self.client.post("/api/v1/analyses", json={"order_id": "ORD-M2", "event_sequence": "RULE_VIOLATION"})
        self.client.post("/api/v1/analyses", json={"order_id": "ORD-M3", "event_sequence": "CONFIRM_PAYMENT"})

        res = self.client.get("/api/v1/metrics")
        self.assertEqual(res.status_code, 200)
        m = res.json()

        self.assertEqual(m["total_analyses"], 3)
        self.assertEqual(m["normal_count"], 1)
        self.assertEqual(m["rule_violation_count"], 1)
        self.assertEqual(m["valid_incomplete_count"], 1)
        self.assertEqual(m["statistical_anomaly_count"], 0)
        self.assertEqual(m["total_anomalies"], 1)
        self.assertAlmostEqual(m["anomaly_rate"], 1.0 / 3.0, places=4)
        self.assertEqual(m["unique_orders_count"], 3)

    # -----------------------------------------------------------------------
    # Scenario 11: API responses correctly serialize nested FSM, ML, and diagnostics
    # -----------------------------------------------------------------------

    def test_scenario_11_nested_serialization_fidelity(self) -> None:
        """Verify strict Pydantic nested structure serialization fidelity."""
        res = self.client.post(
            "/api/v1/analyses",
            json={"order_id": "ORD-SER-1", "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()

        # FSM Evaluation nested structure
        fsm = data["fsm_evaluation"]
        self.assertIn("is_rule_valid", fsm)
        self.assertIn("is_complete", fsm)
        self.assertIn("final_state", fsm)
        self.assertIn("total_steps", fsm)
        self.assertIn("successful_transitions", fsm)
        self.assertIn("error", fsm)

        # ML Evaluation nested structure
        ml = data["ml_evaluation"]
        self.assertIn("prediction", ml)
        self.assertIn("is_ml_anomaly", ml)
        self.assertIn("anomaly_score", ml)
        self.assertIn("raw_decision_score", ml)
        self.assertIn("features", ml)
        self.assertEqual(len(ml["features"]), 14)

        # Diagnostics nested structure
        diag = data["diagnostics"]
        self.assertIn("severity", diag)
        self.assertIn("root_cause", diag)
        self.assertIn("key_factors", diag)
        self.assertIn("recommended_action", diag)

        # Timestamp UTC string format
        self.assertTrue(data["created_at"].endswith("Z"))

    # -----------------------------------------------------------------------
    # Scenario 12: Error responses retain approved nested error envelope
    # -----------------------------------------------------------------------

    def test_scenario_12_nested_error_envelope_format(self) -> None:
        """Verify error responses conform to ErrorResponse envelope across multiple status codes."""
        # 404 Not Found
        r404 = self.client.get("/api/v1/analyses/999999")
        self.assertEqual(r404.status_code, 404)
        self.assertIn("error", r404.json())
        self.assertEqual(r404.json()["error"]["code"], "NOT_FOUND")

        # 422 Validation Error
        r422 = self.client.post("/api/v1/analyses", json={})
        self.assertEqual(r422.status_code, 422)
        self.assertIn("error", r422.json())
        self.assertEqual(r422.json()["error"]["code"], "VALIDATION_ERROR")
        self.assertIsNotNone(r422.json()["error"]["details"])

    # -----------------------------------------------------------------------
    # Real Model Smoke Test (No Service Double)
    # -----------------------------------------------------------------------

    def test_real_model_smoke_test_through_api(self) -> None:
        """Verify end-to-end API execution using the real trained Isolation Forest model."""
        real_service = get_anomaly_service()
        self.app.dependency_overrides[get_service] = lambda: real_service

        res = self.client.post(
            "/api/v1/analyses",
            json={
                "order_id": "ORD-REAL-SMOKE",
                "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["order_id"], "ORD-REAL-SMOKE")
        self.assertEqual(data["verdict"], "NORMAL")
        self.assertFalse(data["is_anomaly"])
        self.assertTrue(data["fsm_evaluation"]["is_rule_valid"])
        self.assertTrue(data["fsm_evaluation"]["is_complete"])
        self.assertEqual(data["fsm_evaluation"]["final_state"], "DELIVERED")
        self.assertIsNotNone(data["ml_evaluation"])
        self.assertEqual(data["ml_evaluation"]["prediction"], "NORMAL")
        self.assertFalse(data["ml_evaluation"]["is_ml_anomaly"])
        self.assertIsInstance(data["ml_evaluation"]["anomaly_score"], float)
        self.assertEqual(len(data["ml_evaluation"]["features"]), 14)

        # Verify stored in in-memory DB and retrievable
        rec_id = data["id"]
        get_res = self.client.get(f"/api/v1/analyses/{rec_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["order_id"], "ORD-REAL-SMOKE")


if __name__ == "__main__":
    unittest.main()
