"""Combined anomaly analysis service orchestrating FSM and ML detectors.

Provides a unified business logic service layer that delegates core analysis
to the hybrid analyzer pipeline (backend.pipeline.analyzer) and enriches the
results with order identifiers, operational diagnostics, and batch summary metrics.

Specifications:
- backend/docs/WORKFLOW_SPEC.md
- backend/docs/DATASET_SPEC.md (v1.4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.ml.inference.predictor import DEFAULT_MODEL_PATH
from backend.pipeline.analyzer import analyze_trace, analyze_trace_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEC_VERSION = "1.4"
SUPPORTED_VERDICTS: List[str] = [
    "NORMAL",
    "RULE_VIOLATION",
    "STATISTICAL_ANOMALY",
    "VALID_INCOMPLETE",
]


# ---------------------------------------------------------------------------
# Diagnostic Enrichment Helpers
# ---------------------------------------------------------------------------

def _build_diagnostics(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured operational diagnostics from an analysis result.

    Severity mapping:
      - RULE_VIOLATION: CRITICAL (structural workflow rule breach)
      - STATISTICAL_ANOMALY: WARNING (statistically unusual behavioral volume)
      - NORMAL: INFO (standard successful workflow)
      - VALID_INCOMPLETE: INFO (legitimate in-progress order)

    Args:
        analysis_result: Output dictionary from backend.pipeline.analyzer.analyze_trace.

    Returns:
        Dictionary containing severity, root_cause, key_factors, and recommended_action.

    Raises:
        ValueError: If analysis_result contains an unsupported verdict.
    """
    verdict = analysis_result["verdict"]

    if verdict == "RULE_VIOLATION":
        fsm_eval = analysis_result.get("fsm_evaluation", {})
        error = fsm_eval.get("error") or {}
        err_type = error.get("error_type", "UNKNOWN_ERROR")
        step_num = error.get("failed_step", "?")
        event = error.get("failed_event", "?")
        state = error.get("state_at_failure", "?")
        allowed = error.get("allowed_events", [])

        root_cause = (
            f"FSM rule violation ({err_type}) at step {step_num}: event '{event}' "
            f"is not permitted in state '{state}'."
        )
        recommendation = (
            f"Review order event progression. Event '{event}' cannot be executed from state '{state}'. "
            f"Permitted events at this state are: {allowed}."
        )
        return {
            "severity": "CRITICAL",
            "root_cause": root_cause,
            "key_factors": [
                f"Error type: {err_type}",
                f"Failed step: {step_num}",
                f"Offending event: {event}",
                f"State at failure: {state}",
            ],
            "recommended_action": recommendation,
        }

    elif verdict == "STATISTICAL_ANOMALY":
        ml_eval = analysis_result.get("ml_evaluation") or {}
        features = ml_eval.get("features", {})
        score = float(ml_eval.get("anomaly_score", 0.0))

        # Extract factors strictly from engineered numerical features without unsupported claims
        key_factors: List[str] = [
            f"Anomaly score: {score:.4f} (positive score indicates statistical outlier)",
        ]

        count_retry = int(features.get("count_RETRY_PAYMENT", 0))
        if count_retry > 0:
            key_factors.append(f"count_RETRY_PAYMENT: {count_retry} payment retry event(s) observed")

        count_fail = int(features.get("count_FAIL_PAYMENT", 0))
        if count_fail > 0:
            key_factors.append(f"count_FAIL_PAYMENT: {count_fail} payment failure event(s) observed")

        failure_rate = float(features.get("payment_failure_rate", 0.0))
        if failure_rate > 0.0:
            key_factors.append(f"payment_failure_rate: {failure_rate * 100:.1f}% failure rate across payment attempts")

        retry_ratio = float(features.get("retry_ratio", 0.0))
        if retry_ratio > 0.0:
            key_factors.append(f"retry_ratio: {retry_ratio * 100:.1f}% of events represented by payment retries")

        seq_len = int(features.get("seq_length", 0))
        if seq_len >= 7:
            key_factors.append(f"seq_length: elevated sequence length of {seq_len} events")

        root_cause = (
            f"Sequence is structurally valid and complete, but exhibits statistically anomalous feature volume "
            f"(anomaly score: {score:.4f})."
        )
        recommendation = (
            "Investigate elevated payment retry loops and checkout friction observed for this order."
        )
        return {
            "severity": "WARNING",
            "root_cause": root_cause,
            "key_factors": key_factors,
            "recommended_action": recommendation,
        }

    elif verdict == "VALID_INCOMPLETE":
        fsm_eval = analysis_result.get("fsm_evaluation", {})
        final_state = fsm_eval.get("final_state", "UNKNOWN")
        steps = fsm_eval.get("total_steps", 0)

        root_cause = (
            f"Order workflow prefix is structurally valid but incomplete; currently in active state '{final_state}'."
        )
        recommendation = (
            f"Order is actively processing in state '{final_state}'. Await subsequent lifecycle events until completion."
        )
        return {
            "severity": "INFO",
            "root_cause": root_cause,
            "key_factors": [
                f"Current active state: {final_state}",
                f"Transitions completed so far: {steps}",
                "Lifecycle status: In-progress (non-terminal)",
            ],
            "recommended_action": recommendation,
        }

    elif verdict == "NORMAL":
        fsm_eval = analysis_result.get("fsm_evaluation", {})
        final_state = fsm_eval.get("final_state", "UNKNOWN")
        ml_eval = analysis_result.get("ml_evaluation") or {}
        score = float(ml_eval.get("anomaly_score", 0.0))

        root_cause = f"Order completed normally terminating in terminal state '{final_state}'."
        recommendation = "Standard order execution following expected baseline paths. No operational intervention required."
        return {
            "severity": "INFO",
            "root_cause": root_cause,
            "key_factors": [
                f"Terminal state: {final_state}",
                f"Anomaly score: {score:.4f} (non-positive score confirms typical profile)",
            ],
            "recommended_action": recommendation,
        }

    else:
        raise ValueError(
            f"Unsupported verdict '{verdict}' for diagnostic enrichment. "
            f"Expected one of: {SUPPORTED_VERDICTS}."
        )


# ---------------------------------------------------------------------------
# Anomaly Service Class
# ---------------------------------------------------------------------------

class AnomalyService:
    """Service orchestrating process workflow anomaly detection."""

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize AnomalyService.

        Args:
            model: Optional pre-loaded Isolation Forest model or (model, metadata) tuple.
            model_path: Optional path to the model artifact. Defaults to DEFAULT_MODEL_PATH.
        """
        self._model = model
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    def analyze_order(
        self,
        event_sequence: Union[str, Sequence[str]],
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a single order event sequence with diagnostic enrichment.

        Delegates core detection to backend.pipeline.analyzer.analyze_trace().

        Args:
            event_sequence: Semicolon-delimited string (e.g. "CONFIRM_PAYMENT;PACK_ITEMS;...")
                or a list of event strings.
            order_id: Optional order identifier.

        Returns:
            Dictionary containing order_id, verdict, anomaly flags, FSM evaluation,
            ML evaluation (or null if skipped), diagnostics, and summary.
        """
        base_result = analyze_trace(
            event_sequence,
            model=self._model,
            model_path=self._model_path,
        )

        diagnostics = _build_diagnostics(base_result)

        return {
            "order_id": str(order_id) if order_id is not None else None,
            "verdict": base_result["verdict"],
            "is_anomaly": base_result["is_anomaly"],
            "anomaly_type": base_result["anomaly_type"],
            "summary": base_result["summary"],
            "diagnostics": diagnostics,
            "fsm_evaluation": base_result["fsm_evaluation"],
            "ml_evaluation": base_result["ml_evaluation"],
            "ml_skip_reason": base_result["ml_skip_reason"],
        }

    def analyze_order_batch(
        self,
        traces: Sequence[Union[str, Sequence[str], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Analyze a batch of orders with summary metrics and lazy model loading.

        Accepts:
          - Semicolon-delimited strings: ["CONFIRM_PAYMENT;...", ...]
          - Event lists: [["CONFIRM_PAYMENT", ...], ...]
          - Dictionaries: [{"order_id": "ORD-1", "event_sequence": "..."}, ...]

        Delegates core batch execution to backend.pipeline.analyzer.analyze_trace_batch().

        Args:
            traces: Sequence of traces or trace dictionaries.

        Returns:
            Dictionary containing 'results' list and 'summary' metrics.
        """
        if not traces:
            return {
                "results": [],
                "summary": {
                    "total_orders": 0,
                    "normal_count": 0,
                    "rule_violation_count": 0,
                    "statistical_anomaly_count": 0,
                    "valid_incomplete_count": 0,
                    "total_anomalies": 0,
                    "anomaly_rate": 0.0,
                },
            }

        # Normalize inputs to extract order_ids and raw event sequences
        extracted_order_ids: List[Optional[str]] = []
        raw_sequences: List[Union[str, Sequence[str]]] = []

        for item in traces:
            if isinstance(item, dict):
                order_id = item.get("order_id")
                seq = item.get("event_sequence", item.get("events", ""))
                extracted_order_ids.append(str(order_id) if order_id is not None else None)
                raw_sequences.append(seq)
            else:
                extracted_order_ids.append(None)
                raw_sequences.append(item)

        # Delegate core batch analysis to pipeline analyzer (preserves lazy model loading)
        base_results = analyze_trace_batch(
            raw_sequences,
            model=self._model,
            model_path=self._model_path,
        )

        enriched_results: List[Dict[str, Any]] = []
        normal_count = 0
        rule_violation_count = 0
        statistical_anomaly_count = 0
        valid_incomplete_count = 0

        for base_res, order_id in zip(base_results, extracted_order_ids):
            verdict = base_res["verdict"]
            if verdict == "NORMAL":
                normal_count += 1
            elif verdict == "RULE_VIOLATION":
                rule_violation_count += 1
            elif verdict == "STATISTICAL_ANOMALY":
                statistical_anomaly_count += 1
            elif verdict == "VALID_INCOMPLETE":
                valid_incomplete_count += 1

            diagnostics = _build_diagnostics(base_res)
            enriched_results.append({
                "order_id": order_id,
                "verdict": verdict,
                "is_anomaly": base_res["is_anomaly"],
                "anomaly_type": base_res["anomaly_type"],
                "summary": base_res["summary"],
                "diagnostics": diagnostics,
                "fsm_evaluation": base_res["fsm_evaluation"],
                "ml_evaluation": base_res["ml_evaluation"],
                "ml_skip_reason": base_res["ml_skip_reason"],
            })

        total_orders = len(enriched_results)
        total_anomalies = rule_violation_count + statistical_anomaly_count
        # Anomaly rate formula: (rule_violation + statistical_anomaly) / total_orders
        # Note: Valid incomplete orders are included in the total_orders denominator.
        anomaly_rate = float(total_anomalies / total_orders) if total_orders > 0 else 0.0

        return {
            "results": enriched_results,
            "summary": {
                "total_orders": total_orders,
                "normal_count": normal_count,
                "rule_violation_count": rule_violation_count,
                "statistical_anomaly_count": statistical_anomaly_count,
                "valid_incomplete_count": valid_incomplete_count,
                "total_anomalies": total_anomalies,
                "anomaly_rate": anomaly_rate,
            },
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Return operational status and metadata without triggering model loading.

        Status determination:
          - OPERATIONAL: if model artifact exists on disk or a model has been injected.
          - DEGRADED: if neither is available.

        Does not load the model from disk to determine status.
        """
        artifact_exists = self._model_path.exists()
        model_is_injected = self._model is not None
        status = "OPERATIONAL" if (artifact_exists or model_is_injected) else "DEGRADED"

        return {
            "service_status": status,
            "dataset_spec_version": SPEC_VERSION,
            "model_artifact_path": str(self._model_path),
            "model_artifact_exists": bool(artifact_exists),
            "model_loaded_in_memory": bool(model_is_injected),
            "supported_verdicts": list(SUPPORTED_VERDICTS),
        }


# ---------------------------------------------------------------------------
# Service Factory
# ---------------------------------------------------------------------------

_service_instance: Optional[AnomalyService] = None


def get_anomaly_service(
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> AnomalyService:
    """Factory function returning an AnomalyService instance.

    If custom model or model_path is provided, instantiates a new service;
    otherwise returns a shared default instance.

    Args:
        model: Optional pre-loaded model or (model, metadata) tuple.
        model_path: Optional path to the model artifact.

    Returns:
        AnomalyService instance.
    """
    global _service_instance
    if model is not None or model_path is not None:
        return AnomalyService(model=model, model_path=model_path)

    if _service_instance is None:
        _service_instance = AnomalyService()
    return _service_instance
