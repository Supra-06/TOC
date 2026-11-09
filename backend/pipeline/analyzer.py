"""Hybrid workflow analyzer combining deterministic FSM and ML anomaly detection.

Evaluates workflow event sequences through a prioritized dual-engine architecture:
1. Deterministic FSM validates state transition syntax and lifecycle completeness.
   - If invalid: verdict is RULE_VIOLATION. ML evaluation is skipped (ml_evaluation = null).
   - If valid but incomplete: verdict is VALID_INCOMPLETE. ML evaluation is skipped (ml_evaluation = null).
2. Isolation Forest ML evaluates statistical behavior ONLY on FSM-valid, completed traces.
   - If normal feature profile: verdict is NORMAL.
   - If statistically unusual: verdict is STATISTICAL_ANOMALY.

Specifications:
- backend/docs/WORKFLOW_SPEC.md
- backend/docs/DATASET_SPEC.md (v1.4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from backend.fsm.validator import validate_trace
from backend.ml.feature_extraction.extractor import parse_event_sequence
from backend.ml.inference.predictor import DEFAULT_MODEL_PATH, load_trained_model, predict_anomaly


def analyze_trace(
    event_sequence: Union[str, Sequence[str]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Analyze a single workflow event sequence using FSM validation and ML anomaly scoring.

    Decision Taxonomy:
      - RULE_VIOLATION: FSM rule check failed (is_rule_valid == False). ML skipped.
      - VALID_INCOMPLETE: In-progress trace (is_rule_valid == True, is_complete == False). ML skipped.
      - NORMAL: Completed trace with typical profile (is_rule_valid == True, is_complete == True, is_ml_anomaly == False).
      - STATISTICAL_ANOMALY: Completed trace with unusual behavior (is_rule_valid == True, is_complete == True, is_ml_anomaly == True).

    Args:
        event_sequence: Semicolon-delimited string (e.g. "CONFIRM_PAYMENT;PACK_ITEMS;...")
            or a sequence of event strings.
        model: Optional pre-loaded ML model estimator or (model, metadata) tuple.
        model_path: Optional path to the model artifact if model is not supplied.

    Returns:
        Structured dictionary containing verdict, anomaly flags, FSM evaluation,
        ML evaluation (or null if skipped), skip reason, and a human-readable summary.
    """
    # 1. Parse and sanitize input sequence
    events = parse_event_sequence(event_sequence)

    # 2. Stage 1: Deterministic FSM Validation
    fsm_raw = validate_trace(events)

    is_rule_valid = bool(fsm_raw["is_valid"])
    is_complete = bool(fsm_raw["is_complete"])
    final_state = str(fsm_raw["final_state"])
    total_steps = int(fsm_raw["total_steps"])
    successful_transitions = int(fsm_raw["successful_transitions"])

    # Format error dict cleanly to guarantee JSON-serializability
    error_raw = fsm_raw.get("error")
    error_dict: Optional[Dict[str, Any]] = None
    if error_raw is not None:
        failed_step = int(error_raw.get("failed_step", error_raw.get("step_number", 0)))
        failed_event = str(error_raw.get("failed_event", error_raw.get("event", "")))
        state_at_failure = str(error_raw.get("state_at_failure", error_raw.get("current_state", "")))
        error_type = str(error_raw.get("error_type", ""))
        allowed_events = [str(e) for e in error_raw.get("allowed_events", [])]
        message = str(error_raw.get("message", ""))

        error_dict = {
            "error_type": error_type,
            "failed_step": failed_step,
            "failed_event": failed_event,
            "state_at_failure": state_at_failure,
            "allowed_events": allowed_events,
            "message": message,
        }

    fsm_evaluation: Dict[str, Any] = {
        "is_rule_valid": is_rule_valid,
        "is_complete": is_complete,
        "final_state": final_state,
        "total_steps": total_steps,
        "successful_transitions": successful_transitions,
        "error": error_dict,
    }

    # 3. Stage 2: Decision Gating & Conditional ML Evaluation

    # Case A: FSM Rule Violation (Priority 1)
    if not is_rule_valid:
        verdict = "RULE_VIOLATION"
        is_anomaly = True
        anomaly_type: Optional[str] = "RULE_VIOLATION"
        ml_evaluation = None
        ml_skip_reason = "ML evaluation skipped: trace failed FSM rule validation (RULE_VIOLATION)."

        if error_dict:
            err_type = error_dict["error_type"]
            step_num = error_dict["failed_step"]
            event = error_dict["failed_event"]
            curr_state = error_dict["state_at_failure"]
            allowed = error_dict["allowed_events"]

            if err_type == "INVALID_TRANSITION":
                summary = (
                    f"FSM rule violation at step {step_num}: event '{event}' is not permitted "
                    f"from state '{curr_state}'. Allowed events: {allowed}."
                )
            elif err_type == "UNKNOWN_EVENT":
                summary = (
                    f"FSM rule violation at step {step_num}: event '{event}' is not recognized "
                    f"in the workflow alphabet."
                )
            elif err_type == "TERMINAL_STATE_VIOLATION":
                summary = (
                    f"FSM rule violation at step {step_num}: event '{event}' arrived after "
                    f"reaching terminal state '{curr_state}'."
                )
            else:
                summary = f"FSM rule violation at step {step_num}: {err_type}."
        else:
            summary = "Trace violates FSM state transition rules."

        return {
            "verdict": verdict,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "summary": summary,
            "fsm_evaluation": fsm_evaluation,
            "ml_evaluation": ml_evaluation,
            "ml_skip_reason": ml_skip_reason,
        }

    # Case B: Valid Incomplete Trace (In-Progress)
    if not is_complete:
        verdict = "VALID_INCOMPLETE"
        is_anomaly = False
        anomaly_type = None
        ml_evaluation = None
        ml_skip_reason = (
            "ML evaluation skipped: trace is in-progress and has not reached a terminal state (VALID_INCOMPLETE)."
        )
        summary = (
            f"Valid in-progress workflow prefix currently at state '{final_state}'; "
            f"order has not yet reached a terminal state."
        )

        return {
            "verdict": verdict,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "summary": summary,
            "fsm_evaluation": fsm_evaluation,
            "ml_evaluation": ml_evaluation,
            "ml_skip_reason": ml_skip_reason,
        }

    # Case C: FSM-Valid and Complete -> Perform Isolation Forest Evaluation
    ml_res = predict_anomaly(events, model=model, model_path=model_path)

    ml_evaluation = {
        "prediction": str(ml_res["prediction"]),
        "is_ml_anomaly": bool(ml_res["is_ml_anomaly"]),
        "anomaly_score": float(ml_res["anomaly_score"]),
        "raw_decision_score": float(ml_res["raw_decision_score"]),
        "features": {k: float(v) for k, v in ml_res["features"].items()},
    }
    ml_skip_reason = None

    if ml_res["is_ml_anomaly"]:
        verdict = "STATISTICAL_ANOMALY"
        is_anomaly = True
        anomaly_type = "STATISTICAL_ANOMALY"
        summary = (
            f"Statistical anomaly detected (anomaly score: {ml_res['anomaly_score']:.4f}): "
            f"sequence is structurally valid and complete, but exhibits unusual behavioral volume (anomaly score > 0)."
        )
    else:
        verdict = "NORMAL"
        is_anomaly = False
        anomaly_type = None
        summary = (
            f"Workflow completed normally and terminated at '{final_state}' with a typical behavioral profile."
        )

    return {
        "verdict": verdict,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "summary": summary,
        "fsm_evaluation": fsm_evaluation,
        "ml_evaluation": ml_evaluation,
        "ml_skip_reason": ml_skip_reason,
    }


def analyze_trace_batch(
    event_sequences: Sequence[Union[str, Sequence[str]]],
    model: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Analyze a batch of workflow event sequences with lazy model loading.

    The model is loaded at most once, and only if at least one trace in the batch
    is eligible for ML evaluation (i.e. FSM-valid and complete).

    Args:
        event_sequences: Collection of event sequences (strings or lists).
        model: Optional pre-loaded model or (model, metadata) tuple.
        model_path: Optional path to the model artifact.

    Returns:
        List of result dictionaries corresponding to each input sequence.
    """
    resolved_model = model
    parsed_traces: List[Tuple[Union[str, Sequence[str]], bool]] = []
    has_eligible = False

    # 1. Perform FSM validation for all input traces to identify ML eligibility
    for seq in event_sequences:
        events = parse_event_sequence(seq)
        fsm_raw = validate_trace(events)
        is_eligible = bool(fsm_raw["is_valid"]) and bool(fsm_raw["is_complete"])
        if is_eligible:
            has_eligible = True
        parsed_traces.append((seq, is_eligible))

    # 2. Lazily load the trained model only if at least one trace is eligible
    if has_eligible and resolved_model is None:
        resolved_model, _ = load_trained_model(model_path=model_path)

    # 3. Analyze all traces, reusing resolved_model for every eligible trace
    results: List[Dict[str, Any]] = []
    for seq, _ in parsed_traces:
        results.append(analyze_trace(seq, model=resolved_model, model_path=model_path))

    return results
