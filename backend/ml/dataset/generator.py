"""Synthetic workflow event log dataset generator.

Generates 5,000 labelled order event traces for the dual-engine anomaly
detection pipeline described in backend/docs/DATASET_SPEC.md.

Allocation (DATASET_SPEC.md Section 4.2):
    NORMAL training       : 2,800 traces  (Isolation Forest training)
    NORMAL testing        :   700 traces  (primary ML test, negative class)
    STATISTICAL_ANOMALY   :   500 traces  (primary ML test, positive class)
    RULE_VIOLATION        : 1,000 traces  (FSM benchmark, excluded from ML metrics)
    ─────────────────────────────────────────────────────────────────────────
    Total                 : 5,000 traces

Authoritative source: backend/docs/DATASET_SPEC.md (v1.4, Approved)
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Dict, List, Optional
import numpy as np

# Reuse the validated FSM validator — do NOT duplicate FSM transition logic
from backend.fsm.validator import validate_trace

# ---------------------------------------------------------------------------
# Constants (DATASET_SPEC.md Section 4.2 & 4.3)
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

N_NORMAL_TRAIN: int = 2_800
N_NORMAL_TEST: int = 700
N_NORMAL_TOTAL: int = N_NORMAL_TRAIN + N_NORMAL_TEST          # 3,500
N_STATISTICAL_ANOMALY: int = 500
N_RULE_VIOLATION: int = 1_000
TOTAL_TRACES: int = N_NORMAL_TOTAL + N_STATISTICAL_ANOMALY + N_RULE_VIOLATION  # 5,000

DATA_DIR: Path = Path(__file__).parent / "data"
OUTPUT_CSV: Path = DATA_DIR / "orders_workflow_dataset.csv"

# CSV column order (DATASET_SPEC.md Section 3.1)
CSV_COLUMNS: List[str] = [
    "trace_id",
    "event_sequence",
    "sequence_length",
    "is_rule_valid",
    "is_complete",
    "anomaly_category",
    "is_ml_anomaly",
]

# Out-of-alphabet tokens used by the UNKNOWN mutation operator
_UNKNOWN_TOKENS: List[str] = [
    "HACK_CHECKOUT",
    "BYPASS_GATEWAY",
    "INJECT_DISCOUNT",
    "SKIP_VERIFICATION",
    "OVERRIDE_INVENTORY",
    "ADMIN_FORCE_SHIP",
    "MANUAL_CREDIT",
    "BYPASS_AUTH_TOKEN",
    "FORCE_CANCEL",
    "SPOOF_DELIVERY",
]

_TERMINAL_STATES = {"DELIVERED", "CANCELLED", "REFUNDED"}


# ===========================================================================
# 1. NORMAL trace generator
#    Specification: DATASET_SPEC.md Section 4.1, Subsection 1
# ===========================================================================

def _generate_normal_trace() -> List[str]:
    """Simulate a single random walk on the FSM transition graph.

    Branch probabilities are taken verbatim from DATASET_SPEC.md Section 4.1:

    At ORDER_CREATED (retry_count == 0):
        85 %  CONFIRM_PAYMENT  →  PAID
        10 %  CANCEL_ORDER     →  CANCELLED  (terminal)
         5 %  FAIL_PAYMENT     →  PAYMENT_FAILED

    At PAYMENT_FAILED:
        80 %  RETRY_PAYMENT    →  ORDER_CREATED  (retry_count becomes 1)
        20 %  CANCEL_ORDER     →  CANCELLED      (terminal)

    At ORDER_CREATED (retry_count >= 1):  FAIL_PAYMENT branch disabled
        90 %  CONFIRM_PAYMENT  →  PAID
        10 %  CANCEL_ORDER     →  CANCELLED  (terminal)

    At PAID:
        95 %  PACK_ITEMS   →  PACKED
         5 %  REFUND_ORDER →  REFUNDED  (terminal)

    At PACKED:
        97 %  SHIP_ITEMS   →  SHIPPED
         3 %  REFUND_ORDER →  REFUNDED  (terminal)

    At SHIPPED:
       100 %  DELIVER_ITEMS →  DELIVERED  (terminal)

    Returns:
        A list of event strings: FSM-valid, terminates at a terminal state,
        and contains at most one RETRY_PAYMENT.
    """
    events: List[str] = []
    state: str = "ORDER_CREATED"
    retry_count: int = 0

    while state not in _TERMINAL_STATES:
        r = random.random()

        if state == "ORDER_CREATED":
            if retry_count == 0:
                if r < 0.85:
                    events.append("CONFIRM_PAYMENT")
                    state = "PAID"
                elif r < 0.95:                  # 0.85 + 0.10 = 0.95
                    events.append("CANCEL_ORDER")
                    state = "CANCELLED"
                else:                           # 0.95 to 1.00 (5 %)
                    events.append("FAIL_PAYMENT")
                    state = "PAYMENT_FAILED"
            else:
                # retry_count >= 1: FAIL_PAYMENT disabled
                if r < 0.90:
                    events.append("CONFIRM_PAYMENT")
                    state = "PAID"
                else:
                    events.append("CANCEL_ORDER")
                    state = "CANCELLED"

        elif state == "PAYMENT_FAILED":
            if r < 0.80:
                events.append("RETRY_PAYMENT")
                state = "ORDER_CREATED"
                retry_count += 1
            else:
                events.append("CANCEL_ORDER")
                state = "CANCELLED"

        elif state == "PAID":
            if r < 0.95:
                events.append("PACK_ITEMS")
                state = "PACKED"
            else:
                events.append("REFUND_ORDER")
                state = "REFUNDED"

        elif state == "PACKED":
            if r < 0.97:
                events.append("SHIP_ITEMS")
                state = "SHIPPED"
            else:
                events.append("REFUND_ORDER")
                state = "REFUNDED"

        elif state == "SHIPPED":
            events.append("DELIVER_ITEMS")
            state = "DELIVERED"

    return events


def generate_normal_traces(n: int) -> List[List[str]]:
    """Return a list of n independently generated NORMAL event traces."""
    traces: List[List[str]] = []
    for _ in range(n):
        trace = _generate_normal_trace()
        traces.append(trace)
    return traces


# ===========================================================================
# 2. STATISTICAL_ANOMALY trace generator
#    Specification: DATASET_SPEC.md Section 4.1, Subsection 3
# ===========================================================================

def _generate_anomaly_trace() -> List[str]:
    """Generate one STATISTICAL_ANOMALY trace.

    Algorithm (DATASET_SPEC.md Section 4.1, Subsection 3):
      1. Sample k in {3, 4, 5, 6} uniformly at random.
      2. For each cycle, append FAIL_PAYMENT followed by RETRY_PAYMENT.
      3. Terminate via either:
         - Successful fulfillment: CONFIRM_PAYMENT -> PACK_ITEMS -> SHIP_ITEMS -> DELIVER_ITEMS
         - Valid cancellation: CANCEL_ORDER
    """
    k: int = random.randint(3, 6)
    events: List[str] = []

    for _ in range(k):
        events.append("FAIL_PAYMENT")
        events.append("RETRY_PAYMENT")

    if random.random() < 0.5:
        events.extend(["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"])
    else:
        events.append("CANCEL_ORDER")

    return events


def generate_anomaly_traces(n: int) -> List[List[str]]:
    """Return a list of n STATISTICAL_ANOMALY event traces."""
    return [_generate_anomaly_trace() for _ in range(n)]


# ===========================================================================
# 3. RULE_VIOLATION trace generator
#    Specification: DATASET_SPEC.md Section 4.1, Subsection 2
# ===========================================================================

def _build_violation_pool() -> List[List[str]]:
    """Build a rich deterministic pool of candidate rule-violation sequences.

    Applies the six mutation operators from DATASET_SPEC.md Section 4.1.2:
      1. DROP: Remove prerequisite step (e.g., skip packing, ship directly).
      2. SWAP: Swap order of two events (e.g., ship before pack).
      3. PREMATURE: Premature action at start (e.g., deliver before payment).
      4. POST_TERMINAL: Event executed after terminal state (zombie mutation).
      5. DUPLICATE: Duplicate event without self-loop (e.g., double confirm).
      6. UNKNOWN: Inject out-of-alphabet event symbol.
    """
    pool: List[List[str]] = []

    # 1. DROP Operator
    pool.append(["CONFIRM_PAYMENT", "SHIP_ITEMS", "DELIVER_ITEMS"])
    pool.append(["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "SHIP_ITEMS", "DELIVER_ITEMS"])
    pool.append(["PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"])
    pool.append(["SHIP_ITEMS", "DELIVER_ITEMS"])
    pool.append(["DELIVER_ITEMS"])
    pool.append(["PACK_ITEMS", "REFUND_ORDER"])
    pool.append(["CONFIRM_PAYMENT", "SHIP_ITEMS"])

    # 2. SWAP Operator
    pool.append(["CONFIRM_PAYMENT", "SHIP_ITEMS", "PACK_ITEMS", "DELIVER_ITEMS"])
    pool.append(["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "SHIP_ITEMS", "PACK_ITEMS", "DELIVER_ITEMS"])
    pool.append(["PACK_ITEMS", "CONFIRM_PAYMENT", "SHIP_ITEMS", "DELIVER_ITEMS"])
    pool.append(["DELIVER_ITEMS", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"])
    pool.append(["SHIP_ITEMS", "CONFIRM_PAYMENT", "PACK_ITEMS", "DELIVER_ITEMS"])

    # 3. PREMATURE Operator
    pool.append(["DELIVER_ITEMS", "CONFIRM_PAYMENT"])
    pool.append(["SHIP_ITEMS", "CONFIRM_PAYMENT"])
    pool.append(["PACK_ITEMS"])
    pool.append(["SHIP_ITEMS"])
    pool.append(["REFUND_ORDER"])
    pool.append(["DELIVER_ITEMS", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS"])
    pool.append(["SHIP_ITEMS", "PACK_ITEMS", "CONFIRM_PAYMENT"])

    # 4. POST_TERMINAL Operator
    terminal_bases = [
        ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
        ["CANCEL_ORDER"],
        ["FAIL_PAYMENT", "CANCEL_ORDER"],
        ["CONFIRM_PAYMENT", "REFUND_ORDER"],
        ["CONFIRM_PAYMENT", "PACK_ITEMS", "REFUND_ORDER"],
    ]
    post_events = [
        "CANCEL_ORDER", "CONFIRM_PAYMENT", "PACK_ITEMS",
        "SHIP_ITEMS", "DELIVER_ITEMS", "REFUND_ORDER",
        "FAIL_PAYMENT", "RETRY_PAYMENT",
    ]
    for base in terminal_bases:
        for ev in post_events:
            pool.append(base + [ev])

    # 5. DUPLICATE Operator
    dup_cases = [
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "CONFIRM_PAYMENT"),
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "PACK_ITEMS"),
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "SHIP_ITEMS"),
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "DELIVER_ITEMS"),
        (["CANCEL_ORDER"], "CANCEL_ORDER"),
        (["CONFIRM_PAYMENT", "REFUND_ORDER"], "REFUND_ORDER"),
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "REFUND_ORDER"], "PACK_ITEMS"),
        (["CONFIRM_PAYMENT", "PACK_ITEMS", "REFUND_ORDER"], "REFUND_ORDER"),
        (["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "CONFIRM_PAYMENT"),
        (["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"], "PACK_ITEMS"),
        (["FAIL_PAYMENT", "RETRY_PAYMENT"], "FAIL_PAYMENT"),
    ]
    for base, event in dup_cases:
        idx = base.index(event)
        mutated = base[:idx + 1] + [event] + base[idx + 1:]
        pool.append(mutated)

    # 6. UNKNOWN Operator
    unknown_bases = [
        ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
        ["CANCEL_ORDER"],
        ["CONFIRM_PAYMENT", "REFUND_ORDER"],
        ["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
    ]
    for token in _UNKNOWN_TOKENS:
        pool.append([token])
        for base in unknown_bases:
            pool.append(base[:1] + [token] + base[1:])
            pool.append([token] + base)

    return pool


def generate_rule_violation_traces(n: int) -> List[List[str]]:
    """Generate n RULE_VIOLATION event traces.

    All traces are guaranteed to fail validate_trace().
    """
    candidate_pool = _build_violation_pool()

    # Defensive validation using existing FSM validator
    verified_pool = [
        seq for seq in candidate_pool
        if not validate_trace(seq)["is_valid"]
    ]

    if not verified_pool:
        raise RuntimeError("No RULE_VIOLATION candidates survived FSM verification.")

    return [random.choice(verified_pool) for _ in range(n)]


# ===========================================================================
# 4. Dataset Assembly & CSV Export
# ===========================================================================

def generate_dataset(output_path: Optional[Path] = None) -> Path:
    """Assemble and persist the complete 5,000-trace labelled dataset.

    Seeding behaviour (DATASET_SPEC.md Section 4.3):
        Both Python's built-in `random` and NumPy's `numpy.random` are
        seeded with RANDOM_SEED (42).

    Rows:
        1 to 3,500   : NORMAL traces (first 2,800 train, next 700 test)
        3,501 to 4,000: STATISTICAL_ANOMALY traces
        4,001 to 5,000: RULE_VIOLATION traces
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    target = Path(output_path) if output_path else OUTPUT_CSV
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    counter: int = 1

    # 1. NORMAL traces (3,500 = 2,800 train + 700 test)
    for events in generate_normal_traces(N_NORMAL_TOTAL):
        trace_id = f"TRACE_{counter:05d}"
        val_res = validate_trace(events)
        if not val_res["is_valid"] or not val_res["is_complete"]:
            raise RuntimeError(
                f"Generated NORMAL trace {trace_id} failed validation or is incomplete: "
                f"is_valid={val_res['is_valid']}, is_complete={val_res['is_complete']}, "
                f"error={val_res.get('error')}. Trace: {events}"
            )
        rows.append({
            "trace_id": trace_id,
            "event_sequence": ";".join(events),
            "sequence_length": len(events),
            "is_rule_valid": val_res["is_valid"],
            "is_complete": val_res["is_complete"],
            "anomaly_category": "NORMAL",
            "is_ml_anomaly": False,
        })
        counter += 1

    # 2. STATISTICAL_ANOMALY traces (500)
    for events in generate_anomaly_traces(N_STATISTICAL_ANOMALY):
        trace_id = f"TRACE_{counter:05d}"
        val_res = validate_trace(events)
        if not val_res["is_valid"] or not val_res["is_complete"]:
            raise RuntimeError(
                f"Generated STATISTICAL_ANOMALY trace {trace_id} failed validation or is incomplete: "
                f"is_valid={val_res['is_valid']}, is_complete={val_res['is_complete']}, "
                f"error={val_res.get('error')}. Trace: {events}"
            )
        rows.append({
            "trace_id": trace_id,
            "event_sequence": ";".join(events),
            "sequence_length": len(events),
            "is_rule_valid": val_res["is_valid"],
            "is_complete": val_res["is_complete"],
            "anomaly_category": "STATISTICAL_ANOMALY",
            "is_ml_anomaly": True,
        })
        counter += 1

    # 3. RULE_VIOLATION traces (1,000)
    for events in generate_rule_violation_traces(N_RULE_VIOLATION):
        rows.append({
            "trace_id": f"TRACE_{counter:05d}",
            "event_sequence": ";".join(events),
            "sequence_length": len(events),
            "is_rule_valid": False,
            "is_complete": False,
            "anomaly_category": "RULE_VIOLATION",
            "is_ml_anomaly": "",  # null / empty string for rule violations
        })
        counter += 1

    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return target


if __name__ == "__main__":
    output = generate_dataset()
    print(f"Dataset generated at: {output}")
    print(f"Total traces: {TOTAL_TRACES}")
    print(f"  NORMAL: {N_NORMAL_TOTAL} (train={N_NORMAL_TRAIN}, test={N_NORMAL_TEST})")
    print(f"  STATISTICAL_ANOMALY: {N_STATISTICAL_ANOMALY}")
    print(f"  RULE_VIOLATION: {N_RULE_VIOLATION}")
