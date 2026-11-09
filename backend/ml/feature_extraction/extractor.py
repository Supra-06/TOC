"""Feature engineering and sequence vectorization for workflow event logs.

Extracts the approved 14 numerical features from order event sequences for
unsupervised anomaly detection (Isolation Forest) as specified in
backend/docs/DATASET_SPEC.md (Section 6).

Approved Feature List (in exact order):
 1. seq_length
 2. count_CONFIRM_PAYMENT
 3. count_FAIL_PAYMENT
 4. count_RETRY_PAYMENT
 5. count_CANCEL_ORDER
 6. count_PACK_ITEMS
 7. count_SHIP_ITEMS
 8. count_DELIVER_ITEMS
 9. count_REFUND_ORDER
10. retry_ratio
11. payment_failure_rate
12. bigram_FAIL_to_RETRY
13. bigram_PACK_to_SHIP
14. unique_event_count

Authoritative source: backend/docs/DATASET_SPEC.md (Section 6.1)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
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

NUM_FEATURES: int = len(FEATURE_NAMES)  # exactly 14


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_event_sequence(sequence: Union[str, Sequence[str]]) -> List[str]:
    """Parse an event sequence into a clean list of event strings.

    Args:
        sequence: Either a semicolon-delimited string (as formatted in CSV,
            e.g. "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS")
            or an iterable of event strings.

    Returns:
        A list of non-empty event token strings.
    """
    if isinstance(sequence, str):
        trimmed = sequence.strip()
        if not trimmed:
            return []
        return [token.strip() for token in trimmed.split(";") if token.strip()]

    return [str(token).strip() for token in sequence if str(token).strip()]


# ---------------------------------------------------------------------------
# Core Feature Extraction
# ---------------------------------------------------------------------------

def extract_features_dict(sequence: Union[str, Sequence[str]]) -> Dict[str, Union[int, float]]:
    """Extract the 14 approved numerical features as a dictionary.

    Args:
        sequence: Raw event sequence (semicolon-delimited string or list of events).

    Returns:
        Ordered dictionary mapping each feature name to its computed numerical value.
    """
    events = parse_event_sequence(sequence)
    n = len(events)

    # 1. Sequence Length
    seq_length = n

    # 2–9. Individual Event Counts
    count_confirm = events.count("CONFIRM_PAYMENT")
    count_fail = events.count("FAIL_PAYMENT")
    count_retry = events.count("RETRY_PAYMENT")
    count_cancel = events.count("CANCEL_ORDER")
    count_pack = events.count("PACK_ITEMS")
    count_ship = events.count("SHIP_ITEMS")
    count_deliver = events.count("DELIVER_ITEMS")
    count_refund = events.count("REFUND_ORDER")

    # 10. Retry Ratio: count_RETRY_PAYMENT / max(1, seq_length)
    retry_ratio = count_retry / max(1, n)

    # 11. Payment Failure Rate: count_FAIL_PAYMENT / max(1, count_CONFIRM_PAYMENT + count_FAIL_PAYMENT)
    payment_attempts = count_confirm + count_fail
    payment_failure_rate = count_fail / max(1, payment_attempts)

    # 12. Bigram FAIL_PAYMENT -> RETRY_PAYMENT
    bigram_fail_to_retry = sum(
        1 for i in range(n - 1)
        if events[i] == "FAIL_PAYMENT" and events[i + 1] == "RETRY_PAYMENT"
    )

    # 13. Bigram PACK_ITEMS -> SHIP_ITEMS
    bigram_pack_to_ship = sum(
        1 for i in range(n - 1)
        if events[i] == "PACK_ITEMS" and events[i + 1] == "SHIP_ITEMS"
    )

    # 14. Unique Event Count
    unique_event_count = len(set(events))

    return {
        "seq_length": seq_length,
        "count_CONFIRM_PAYMENT": count_confirm,
        "count_FAIL_PAYMENT": count_fail,
        "count_RETRY_PAYMENT": count_retry,
        "count_CANCEL_ORDER": count_cancel,
        "count_PACK_ITEMS": count_pack,
        "count_SHIP_ITEMS": count_ship,
        "count_DELIVER_ITEMS": count_deliver,
        "count_REFUND_ORDER": count_refund,
        "retry_ratio": float(retry_ratio),
        "payment_failure_rate": float(payment_failure_rate),
        "bigram_FAIL_to_RETRY": bigram_fail_to_retry,
        "bigram_PACK_to_SHIP": bigram_pack_to_ship,
        "unique_event_count": unique_event_count,
    }


def extract_features(sequence: Union[str, Sequence[str]]) -> List[float]:
    """Extract the 14 approved features as an ordered numerical list.

    Args:
        sequence: Raw event sequence (semicolon-delimited string or list of events).

    Returns:
        List of 14 float values matching FEATURE_NAMES order.
    """
    f_dict = extract_features_dict(sequence)
    return [float(f_dict[name]) for name in FEATURE_NAMES]


def extract_feature_matrix(sequences: Sequence[Union[str, Sequence[str]]]) -> np.ndarray:
    """Convert multiple event sequences into a 2D numerical feature matrix.

    This function produces the input matrix X for Scikit-learn estimators
    (such as Isolation Forest).

    Args:
        sequences: Sequence of event sequences (strings or lists).

    Returns:
        NumPy 2D array of shape (N, 14) and dtype float64, where N = len(sequences).
    """
    if len(sequences) == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float64)

    matrix = [extract_features(seq) for seq in sequences]
    return np.asarray(matrix, dtype=np.float64)
