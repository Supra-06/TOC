"""Deterministic state transition table and helper functions.

Authoritative source: backend/docs/WORKFLOW_SPEC.md
"""

from typing import Dict, List, Optional, Tuple, Union
from backend.fsm.states import OrderEvent, OrderState

# Transition table mapping: (current_state, event) -> next_state
# Formal partial transition function: delta: Q x Sigma -> Q
TRANSITION_TABLE: Dict[Tuple[str, str], str] = {
    # From ORDER_CREATED
    (OrderState.ORDER_CREATED.value, OrderEvent.CONFIRM_PAYMENT.value): OrderState.PAID.value,
    (OrderState.ORDER_CREATED.value, OrderEvent.FAIL_PAYMENT.value): OrderState.PAYMENT_FAILED.value,
    (OrderState.ORDER_CREATED.value, OrderEvent.CANCEL_ORDER.value): OrderState.CANCELLED.value,

    # From PAYMENT_FAILED
    (OrderState.PAYMENT_FAILED.value, OrderEvent.RETRY_PAYMENT.value): OrderState.ORDER_CREATED.value,
    (OrderState.PAYMENT_FAILED.value, OrderEvent.CONFIRM_PAYMENT.value): OrderState.PAID.value,
    (OrderState.PAYMENT_FAILED.value, OrderEvent.CANCEL_ORDER.value): OrderState.CANCELLED.value,

    # From PAID
    (OrderState.PAID.value, OrderEvent.PACK_ITEMS.value): OrderState.PACKED.value,
    (OrderState.PAID.value, OrderEvent.REFUND_ORDER.value): OrderState.REFUNDED.value,

    # From PACKED
    (OrderState.PACKED.value, OrderEvent.SHIP_ITEMS.value): OrderState.SHIPPED.value,
    (OrderState.PACKED.value, OrderEvent.REFUND_ORDER.value): OrderState.REFUNDED.value,

    # From SHIPPED
    (OrderState.SHIPPED.value, OrderEvent.DELIVER_ITEMS.value): OrderState.DELIVERED.value,
}


def get_next_state(
    current_state: Union[str, OrderState],
    event: Union[str, OrderEvent],
) -> Optional[str]:
    """Return the next state for a (current_state, event) pair, or None if undefined."""
    state_key = current_state.value if isinstance(current_state, OrderState) else str(current_state)
    event_key = event.value if isinstance(event, OrderEvent) else str(event)
    return TRANSITION_TABLE.get((state_key, event_key))


def get_allowed_events(current_state: Union[str, OrderState]) -> List[str]:
    """Return a sorted list of permitted event names from the given state."""
    state_key = current_state.value if isinstance(current_state, OrderState) else str(current_state)
    allowed = [
        event for (st, event) in TRANSITION_TABLE.keys()
        if st == state_key
    ]
    return sorted(allowed)
