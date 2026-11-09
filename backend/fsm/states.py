"""Order workflow states and events definition.

Authoritative source: backend/docs/WORKFLOW_SPEC.md
"""

from enum import Enum


class OrderState(str, Enum):
    """Order lifecycle states."""

    ORDER_CREATED = "ORDER_CREATED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    PAID = "PAID"
    PACKED = "PACKED"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"


class OrderEvent(str, Enum):
    """Order lifecycle events (input alphabet Sigma)."""

    CONFIRM_PAYMENT = "CONFIRM_PAYMENT"
    FAIL_PAYMENT = "FAIL_PAYMENT"
    RETRY_PAYMENT = "RETRY_PAYMENT"
    CANCEL_ORDER = "CANCEL_ORDER"
    PACK_ITEMS = "PACK_ITEMS"
    SHIP_ITEMS = "SHIP_ITEMS"
    DELIVER_ITEMS = "DELIVER_ITEMS"
    REFUND_ORDER = "REFUND_ORDER"


# Initial state of the order lifecycle
INITIAL_STATE: OrderState = OrderState.ORDER_CREATED

# Terminal states (dead-end states with no permitted outgoing transitions)
TERMINAL_STATES: frozenset[str] = frozenset({
    OrderState.DELIVERED.value,
    OrderState.CANCELLED.value,
    OrderState.REFUNDED.value,
})

# Complete sets for quick membership testing
ALL_STATES: frozenset[str] = frozenset(state.value for state in OrderState)
ALL_EVENTS: frozenset[str] = frozenset(event.value for event in OrderEvent)
