"""Unit test suite for the Finite State Machine (FSM) validator.

Authoritative source: backend/docs/WORKFLOW_SPEC.md
"""

import unittest
from backend.fsm.states import OrderEvent, OrderState
from backend.fsm.transitions import TRANSITION_TABLE, get_allowed_events, get_next_state
from backend.fsm.validator import validate_trace


class TestFSMValidator(unittest.TestCase):
    """Test suite covering FSM validation, state transitions, and edge cases."""

    def test_01_complete_successful_order(self):
        """Test a complete normal fulfillment lifecycle (Happy Path).

        CONFIRM_PAYMENT -> PACK_ITEMS -> SHIP_ITEMS -> DELIVER_ITEMS
        """
        trace = [
            OrderEvent.CONFIRM_PAYMENT,
            OrderEvent.PACK_ITEMS,
            OrderEvent.SHIP_ITEMS,
            OrderEvent.DELIVER_ITEMS,
        ]
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.DELIVERED.value)
        self.assertEqual(result["total_steps"], 4)
        self.assertEqual(result["successful_transitions"], 4)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["history"]), 4)
        self.assertEqual(result["history"][-1]["to_state"], OrderState.DELIVERED.value)

    def test_02_successful_cancellation_from_created(self):
        """Test immediate cancellation before payment."""
        trace = [OrderEvent.CANCEL_ORDER]
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.CANCELLED.value)
        self.assertEqual(result["total_steps"], 1)
        self.assertEqual(result["successful_transitions"], 1)
        self.assertIsNone(result["error"])
        self.assertEqual(result["history"][0]["from_state"], OrderState.ORDER_CREATED.value)
        self.assertEqual(result["history"][0]["to_state"], OrderState.CANCELLED.value)

    def test_03_successful_refund_from_paid(self):
        """Test cancellation and refund of a paid order before packaging."""
        trace = [OrderEvent.CONFIRM_PAYMENT, OrderEvent.REFUND_ORDER]
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.REFUNDED.value)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_transitions"], 2)
        self.assertIsNone(result["error"])
        self.assertEqual(result["history"][-1]["to_state"], OrderState.REFUNDED.value)

    def test_04_valid_but_incomplete_sequence(self):
        """Test a valid in-progress workflow prefix (active order in warehouse)."""
        trace = [OrderEvent.CONFIRM_PAYMENT, OrderEvent.PACK_ITEMS]
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.PACKED.value)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_transitions"], 2)
        self.assertIsNone(result["error"])

    def test_05_empty_event_sequence(self):
        """Test an empty event sequence (newly created order with no events)."""
        trace = []
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.ORDER_CREATED.value)
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["successful_transitions"], 0)
        self.assertEqual(result["history"], [])
        self.assertIsNone(result["error"])

    def test_06_invalid_transition_skipping_packing(self):
        """Test an illegal transition: attempting to ship directly from PAID."""
        trace = [OrderEvent.CONFIRM_PAYMENT, OrderEvent.SHIP_ITEMS]
        result = validate_trace(trace)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.PAID.value)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_transitions"], 1)
        self.assertIsNotNone(result["error"])

        err = result["error"]
        self.assertEqual(err["failed_step"], 2)
        self.assertEqual(err["failed_event"], OrderEvent.SHIP_ITEMS.value)
        self.assertEqual(err["state_at_failure"], OrderState.PAID.value)
        self.assertEqual(err["error_type"], "INVALID_TRANSITION")
        self.assertIn(OrderEvent.PACK_ITEMS.value, err["allowed_events"])
        self.assertIn(OrderEvent.REFUND_ORDER.value, err["allowed_events"])

    def test_07_unknown_event(self):
        """Test an unrecognized event token not present in alphabet Sigma."""
        trace = [OrderEvent.CONFIRM_PAYMENT, "BYPASS_SECURITY_INJECTION"]
        result = validate_trace(trace)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.PAID.value)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_transitions"], 1)
        self.assertIsNotNone(result["error"])

        err = result["error"]
        self.assertEqual(err["failed_step"], 2)
        self.assertEqual(err["failed_event"], "BYPASS_SECURITY_INJECTION")
        self.assertEqual(err["state_at_failure"], OrderState.PAID.value)
        self.assertEqual(err["error_type"], "UNKNOWN_EVENT")
        self.assertEqual(err["allowed_events"], [])

    def test_08_event_after_terminal_state(self):
        """Test attempting to execute an event after reaching terminal state DELIVERED."""
        trace = [
            OrderEvent.CONFIRM_PAYMENT,
            OrderEvent.PACK_ITEMS,
            OrderEvent.SHIP_ITEMS,
            OrderEvent.DELIVER_ITEMS,
            OrderEvent.CANCEL_ORDER,  # Event arriving in terminal state DELIVERED
        ]
        result = validate_trace(trace)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.DELIVERED.value)
        self.assertEqual(result["total_steps"], 5)
        self.assertEqual(result["successful_transitions"], 4)
        self.assertIsNotNone(result["error"])

        err = result["error"]
        self.assertEqual(err["failed_step"], 5)
        self.assertEqual(err["failed_event"], OrderEvent.CANCEL_ORDER.value)
        self.assertEqual(err["state_at_failure"], OrderState.DELIVERED.value)
        self.assertEqual(err["error_type"], "TERMINAL_STATE_VIOLATION")
        self.assertEqual(err["allowed_events"], [])

    def test_09_terminal_state_precedence_over_unknown_event(self):
        """Test terminal-state precedence: unknown event after terminal state

        must produce TERMINAL_STATE_VIOLATION, not UNKNOWN_EVENT.
        """
        trace = [
            OrderEvent.CONFIRM_PAYMENT,
            OrderEvent.PACK_ITEMS,
            OrderEvent.SHIP_ITEMS,
            OrderEvent.DELIVER_ITEMS,
            "COMPLETELY_UNKNOWN_CORRUPTED_ACTION",
        ]
        result = validate_trace(trace)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.DELIVERED.value)
        self.assertEqual(result["total_steps"], 5)
        self.assertEqual(result["successful_transitions"], 4)
        self.assertIsNotNone(result["error"])

        err = result["error"]
        self.assertEqual(err["failed_step"], 5)
        self.assertEqual(err["failed_event"], "COMPLETELY_UNKNOWN_CORRUPTED_ACTION")
        self.assertEqual(err["state_at_failure"], OrderState.DELIVERED.value)
        self.assertEqual(err["error_type"], "TERMINAL_STATE_VIOLATION")

    def test_10_duplicate_payment_confirmation(self):
        """Test repeated event with no self-loop: duplicate CONFIRM_PAYMENT in PAID."""
        trace = [OrderEvent.CONFIRM_PAYMENT, OrderEvent.CONFIRM_PAYMENT]
        result = validate_trace(trace)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.PAID.value)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_transitions"], 1)
        self.assertIsNotNone(result["error"])

        err = result["error"]
        self.assertEqual(err["failed_step"], 2)
        self.assertEqual(err["failed_event"], OrderEvent.CONFIRM_PAYMENT.value)
        self.assertEqual(err["state_at_failure"], OrderState.PAID.value)
        self.assertEqual(err["error_type"], "INVALID_TRANSITION")

    def test_11_payment_failure_retry_and_recovery(self):
        """Test valid payment recovery loop:

        FAIL_PAYMENT -> RETRY_PAYMENT -> CONFIRM_PAYMENT -> PACK -> SHIP -> DELIVER.
        """
        trace = [
            OrderEvent.FAIL_PAYMENT,
            OrderEvent.RETRY_PAYMENT,
            OrderEvent.CONFIRM_PAYMENT,
            OrderEvent.PACK_ITEMS,
            OrderEvent.SHIP_ITEMS,
            OrderEvent.DELIVER_ITEMS,
        ]
        result = validate_trace(trace)

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["final_state"], OrderState.DELIVERED.value)
        self.assertEqual(result["total_steps"], 6)
        self.assertEqual(result["successful_transitions"], 6)
        self.assertIsNone(result["error"])

    def test_12_all_eleven_transitions_defined_in_table(self):
        """Explicitly test every single one of the 11 transitions defined in WORKFLOW_SPEC.md."""
        # 1. ORDER_CREATED + CONFIRM_PAYMENT -> PAID
        self.assertEqual(
            get_next_state(OrderState.ORDER_CREATED, OrderEvent.CONFIRM_PAYMENT),
            OrderState.PAID.value,
        )

        # 2. ORDER_CREATED + FAIL_PAYMENT -> PAYMENT_FAILED
        self.assertEqual(
            get_next_state(OrderState.ORDER_CREATED, OrderEvent.FAIL_PAYMENT),
            OrderState.PAYMENT_FAILED.value,
        )

        # 3. ORDER_CREATED + CANCEL_ORDER -> CANCELLED
        self.assertEqual(
            get_next_state(OrderState.ORDER_CREATED, OrderEvent.CANCEL_ORDER),
            OrderState.CANCELLED.value,
        )

        # 4. PAYMENT_FAILED + RETRY_PAYMENT -> ORDER_CREATED
        self.assertEqual(
            get_next_state(OrderState.PAYMENT_FAILED, OrderEvent.RETRY_PAYMENT),
            OrderState.ORDER_CREATED.value,
        )

        # 5. PAYMENT_FAILED + CONFIRM_PAYMENT -> PAID
        self.assertEqual(
            get_next_state(OrderState.PAYMENT_FAILED, OrderEvent.CONFIRM_PAYMENT),
            OrderState.PAID.value,
        )

        # 6. PAYMENT_FAILED + CANCEL_ORDER -> CANCELLED
        self.assertEqual(
            get_next_state(OrderState.PAYMENT_FAILED, OrderEvent.CANCEL_ORDER),
            OrderState.CANCELLED.value,
        )

        # 7. PAID + PACK_ITEMS -> PACKED
        self.assertEqual(
            get_next_state(OrderState.PAID, OrderEvent.PACK_ITEMS),
            OrderState.PACKED.value,
        )

        # 8. PAID + REFUND_ORDER -> REFUNDED
        self.assertEqual(
            get_next_state(OrderState.PAID, OrderEvent.REFUND_ORDER),
            OrderState.REFUNDED.value,
        )

        # 9. PACKED + SHIP_ITEMS -> SHIPPED
        self.assertEqual(
            get_next_state(OrderState.PACKED, OrderEvent.SHIP_ITEMS),
            OrderState.SHIPPED.value,
        )

        # 10. PACKED + REFUND_ORDER -> REFUNDED
        self.assertEqual(
            get_next_state(OrderState.PACKED, OrderEvent.REFUND_ORDER),
            OrderState.REFUNDED.value,
        )

        # 11. SHIPPED + DELIVER_ITEMS -> DELIVERED
        self.assertEqual(
            get_next_state(OrderState.SHIPPED, OrderEvent.DELIVER_ITEMS),
            OrderState.DELIVERED.value,
        )

        # Verify the transition table size is exactly 11
        self.assertEqual(len(TRANSITION_TABLE), 11)

    def test_13_allowed_events_helper(self):
        """Test get_allowed_events for each active state."""
        self.assertEqual(
            get_allowed_events(OrderState.ORDER_CREATED),
            sorted([
                OrderEvent.CANCEL_ORDER.value,
                OrderEvent.CONFIRM_PAYMENT.value,
                OrderEvent.FAIL_PAYMENT.value,
            ]),
        )
        self.assertEqual(
            get_allowed_events(OrderState.PAYMENT_FAILED),
            sorted([
                OrderEvent.CANCEL_ORDER.value,
                OrderEvent.CONFIRM_PAYMENT.value,
                OrderEvent.RETRY_PAYMENT.value,
            ]),
        )
        self.assertEqual(
            get_allowed_events(OrderState.PAID),
            sorted([
                OrderEvent.PACK_ITEMS.value,
                OrderEvent.REFUND_ORDER.value,
            ]),
        )
        self.assertEqual(
            get_allowed_events(OrderState.PACKED),
            sorted([
                OrderEvent.REFUND_ORDER.value,
                OrderEvent.SHIP_ITEMS.value,
            ]),
        )
        self.assertEqual(
            get_allowed_events(OrderState.SHIPPED),
            [OrderEvent.DELIVER_ITEMS.value],
        )
        self.assertEqual(get_allowed_events(OrderState.DELIVERED), [])
        self.assertEqual(get_allowed_events(OrderState.CANCELLED), [])
        self.assertEqual(get_allowed_events(OrderState.REFUNDED), [])


if __name__ == "__main__":
    unittest.main()
