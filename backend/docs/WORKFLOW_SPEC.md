# Order Workflow & Finite State Machine (FSM) Specification

**Project:** Anomaly Detection in Process Workflows Using Automata and ML  
**Domain:** Online Shopping Order Management Workflow  
**Document Version:** 1.2  
**Status:** Approved Specification (Revised)  

---

## 1. Overview & Conceptual Boundary

This document defines the formal workflow model for an online shopping order management lifecycle. The project employs a dual-engine anomaly detection architecture:

1. **Finite State Machine (FSM):** Provides **deterministic** verification of process structure and transition rules. It answers: *"Does this sequence obey the formal grammar and operational logic of an order lifecycle?"*
2. **Machine Learning (Isolation Forest):** Provides **statistical** anomaly detection on numeric sequence features. It answers: *"Is this sequence statistically rare, unusually delayed, or behaviorally abnormal, even if structurally legal?"*

```text
               ┌────────────────────────────────────────────────────────┐
               │              Incoming Order Event Trace                │
               └───────────┬────────────────────────────────┬───────────┘
                           │                                │
                           ▼                                ▼
              ┌─────────────────────────┐      ┌─────────────────────────┐
              │  FSM Validation Engine  │      │  ML Feature Extractor   │
              │  (Deterministic Rules)  │      │   & Isolation Forest    │
              └────────────┬────────────┘      └────────────┬────────────┘
                           │                                │
                           │  Structural                    │  Statistical
                           │  Violation Report              │  Anomaly Score
                           │                                │
                           └───────────────┬────────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │    Combined Analyzer    │
                              │ (Hybrid Verdict Report) │
                              └─────────────────────────┘
```

> [!IMPORTANT]
> **Separation of Concerns:** The FSM must **never** perform probabilistic scoring, and ML must **never** replace strict transition enforcement. A sequence with multiple failed payment retries may be fully **valid** under FSM rules, yet flagged as an anomaly by the **Isolation Forest**.

---

## 2. Workflow States (Section A)

In formal automata theory, the state space is defined as a finite set $Q$. Each state represents a distinct, mutually exclusive condition of an order entity.

### 2.1 State Definitions

| State Name | State Type | Description / Meaning |
| :--- | :--- | :--- |
| `ORDER_CREATED` | **Initial State ($q_0$)** | The customer has placed the order. The order record exists in the system, inventory is reserved, and the order is awaiting payment processing. |
| `PAYMENT_FAILED` | Intermediate State | A payment attempt was rejected, timed out, or declined by the payment gateway. The order is awaiting a customer retry or cancellation. |
| `PAID` | Intermediate State | Payment has been successfully verified, captured, and cleared. The order is authorized for warehouse fulfillment. |
| `PACKED` | Intermediate State | The warehouse team has picked the items from inventory, verified them, and boxed the package for courier dispatch. |
| `SHIPPED` | Intermediate State | The parcel has been handed over to the courier/logistics carrier, a tracking number is assigned, and goods are in transit. |
| `DELIVERED` | **Terminal State ($F_1$)** | The package has been successfully delivered and confirmed by the customer/carrier. Fulfillment succeeded. |
| `CANCELLED` | **Terminal State ($F_2$)** | The order was cancelled prior to payment or after payment failure. No money was captured, so no refund is required. |
| `REFUNDED` | **Terminal State ($F_3$)** | A paid order was cancelled before dispatch (while in `PAID` or `PACKED`), and captured funds were credited back to the customer. |

### 2.2 Initial State ($q_0$)
- **`ORDER_CREATED`**: Every order lifecycle begins in this state. An order is registered when a customer checks out.

### 2.3 Terminal States ($F \subseteq Q$)
A terminal state (or dead-end state) has **no permitted outgoing transitions** ($\delta(q_{\text{term}}, e) = \emptyset$ for all $e \in \Sigma$). Once reached, the order lifecycle is permanently closed:
- **`DELIVERED`**: Successful completion of fulfillment.
- **`CANCELLED`**: Clean cancellation of an unpaid order.
- **`REFUNDED`**: Order revoked after payment with funds reversed.

Any event arriving when an order is already in a terminal state is strictly rejected as a `TERMINAL_STATE_VIOLATION`.

---

## 3. Workflow Events (Section B)

Events constitute the alphabet $\Sigma$ of input symbols processed by the automaton. An **event** is an instantaneous occurrence or external trigger that drives an order from one state to another.

### 3.1 Event Definitions

| Event Name | Trigger Source | Description / Meaning |
| :--- | :--- | :--- |
| `CONFIRM_PAYMENT` | Payment Gateway | Payment gateway authorizes and captures the customer's payment. |
| `FAIL_PAYMENT` | Payment Gateway | Payment gateway returns a failure (insufficient funds, timeout, declined). |
| `RETRY_PAYMENT` | Customer / System | Customer re-initiates checkout following a failed payment attempt. |
| `CANCEL_ORDER` | Customer / System | Order is cancelled while in an unpaid state (`ORDER_CREATED` or `PAYMENT_FAILED`). |
| `PACK_ITEMS` | Warehouse System | Warehouse marks goods as picked, verified, and packed. |
| `SHIP_ITEMS` | Logistics Carrier | Carrier scans parcel pickup and begins transit to destination. |
| `DELIVER_ITEMS` | Logistics Carrier | Carrier records successful delivery at destination address. |
| `REFUND_ORDER` | Merchant / Billing | Payment reversal is issued for an already paid order (`PAID` or `PACKED`). |

### 3.2 Distinction Between States and Events

To prevent architectural ambiguity, states and events follow strict naming and semantic conventions:

| Category | Grammatical Form | Semantics | Examples |
| :--- | :--- | :--- | :--- |
| **State ($Q$)** | Past Participle / Adjective (Condition) | A **persistent condition** an order resides in over a duration of time. | `ORDER_CREATED`, `PAID`, `PACKED`, `SHIPPED` |
| **Event ($\Sigma$)** | Imperative / Action Verb (Trigger) | An **instantaneous stimulus** that arrives at a discrete point in time. | `CONFIRM_PAYMENT`, `PACK_ITEMS`, `SHIP_ITEMS`, `CANCEL_ORDER` |

#### Valid vs. Invalid Event Scenarios
- **Valid Event Application:** In state `ORDER_CREATED`, applying `CONFIRM_PAYMENT` transitions the order to `PAID`.
- **Invalid Event Application:** In state `ORDER_CREATED`, applying `SHIP_ITEMS` is rejected because shipping cannot occur before payment and packing.
- **Syntactic Event Violation:** Applying an unrecognized string (e.g., `APPLY_DISCOUNT_CODE` or `HACK_ORDER`) is rejected as an alphabet violation ($e \notin \Sigma$).

---

## 4. Formal Transition Model & Edge Cases (Section C)

### 4.1 Automata Model: Partial Transition Function vs. Complete DFA

In theoretical computer science, a standard textbook **complete DFA** requires the transition function to be total:
$$\delta: Q \times \Sigma \to Q$$
Under a complete DFA, every single state must define a transition for every symbol in $\Sigma$. To handle invalid operations, a complete DFA introduces an artificial, non-accepting "trap" (or "sink" / "dead") state $q_{\text{trap}}$ where all disallowed transitions lead, looping indefinitely: $\delta(q_{\text{trap}}, e) = q_{\text{trap}}$.

In **workflow management systems, business process automation, and process mining**, state machines are modeled using a **partial transition function**:
$$\delta: Q \times \Sigma \rightharpoonup Q$$

- If a state-event pair $(q, e)$ is defined in the transition table, the transition is **valid**, and the engine updates the current state to $\delta(q, e)$.
- If a state-event pair $(q, e)$ is **undefined** (i.e., $\delta(q, e) = \emptyset$), the transition is **invalid**. The validator halts execution immediately at that exact event, reports an `INVALID_TRANSITION`, and preserves the current state without advancing to an artificial trap state.

This partial model avoids cluttering the state space with synthetic trap states and provides immediate, pinpoint diagnostic feedback for workflow auditing.

---

### 4.2 Complete Transition Table

The table below lists all permitted transitions. Any state-event combination not listed is undefined and therefore **invalid**.

| Current State ($q$) | Event ($e$) | Next State ($\delta(q, e)$) | Status | Business Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `ORDER_CREATED` | `CONFIRM_PAYMENT` | `PAID` | **Valid** | Successful checkout payment captures funds. |
| `ORDER_CREATED` | `FAIL_PAYMENT` | `PAYMENT_FAILED` | **Valid** | Payment gateway declines transaction. |
| `ORDER_CREATED` | `CANCEL_ORDER` | `CANCELLED` | **Valid** | Customer cancels prior to making payment. |
| `PAYMENT_FAILED` | `RETRY_PAYMENT` | `ORDER_CREATED` | **Valid** | Customer updates card or retries checkout. |
| `PAYMENT_FAILED` | `CONFIRM_PAYMENT` | `PAID` | **Valid** | Direct payment success on secondary attempt. |
| `PAYMENT_FAILED` | `CANCEL_ORDER` | `CANCELLED` | **Valid** | Customer abandons checkout after payment failure. |
| `PAID` | `PACK_ITEMS` | `PACKED` | **Valid** | Warehouse fulfills and boxes paid order. |
| `PAID` | `REFUND_ORDER` | `REFUNDED` | **Valid** | Customer revokes paid order before packaging; money refunded. |
| `PACKED` | `SHIP_ITEMS` | `SHIPPED` | **Valid** | Parcel handed to courier for transit. |
| `PACKED` | `REFUND_ORDER` | `REFUNDED` | **Valid** | Order cancelled before dispatch; parcel unpacked and refunded. |
| `SHIPPED` | `DELIVER_ITEMS` | `DELIVERED` | **Valid** | Package successfully handed to recipient. |
| `DELIVERED` | *Any Event* | — | **Invalid** | Terminal state with no permitted outgoing transitions. |
| `CANCELLED` | *Any Event* | — | **Invalid** | Terminal state with no permitted outgoing transitions. |
| `REFUNDED` | *Any Event* | — | **Invalid** | Terminal state with no permitted outgoing transitions. |

---

### 4.3 Handling Exceptional Conditions & Evaluation Precedence

The FSM validation engine evaluates each incoming event against three prioritized checks. This establishes strict **terminal-state precedence**: **any event received after an order enters a terminal state must produce a `TERMINAL_STATE_VIOLATION`, even if the event is unknown or malformed.**

```text
                           ┌──────────────────────────┐
                           │      Incoming Event e    │
                           └────────────┬─────────────┘
                                        │
                    [Check 1] Is current_state Terminal?
                                        │
                       ┌────────────────┴────────────────┐
                      YES                                NO
                       │                                 │
         [Report TERMINAL_STATE]            [Check 2] Is e in alphabet Σ?
                                                         │
                                        ┌────────────────┴────────────────┐
                                       YES                                NO
                                        │                                 │
                        [Check 3] Does δ(q, e) exist?           [Report UNKNOWN_EVENT]
                                        │
                              ┌─────────┴─────────┐
                             YES                  NO
                              │                   │
                      [Update State]      [Report INVALID_TRANSITION]
                      Current State :=    (covers disallowed steps &
                        δ(q, e)            unpermitted duplicate events)
```

#### 1. Event Occurring After Reaching a Terminal State (Precedence 1)
- **Condition:** The current state $q \in \{ \text{DELIVERED}, \text{CANCELLED}, \text{REFUNDED} \}$ and any subsequent event $e$ arrives.
- **FSM Behavior:**
  - Evaluated **first** before checking whether $e$ is recognized in the alphabet or valid in the transition table.
  - Terminal states have no permitted outgoing transitions.
  - Return `is_valid = False` with error type `TERMINAL_STATE_VIOLATION`.
  - Report that the workflow instance has already reached its final lifecycle state and cannot accept further mutations.

#### 2. Unknown Event Received (Precedence 2)
- **Condition:** Current state is not terminal, but $e \notin \Sigma$ (e.g., `PROCESS_FRAUD_CHECK` or mistyped `SHIPP_ITEM`).
- **FSM Behavior:**
  - Evaluated **second** after confirming the order is in an active (non-terminal) state.
  - Immediately halt trace evaluation.
  - Return `is_valid = False` with error type `UNKNOWN_EVENT`.
  - Report the unknown token and the valid alphabet $\Sigma$.

#### 3. Event Not Allowed in Current State / Repeated Events (Precedence 3)
- **Condition:** Current state is active and $e \in \Sigma$, but $\delta(q, e) = \emptyset$ (e.g., `SHIP_ITEMS` while in `PAID`, or duplicate `CONFIRM_PAYMENT` while already in `PAID`).
- **FSM Behavior:** 
  - Evaluated **third** when checking the transition table.
  - Immediately halt trace evaluation.
  - Return `is_valid = False` with error type `INVALID_TRANSITION`.
  - Provide `failed_step`, `failed_event`, `state_at_failure`, and the list of permitted events in that state.
  - *ML Note:* In contrast, legitimate loops (such as `FAIL_PAYMENT` followed by `RETRY_PAYMENT`) are structurally valid. However, repeating that loop an excessive number of times (e.g., 6 retries) represents a **statistical anomaly** detected by the Isolation Forest.

---

## 5. Example Event Sequences (Section D)

### 5.1 Valid Event Sequences

A sequence is **valid** if every transition obeys the transition table. It is **complete** if and only if it terminates in a terminal state (`DELIVERED`, `CANCELLED`, `REFUNDED`).

#### Sequence V1: Standard Purchase (Happy Path) — Valid & Complete
- **Trace:** `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID`
  2. `PAID` + `PACK_ITEMS` $\to$ `PACKED`
  3. `PACKED` + `SHIP_ITEMS` $\to$ `SHIPPED`
  4. `SHIPPED` + `DELIVER_ITEMS` $\to$ `DELIVERED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 4`, `successful_transitions = 4`, `final_state = "DELIVERED"`
- **Explanation:** Linear progression through full fulfillment ending in terminal state `DELIVERED`.

#### Sequence V2: Immediate Cancellation — Valid & Complete
- **Trace:** `[CANCEL_ORDER]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `CANCEL_ORDER` $\to$ `CANCELLED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 1`, `successful_transitions = 1`, `final_state = "CANCELLED"`
- **Explanation:** Customer aborts checkout before payment. Ends in terminal state `CANCELLED`.

#### Sequence V3: Payment Failure with Retry and Fulfillment — Valid & Complete
- **Trace:** `[FAIL_PAYMENT, RETRY_PAYMENT, CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `FAIL_PAYMENT` $\to$ `PAYMENT_FAILED`
  2. `PAYMENT_FAILED` + `RETRY_PAYMENT` $\to$ `ORDER_CREATED`
  3. `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID`
  4. `PAID` + `PACK_ITEMS` $\to$ `PACKED`
  5. `PACKED` + `SHIP_ITEMS` $\to$ `SHIPPED`
  6. `SHIPPED` + `DELIVER_ITEMS` $\to$ `DELIVERED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 6`, `successful_transitions = 6`, `final_state = "DELIVERED"`
- **Explanation:** Handles payment decline, recovers via retry, and completes fulfillment.

#### Sequence V4: Abandonment Following Payment Failure — Valid & Complete
- **Trace:** `[FAIL_PAYMENT, CANCEL_ORDER]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `FAIL_PAYMENT` $\to$ `PAYMENT_FAILED`
  2. `PAYMENT_FAILED` + `CANCEL_ORDER` $\to$ `CANCELLED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 2`, `successful_transitions = 2`, `final_state = "CANCELLED"`
- **Explanation:** Customer declines to retry after payment failure and cancels the order.

#### Sequence V5: Paid Order Cancelled Before Packing — Valid & Complete
- **Trace:** `[CONFIRM_PAYMENT, REFUND_ORDER]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID`
  2. `PAID` + `REFUND_ORDER` $\to$ `REFUNDED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 2`, `successful_transitions = 2`, `final_state = "REFUNDED"`
- **Explanation:** Customer cancels paid order prior to packaging; refund is issued cleanly.

#### Sequence V6: Paid Order Cancelled After Packing — Valid & Complete
- **Trace:** `[CONFIRM_PAYMENT, PACK_ITEMS, REFUND_ORDER]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID`
  2. `PAID` + `PACK_ITEMS` $\to$ `PACKED`
  3. `PACKED` + `REFUND_ORDER` $\to$ `REFUNDED`
- **Result:** `is_valid = True`, `is_complete = True`
- **Metrics:** `total_steps = 3`, `successful_transitions = 3`, `final_state = "REFUNDED"`
- **Explanation:** Order was packed, but cancellation arrived before dispatch. Parcel returned to inventory and funds refunded.

#### Sequence V7: In-Progress Order (Valid Prefix) — Valid & Incomplete
- **Trace:** `[CONFIRM_PAYMENT, PACK_ITEMS]`
- **Trace Execution:**
  1. `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID`
  2. `PAID` + `PACK_ITEMS` $\to$ `PACKED`
- **Result:** `is_valid = True`, `is_complete = False`
- **Metrics:** `total_steps = 2`, `successful_transitions = 2`, `final_state = "PACKED"`
- **Explanation:** Every transition is strictly valid, but the order is actively in progress at the warehouse (`PACKED`), not in a terminal state.

#### Sequence V8: Empty Event Sequence — Valid & Incomplete
- **Trace:** `[]`
- **Trace Execution:**
  - No events processed. Initial state unchanged.
- **Result:** `is_valid = True`, `is_complete = False`
- **Metrics:** `total_steps = 0`, `successful_transitions = 0`, `final_state = "ORDER_CREATED"`
- **Explanation:** Represents a newly created order that has not yet received any lifecycle events. Fully valid prefix, but incomplete.

---

### 5.2 Invalid Event Sequences

An invalid sequence contains at least one violation. It immediately halts at the failing event and is always `is_valid = False` and `is_complete = False`.

#### Sequence I1: Skipping Packaging (Direct Dispatch)
- **Trace:** `[CONFIRM_PAYMENT, SHIP_ITEMS, DELIVER_ITEMS]`
- **Trace Execution:**
  1. Step 1: `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID` (Success)
  2. Step 2: `PAID` + `SHIP_ITEMS` $\to$ **HALT! Undefined transition.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 2`, `successful_transitions = 1`, `failed_step = 2`, `final_state = "PAID"`
- **Error Type:** `INVALID_TRANSITION`
- **Explanation:** In state `PAID`, `SHIP_ITEMS` is not permitted. Items must first be assembled and packaged via `PACK_ITEMS`.

#### Sequence I2: Delivery Attempt on Unpaid Order
- **Trace:** `[DELIVER_ITEMS, CONFIRM_PAYMENT]`
- **Trace Execution:**
  1. Step 1: `ORDER_CREATED` + `DELIVER_ITEMS` $\to$ **HALT! Undefined transition.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 1`, `successful_transitions = 0`, `failed_step = 1`, `final_state = "ORDER_CREATED"`
- **Error Type:** `INVALID_TRANSITION`
- **Explanation:** An order in `ORDER_CREATED` cannot be delivered. Prerequisite payment, packing, and shipping steps were all skipped.

#### Sequence I3: Action Post-Delivery (Terminal State Violation with Precedence)
- **Trace:** `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS, CANCEL_ORDER]`
- **Trace Execution:**
  1. Steps 1–4 transition cleanly to `DELIVERED`.
  2. Step 5: `DELIVERED` + `CANCEL_ORDER` $\to$ **HALT! Terminal state check triggers first.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 5`, `successful_transitions = 4`, `failed_step = 5`, `final_state = "DELIVERED"`
- **Error Type:** `TERMINAL_STATE_VIOLATION`
- **Explanation:** `DELIVERED` is a terminal state with no permitted outgoing transitions. By terminal-state precedence, any subsequent event (even if unrecognized) immediately triggers `TERMINAL_STATE_VIOLATION`.

#### Sequence I4: Refunding an Unpaid Order
- **Trace:** `[REFUND_ORDER]`
- **Trace Execution:**
  1. Step 1: `ORDER_CREATED` + `REFUND_ORDER` $\to$ **HALT! Undefined transition.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 1`, `successful_transitions = 0`, `failed_step = 1`, `final_state = "ORDER_CREATED"`
- **Error Type:** `INVALID_TRANSITION`
- **Explanation:** An order in `ORDER_CREATED` has not captured any money. It must be cancelled via `CANCEL_ORDER`, not `REFUND_ORDER`.

#### Sequence I5: Duplicate Payment Confirmation (Repeated Event)
- **Trace:** `[CONFIRM_PAYMENT, CONFIRM_PAYMENT, PACK_ITEMS]`
- **Trace Execution:**
  1. Step 1: `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID` (Success)
  2. Step 2: `PAID` + `CONFIRM_PAYMENT` $\to$ **HALT! Undefined transition.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 2`, `successful_transitions = 1`, `failed_step = 2`, `final_state = "PAID"`
- **Error Type:** `INVALID_TRANSITION`
- **Explanation:** An order in `PAID` cannot receive another `CONFIRM_PAYMENT`. Prevents double-charging. Classified uniformly as an `INVALID_TRANSITION`.

#### Sequence I6: Cancellation During Active Transit
- **Trace:** `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, CANCEL_ORDER]`
- **Trace Execution:**
  1. Steps 1–3 transition order to `SHIPPED`.
  2. Step 4: `SHIPPED` + `CANCEL_ORDER` $\to$ **HALT! Undefined transition.**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 4`, `successful_transitions = 3`, `failed_step = 4`, `final_state = "SHIPPED"`
- **Error Type:** `INVALID_TRANSITION`
- **Explanation:** In state `SHIPPED`, the parcel is in custody of the courier. The order cannot be cancelled mid-transit.

#### Sequence I7: Unrecognized Event Injection
- **Trace:** `[CONFIRM_PAYMENT, BYPASS_PAYWALL_INJECTION, PACK_ITEMS]`
- **Trace Execution:**
  1. Step 1: `ORDER_CREATED` + `CONFIRM_PAYMENT` $\to$ `PAID` (Success)
  2. Step 2: `PAID` is not terminal. Alphabet check fails: `BYPASS_PAYWALL_INJECTION` $\notin \Sigma$. **HALT!**
- **Result:** `is_valid = False`, `is_complete = False`
- **Metrics:** `total_steps = 2`, `successful_transitions = 1`, `failed_step = 2`, `final_state = "PAID"`
- **Error Type:** `UNKNOWN_EVENT`
- **Explanation:** The order is in active state `PAID`. The event `BYPASS_PAYWALL_INJECTION` is not a member of the defined event alphabet $\Sigma$.

---

## 6. FSM Implementation & Validator Specification (Section E)

### 6.1 Step-by-Step Processing Algorithm

The FSM validation engine evaluates an incoming sequence of events using the following formal procedure, adhering strictly to **terminal-state precedence**:

```text
Input: Event sequence E = [e_1, e_2, ..., e_n]
Output: ValidationResult object

1. Set current_state := ORDER_CREATED
2. Set history := []
3. Set successful_transitions := 0

4. For index i, event e in enumerate(E):
     Set step_number := i + 1   // 1-based position

     // Priority Check 1: Terminal state violation (Precedence 1)
     // Any event received after reaching a terminal state triggers this,
     // even if the event is unrecognized or malformed.
     If current_state in TERMINAL_STATES:
          Return FAILURE(
            is_valid = False,
            is_complete = False,
            final_state = current_state,
            total_steps = step_number,
            successful_transitions = successful_transitions,
            error = {
              failed_step: step_number,
              failed_event: e,
              state_at_failure: current_state,
              error_type: "TERMINAL_STATE_VIOLATION",
              allowed_events: [],
              message: "Cannot execute event after reaching terminal state"
            }
          )

     // Priority Check 2: Alphabet membership (Precedence 2)
     // Verified only while the order is in an active (non-terminal) state.
     If e not in ALPHABET:
          Return FAILURE(
            is_valid = False,
            is_complete = False,
            final_state = current_state,
            total_steps = step_number,
            successful_transitions = successful_transitions,
            error = {
              failed_step: step_number,
              failed_event: e,
              state_at_failure: current_state,
              error_type: "UNKNOWN_EVENT",
              allowed_events: [],
              message: "Event not recognized in workflow alphabet"
            }
          )

     // Priority Check 3: Transition existence in partial function δ (Precedence 3)
     next_state := TRANSITION_TABLE.get((current_state, e))
     If next_state is None:
          Return FAILURE(
            is_valid = False,
            is_complete = False,
            final_state = current_state,
            total_steps = step_number,
            successful_transitions = successful_transitions,
            error = {
              failed_step: step_number,
              failed_event: e,
              state_at_failure: current_state,
              error_type: "INVALID_TRANSITION",
              allowed_events: GET_ALLOWED_EVENTS(current_state),
              message: "Transition not permitted from current state"
            }
          )

     // Transition succeeded
     Append {step: step_number, from_state: current_state, event: e, to_state: next_state} to history
     current_state := next_state
     successful_transitions := successful_transitions + 1

5. Set is_complete := (current_state in TERMINAL_STATES)
6. Return SUCCESS(
     is_valid = True,
     is_complete = is_complete,
     final_state = current_state,
     total_steps = len(E),
     successful_transitions = successful_transitions,
     history = history,
     error = None
   )
```

---

### 6.2 Step Counting Definitions

| Metric | Definition | Example (`[CONFIRM_PAYMENT, SHIP_ITEMS]`) |
| :--- | :--- | :--- |
| `total_steps` | Count of input events processed, **including** the event that caused failure. For empty sequences, `total_steps = 0`. | `2` |
| `successful_transitions` | Count of transitions successfully completed before termination or failure. | `1` |
| `failed_step` | 1-based index of the event that caused failure (null on success). | `2` |

---

### 6.3 Validator Return Contract (Data Structure Examples)

#### Example 1: Failure Response (e.g., `[CONFIRM_PAYMENT, SHIP_ITEMS]`)
```json
{
  "is_valid": false,
  "is_complete": false,
  "final_state": "PAID",
  "total_steps": 2,
  "successful_transitions": 1,
  "history": [
    {
      "step": 1,
      "from_state": "ORDER_CREATED",
      "event": "CONFIRM_PAYMENT",
      "to_state": "PAID"
    }
  ],
  "error": {
    "failed_step": 2,
    "failed_event": "SHIP_ITEMS",
    "state_at_failure": "PAID",
    "error_type": "INVALID_TRANSITION",
    "allowed_events": ["PACK_ITEMS", "REFUND_ORDER"],
    "message": "Illegal transition: Event 'SHIP_ITEMS' is not permitted in state 'PAID'. Permitted events are: ['PACK_ITEMS', 'REFUND_ORDER']."
  }
}
```

#### Example 2: Incomplete Valid Sequence (e.g., `[CONFIRM_PAYMENT, PACK_ITEMS]`)
```json
{
  "is_valid": true,
  "is_complete": false,
  "final_state": "PACKED",
  "total_steps": 2,
  "successful_transitions": 2,
  "history": [
    {
      "step": 1,
      "from_state": "ORDER_CREATED",
      "event": "CONFIRM_PAYMENT",
      "to_state": "PAID"
    },
    {
      "step": 2,
      "from_state": "PAID",
      "event": "PACK_ITEMS",
      "to_state": "PACKED"
    }
  ],
  "error": null
}
```

#### Example 3: Complete Valid Sequence (e.g., `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`)
```json
{
  "is_valid": true,
  "is_complete": true,
  "final_state": "DELIVERED",
  "total_steps": 4,
  "successful_transitions": 4,
  "history": [
    {
      "step": 1,
      "from_state": "ORDER_CREATED",
      "event": "CONFIRM_PAYMENT",
      "to_state": "PAID"
    },
    {
      "step": 2,
      "from_state": "PAID",
      "event": "PACK_ITEMS",
      "to_state": "PACKED"
    },
    {
      "step": 3,
      "from_state": "PACKED",
      "event": "SHIP_ITEMS",
      "to_state": "SHIPPED"
    },
    {
      "step": 4,
      "from_state": "SHIPPED",
      "event": "DELIVER_ITEMS",
      "to_state": "DELIVERED"
    }
  ],
  "error": null
}
```

#### Example 4: Empty Sequence (`[]`)
```json
{
  "is_valid": true,
  "is_complete": false,
  "final_state": "ORDER_CREATED",
  "total_steps": 0,
  "successful_transitions": 0,
  "history": [],
  "error": null
}
```

---

### 6.4 Field Specifications

- `is_valid` (`bool`): `True` if every evaluated event transitioned legally; `False` if any failure occurred.
- `is_complete` (`bool`): `True` if the sequence ends in one of the terminal states (`DELIVERED`, `CANCELLED`, `REFUNDED`) and `is_valid` is `True`; `False` otherwise.
- `final_state` (`str`): The state of the order when validation completed or halted.
- `total_steps` (`int`): Count of input events processed, including the event that caused failure.
- `successful_transitions` (`int`): Count of transitions successfully completed before termination or failure.
- `history` (`List[dict]`): Step-by-step transition audit trail.
- `error` (`Optional[dict]`): `null` on success; otherwise contains:
  - `failed_step` (`int`): 1-based index of the invalid event in the input array.
  - `failed_event` (`str`): The exact string of the rejected event.
  - `state_at_failure` (`str`): The state in which the rejection occurred.
  - `error_type` (`str`): Machine-readable failure classification, evaluated in strict order of precedence:
    1. `"TERMINAL_STATE_VIOLATION"`: Event received after order reached a terminal state (**Precedence 1**).
    2. `"UNKNOWN_EVENT"`: Event token not present in alphabet $\Sigma$ (**Precedence 2**).
    3. `"INVALID_TRANSITION"`: Disallowed transition or unpermitted repeated event (**Precedence 3**).
  - `allowed_events` (`List[str]`): Permitted events in `state_at_failure`.
  - `message` (`str`): Human-readable diagnostic explanation.

---

## 7. FSM vs. Machine Learning Distinction

To ensure the FSM design does not conflate logical validity with statistical likelihood, the following table clarifies the boundary between the two detection mechanisms:

| Scenario / Pattern | FSM Verdict (`is_valid`, `is_complete`) | Isolation Forest (ML) Verdict | Explanation |
| :--- | :--- | :--- | :--- |
| `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]` | **Valid, Complete** | **Normal** | Standard, typical linear workflow trace. |
| `[CONFIRM_PAYMENT, PACK_ITEMS]` | **Valid, Incomplete** | **Normal (In-Progress)** | Valid intermediate prefix; active order in warehouse. |
| `[CONFIRM_PAYMENT, SHIP_ITEMS, DELIVER_ITEMS]` | **Invalid, Incomplete** | *(Bypassed / Outlier)* | FSM rejects at step 2: packing skipped. Deterministic rule failure. |
| 5 consecutive `FAIL_PAYMENT` $\to$ `RETRY_PAYMENT` loops, then success | **Valid, Complete** | **Statistical Anomaly** | Allowed by FSM grammar, but statistically rare in customer population (outlier frequency). |
| Order completed in 1.2 seconds from checkout to delivery | **Valid, Complete** | **Statistical Anomaly** | Transition sequence is valid, but duration feature vector is an extreme temporal outlier. |
| `[CONFIRM_PAYMENT, PACK_ITEMS, REFUND_ORDER]` | **Valid, Complete** | **Mild Outlier / Rare Path** | Legal cancellation path, but represents < 2% of overall production traces. |
