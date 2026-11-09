"""FSM validation engine for order workflow traces.

Authoritative source: backend/docs/WORKFLOW_SPEC.md
"""

from typing import Any, Dict, List, Optional, Sequence, Union
from backend.fsm.states import ALL_EVENTS, INITIAL_STATE, TERMINAL_STATES, OrderEvent, OrderState
from backend.fsm.transitions import get_allowed_events, get_next_state


def validate_trace(
    events: Sequence[Union[str, OrderEvent]],
) -> Dict[str, Any]:
    """Validate an ordered sequence of events against the order workflow FSM.

    Evaluation checks are performed in strict order of precedence:
      1. Terminal-state check: If the current state is terminal, any subsequent
         event produces a TERMINAL_STATE_VIOLATION.
      2. Alphabet check: If the event is not in the defined alphabet, produces an
         UNKNOWN_EVENT.
      3. Transition check: If (current_state, event) is undefined, produces an
         INVALID_TRANSITION.

    Args:
        events: Ordered sequence of events (strings or OrderEvent enum values).

    Returns:
        A dictionary containing:
          - is_valid (bool): True if all evaluated transitions succeeded.
          - is_complete (bool): True if is_valid is True and final_state is terminal.
          - final_state (str): State reached when validation finished or halted.
          - total_steps (int): Count of input events processed, including any failing event.
          - successful_transitions (int): Count of transitions successfully completed.
          - history (List[Dict]): Transition audit trail.
          - error (Optional[Dict]): Error diagnostic payload on failure, or None on success.
    """
    current_state: str = INITIAL_STATE.value
    history: List[Dict[str, Any]] = []
    successful_transitions: int = 0

    for idx, raw_event in enumerate(events):
        step_number = idx + 1
        event_str = raw_event.value if isinstance(raw_event, OrderEvent) else str(raw_event)

        # Priority Check 1: Terminal state violation (Precedence 1)
        # Any event arriving after reaching a terminal state triggers this,
        # even if the event is unknown or malformed.
        if current_state in TERMINAL_STATES:
            return {
                "is_valid": False,
                "is_complete": False,
                "final_state": current_state,
                "total_steps": step_number,
                "successful_transitions": successful_transitions,
                "history": history,
                "error": {
                    "failed_step": step_number,
                    "failed_event": event_str,
                    "state_at_failure": current_state,
                    "error_type": "TERMINAL_STATE_VIOLATION",
                    "allowed_events": [],
                    "message": (
                        f"Cannot execute event '{event_str}' after reaching "
                        f"terminal state '{current_state}'."
                    ),
                },
            }

        # Priority Check 2: Alphabet membership (Precedence 2)
        # Evaluated only while the order is in an active (non-terminal) state.
        if event_str not in ALL_EVENTS:
            return {
                "is_valid": False,
                "is_complete": False,
                "final_state": current_state,
                "total_steps": step_number,
                "successful_transitions": successful_transitions,
                "history": history,
                "error": {
                    "failed_step": step_number,
                    "failed_event": event_str,
                    "state_at_failure": current_state,
                    "error_type": "UNKNOWN_EVENT",
                    "allowed_events": [],
                    "message": (
                        f"Event '{event_str}' is not recognized in the workflow alphabet."
                    ),
                },
            }

        # Priority Check 3: Transition existence in partial function delta (Precedence 3)
        next_state = get_next_state(current_state, event_str)
        if next_state is None:
            allowed = get_allowed_events(current_state)
            return {
                "is_valid": False,
                "is_complete": False,
                "final_state": current_state,
                "total_steps": step_number,
                "successful_transitions": successful_transitions,
                "history": history,
                "error": {
                    "failed_step": step_number,
                    "failed_event": event_str,
                    "state_at_failure": current_state,
                    "error_type": "INVALID_TRANSITION",
                    "allowed_events": allowed,
                    "message": (
                        f"Illegal transition: Event '{event_str}' is not permitted "
                        f"in state '{current_state}'. Permitted events are: {allowed}."
                    ),
                },
            }

        # Successful transition
        history.append({
            "step": step_number,
            "from_state": current_state,
            "event": event_str,
            "to_state": next_state,
        })
        current_state = next_state
        successful_transitions += 1

    # Loop completed without any transition failure
    is_complete = current_state in TERMINAL_STATES

    return {
        "is_valid": True,
        "is_complete": is_complete,
        "final_state": current_state,
        "total_steps": len(events),
        "successful_transitions": successful_transitions,
        "history": history,
        "error": None,
    }


# Convenient aliases
validate_workflow = validate_trace
validate_event_sequence = validate_trace
