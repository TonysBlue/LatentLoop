from __future__ import annotations

import pytest
from contracts import ControlKind, ControlSignal
from harness.action.safety import SafetyGate
from harness.session import Session


def test_safety_accepts_pointer_without_screen_revision() -> None:
    SafetyGate().validate(ControlSignal(ControlKind.POINTER_MOVE, "event", x=0.1, y=0.2))


def test_session_rejects_out_of_order_unit() -> None:
    session = Session("s")
    observation = type(
        "Observation",
        (),
        {"session_id": "s", "unit_index": 1},
    )()
    with pytest.raises(ValueError, match="out of order"):
        session.accept_observation(observation)
