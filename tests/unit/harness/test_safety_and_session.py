from __future__ import annotations

import pytest
from contracts import ControlKind, ControlSignal
from harness.action.safety import SafetyGate
from harness.session import Session


def test_safety_rejects_stale_revision() -> None:
    gate = SafetyGate()
    with pytest.raises(PermissionError, match="stale"):
        gate.validate(
            ControlSignal(ControlKind.POINTER_MOVE, "event", screen_revision=2, x=0.1, y=0.2),
            current_revision=1,
        )


def test_session_rejects_out_of_order_unit() -> None:
    session = Session("s")
    observation = type(
        "Observation",
        (),
        {"session_id": "s", "unit_index": 1, "screen": type("Screen", (), {"revision": 0})()},
    )()
    with pytest.raises(ValueError, match="out of order"):
        session.accept_observation(observation)
