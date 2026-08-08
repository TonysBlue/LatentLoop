from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "packages/contracts/src")

from contracts import (  # noqa: E402
    ActuationSignal,
    ControlKind,
    ControlSignal,
    MicSignal,
    ObservationSignal,
    ScreenSignal,
    SpeechSignal,
)
from contracts.protocol import (  # noqa: E402
    actuation_to_message,
    message_to_actuation,
    message_to_observation,
    observation_to_message,
)


def test_observation_physical_signal_round_trip() -> None:
    value = ObservationSignal(
        "session",
        0,
        0,
        80,
        MicSignal(b"x" * 7680),
        ScreenSignal(b"rgb", 1, 1, 3),
    )
    restored = message_to_observation(observation_to_message(value))
    assert restored == value


def test_actuation_does_not_contain_raw_action_tokens() -> None:
    value = ActuationSignal(
        "session",
        0,
        SpeechSignal(b"x" * 7680, silent=True),
        (ControlSignal(ControlKind.POINTER_BUTTON, "event", x=0.5, y=0.5, button=0),),
    )
    restored = message_to_actuation(actuation_to_message(value))
    assert restored == value
    assert not hasattr(restored, "action_tokens")


def test_physical_signal_rejects_wrong_audio_clock() -> None:
    with pytest.raises(ValueError, match="24 kHz"):
        MicSignal(b"x", sample_rate_hz=16_000)
