from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "packages/contracts/src")

from contracts import (  # noqa: E402
    ActionFrame,
    ActionKind,
    ActuationSignal,
    ButtonPhase,
    ControlKind,
    ControlSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    RewardBreakdown,
    ScreenSignal,
    SpeechSignal,
    payload_to_receipt,
    payload_to_reward,
    receipt_to_payload,
    reward_to_payload,
)
from contracts.protocol import (  # noqa: E402
    actuation_to_message,
    message_to_actuation,
    message_to_observation,
    observation_to_message,
)
from runtime.action import ActionFrameDecoder


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
        (
            ControlSignal(
                ControlKind.POINTER_BUTTON,
                "event",
                button=0,
                button_phase=ButtonPhase.CLICK,
            ),
        ),
    )
    restored = message_to_actuation(actuation_to_message(value))
    assert restored == value
    assert not hasattr(restored, "action_tokens")


def test_physical_signal_rejects_wrong_audio_clock() -> None:
    with pytest.raises(ValueError, match="24 kHz"):
        MicSignal(b"x", sample_rate_hz=16_000)


def test_receipt_and_reward_protobuf_round_trip() -> None:
    receipt = EnvironmentReceipt("session", 3, True, execution_latency_ms=1.5, terminated=True)
    reward = RewardBreakdown(1.0, 0.2, 0.3, 0.4, 0.5)
    assert payload_to_receipt(receipt_to_payload(receipt)) == receipt
    assert payload_to_reward(reward_to_payload(reward)) == reward


def test_action_frame_is_executed_in_its_unit() -> None:
    decoder = ActionFrameDecoder()
    controls = decoder.push(ActionFrame(ActionKind.TYPE, text_bytes=b"ok"), event_id="event")
    assert [control.text for control in controls] == ["ok"]
