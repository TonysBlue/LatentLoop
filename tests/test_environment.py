from __future__ import annotations

import pytest
from contracts import MicSignal, ObservationSignal, ProtocolIdentity, RewardBreakdown, ScreenSignal


def test_observation_cannot_contain_privileged_reward_fields() -> None:
    observation = ObservationSignal(
        "session", 0, 0, 80, MicSignal(b"x" * 7680), ScreenSignal(b"x" * 48, 4, 4, 0)
    )
    assert not hasattr(observation, "task_success")


def test_environment_identity_requires_action_vocabulary() -> None:
    with pytest.raises(ValueError, match="protocol identity"):
        ProtocolIdentity(action_vocabulary_id="")


def test_reward_breakdown_has_fixed_components() -> None:
    reward = RewardBreakdown(
        task=1.0, speech_quality=0.5, latency_quality=0.5, action_efficiency=0.5, safety=-0.1
    )
    assert reward.total == pytest.approx(1.0 + 0.2 * 0.5 - 0.1)
