from __future__ import annotations

import pytest
import torch

from latentloop.environment import (
    EnvironmentIdentity,
    Observation,
    RewardBreakdown,
    validate_observation,
)


def test_observation_cannot_contain_privileged_reward_fields() -> None:
    observation = Observation(
        timestamp_ms=0,
        delta_ms=80,
        mixed_microphone=torch.zeros(1920),
        screen=torch.zeros(3, 4, 4),
        screen_valid=True,
        screen_revision=0,
        terminated=False,
    )
    validate_observation(observation, audio_samples=1920)
    assert not hasattr(observation, "task_success")


def test_environment_identity_requires_action_vocabulary() -> None:
    with pytest.raises(ValueError, match="action vocabulary"):
        EnvironmentIdentity("desktop", "1", "1", "")


def test_reward_breakdown_has_fixed_components() -> None:
    reward = RewardBreakdown(
        task=1.0, speech_quality=0.5, latency_quality=0.5, action_efficiency=0.5, safety=-0.1
    )
    assert reward.total == pytest.approx(1.0 + 0.2 * 0.5 - 0.1)
