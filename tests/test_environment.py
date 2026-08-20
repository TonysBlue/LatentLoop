from __future__ import annotations

import pytest
from contracts import MicSignal, ObservationSignal, ProtocolIdentity, ScreenSignal


def test_observation_cannot_contain_privileged_reward_fields() -> None:
    observation = ObservationSignal(
        "session", 0, 0, 80, MicSignal(b"x" * 7680), ScreenSignal(b"x" * 48, 4, 4)
    )
    assert not hasattr(observation, "task_success")


def test_environment_identity_requires_action_schema() -> None:
    with pytest.raises(ValueError, match="protocol identity"):
        ProtocolIdentity(action_schema_id="")
