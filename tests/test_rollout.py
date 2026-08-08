from __future__ import annotations

import torch

from latentloop.environment import EnvironmentIdentity, Observation, RewardBreakdown
from latentloop.rollout import RolloutGroup


class _Env:
    def __init__(self) -> None:
        self._identity = EnvironmentIdentity("test", "1", "1", "action-v3")
        self.resets: list[tuple[str, int]] = []

    def identity(self):
        return self._identity

    def reset(self, task_id: str, seed: int):
        self.resets.append((task_id, seed))
        return Observation(0, 80, torch.zeros(320), torch.zeros(3, 32, 32), False, 0, False)

    def submit_unit(self, *args, **kwargs):
        return self.reset("task", 1), None

    def evaluate(self, task_id: str):
        return RewardBreakdown(
            task=1.0, speech_quality=0.0, latency_quality=0.0, action_efficiency=0.0, safety=0.0
        )

    def close(self):
        pass


def test_rollout_group_records_group_identity() -> None:
    group = RolloutGroup(group_id="g", task_id="task", seed=1, rollouts=[])
    assert group.group_id == "g"
