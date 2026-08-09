"""Physical online rollout trace schema owned by the Training System."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, RewardBreakdown


@dataclass(slots=True)
class Rollout:
    rollout_id: str
    group_id: str
    task_id: str
    seed: int
    observations: list[ObservationSignal] = field(default_factory=list)
    outputs: list[ActuationSignal] = field(default_factory=list)
    receipts: list[EnvironmentReceipt] = field(default_factory=list)
    speech_modes: list[torch.Tensor] = field(default_factory=list)
    speech_codes: list[torch.Tensor] = field(default_factory=list)
    action_tokens: list[torch.Tensor] = field(default_factory=list)
    old_logprobs: list[torch.Tensor] = field(default_factory=list)
    reference_logprobs: list[torch.Tensor] = field(default_factory=list)
    reward_breakdown: RewardBreakdown | None = None
    infrastructure_failure: str | None = None

    @property
    def reward(self) -> float:
        if self.reward_breakdown is None:
            raise ValueError("rollout has not been evaluated")
        return self.reward_breakdown.total


@dataclass(slots=True)
class RolloutGroup:
    group_id: str
    task_id: str
    seed: int
    rollouts: list[Rollout]


def rollout_group(*_args: object, **_kwargs: object) -> RolloutGroup:
    raise NotImplementedError(
        "formal rollout uses PhysicalRolloutClient and the shared Online GRPO loop"
    )


__all__ = ["Rollout", "RolloutGroup", "rollout_group"]
