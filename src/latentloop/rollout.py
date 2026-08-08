from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from latentloop.environment import (
    EnvironmentReceipt,
    IsolatedComputerEnvironment,
    Observation,
    RewardBreakdown,
    validate_observation,
)


@dataclass(slots=True)
class Rollout:
    rollout_id: str
    group_id: str
    task_id: str
    seed: int
    observations: list[Observation] = field(default_factory=list)
    receipts: list[EnvironmentReceipt] = field(default_factory=list)
    speech_modes: list[Tensor] = field(default_factory=list)
    speech_codes: list[Tensor] = field(default_factory=list)
    action_tokens: list[Tensor] = field(default_factory=list)
    old_logprobs: list[Tensor] = field(default_factory=list)
    reference_logprobs: list[Tensor] = field(default_factory=list)
    reward_breakdown: RewardBreakdown | None = None

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


def rollout_group(
    environment_factory: Any,
    *,
    task_id: str,
    seed: int,
    group_id: str,
    group_size: int,
    policy_step: Any,
    horizon_units: int,
    audio_samples: int,
) -> RolloutGroup:
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least 2")
    rollouts: list[Rollout] = []
    for index in range(group_size):
        environment: IsolatedComputerEnvironment = environment_factory()
        observation = environment.reset(task_id, seed)
        validate_observation(observation, audio_samples=audio_samples)
        rollout = Rollout(f"{group_id}-{index}", group_id, task_id, seed)
        rollout.observations.append(observation)
        try:
            for unit_index in range(horizon_units):
                if observation.terminated:
                    break
                speech_mode, speech_codes, action_tokens, old_logprob, reference_logprob = (
                    policy_step(observation, index)
                )
                next_observation, receipt = environment.submit_unit(
                    task_id, unit_index, int(speech_mode), speech_codes, action_tokens
                )
                validate_observation(next_observation, audio_samples=audio_samples)
                rollout.speech_modes.append(speech_mode)
                rollout.speech_codes.append(speech_codes)
                rollout.action_tokens.append(action_tokens)
                rollout.old_logprobs.append(old_logprob)
                rollout.reference_logprobs.append(reference_logprob)
                rollout.receipts.append(receipt)
                rollout.observations.append(next_observation)
                observation = next_observation
            rollout.reward_breakdown = environment.evaluate(task_id)
        finally:
            environment.close()
        rollouts.append(rollout)
    return RolloutGroup(group_id, task_id, seed, rollouts)
