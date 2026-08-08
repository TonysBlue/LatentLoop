from __future__ import annotations

import torch
from torch import Tensor


def compute_group_advantages(rewards: Tensor, epsilon: float = 1e-6) -> tuple[Tensor, bool]:
    """Return group-normalized rewards and whether the group is trainable."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO rewards must be a one-dimensional group of at least two")
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std.detach()) <= epsilon:
        return torch.zeros_like(rewards), False
    return (rewards - mean) / (std + epsilon), True


def clipped_policy_loss(
    current_logprob: Tensor,
    old_logprob: Tensor,
    advantage: Tensor,
    clip_epsilon: float,
    mask: Tensor | None = None,
) -> Tensor:
    if current_logprob.shape != old_logprob.shape:
        raise ValueError("current and old log-probability shapes must match")
    ratio = torch.exp((current_logprob - old_logprob).clamp(-30, 30))
    expanded = advantage.to(ratio).reshape(-1, *([1] * (ratio.ndim - 1)))
    unclipped = ratio * expanded
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * expanded
    values = torch.minimum(unclipped, clipped)
    if mask is not None:
        values = values * mask.to(values)
        return -values.sum() / mask.to(values).sum().clamp_min(1)
    return -values.mean()


def reference_kl_loss(
    current_logprob: Tensor, reference_logprob: Tensor, mask: Tensor | None = None
) -> Tensor:
    log_ratio = reference_logprob.detach() - current_logprob
    values = torch.expm1(log_ratio) - log_ratio
    if mask is not None:
        values = values * mask.to(values)
        return values.sum() / mask.to(values).sum().clamp_min(1)
    return values.mean()


def grpo_loss(
    current_logprob: Tensor,
    old_logprob: Tensor,
    reference_logprob: Tensor,
    advantage: Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
    mask: Tensor | None = None,
) -> Tensor:
    policy = clipped_policy_loss(current_logprob, old_logprob, advantage, clip_epsilon, mask)
    return policy + kl_beta * reference_kl_loss(current_logprob, reference_logprob, mask)
