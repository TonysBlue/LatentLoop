from __future__ import annotations

import torch
from training.grpo import clipped_policy_loss, compute_group_advantages, grpo_loss


def test_group_advantages_are_normalized_and_zero_variance_is_skipped() -> None:
    advantages, active = compute_group_advantages(torch.tensor([1.0, 2.0, 3.0, 2.0]))
    assert active
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    _, inactive = compute_group_advantages(torch.ones(4))
    assert not inactive


def test_clipped_policy_ratio_and_reference_kl_are_differentiable() -> None:
    current = torch.tensor([0.0, 0.5], requires_grad=True)
    old = torch.tensor([0.0, 0.0])
    reference = torch.tensor([0.0, 0.0])
    advantage = torch.tensor([1.0, -1.0])
    loss = grpo_loss(current, old, reference, advantage, clip_epsilon=0.2, kl_beta=0.1)
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert clipped_policy_loss(current.detach(), old, advantage, 0.2).ndim == 0
