"""Online GRPO primitives shared by the policy runner."""

from latentloop.grpo import (
    clipped_policy_loss,
    compute_group_advantages,
    grpo_loss,
    reference_kl_loss,
)

__all__ = ["clipped_policy_loss", "compute_group_advantages", "grpo_loss", "reference_kl_loss"]
