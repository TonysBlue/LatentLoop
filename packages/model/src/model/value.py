from __future__ import annotations

import torch
from torch import Tensor, nn


class ValueHead(nn.Module):
    """Training-only value estimate from the causal recurrent representation."""

    def __init__(self, model_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, hidden: Tensor, latent: Tensor) -> Tensor:
        state_query = hidden[:, -1]
        pooled_latent = self.latent_projection(latent.mean(dim=1))
        return self.network(torch.cat((state_query, pooled_latent), dim=-1)).squeeze(-1)


__all__ = ["ValueHead"]
