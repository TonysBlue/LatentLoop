from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StreamingAudioEncoder(nn.Module):
    def __init__(self, dim: int, output_tokens: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.cache_samples = kernel - stride
        self.conv = nn.Conv1d(1, dim, kernel_size=kernel, stride=stride)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat((cache, audio), dim=1)
        encoded = self.conv(combined.unsqueeze(1))
        encoded = F.adaptive_avg_pool1d(encoded, self.output_tokens).transpose(1, 2)
        next_cache = combined[:, -self.cache_samples :] if self.cache_samples else combined[:, :0]
        return self.norm(encoded), next_cache


class VisionEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner = max(dim // 8, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, inner, 3, stride=2, padding=1),
            nn.GroupNorm(4, inner),
            nn.SiLU(),
            nn.Conv2d(inner, inner * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, inner * 2),
            nn.SiLU(),
            nn.Conv2d(inner * 2, dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.position = nn.Parameter(torch.zeros(1, 16, dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, screen: Tensor) -> Tensor:
        features = self.encoder(screen)
        return features.flatten(2).transpose(1, 2) + self.position


class TimeEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestamp_ms: Tensor, delta_ms: Tensor) -> Tensor:
        dtype = self.network[0].weight.dtype
        features = torch.stack((timestamp_ms / 60_000.0, delta_ms / 1_000.0), dim=-1).to(
            dtype=dtype
        )
        return self.network(features)[:, None]
