from __future__ import annotations

import math

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


class DeltaTimeEncoder(nn.Module):
    def __init__(self, dim: int, bands: int = 8, base_period_ms: int = 80) -> None:
        super().__init__()
        if bands < 1 or base_period_ms < 1:
            raise ValueError("delta time Fourier bands and base period must be positive")
        self.bands = bands
        self.base_period_ms = base_period_ms
        periods = base_period_ms * (2.0 ** torch.arange(bands))
        self.register_buffer("periods_ms", periods, persistent=False)
        self.network = nn.Sequential(
            nn.Linear(1 + 2 * bands, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, delta_ms: Tensor) -> Tensor:
        if delta_ms.ndim != 1:
            raise ValueError("delta_ms must have shape [B]")
        if torch.any(delta_ms <= 0):
            raise ValueError("delta_ms must be positive")
        dtype = self.network[0].weight.dtype
        delta = delta_ms.to(dtype=dtype)
        log_delta = torch.log1p(delta / float(self.base_period_ms))[:, None]
        phase = 2.0 * math.pi * delta[:, None] / self.periods_ms.to(dtype=dtype)[None]
        features = torch.cat((log_delta, phase.sin(), phase.cos()), dim=-1)
        return self.network(features)[:, None]
