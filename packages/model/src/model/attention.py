from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from model.types import LayerKV


class CachedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        cache: LayerKV,
        current_is_visual: Tensor,
        max_temporal_tokens: int,
        max_visual_tokens: int,
    ) -> tuple[Tensor, LayerKV]:
        batch, current_tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, current_tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cached_tokens = cache.key.shape[2]
        all_key = torch.cat((cache.key, key), dim=2)
        all_value = torch.cat((cache.value, value), dim=2)
        all_is_visual = torch.cat((cache.is_visual, current_is_visual), dim=0)
        scores = torch.matmul(query, all_key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        current_mask = torch.triu(
            torch.ones(current_tokens, current_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        mask = torch.zeros(
            current_tokens,
            cached_tokens + current_tokens,
            device=x.device,
            dtype=torch.bool,
        )
        mask[:, cached_tokens:] = current_mask
        scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = torch.dropout(weights, self.dropout, self.training)
        output = (
            torch.matmul(weights, all_value).transpose(1, 2).reshape(batch, current_tokens, dim)
        )

        temporal_indices = torch.nonzero(~all_is_visual, as_tuple=False).flatten()
        visual_indices = torch.nonzero(all_is_visual, as_tuple=False).flatten()
        temporal_indices = temporal_indices[-max_temporal_tokens:]
        visual_indices = visual_indices[-max_visual_tokens:]
        kept_indices = torch.cat((temporal_indices, visual_indices)).sort().values
        new_cache = LayerKV(
            key=all_key.index_select(2, kept_indices),
            value=all_value.index_select(2, kept_indices),
            is_visual=all_is_visual.index_select(0, kept_indices),
        )
        return self.out(output), new_cache


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int, ffn_dim: int, dropout: float, cross_latent: bool
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = CachedSelfAttention(dim, heads, dropout)
        self.cross_norm = nn.LayerNorm(dim) if cross_latent else None
        self.cross_attention = (
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            if cross_latent
            else None
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        latent: Tensor,
        cache: LayerKV,
        current_is_visual: Tensor,
        max_temporal_tokens: int,
        max_visual_tokens: int,
    ) -> tuple[Tensor, LayerKV]:
        attended, new_cache = self.self_attention(
            self.self_norm(hidden),
            cache,
            current_is_visual,
            max_temporal_tokens,
            max_visual_tokens,
        )
        hidden = hidden + self.dropout(attended)
        if self.cross_attention is not None and self.cross_norm is not None:
            normalized = self.cross_norm(hidden)
            crossed, _ = self.cross_attention(normalized, latent, latent, need_weights=False)
            hidden = hidden + self.dropout(crossed)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden, new_cache
