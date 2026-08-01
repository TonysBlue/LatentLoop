from __future__ import annotations

import torch
from torch import Tensor, nn

from latentloop.config import ModelConfig
from latentloop.types import SpeechLocalState, SpeechSamplingConfig


class FactorizedSpeechHead(nn.Module):
    """Causal temporal and within-frame model for residual codec codebooks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.temporal = nn.GRUCell(dim * 2 + config.latent_dim, dim)
        self.depth_embeddings = nn.ModuleList(
            nn.Embedding(config.speech_codebook_size, dim)
            for _ in range(config.speech_codebooks)
        )
        self.bos = nn.Parameter(torch.zeros(dim))
        self.positions = nn.Parameter(torch.zeros(config.speech_codebooks, dim))
        depth_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.speech_depth_heads,
            dim_feedforward=config.speech_depth_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth = nn.TransformerEncoder(
            depth_layer,
            num_layers=config.speech_depth_layers,
            enable_nested_tensor=False,
        )
        self.depth_norm = nn.LayerNorm(dim)
        self.output_bias = nn.Parameter(
            torch.zeros(config.speech_codebooks, config.speech_codebook_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.positions, std=0.02)

    def update_temporal(
        self, query: Tensor, pooled_latent: Tensor, state: SpeechLocalState
    ) -> Tensor:
        active = state.utterance_active[:, None]
        previous_codes = torch.where(
            active,
            state.previous_codes,
            torch.zeros_like(state.previous_codes),
        )
        previous = torch.stack(
            [
                embedding(previous_codes[:, index])
                for index, embedding in enumerate(self.depth_embeddings)
            ],
            dim=1,
        ).mean(dim=1)
        previous = torch.where(active, previous, torch.zeros_like(previous))
        return self.temporal(
            torch.cat((query, pooled_latent, previous), dim=-1),
            torch.where(active, state.temporal, torch.zeros_like(state.temporal)),
        )

    def _inputs(self, temporal: Tensor, previous_in_frame: list[Tensor]) -> Tensor:
        batch = temporal.shape[0]
        values = [self.bos[None].expand(batch, -1)]
        values.extend(
            embedding(codes)
            for embedding, codes in zip(
                self.depth_embeddings, previous_in_frame, strict=False
            )
        )
        inputs = torch.stack(values, dim=1)
        positions = self.positions[: inputs.shape[1]][None]
        return inputs + temporal[:, None] + positions

    def _depth_hidden(self, temporal: Tensor, previous_in_frame: list[Tensor]) -> Tensor:
        inputs = self._inputs(temporal, previous_in_frame)
        length = inputs.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=inputs.device, dtype=torch.bool), diagonal=1
        )
        return self.depth_norm(self.depth(inputs, mask=causal_mask))

    def _logits(self, hidden: Tensor, codebook: int) -> Tensor:
        return torch.nn.functional.linear(
            hidden,
            self.depth_embeddings[codebook].weight,
            self.output_bias[codebook],
        )

    def teacher_logits(self, temporal: Tensor, codes: Tensor) -> Tensor:
        if codes.shape[1:] != (1, self.config.speech_codebooks):
            raise ValueError("teacher speech codes must have shape [B, 1, codebooks]")
        previous = [codes[:, 0, index] for index in range(self.config.speech_codebooks - 1)]
        hidden = self._depth_hidden(temporal, previous)
        logits = [self._logits(hidden[:, index], index) for index in range(len(hidden[0]))]
        return torch.stack(logits, dim=1)[:, None]

    def generate(
        self, temporal: Tensor, sampling: SpeechSamplingConfig
    ) -> tuple[Tensor, Tensor]:
        selected: list[Tensor] = []
        logits: list[Tensor] = []
        for codebook in range(self.config.speech_codebooks):
            hidden = self._depth_hidden(temporal, selected)[:, -1]
            current_logits = self._logits(hidden, codebook)
            logits.append(current_logits)
            selected.append(self._sample(current_logits, sampling))
        return torch.stack(logits, dim=1)[:, None], torch.stack(selected, dim=1)[:, None]

    @staticmethod
    def _sample(logits: Tensor, sampling: SpeechSamplingConfig) -> Tensor:
        if sampling.greedy or sampling.temperature <= 0:
            return logits.argmax(dim=-1)
        scaled = logits / sampling.temperature
        if 0 < sampling.top_k < scaled.shape[-1]:
            top_values, top_indices = torch.topk(scaled, sampling.top_k, dim=-1)
            sampled = torch.multinomial(torch.softmax(top_values, dim=-1), 1).squeeze(-1)
            return top_indices.gather(-1, sampled[:, None]).squeeze(-1)
        return torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(-1)
