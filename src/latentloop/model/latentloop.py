from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from latentloop.config import ModelConfig
from latentloop.model.attention import StreamingTransformerLayer
from latentloop.model.encoders import StreamingAudioEncoder, TimeEncoder, VisionEncoder
from latentloop.types import (
    ActionOutput,
    ControlOutput,
    LayerKV,
    RecurrentState,
    StepOutput,
    StreamUnit,
)


class LatentUpdater(nn.Module):
    def __init__(self, model_dim: int, latent_dim: int, heads: int) -> None:
        super().__init__()
        self.latent_to_model = nn.Linear(latent_dim, model_dim)
        self.read = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.candidate = nn.Sequential(
            nn.Linear(model_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.gate = nn.Linear(model_dim * 2, 1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent: Tensor, query: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        latent_query = self.latent_to_model(latent)
        context, _ = self.read(latent_query, observed, observed, need_weights=False)
        repeated_query = query[:, None].expand(-1, latent.shape[1], -1)
        combined = torch.cat((context, repeated_query), dim=-1)
        candidate = self.candidate(combined)
        gate = torch.sigmoid(self.gate(combined))
        updated = self.norm((1 - gate) * latent + gate * candidate)
        return updated, gate.squeeze(-1)


class StreamingLatentLoop(nn.Module):
    """Small but structurally faithful streaming LatentLoop reference model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.audio_encoder = StreamingAudioEncoder(
            dim, config.audio_tokens, config.audio_kernel, config.audio_stride
        )
        self.vision_encoder = VisionEncoder(dim)
        self.time_encoder = TimeEncoder(dim)
        self.type_embedding = nn.Embedding(4, dim)
        self.state_query = nn.Parameter(torch.zeros(dim))
        self.latent_reader = nn.Linear(config.latent_dim, dim)
        self.layers = nn.ModuleList(
            StreamingTransformerLayer(
                dim=dim,
                heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                cross_latent=(index + 1) % config.cross_attention_every == 0,
            )
            for index in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.latent_updater = LatentUpdater(dim, config.latent_dim, config.num_heads)
        head_input = dim + config.latent_dim
        self.speech_state = nn.GRUCell(head_input, dim)
        self.speech_frame_embedding = nn.Parameter(torch.zeros(config.speech_frames_per_unit, dim))
        self.speech_heads = nn.ModuleList(
            nn.Linear(dim, config.speech_codebook_size) for _ in range(config.speech_codebooks)
        )
        self.action_type_head = nn.Linear(head_input, 10)
        self.action_coord_head = nn.Linear(head_input, 4)
        self.action_scroll_head = nn.Linear(head_input, 2)
        self.action_duration_head = nn.Linear(head_input, 1)
        self.action_text_position = nn.Parameter(torch.zeros(config.action_text_tokens, head_input))
        self.action_text_head = nn.Linear(head_input, config.action_text_vocab_size)
        self.action_key_head = nn.Linear(head_input, config.action_key_vocab_size)
        self.action_confidence_head = nn.Linear(head_input, 1)
        self.speech_control_head = nn.Linear(head_input, 5)
        self.action_control_head = nn.Linear(head_input, 4)
        self.cognitive_control_head = nn.Linear(head_input, 5)
        self.memory_probe = nn.Linear(head_input, config.memory_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_query, std=0.02)
        nn.init.normal_(self.speech_frame_embedding, std=0.02)
        nn.init.normal_(self.action_text_position, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        dtype = next(self.parameters()).dtype
        head_dim = self.config.model_dim // self.config.num_heads
        empty_kv = tuple(
            LayerKV(
                key=torch.empty(
                    batch_size,
                    self.config.num_heads,
                    0,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                value=torch.empty(
                    batch_size, self.config.num_heads, 0, head_dim, device=device, dtype=dtype
                ),
            )
            for _ in self.layers
        )
        return RecurrentState(
            layer_kv=empty_kv,
            latent=torch.zeros(
                batch_size,
                self.config.latent_slots,
                self.config.latent_dim,
                device=device,
                dtype=dtype,
            ),
            audio_cache=torch.zeros(
                batch_size, self.audio_encoder.cache_samples, device=device, dtype=dtype
            ),
            speech_local=torch.zeros(batch_size, self.config.model_dim, device=device, dtype=dtype),
            unit_index=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _pack_unit(self, unit: StreamUnit, audio: Tensor, vision: Tensor) -> Tensor:
        time = self.time_encoder(unit.timestamp_ms, unit.delta_ms)
        query = self.state_query[None, None].expand(unit.batch_size, -1, -1)
        time = time + self.type_embedding.weight[0]
        audio = audio + self.type_embedding.weight[1]
        vision = vision + self.type_embedding.weight[2]
        query = query + self.type_embedding.weight[3]
        return torch.cat((time, audio, vision, query), dim=1)

    def forward_step(self, unit: StreamUnit, state: RecurrentState) -> StepOutput:
        audio, audio_cache = self.audio_encoder(unit.mic_audio, state.audio_cache)
        vision = self.vision_encoder(unit.screen, unit.screen_valid)
        hidden = self._pack_unit(unit, audio, vision)
        latent_for_read = self.latent_reader(state.latent)
        new_caches: list[LayerKV] = []
        max_tokens = self.config.kv_units * self.config.tokens_per_unit
        for layer, cache in zip(self.layers, state.layer_kv, strict=True):
            if self.training and self.config.activation_checkpointing:

                def layer_forward(
                    current: Tensor,
                    latent: Tensor,
                    key: Tensor,
                    value: Tensor,
                    layer: StreamingTransformerLayer = layer,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    output, updated = layer(current, latent, LayerKV(key, value), max_tokens)
                    return output, updated.key, updated.value

                hidden, key, value = checkpoint(
                    layer_forward,
                    hidden,
                    latent_for_read,
                    cache.key,
                    cache.value,
                    use_reentrant=False,
                )
                new_cache = LayerKV(key, value)
            else:
                hidden, new_cache = layer(hidden, latent_for_read, cache, max_tokens)
            new_caches.append(new_cache)
        hidden = self.final_norm(hidden)
        query = hidden[:, -1]
        updated_latent, gate = self.latent_updater(state.latent, query, hidden)
        pooled_latent = updated_latent.mean(dim=1)
        head_input = torch.cat((query, pooled_latent), dim=-1)

        speech_local = self.speech_state(head_input, state.speech_local)
        speech_frames = speech_local[:, None] + self.speech_frame_embedding[None]
        speech_logits = torch.stack(
            [head(speech_frames) for head in self.speech_heads],
            dim=2,
        )
        action = ActionOutput(
            type_logits=self.action_type_head(head_input),
            coordinates=torch.sigmoid(self.action_coord_head(head_input)),
            scroll_delta=torch.tanh(self.action_scroll_head(head_input)),
            duration_ms=torch.sigmoid(self.action_duration_head(head_input).squeeze(-1))
            * self.config.max_action_duration_ms,
            text_logits=self.action_text_head(
                head_input[:, None] + self.action_text_position[None]
            ),
            key_logits=self.action_key_head(head_input),
            confidence=torch.sigmoid(self.action_confidence_head(head_input)).squeeze(-1),
            observed_screen_revision=unit.screen_revision,
        )
        controls = ControlOutput(
            speech_logits=self.speech_control_head(head_input),
            action_logits=self.action_control_head(head_input),
            cognitive_logits=self.cognitive_control_head(head_input),
        )
        next_state = RecurrentState(
            layer_kv=tuple(new_caches),
            latent=updated_latent,
            audio_cache=audio_cache,
            speech_local=speech_local,
            unit_index=state.unit_index + 1,
        )
        return StepOutput(
            state=next_state,
            speech_logits=speech_logits,
            action=action,
            controls=controls,
            memory_logits=self.memory_probe(head_input),
            latent_gate=gate,
            query=query,
        )

    def forward(self, unit: StreamUnit, state: RecurrentState) -> StepOutput:
        return self.forward_step(unit, state)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
