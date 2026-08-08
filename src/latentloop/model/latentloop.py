from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from latentloop.config import ModelConfig
from latentloop.model.action import ActionHead
from latentloop.model.attention import StreamingTransformerLayer
from latentloop.model.encoders import StreamingAudioEncoder, TimeEncoder, VisionEncoder
from latentloop.model.speech import FactorizedSpeechHead
from latentloop.types import (
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SpeechLocalState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
)


class LatentUpdater(nn.Module):
    def __init__(self, model_dim: int, latent_dim: int, heads: int, slots: int) -> None:
        super().__init__()
        self.latent_to_model = nn.Linear(latent_dim, model_dim)
        self.slot_identity = nn.Parameter(torch.zeros(slots, model_dim))
        self.slot_identity_latent = nn.Parameter(torch.zeros(slots, latent_dim))
        self.read = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.candidate = nn.Sequential(
            nn.Linear(latent_dim + model_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.gate = nn.Linear(latent_dim + model_dim, 1)
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.normal_(self.slot_identity, std=0.02)
        nn.init.normal_(self.slot_identity_latent, std=0.02)

    def forward(self, latent: Tensor, previous_hidden: Tensor) -> Tensor:
        query = self.latent_to_model(latent) + self.slot_identity[None]
        context, _ = self.read(query, previous_hidden, previous_hidden, need_weights=False)
        combined = torch.cat((latent, context), dim=-1)
        # Slot identity must affect the first write even when Z_0 and H_0 are
        # both zero; otherwise every slot remains exactly symmetric.
        candidate = self.candidate(combined) + self.slot_identity_latent[None]
        gate = torch.sigmoid(self.gate(combined))
        return self.norm((1 - gate) * latent + gate * candidate)


class StreamingLatentLoop(nn.Module):
    """Final target: bounded KV, recurrent Z/H state, independent speech/action heads."""

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
        self.latent_updater = LatentUpdater(
            dim, config.latent_dim, config.num_heads, config.latent_slots
        )
        self.speech_head = FactorizedSpeechHead(config)
        self.action_head = ActionHead(config)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_query, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        dtype = next(self.parameters()).dtype
        device = torch.device(device)
        head_dim = self.config.model_dim // self.config.num_heads
        empty_kv = tuple(
            LayerKV(
                key=torch.empty(
                    batch_size, self.config.num_heads, 0, head_dim, device=device, dtype=dtype
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
            hidden=torch.zeros(
                batch_size,
                self.config.tokens_per_unit,
                self.config.model_dim,
                device=device,
                dtype=dtype,
            ),
            speech_local=SpeechLocalState(
                temporal=torch.zeros(batch_size, self.config.model_dim, device=device, dtype=dtype),
                previous_codes=torch.zeros(
                    batch_size, self.config.speech_codebooks, device=device, dtype=torch.long
                ),
            ),
            action_local=self.action_head.initial_state(batch_size, device, dtype),
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

    def forward_step(
        self,
        unit: StreamUnit,
        state: RecurrentState,
        *,
        speech_teacher_codes: Tensor | None = None,
        speech_teacher_mode: Tensor | None = None,
        action_teacher_tokens: Tensor | None = None,
        sampling: SpeechSamplingConfig | None = None,
    ) -> StepOutput:
        audio, audio_cache = self.audio_encoder(unit.mic_audio, state.audio_cache)
        vision = self.vision_encoder(unit.screen, unit.screen_valid)
        encoded = self._pack_unit(unit, audio, vision)
        updated_latent = self.latent_updater(state.latent, state.hidden)
        latent_for_read = self.latent_reader(updated_latent)
        new_caches: list[LayerKV] = []
        max_tokens = self.config.kv_units * self.config.tokens_per_unit
        hidden = encoded
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
        speech_temporal = self.speech_head.update_temporal(hidden, state.speech_local)
        speech_mode_logits = self.speech_head.mode_logits(hidden)
        if speech_teacher_mode is not None:
            mode = speech_teacher_mode
        else:
            mode = speech_mode_logits.argmax(dim=-1)
        if speech_teacher_codes is not None:
            speech_codec_logits = self.speech_head.teacher_logits(
                speech_temporal, speech_teacher_codes
            )
            next_codes = speech_teacher_codes[:, 0]
        else:
            speech_codec_logits, generated_codes = self.speech_head.generate(
                speech_temporal, sampling or SpeechSamplingConfig(greedy=True)
            )
            next_codes = generated_codes[:, 0]
        next_codes = torch.where(
            mode[:, None] == int(SpeechMode.SPEECH), next_codes, torch.zeros_like(next_codes)
        )
        action_logits, action_tokens, action_local, action_token_mask = self.action_head(
            hidden,
            state.action_local,
            action_teacher_tokens,
            sampling_temperature=(sampling.temperature if sampling is not None else None),
        )
        next_state = RecurrentState(
            layer_kv=tuple(new_caches),
            latent=updated_latent,
            audio_cache=audio_cache,
            hidden=hidden,
            speech_local=SpeechLocalState(temporal=speech_temporal, previous_codes=next_codes),
            action_local=action_local,
            unit_index=state.unit_index + 1,
        )
        return StepOutput(
            state=next_state,
            speech_mode_logits=speech_mode_logits,
            speech_codec_logits=speech_codec_logits,
            action_logits=action_logits,
            action_token_mask=action_token_mask,
            hidden=hidden,
        )

    def forward(
        self,
        unit: StreamUnit,
        state: RecurrentState,
        speech_teacher_codes: Tensor | None = None,
        *,
        speech_teacher_mode: Tensor | None = None,
        action_teacher_tokens: Tensor | None = None,
    ) -> StepOutput:
        return self.forward_step(
            unit,
            state,
            speech_teacher_codes=speech_teacher_codes,
            speech_teacher_mode=speech_teacher_mode,
            action_teacher_tokens=action_teacher_tokens,
        )

    @torch.no_grad()
    def generate_step(
        self, unit: StreamUnit, state: RecurrentState, sampling: SpeechSamplingConfig | None = None
    ) -> GenerationOutput:
        output = self.forward_step(unit, state, sampling=sampling)
        return GenerationOutput(
            output=output,
            speech_mode=output.speech_mode_logits.argmax(dim=-1),
            speech_codes=output.state.speech_local.previous_codes[:, None],
            action_tokens=output.action_logits.argmax(dim=-1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
