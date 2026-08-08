from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from torch import Tensor


class ActionType(IntEnum):
    NOOP = 0
    CLICK = 1
    DOUBLE_CLICK = 2
    RIGHT_CLICK = 3
    DRAG = 4
    SCROLL = 5
    TYPE = 6
    HOTKEY = 7
    WAIT = 8
    CANCEL = 9


class SpeechMode(IntEnum):
    SILENCE = 0
    SPEECH = 1


@dataclass(slots=True)
class StreamUnit:
    """One batched, time-aligned multimodal training unit."""

    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    screen_valid: Tensor
    screen_revision: Tensor
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
    action_tokens: Tensor
    action_token_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.mic_audio.shape[0]

    def to(self, device: torch.device | str) -> StreamUnit:
        return StreamUnit(
            timestamp_ms=self.timestamp_ms.to(device),
            delta_ms=self.delta_ms.to(device),
            mic_audio=self.mic_audio.to(device),
            screen=self.screen.to(device),
            screen_valid=self.screen_valid.to(device),
            screen_revision=self.screen_revision.to(device),
            speech_mode=self.speech_mode.to(device),
            speech_mode_mask=self.speech_mode_mask.to(device),
            speech_codes=self.speech_codes.to(device),
            speech_codec_mask=self.speech_codec_mask.to(device),
            action_tokens=self.action_tokens.to(device),
            action_token_mask=self.action_token_mask.to(device),
        )

    def validate(
        self,
        *,
        audio_samples: int,
        speech_frames: int,
        speech_codebooks: int,
        speech_codebook_size: int = 2**16,
        action_vocab_size: int | None = None,
    ) -> None:
        batch = self.batch_size
        if self.mic_audio.shape != (batch, audio_samples):
            raise ValueError(f"mic_audio must have shape [B, {audio_samples}]")
        if self.screen.ndim != 4 or self.screen.shape[:2] != (batch, 3):
            raise ValueError("screen must have shape [B, 3, H, W]")
        if self.speech_mode.shape != (batch,) or self.speech_mode_mask.shape != (batch,):
            raise ValueError("speech mode tensors must have shape [B]")
        if torch.any((self.speech_mode < 0) | (self.speech_mode > 1)):
            raise ValueError("speech mode must be SILENCE or SPEECH")
        if self.speech_codes.shape != (batch, speech_frames, speech_codebooks):
            raise ValueError("speech_codes has an incompatible shape")
        if self.speech_codec_mask.shape != (batch, speech_frames):
            raise ValueError("speech_codec_mask has an incompatible shape")
        valid_codes = self.speech_codes[self.speech_codec_mask]
        if valid_codes.numel() and (
            valid_codes.min() < 0 or valid_codes.max() >= speech_codebook_size
        ):
            raise ValueError("speech code is outside the configured codebook")
        if self.action_tokens.ndim != 2 or self.action_tokens.shape[0] != batch:
            raise ValueError("action_tokens must have shape [B, max_tokens]")
        if self.action_token_mask.shape != self.action_tokens.shape:
            raise ValueError("action_token_mask must match action_tokens")
        if action_vocab_size is not None:
            valid_actions = self.action_tokens[self.action_token_mask]
            if valid_actions.numel() and (
                valid_actions.min() < 0 or valid_actions.max() >= action_vocab_size
            ):
                raise ValueError("action token is outside the configured vocabulary")
        if torch.any(self.delta_ms <= 0):
            raise ValueError("delta_ms must be positive")


@dataclass(slots=True)
class Episode:
    episode_id: str
    units: list[StreamUnit]
    metadata: dict[str, Any]
    target_speech: Tensor | None = None
    ordered_shard_index: int = 0
    sample_index_in_shard: int = 0

    def validate(self, **unit_shape: int) -> None:
        if not self.units:
            raise ValueError("episode must contain at least one stream unit")
        previous = -1
        for unit in self.units:
            unit.validate(**unit_shape)
            timestamp = int(unit.timestamp_ms[0].item())
            if timestamp <= previous:
                raise ValueError("episode timestamps must be strictly increasing")
            previous = timestamp
        if self.target_speech is not None:
            expected = len(self.units) * unit_shape["audio_samples"]
            if self.target_speech.numel() != expected:
                raise ValueError("target_speech must exactly cover the episode timeline")


@dataclass(slots=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(self.key.detach(), self.value.detach())


@dataclass(slots=True)
class SpeechLocalState:
    temporal: Tensor
    previous_codes: Tensor

    def detach(self) -> SpeechLocalState:
        return SpeechLocalState(
            temporal=self.temporal.detach(),
            previous_codes=self.previous_codes.detach(),
        )


@dataclass(slots=True)
class ActionLocalState:
    hidden: Tensor
    previous_token: Tensor
    active: Tensor
    event_type: Tensor
    burst_tokens: Tensor

    def detach(self) -> ActionLocalState:
        return ActionLocalState(
            hidden=self.hidden.detach(),
            previous_token=self.previous_token.detach(),
            active=self.active.detach(),
            event_type=self.event_type.detach(),
            burst_tokens=self.burst_tokens.detach(),
        )


@dataclass(slots=True)
class RecurrentState:
    layer_kv: tuple[LayerKV, ...]
    latent: Tensor
    audio_cache: Tensor
    hidden: Tensor
    speech_local: SpeechLocalState
    action_local: ActionLocalState
    unit_index: Tensor

    def detach(self) -> RecurrentState:
        return RecurrentState(
            layer_kv=tuple(cache.detach() for cache in self.layer_kv),
            latent=self.latent.detach(),
            audio_cache=self.audio_cache.detach(),
            hidden=self.hidden.detach(),
            speech_local=self.speech_local.detach(),
            action_local=self.action_local.detach(),
            unit_index=self.unit_index.detach(),
        )


@dataclass(slots=True)
class StepOutput:
    state: RecurrentState
    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    action_logits: Tensor
    action_token_mask: Tensor
    hidden: Tensor


@dataclass(frozen=True, slots=True)
class SpeechSamplingConfig:
    temperature: float = 0.8
    top_k: int = 250
    greedy: bool = False


@dataclass(slots=True)
class GenerationOutput:
    output: StepOutput
    speech_mode: Tensor
    speech_codes: Tensor
    action_tokens: Tensor
