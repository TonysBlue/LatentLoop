from __future__ import annotations

from dataclasses import dataclass, fields
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


class SpeechControl(IntEnum):
    SILENT = 0
    START = 1
    CONTINUE = 2
    PAUSE = 3
    STOP = 4


class ActionControl(IntEnum):
    NOOP = 0
    EXECUTE = 1
    CANCEL = 2
    WAIT_CONFIRMATION = 3


class CognitiveControl(IntEnum):
    OBSERVE = 0
    UPDATE = 1
    SILENT_THINK = 2
    COMPACT = 3
    RESET = 4


@dataclass(slots=True)
class ActionTarget:
    type: Tensor
    coordinates: Tensor
    coordinate_mask: Tensor
    scroll_delta: Tensor
    scroll_mask: Tensor
    duration_ms: Tensor
    duration_mask: Tensor
    text_tokens: Tensor
    text_mask: Tensor
    key_mask: Tensor

    def to(self, device: torch.device | str) -> ActionTarget:
        values = {field.name: getattr(self, field.name).to(device) for field in fields(self)}
        return ActionTarget(**values)


@dataclass(slots=True)
class ControlTarget:
    speech: Tensor
    action: Tensor
    cognitive: Tensor

    def to(self, device: torch.device | str) -> ControlTarget:
        values = {field.name: getattr(self, field.name).to(device) for field in fields(self)}
        return ControlTarget(**values)


@dataclass(slots=True)
class StreamUnit:
    """One batched, time-aligned multimodal training unit."""

    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    screen_valid: Tensor
    screen_revision: Tensor
    speech_codes: Tensor
    speech_mask: Tensor
    action_target: ActionTarget
    control_target: ControlTarget
    memory_target: Tensor

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
            speech_codes=self.speech_codes.to(device),
            speech_mask=self.speech_mask.to(device),
            action_target=self.action_target.to(device),
            control_target=self.control_target.to(device),
            memory_target=self.memory_target.to(device),
        )

    def validate(
        self,
        *,
        audio_samples: int,
        speech_frames: int,
        speech_codebooks: int,
    ) -> None:
        batch = self.batch_size
        if self.mic_audio.shape != (batch, audio_samples):
            raise ValueError(f"mic_audio must have shape [B, {audio_samples}]")
        if self.screen.ndim != 4 or self.screen.shape[:2] != (batch, 3):
            raise ValueError("screen must have shape [B, 3, H, W]")
        if self.speech_codes.shape != (batch, speech_frames, speech_codebooks):
            raise ValueError("speech_codes has an incompatible shape")
        action = self.action_target
        if action.coordinates.shape != (batch, 4):
            raise ValueError("action coordinates must have shape [B, 4]")
        if action.coordinate_mask.shape != (batch, 4):
            raise ValueError("action coordinate_mask must have shape [B, 4]")
        if torch.any(action.coordinates < 0) or torch.any(action.coordinates > 1):
            raise ValueError("action coordinates must be normalized to [0, 1]")
        if action.scroll_delta.shape != (batch, 2):
            raise ValueError("action scroll_delta must have shape [B, 2]")
        if action.text_tokens.shape != action.text_mask.shape:
            raise ValueError("action text tokens and mask must have identical shapes")
        if action.key_mask.shape[0] != batch:
            raise ValueError("action key_mask must start with the batch dimension")
        for index, action_type in enumerate(action.type.tolist()):
            coordinate_count = int(action.coordinate_mask[index].sum().item())
            if (
                action_type
                in {
                    ActionType.CLICK,
                    ActionType.DOUBLE_CLICK,
                    ActionType.RIGHT_CLICK,
                }
                and coordinate_count != 2
            ):
                raise ValueError("point actions require exactly two coordinate values")
            if action_type == ActionType.DRAG and coordinate_count != 4:
                raise ValueError("drag requires four coordinate values")
            if action_type == ActionType.SCROLL and not action.scroll_mask[index]:
                raise ValueError("scroll requires a scroll delta")
            if action_type == ActionType.TYPE and not action.text_mask[index].any():
                raise ValueError("type requires at least one text token")
            if action_type == ActionType.HOTKEY and not action.key_mask[index].any():
                raise ValueError("hotkey requires at least one key")
            if action_type == ActionType.WAIT and not action.duration_mask[index]:
                raise ValueError("wait requires a duration")
        if torch.any(self.delta_ms <= 0):
            raise ValueError("delta_ms must be positive")


@dataclass(slots=True)
class Episode:
    episode_id: str
    units: list[StreamUnit]
    metadata: dict[str, Any]
    ordered_shard_index: int = 0
    sample_index_in_shard: int = 0

    def validate(self, **unit_shape: int) -> None:
        previous = -1
        for unit in self.units:
            unit.validate(**unit_shape)
            timestamp = int(unit.timestamp_ms[0].item())
            if timestamp <= previous:
                raise ValueError("episode timestamps must be strictly increasing")
            previous = timestamp


@dataclass(slots=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(self.key.detach(), self.value.detach())


@dataclass(slots=True)
class RecurrentState:
    layer_kv: tuple[LayerKV, ...]
    latent: Tensor
    audio_cache: Tensor
    speech_local: Tensor
    unit_index: Tensor

    def detach(self) -> RecurrentState:
        return RecurrentState(
            layer_kv=tuple(cache.detach() for cache in self.layer_kv),
            latent=self.latent.detach(),
            audio_cache=self.audio_cache.detach(),
            speech_local=self.speech_local.detach(),
            unit_index=self.unit_index.detach(),
        )


@dataclass(slots=True)
class ActionOutput:
    type_logits: Tensor
    coordinates: Tensor
    scroll_delta: Tensor
    duration_ms: Tensor
    text_logits: Tensor
    key_logits: Tensor
    confidence: Tensor
    observed_screen_revision: Tensor


@dataclass(slots=True)
class ControlOutput:
    speech_logits: Tensor
    action_logits: Tensor
    cognitive_logits: Tensor


@dataclass(slots=True)
class StepOutput:
    state: RecurrentState
    speech_logits: Tensor
    action: ActionOutput
    controls: ControlOutput
    memory_logits: Tensor
    latent_gate: Tensor
    query: Tensor
