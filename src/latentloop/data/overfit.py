from __future__ import annotations

from collections.abc import Iterator

import torch

from latentloop.config import DataConfig, ModelConfig
from latentloop.types import (
    ActionControl,
    ActionTarget,
    ActionType,
    CognitiveControl,
    ControlTarget,
    Episode,
    SpeechControl,
    StreamUnit,
)


class SpeechOverfitDataset:
    """Small deterministic speech-response trajectories for the E2 overfit gate."""

    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        if data.episode_units < 12:
            raise ValueError("speech overfit trajectories require at least 12 units")
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return self.data.train_episodes

    def __iter__(self) -> Iterator[Episode]:
        for episode_index in range(self.data.train_episodes):
            yield self.make_episode(episode_index)

    def make_episode(self, episode_index: int) -> Episode:
        response_class = episode_index % 8
        variant = episode_index // 8
        mic = self._microphone_prompt(response_class, variant)
        target, controls = self._target_response(response_class)
        units = [
            self._unit(index, mic, controls, response_class)
            for index in range(self.data.episode_units)
        ]
        episode = Episode(
            episode_id=f"e2-overfit-{episode_index:04d}",
            units=units,
            metadata={
                "schema_version": self.data.schema_version,
                "source": "synthetic",
                "source_license": "project-generated",
                "redistribution_allowed": True,
                "language": "zxx-procedural",
                "split": "train",
                "scenario": "e2-codec-overfit-gate",
                "device_id_hash": "e2-overfit-device",
                "session_id_hash": f"e2-overfit-session-{episode_index:04d}",
                "sample_rate": self.data.audio_sample_rate,
                "unit_ms": self.data.unit_ms,
                "codec_frame_rate": self.data.codec_frame_rate,
                "codec_id": self.data.codec_id,
                "codec_weight_hash": self.data.codec_weight_hash,
                "codec_revision": self.data.codec_revision,
                "speech_codes_encoded": False,
                "response_class": response_class,
                "variant": variant,
                "turns": [
                    {"speaker": "user", "start_tick": 0, "end_tick": 3},
                    {"speaker": "assistant", "start_tick": 5, "end_tick": 13},
                ],
            },
            target_speech=target,
            sample_index_in_shard=episode_index,
        )
        episode.validate(
            audio_samples=self.data.unit_audio_samples,
            speech_frames=self.model.speech_frames_per_unit,
            speech_codebooks=self.model.speech_codebooks,
            speech_codebook_size=self.model.speech_codebook_size,
        )
        return episode

    def _microphone_prompt(self, response_class: int, variant: int) -> torch.Tensor:
        total = self.data.episode_units * self.data.unit_audio_samples
        waveform = torch.zeros(total)
        prompt_samples = 3 * self.data.unit_audio_samples
        time = torch.arange(prompt_samples) / self.data.audio_sample_rate
        fundamental = 150.0 + 25.0 * response_class
        secondary = 410.0 + 15.0 * variant
        prompt = 0.12 * torch.sin(2 * torch.pi * fundamental * time)
        prompt += 0.04 * torch.sin(2 * torch.pi * secondary * time)
        prompt *= self._edge_envelope(prompt_samples)
        waveform[:prompt_samples] = prompt
        return waveform

    def _target_response(
        self, response_class: int
    ) -> tuple[torch.Tensor, list[SpeechControl]]:
        total = self.data.episode_units * self.data.unit_audio_samples
        waveform = torch.zeros(total)
        start_tick = 5
        active_ticks = 8
        active_samples = active_ticks * self.data.unit_audio_samples
        time = torch.arange(active_samples) / self.data.audio_sample_rate
        fundamental = 180.0 + 30.0 * response_class
        response = 0.16 * torch.sin(2 * torch.pi * fundamental * time)
        response += 0.05 * torch.sin(2 * torch.pi * fundamental * 2.0 * time + 0.2)
        response += 0.02 * torch.sin(2 * torch.pi * fundamental * 3.0 * time + 0.4)
        response *= self._edge_envelope(active_samples)
        offset = start_tick * self.data.unit_audio_samples
        waveform[offset : offset + active_samples] = response

        controls = [SpeechControl.SILENT] * self.data.episode_units
        controls[start_tick] = SpeechControl.START
        for tick in range(start_tick + 1, start_tick + active_ticks):
            controls[tick] = SpeechControl.CONTINUE
        controls[start_tick + active_ticks] = SpeechControl.STOP
        return waveform, controls

    def _edge_envelope(self, samples: int) -> torch.Tensor:
        ramp_samples = min(self.data.audio_sample_rate // 100, samples // 2)
        envelope = torch.ones(samples)
        ramp = torch.linspace(0.0, 1.0, ramp_samples)
        envelope[:ramp_samples] = ramp
        envelope[-ramp_samples:] = ramp.flip(0)
        return envelope

    def _unit(
        self,
        index: int,
        microphone: torch.Tensor,
        controls: list[SpeechControl],
        response_class: int,
    ) -> StreamUnit:
        frame = self.data.unit_audio_samples
        control = controls[index]
        speech_valid = control is not SpeechControl.SILENT
        return StreamUnit(
            timestamp_ms=torch.tensor([index * self.data.unit_ms]),
            delta_ms=torch.tensor([self.data.unit_ms]),
            mic_audio=microphone[None, index * frame : (index + 1) * frame],
            screen=torch.zeros(
                1, 3, self.data.screen_height, self.data.screen_width
            ),
            screen_valid=torch.tensor([False]),
            screen_revision=torch.tensor([-1]),
            speech_codes=torch.zeros(
                1,
                self.model.speech_frames_per_unit,
                self.model.speech_codebooks,
                dtype=torch.long,
            ),
            speech_mask=torch.tensor([[speech_valid]]),
            action_mask=torch.tensor([False]),
            speech_control_mask=torch.tensor([True]),
            action_control_mask=torch.tensor([False]),
            cognitive_control_mask=torch.tensor([False]),
            memory_mask=torch.tensor([False]),
            action_target=self._empty_action(),
            control_target=ControlTarget(
                speech=torch.tensor([int(control)]),
                action=torch.tensor([int(ActionControl.NOOP)]),
                cognitive=torch.tensor([int(CognitiveControl.OBSERVE)]),
            ),
            memory_target=torch.tensor([response_class]),
        )

    def _empty_action(self) -> ActionTarget:
        return ActionTarget(
            type=torch.tensor([int(ActionType.NOOP)]),
            coordinates=torch.zeros(1, 4),
            coordinate_mask=torch.zeros(1, 4, dtype=torch.bool),
            scroll_delta=torch.zeros(1, 2),
            scroll_mask=torch.tensor([False]),
            duration_ms=torch.zeros(1),
            duration_mask=torch.tensor([False]),
            text_tokens=torch.zeros(
                1, self.model.action_text_tokens, dtype=torch.long
            ),
            text_mask=torch.zeros(
                1, self.model.action_text_tokens, dtype=torch.bool
            ),
            key_mask=torch.zeros(
                1, self.model.action_key_vocab_size, dtype=torch.bool
            ),
        )
