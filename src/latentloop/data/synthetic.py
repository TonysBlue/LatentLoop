from __future__ import annotations

from collections.abc import Iterator

import torch

from latentloop.codec import codec_frame_bounds, codec_frame_mask
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


class SyntheticEpisodeDataset:
    """Deterministic trajectories for state, cache, and checkpoint validation."""

    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return self.data.train_episodes

    def __iter__(self) -> Iterator[Episode]:
        for episode_index in range(self.data.train_episodes):
            yield self.make_episode(episode_index)

    def make_episode(self, episode_index: int) -> Episode:
        generator = torch.Generator().manual_seed(self.data.seed + episode_index)
        memory_target = episode_index % self.model.memory_classes
        units: list[StreamUnit] = []
        for unit_index in range(self.data.episode_units):
            phase = (memory_target + unit_index) % 10
            audio = torch.randn(self.data.unit_audio_samples, generator=generator) * 0.03
            audio += memory_target / max(self.model.memory_classes - 1, 1) * 0.1
            screen = torch.zeros(3, self.data.screen_height, self.data.screen_width)
            row = memory_target % self.data.screen_height
            column = (memory_target * 3 + unit_index) % self.data.screen_width
            screen[:, row, :] = 0.5
            screen[:, :, column] = 1.0
            codes = torch.empty(
                self.model.speech_frames_per_unit,
                self.model.speech_codebooks,
                dtype=torch.long,
            )
            for codebook in range(self.model.speech_codebooks):
                frame_ids = torch.arange(self.model.speech_frames_per_unit)
                codes[:, codebook] = (
                    memory_target * 7 + unit_index * 3 + frame_ids + codebook
                ) % self.model.speech_codebook_size

            action_type = ActionType.CLICK if unit_index % 4 == 2 else ActionType.NOOP
            coord_valid = action_type != ActionType.NOOP
            coordinate_mask = torch.tensor([coord_valid, coord_valid, False, False])
            coordinates = torch.tensor(
                [
                    column / max(self.data.screen_width - 1, 1),
                    row / max(self.data.screen_height - 1, 1),
                    column / max(self.data.screen_width - 1, 1),
                    row / max(self.data.screen_height - 1, 1),
                ],
                dtype=torch.float32,
            )
            units.append(
                StreamUnit(
                    timestamp_ms=torch.tensor([unit_index * self.data.unit_ms], dtype=torch.long),
                    delta_ms=torch.tensor([self.data.unit_ms], dtype=torch.long),
                    mic_audio=audio[None],
                    screen=screen[None],
                    screen_valid=torch.tensor([unit_index % 3 != 1]),
                    screen_revision=torch.tensor([unit_index], dtype=torch.long),
                    speech_codes=codes[None],
                    speech_mask=codec_frame_mask(
                        unit_index * self.data.unit_ms,
                        self.data.unit_ms,
                        self.data.codec_frame_rate,
                        self.model.speech_frames_per_unit,
                    )[None],
                    action_mask=torch.tensor([True]),
                    speech_control_mask=torch.tensor([True]),
                    action_control_mask=torch.tensor([True]),
                    cognitive_control_mask=torch.tensor([True]),
                    memory_mask=torch.tensor([True]),
                    action_target=ActionTarget(
                        type=torch.tensor([int(action_type)], dtype=torch.long),
                        coordinates=coordinates[None],
                        coordinate_mask=coordinate_mask[None],
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
                    ),
                    control_target=ControlTarget(
                        speech=torch.tensor(
                            [
                                int(
                                    SpeechControl.CONTINUE
                                    if phase < 7
                                    else SpeechControl.SILENT
                                )
                            ]
                        ),
                        action=torch.tensor(
                            [int(ActionControl.EXECUTE if coord_valid else ActionControl.NOOP)]
                        ),
                        cognitive=torch.tensor([int(CognitiveControl.UPDATE)]),
                    ),
                    memory_target=torch.tensor([memory_target], dtype=torch.long),
                )
            )
        episode = Episode(
            episode_id=f"synthetic-{episode_index:08d}",
            units=units,
            metadata={
                "schema_version": self.data.schema_version,
                "source": "synthetic",
                "source_license": "project-generated",
                "redistribution_allowed": True,
                "split": "train",
                "scenario": "state-memory-click",
                "device_id_hash": "synthetic-device",
                "session_id_hash": f"synthetic-session-{episode_index:08d}",
                "seed": self.data.seed + episode_index,
                "memory_target": memory_target,
                "sample_rate": self.data.audio_sample_rate,
                "unit_ms": self.data.unit_ms,
                "codec_frame_rate": self.data.codec_frame_rate,
                "codec_id": self.data.codec_id,
                "codec_weight_hash": self.data.codec_weight_hash,
                "codec_revision": self.data.codec_revision,
                "codec_frame_bounds": [
                    codec_frame_bounds(
                        unit_index * self.data.unit_ms,
                        self.data.unit_ms,
                        self.data.codec_frame_rate,
                    )
                    for unit_index in range(self.data.episode_units)
                ],
            },
            target_speech=torch.zeros(
                self.data.episode_units * self.data.unit_audio_samples,
                dtype=torch.float32,
            ),
            sample_index_in_shard=episode_index,
        )
        episode.validate(
            audio_samples=self.data.unit_audio_samples,
            speech_frames=self.model.speech_frames_per_unit,
            speech_codebooks=self.model.speech_codebooks,
            speech_codebook_size=self.model.speech_codebook_size,
        )
        return episode
