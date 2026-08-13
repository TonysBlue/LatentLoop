from __future__ import annotations

from collections.abc import Iterator

import torch
from contracts import ACTION_SCHEMA_ID, ActionKind
from model.types import ActionFrame, Episode, SpeechMode, StreamUnit
from runtime.codec import codec_frame_mask
from runtime.config import DataConfig, ModelConfig


class SyntheticEpisodeDataset:
    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return self.data.train_episodes

    def __iter__(self) -> Iterator[Episode]:
        for index in range(self.data.train_episodes):
            yield self.make_episode(index)

    def make_episode(self, episode_index: int) -> Episode:
        generator = torch.Generator().manual_seed(self.data.seed + episode_index)
        units: list[StreamUnit] = []
        scenario = episode_index % 64
        for unit_index in range(self.data.episode_units):
            units.append(self._unit(generator, scenario, unit_index))
        episode = Episode(
            episode_id=f"synthetic-{episode_index:08d}",
            units=units,
            metadata={
                "source": "synthetic",
                "source_license": "project-generated",
                "redistribution_allowed": True,
                "split": "train",
                "scenario": "state-memory-click",
                "device_id_hash": "synthetic-device",
                "session_id_hash": f"synthetic-session-{episode_index:08d}",
                "seed": self.data.seed + episode_index,
                "sample_rate": self.data.audio_sample_rate,
                "unit_ms": self.data.unit_ms,
                "codec_frame_rate": self.data.codec_frame_rate,
                "codec_id": self.data.codec_id,
                "codec_weight_hash": self.data.codec_weight_hash,
                "codec_revision": self.data.codec_revision,
                "stage": "pretrain",
                "dataset_scale": self.data.dataset,
                "sample_kind": "supervised_episode",
                "supervision_kind": "speech_action",
                "action_source": "synthetic_expert",
                "task_id": f"synthetic-task-{scenario}",
                "environment_id": "synthetic-test-only",
                "environment_version": "1",
                "protocol_version": "realtime-v2",
                "action_schema_id": ACTION_SCHEMA_ID,
            },
            target_speech=torch.zeros(self.data.episode_units * self.data.unit_audio_samples),
            sample_index_in_shard=episode_index,
        )
        episode.validate(
            audio_samples=self.data.unit_audio_samples,
            speech_frames=self.model.speech_frames_per_unit,
            speech_codebooks=self.model.speech_codebooks,
            speech_codebook_size=self.model.speech_codebook_size,
        )
        return episode

    def _unit(
        self,
        generator: torch.Generator,
        scenario: int,
        unit_index: int,
    ) -> StreamUnit:
        audio = (
            torch.randn(
                self.data.unit_audio_samples,
                generator=generator,
            )
            * 0.03
        )
        audio += scenario / 63 * 0.1
        screen = torch.zeros(3, self.data.screen_height, self.data.screen_width)
        row = scenario % self.data.screen_height
        column = (scenario * 3 + unit_index) % self.data.screen_width
        screen[:, row, :] = 0.5
        screen[:, :, column] = 1.0
        codes = torch.empty(
            self.model.speech_frames_per_unit,
            self.model.speech_codebooks,
            dtype=torch.long,
        )
        for codebook in range(self.model.speech_codebooks):
            codes[:, codebook] = (
                scenario * 7
                + unit_index * 3
                + torch.arange(self.model.speech_frames_per_unit)
                + codebook
            ) % self.model.speech_codebook_size
        action = ActionFrame.no_action(1)
        if unit_index % 4 == 2:
            x = column / max(self.data.screen_width - 1, 1)
            y = row / max(self.data.screen_height - 1, 1)
            cell_x = min(int(x * 32), 31)
            cell_y = min(int(y * 32), 31)
            action.kind[0] = int(ActionKind.POINTER_MOVE)
            action.coordinate_cell[0] = cell_y * 32 + cell_x
            action.coordinate_residual[0] = torch.tensor((x * 32 - cell_x, y * 32 - cell_y))
        return StreamUnit(
            timestamp_ms=torch.tensor([unit_index * self.data.unit_ms]),
            delta_ms=torch.tensor([self.data.unit_ms]),
            mic_audio=audio[None],
            screen=screen[None],
            speech_mode=torch.tensor([int(SpeechMode.SPEECH)]),
            speech_mode_mask=torch.ones(1, dtype=torch.bool),
            speech_codes=codes[None],
            speech_codec_mask=codec_frame_mask(
                unit_index * self.data.unit_ms,
                self.data.unit_ms,
                self.data.codec_frame_rate,
                self.model.speech_frames_per_unit,
            )[None],
            action=action,
            action_supervision_mask=torch.ones(1, dtype=torch.bool),
        )
