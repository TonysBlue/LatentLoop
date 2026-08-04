from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from latentloop.checkpoint import CheckpointMetadata, file_sha256
from latentloop.config import ProjectConfig
from latentloop.data import SpeechOverfitDataset, write_episode_shards
from latentloop.evaluation import evaluate_overfit_checkpoint
from latentloop.model import StreamingLatentLoop
from latentloop.types import SpeechControl


def test_speech_overfit_fixture_has_expected_timeline(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.data.episode_units = 16
    smoke_config.data.train_episodes = 32
    dataset = SpeechOverfitDataset(smoke_config.data, smoke_config.model)
    episode = dataset.make_episode(9)

    assert len(dataset) == 32
    assert len(episode.units) == 16
    assert episode.metadata["response_class"] == 1
    assert episode.metadata["variant"] == 1
    controls = [unit.control_target.speech.item() for unit in episode.units]
    assert controls == [
        SpeechControl.SILENT,
        SpeechControl.SILENT,
        SpeechControl.SILENT,
        SpeechControl.SILENT,
        SpeechControl.SILENT,
        SpeechControl.START,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.CONTINUE,
        SpeechControl.STOP,
        SpeechControl.SILENT,
        SpeechControl.SILENT,
    ]
    assert [unit.speech_mask.item() for unit in episode.units] == [
        control != SpeechControl.SILENT for control in controls
    ]
    assert all(not unit.action_mask.item() for unit in episode.units)
    assert all(not unit.memory_mask.item() for unit in episode.units)
    assert episode.target_speech is not None
    assert episode.target_speech[: 5 * smoke_config.data.unit_audio_samples].count_nonzero() == 0


def test_overfit_evaluation_covers_complete_encoded_dataset(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    smoke_config.data.episode_units = 16
    smoke_config.data.train_episodes = 2
    episodes = list(SpeechOverfitDataset(smoke_config.data, smoke_config.model))
    for episode in episodes:
        episode.metadata["speech_codes_encoded"] = True
    processed = tmp_path / "processed"
    write_episode_shards(episodes, processed / "train-%06d.tar")
    smoke_config.data.source = "webdataset"
    smoke_config.data.shards = str(processed / "train-*.tar")
    smoke_config.data.manifest = str(processed / "train-manifest.jsonl")

    model = StreamingLatentLoop(smoke_config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = CheckpointMetadata(
        data_identity=file_sha256(processed / "train-manifest.jsonl"),
        codec_id=smoke_config.data.codec_id,
        codec_weight_hash=smoke_config.data.codec_weight_hash,
        git_commit="test",
        codec_revision=smoke_config.data.codec_revision,
    )
    torch.save(
        {
            "format_version": 3,
            "config_hash": "training-only-hash-is-not-used-for-evaluation",
            "metadata": asdict(metadata),
            "model": model.state_dict(),
        },
        checkpoint,
    )

    result = evaluate_overfit_checkpoint(
        smoke_config,
        checkpoint,
        device="cpu",
        codec_threshold=0.0,
        control_f1_threshold=0.0,
    )

    assert result.episodes == 2
    assert result.speech_frames == 18
    assert len(result.teacher_codec_accuracy) == smoke_config.model.speech_codebooks
    assert len(result.autoregressive_codec_accuracy) == smoke_config.model.speech_codebooks
    assert len(result.speech_control_per_class_f1) == 5
    assert len(result.speech_control_confusion) == 5
    assert result.passed
