from __future__ import annotations

from pathlib import Path

import pytest
import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.losses import compute_losses
from runtime.config import ProjectConfig
from training.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    DataCursor,
    file_sha256,
)


def test_checkpoint_restores_full_recurrent_step(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    first = model(episode.units[0], model.initial_state(1, "cpu"))
    loss = compute_losses(first, episode.units[0])["total"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    manager = CheckpointManager(tmp_path)
    path, digest = manager.save(
        "state",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        recurrent_state=first.state.detach(),
        train_state={"update": 1, "episode": 1, "unit": 1},
        data_cursor=DataCursor(epoch=0, episode=0, unit=1),
        metadata=CheckpointMetadata(
            data_identity="test-data",
            codec_id=smoke_config.data.codec_id,
            codec_weight_hash=smoke_config.data.codec_weight_hash,
            git_commit="test",
            codec_revision=smoke_config.data.codec_revision,
        ),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 8
    assert payload["metadata"]["world_state_update_version"] == "gated-residual-v1"
    assert payload["metadata"]["delta_time_encoder_version"] == "fourier-delta-v1"
    expected = model(episode.units[1], first.state.detach()).speech_codec_logits.detach()

    restored_model = StreamingLatentLoop(smoke_config.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    train_state, cursor, recurrent, metadata = manager.load(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        device=torch.device("cpu"),
        config=smoke_config.as_dict(),
        expected_metadata=CheckpointMetadata(
            data_identity="test-data",
            codec_id=smoke_config.data.codec_id,
            codec_weight_hash=smoke_config.data.codec_weight_hash,
            git_commit="different-commit-is-allowed",
            codec_revision=smoke_config.data.codec_revision,
        ),
    )
    assert recurrent is not None
    actual = restored_model(episode.units[1], recurrent).speech_codec_logits.detach()
    assert train_state["update"] == 1
    assert cursor.unit == 1
    assert metadata.codec_id == smoke_config.data.codec_id
    assert digest == file_sha256(path)
    assert torch.equal(actual, expected)
    assert (tmp_path / "manifest.json").exists()


def test_checkpoint_rejects_codec_mismatch(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "codec",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec-a", "hash-a", "test", "revision"),
        config=smoke_config.as_dict(),
    )
    try:
        manager.load(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata("data", "codec-b", "hash-a", "test", "revision"),
        )
    except ValueError as error:
        assert "codec_id" in str(error)
    else:
        raise AssertionError("codec mismatch must be rejected")


def test_checkpoint_rejects_pre_dense_visual_format(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path, _ = manager.save(
        "v7",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 7
    old_path = tmp_path / "v6.pt"
    torch.save(payload, old_path)

    with pytest.raises(ValueError, match="abstract world state requires v8"):
        manager.load(
            old_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            device=torch.device("cpu"),
            config=smoke_config.as_dict(),
            expected_metadata=CheckpointMetadata("data", "codec", "hash", "test"),
        )
