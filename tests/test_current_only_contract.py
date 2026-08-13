from __future__ import annotations

from pathlib import Path

import torch
from data import SyntheticEpisodeDataset, write_episode_shards
from data.curation.common import dataset_root
from model import StreamingLatentLoop
from training.checkpoint import CheckpointManager, CheckpointMetadata, DataCursor


def test_current_contract_has_no_project_version_numbers(smoke_config, tmp_path: Path) -> None:
    assert not hasattr(smoke_config.data, "schema_version")
    assert dataset_root(tmp_path, "canary") == (tmp_path / "canary").resolve()

    model = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path / "checkpoints")
    path, _ = manager.save(
        "current",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={"update": 0},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata(
            data_identity="data",
            codec_id=smoke_config.data.codec_id,
            codec_weight_hash=smoke_config.data.codec_weight_hash,
            git_commit="test",
            codec_revision=smoke_config.data.codec_revision,
        ),
        config=smoke_config.as_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "format_version" not in payload
    assert "schema_version" not in payload["metadata"]
    assert "world_state_update_version" not in payload["metadata"]
    assert "delta_time_encoder_version" not in payload["metadata"]


def test_current_trajectory_has_no_schema_number(smoke_config, tmp_path: Path) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    write_episode_shards([episode], tmp_path / "train-%06d.tar", max_size=10_000_000)
    import json

    manifest = json.loads((tmp_path / "train-manifest.jsonl").read_text(encoding="utf-8"))
    assert "schema_version" not in manifest
