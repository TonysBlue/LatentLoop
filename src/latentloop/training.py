from __future__ import annotations

import random
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator

from latentloop.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    DataCursor,
    config_hash,
    file_sha256,
)
from latentloop.config import ProjectConfig
from latentloop.data import EpisodeShardReader, SyntheticEpisodeDataset
from latentloop.losses import compute_losses
from latentloop.model import StreamingLatentLoop
from latentloop.tracking import Tracker
from latentloop.types import Episode, RecurrentState


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_episodes(config: ProjectConfig) -> Iterable[Episode]:
    if config.data.source == "synthetic":
        return SyntheticEpisodeDataset(config.data, config.model)
    assert config.data.shards is not None
    return EpisodeShardReader(config.data.shards, config.data, config.model)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _checkpoint_metadata(
    config: ProjectConfig, parent_sha256: str | None = None
) -> CheckpointMetadata:
    return CheckpointMetadata(
        data_identity=_data_identity(config),
        codec_id=config.data.codec_id,
        codec_weight_hash=config.data.codec_weight_hash,
        git_commit=_git_commit(),
        parent_sha256=parent_sha256,
    )


def _data_identity(config: ProjectConfig) -> str:
    if config.data.manifest:
        manifest_path = Path(config.data.manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = config.runtime.root_path() / manifest_path
        return file_sha256(manifest_path)
    return config_hash(config.as_dict()["data"])


def train(
    config: ProjectConfig,
    *,
    resume: str | None = None,
    model: StreamingLatentLoop | None = None,
    stop_after_updates: int | None = None,
) -> dict[str, Any]:
    seed_everything(config.data.seed)
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    model = model or StreamingLatentLoop(config.model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    root = config.runtime.root_path()
    checkpoint_manager = CheckpointManager(root / "checkpoints")
    unwrapped_model = accelerator.unwrap_model(model)
    tracker = Tracker(
        config,
        stage="state",
        model_name=f"{unwrapped_model.parameter_count()}p",
        parameter_count=unwrapped_model.parameter_count(),
        data_identity=_data_identity(config),
    )
    train_state: dict[str, Any] = {"update": 0, "epoch": 0, "episode": 0, "unit": 0}
    cursor = DataCursor()
    recurrent: RecurrentState | None = None
    parent_sha256: str | None = None
    if resume:
        parent_sha256 = file_sha256(resume)
        train_state, cursor, recurrent, _ = checkpoint_manager.load(
            resume,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=accelerator.scaler,
            device=accelerator.device,
            config=config.as_dict(),
            expected_metadata=_checkpoint_metadata(config),
        )

    optimizer.zero_grad(set_to_none=True)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    training_started = time.perf_counter()
    last_metrics: dict[str, float] = {}
    target_updates = min(
        config.training.max_updates,
        stop_after_updates if stop_after_updates is not None else config.training.max_updates,
    )
    try:
        epoch = cursor.epoch
        while train_state["update"] < target_updates:
            yielded = False
            for episode_index, episode in enumerate(build_episodes(config)):
                yielded = True
                episode_location = (
                    episode.ordered_shard_index,
                    episode.sample_index_in_shard,
                )
                cursor_location = (
                    cursor.ordered_shard_index,
                    cursor.sample_index_in_shard,
                )
                if epoch == cursor.epoch and episode_location < cursor_location:
                    continue
                is_resumed_episode = epoch == cursor.epoch and episode_location == cursor_location
                unit_start = cursor.unit if is_resumed_episode else 0
                if not is_resumed_episode or unit_start == 0:
                    recurrent = None
                chunk_size = config.training.tbptt_units
                for chunk_start in range(unit_start, len(episode.units), chunk_size):
                    if train_state["update"] >= target_updates:
                        break
                    chunk = episode.units[chunk_start : chunk_start + chunk_size]
                    moved = [unit.to(accelerator.device) for unit in chunk]
                    if recurrent is None:
                        recurrent = accelerator.unwrap_model(model).initial_state(
                            moved[0].batch_size, accelerator.device
                        )
                    with accelerator.accumulate(model):
                        chunk_losses: dict[str, torch.Tensor] = {}
                        output = None
                        for unit in moved:
                            output = model(unit, recurrent)
                            recurrent = output.state
                            unit_losses = compute_losses(output, unit)
                            for name, value in unit_losses.items():
                                accumulated = chunk_losses.get(name, torch.zeros_like(value))
                                chunk_losses[name] = accumulated + value
                        losses = {name: value / len(moved) for name, value in chunk_losses.items()}
                        accelerator.backward(losses["total"])
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.training.max_grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    assert output is not None
                    recurrent = recurrent.detach()
                    next_unit = chunk_start + len(chunk)
                    if next_unit >= len(episode.units):
                        cursor = DataCursor(
                            epoch=epoch,
                            episode=episode_index + 1,
                            unit=0,
                            ordered_shard_index=episode.ordered_shard_index,
                            sample_index_in_shard=episode.sample_index_in_shard + 1,
                        )
                        checkpoint_recurrent = None
                    else:
                        cursor = DataCursor(
                            epoch=epoch,
                            episode=episode_index,
                            unit=next_unit,
                            ordered_shard_index=episode.ordered_shard_index,
                            sample_index_in_shard=episode.sample_index_in_shard,
                        )
                        checkpoint_recurrent = recurrent
                    train_state.update(
                        {
                            "epoch": epoch,
                            "episode": episode_index,
                            "unit": next_unit,
                            "consumed_units": train_state.get("consumed_units", 0) + len(chunk),
                        }
                    )
                    if not accelerator.sync_gradients:
                        continue

                    train_state["update"] += 1
                    last_metrics = {
                        f"train/loss_{name}": float(value.detach().float().item())
                        for name, value in losses.items()
                    }
                    last_metrics.update(
                        {
                            "latent/gate_mean": float(
                                output.latent_gate.detach().float().mean().item()
                            ),
                            "stream/kv_tokens": float(output.state.layer_kv[0].key.shape[2]),
                            "data/episode": float(episode_index),
                            "data/unit": float(next_unit),
                            "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        }
                    )
                    should_log = (
                        accelerator.is_main_process
                        and train_state["update"] % config.training.log_every == 0
                    )
                    if should_log:
                        tracker.log(last_metrics, train_state["update"])
                    should_checkpoint = (
                        accelerator.is_main_process
                        and train_state["update"] % config.training.checkpoint_every == 0
                    )
                    if should_checkpoint:
                        path, digest = checkpoint_manager.save(
                            f"step-{train_state['update']:08d}",
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=accelerator.scaler,
                            recurrent_state=checkpoint_recurrent,
                            train_state=train_state,
                            data_cursor=cursor,
                            metadata=_checkpoint_metadata(config, parent_sha256),
                            config=config.as_dict(),
                        )
                        parent_sha256 = digest
                        tracker.record_checkpoint(path, digest, train_state["update"])
                if train_state["update"] >= target_updates:
                    break
            if not yielded:
                raise RuntimeError("training source produced no episodes")
            if train_state["update"] < target_updates:
                epoch += 1
                cursor = DataCursor(epoch=epoch, episode=0, unit=0)
                recurrent = None
    finally:
        elapsed_seconds = time.perf_counter() - training_started
        runtime_metrics = {
            "runtime/elapsed_seconds": elapsed_seconds,
            "runtime/units_per_second": train_state.get("consumed_units", 0)
            / max(elapsed_seconds, 1e-9),
        }
        if accelerator.device.type == "cuda":
            runtime_metrics.update(
                {
                    "runtime/peak_memory_allocated_bytes": float(
                        torch.cuda.max_memory_allocated(accelerator.device)
                    ),
                    "runtime/peak_memory_reserved_bytes": float(
                        torch.cuda.max_memory_reserved(accelerator.device)
                    ),
                }
            )
        last_metrics.update(runtime_metrics)
        if accelerator.is_main_process and train_state["update"]:
            tracker.log(runtime_metrics, train_state["update"])
        tracker.finish()
    return {
        "train_state": train_state,
        "data_cursor": cursor,
        "metrics": last_metrics,
        "model": model,
    }
