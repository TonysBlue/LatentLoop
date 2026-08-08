from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from latentloop.types import ActionLocalState, LayerKV, RecurrentState, SpeechLocalState


@dataclass(frozen=True, slots=True)
class DataCursor:
    epoch: int = 0
    episode: int = 0
    unit: int = 0
    ordered_shard_index: int = 0
    sample_index_in_shard: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    data_identity: str
    codec_id: str
    codec_weight_hash: str
    git_commit: str
    codec_revision: str = "unknown"
    parent_sha256: str | None = None


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialize_state(state: RecurrentState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "layer_kv": [(cache.key.cpu(), cache.value.cpu()) for cache in state.layer_kv],
        "latent": state.latent.cpu(),
        "audio_cache": state.audio_cache.cpu(),
        "hidden": state.hidden.cpu(),
        "speech_local": {
            "temporal": state.speech_local.temporal.cpu(),
            "previous_codes": state.speech_local.previous_codes.cpu(),
        },
        "action_local": {
            "hidden": state.action_local.hidden.cpu(),
            "previous_token": state.action_local.previous_token.cpu(),
            "active": state.action_local.active.cpu(),
            "event_type": state.action_local.event_type.cpu(),
            "burst_tokens": state.action_local.burst_tokens.cpu(),
        },
        "unit_index": state.unit_index.cpu(),
    }


def _deserialize_state(
    payload: dict[str, Any] | None, device: torch.device
) -> RecurrentState | None:
    if payload is None:
        return None
    return RecurrentState(
        layer_kv=tuple(
            LayerKV(key=key.to(device), value=value.to(device))
            for key, value in payload["layer_kv"]
        ),
        latent=payload["latent"].to(device),
        audio_cache=payload["audio_cache"].to(device),
        hidden=payload["hidden"].to(device),
        speech_local=SpeechLocalState(
            temporal=payload["speech_local"]["temporal"].to(device),
            previous_codes=payload["speech_local"]["previous_codes"].to(device),
        ),
        action_local=ActionLocalState(
            hidden=payload["action_local"]["hidden"].to(device),
            previous_token=payload["action_local"]["previous_token"].to(device),
            active=payload["action_local"]["active"].to(device),
            event_type=payload["action_local"]["event_type"].to(device),
            burst_tokens=payload["action_local"]["burst_tokens"].to(device),
        ),
        unit_index=payload["unit_index"].to(device),
    )


class CheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
        recurrent_state: RecurrentState | None,
        train_state: dict[str, Any],
        data_cursor: DataCursor,
        metadata: CheckpointMetadata,
        config: dict[str, Any],
    ) -> tuple[Path, str]:
        target = self.directory / f"{name}.pt"
        payload = {
            "format_version": 4,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "recurrent_state": _serialize_state(recurrent_state),
            "train_state": train_state,
            "data_cursor": asdict(data_cursor),
            "metadata": asdict(metadata),
            "config_hash": config_hash(config),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        with tempfile.NamedTemporaryFile(
            dir=self.directory,
            prefix=f"checkpoint-{name}-",
            suffix=".tmp.pt",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            torch.save(payload, temp)
            temp.flush()
            os.fsync(temp.fileno())
        try:
            os.replace(temp_path, target)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
        digest = file_sha256(target)
        self._write_manifest_entry(target, digest, train_state, data_cursor, metadata)
        return target, digest

    def _write_manifest_entry(
        self,
        path: Path,
        digest: str,
        train_state: dict[str, Any],
        data_cursor: DataCursor,
        metadata: CheckpointMetadata,
    ) -> None:
        manifest_path = self.directory / "manifest.json"
        manifest = {"format_version": 1, "checkpoints": []}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = {
            "path": path.name,
            "sha256": digest,
            "train_state": train_state,
            "data_cursor": asdict(data_cursor),
            "metadata": asdict(metadata),
        }
        entries = [item for item in manifest["checkpoints"] if item["path"] != path.name]
        entries.append(entry)
        manifest["checkpoints"] = entries
        with tempfile.NamedTemporaryFile(
            dir=self.directory,
            prefix="checkpoint-manifest-",
            suffix=".tmp.json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            json.dump(manifest, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        try:
            os.replace(temp_path, manifest_path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
        device: torch.device,
        config: dict[str, Any],
        expected_metadata: CheckpointMetadata,
    ) -> tuple[dict[str, Any], DataCursor, RecurrentState | None, CheckpointMetadata]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("format_version") != 4:
            raise ValueError("unsupported checkpoint format")
        if payload["config_hash"] != config_hash(config):
            raise ValueError("checkpoint configuration does not match the current run")
        restored_metadata = CheckpointMetadata(**payload["metadata"])
        for field_name in (
            "data_identity",
            "codec_id",
            "codec_weight_hash",
            "codec_revision",
        ):
            if getattr(restored_metadata, field_name) != getattr(expected_metadata, field_name):
                raise ValueError(f"checkpoint {field_name} does not match the current run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
        recurrent = _deserialize_state(payload["recurrent_state"], device)
        cursor = DataCursor(**payload["data_cursor"])
        return payload["train_state"], cursor, recurrent, restored_metadata
