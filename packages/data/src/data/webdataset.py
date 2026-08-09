from __future__ import annotations

import glob
import hashlib
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import webdataset as wds
from braceexpand import braceexpand
from model.types import Episode, StreamUnit
from runtime.config import DataConfig, ModelConfig


def _numpy_bytes(array: np.ndarray, *, compressed: bool = False) -> bytes:
    output = io.BytesIO()
    if compressed:
        np.savez_compressed(output, values=array)
    else:
        np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _audio_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, audio.reshape(-1), sample_rate, format="FLAC", subtype="PCM_16")
    return output.getvalue()


def _timeline_bytes(values: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    return output.getvalue()


def _load_numpy(data: bytes, *, compressed: bool = False) -> np.ndarray:
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    if compressed:
        try:
            return loaded["values"]
        finally:
            loaded.close()
    return loaded


def _load_timeline(data: bytes) -> dict[str, np.ndarray]:
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    try:
        return {name: loaded[name] for name in loaded.files}
    finally:
        loaded.close()


def episode_to_sample(episode: Episode) -> dict[str, bytes | str]:
    units = episode.units
    audio = torch.cat([unit.mic_audio for unit in units], dim=0).cpu().numpy()
    sparse_screens: list[np.ndarray] = []
    screen_indices: list[int] = []
    for unit in units:
        if bool(unit.screen_valid.item()):
            screen_indices.append(len(sparse_screens))
            sparse_screens.append(unit.screen[0].cpu().numpy())
        else:
            screen_indices.append(-1)
    if sparse_screens:
        screens = np.stack(sparse_screens)
    else:
        height, width = units[0].screen.shape[-2:]
        screens = np.empty((0, 3, height, width), dtype=np.float32)
    metadata = {
        **episode.metadata,
        "episode_id": episode.episode_id,
        "unit_count": len(units),
        "unit_audio_samples": units[0].mic_audio.shape[1],
        "schema_version": int(episode.metadata.get("schema_version", 5)),
    }
    metadata.setdefault(
        "runtime_identity",
        {
            "protocol_version": metadata.get("protocol_version", "realtime-v1"),
            "environment_id": metadata.get("environment_id", "recorded"),
            "environment_version": metadata.get("environment_version", "1"),
            "action_vocabulary_id": metadata.get("action_vocabulary_id", "unified-action-v4"),
        },
    )
    decoded_controls = metadata.get("decoded_controls", [])
    receipts = metadata.get("receipts", [])
    timeline = {
        "timestamps_ms": np.asarray([int(u.timestamp_ms.item()) for u in units]),
        "delta_ms": np.asarray([int(u.delta_ms.item()) for u in units]),
        "screen_indices": np.asarray(screen_indices),
        "screen_revision": np.asarray([int(u.screen_revision.item()) for u in units]),
        "speech_mode": np.asarray([int(u.speech_mode.item()) for u in units], dtype=np.int64),
        "speech_mode_mask": np.asarray([bool(u.speech_mode_mask.item()) for u in units]),
        "speech_codec_mask": np.stack([u.speech_codec_mask[0].cpu().numpy() for u in units]),
        "action_tokens": torch.cat([u.action_tokens for u in units], dim=0)
        .cpu()
        .numpy()
        .astype(np.int64),
        "action_token_mask": torch.cat([u.action_token_mask for u in units], dim=0).cpu().numpy(),
    }
    codes = torch.cat([u.speech_codes for u in units], dim=0).cpu().numpy().astype(np.uint16)
    target_speech = (
        episode.target_speech.cpu().numpy()
        if episode.target_speech is not None
        else np.zeros(audio.size, dtype=np.float32)
    )
    return {
        "__key__": episode.episode_id,
        "meta.json": json.dumps(metadata, sort_keys=True).encode("utf-8"),
        "mic.flac": _audio_bytes(audio, int(metadata["sample_rate"])),
        "target_speech.flac": _audio_bytes(target_speech, int(metadata["sample_rate"])),
        "screen.npz": _numpy_bytes(screens, compressed=True),
        "timeline.npz": _timeline_bytes(timeline),
        "speech_codes.npy": _numpy_bytes(codes),
        "turns.json": json.dumps(episode.metadata.get("turns", [])).encode("utf-8"),
        "controls.json": json.dumps(decoded_controls, sort_keys=True).encode("utf-8"),
        "receipts.json": json.dumps(receipts, sort_keys=True).encode("utf-8"),
    }


def _sample_sha256(sample: dict[str, bytes | str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in sample if not key.startswith("__")):
        digest.update(key.encode("utf-8"))
        value = sample[key]
        digest.update(value.encode("utf-8") if isinstance(value, str) else value)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    session_splits: dict[tuple[str, str], set[str]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("source") != "synthetic":
                if (
                    not isinstance(entry.get("source_license"), str)
                    or not entry["source_license"].strip()
                ):
                    raise ValueError(f"source_license is required at line {line_number}")
                if not isinstance(entry.get("redistribution_allowed"), bool):
                    raise ValueError(
                        f"redistribution_allowed must be a boolean at line {line_number}"
                    )
            episode_id = str(entry["episode_id"])
            if episode_id in entries:
                raise ValueError(f"duplicate episode_id in manifest at line {line_number}")
            entries[episode_id] = entry
            session = (str(entry.get("device_id_hash", "")), str(entry.get("session_id_hash", "")))
            if all(session):
                session_splits.setdefault(session, set()).add(str(entry.get("split", "")))
    leaking = [session for session, splits in session_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"sessions cross dataset splits: {leaking[:3]}")
    return entries


def write_episode_shards(
    episodes: Iterable[Episode], output_pattern: str | Path, max_size: int = 1_000_000_000
) -> list[dict[str, Any]]:
    output_pattern = str(Path(output_pattern).expanduser())
    Path(output_pattern).parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    with wds.ShardWriter(output_pattern, maxsize=max_size, maxcount=100_000) as sink:
        for episode in episodes:
            sample = episode_to_sample(episode)
            sink.write(sample)
            sample_metadata = json.loads(sample["meta.json"])
            manifest.append(
                {
                    **sample_metadata,
                    "episode_id": episode.episode_id,
                    "units": len(episode.units),
                    "duration_ms": sum(int(u.delta_ms.item()) for u in episode.units),
                    "content_sha256": _sample_sha256(sample),
                }
            )
    manifest_path = Path(output_pattern.replace("%06d", "manifest")).with_suffix(".jsonl")
    with manifest_path.open("w", encoding="utf-8") as output:
        for item in manifest:
            output.write(json.dumps(item, sort_keys=True) + "\n")
    return manifest


class EpisodeShardReader:
    def __init__(
        self,
        shards: str,
        data: DataConfig,
        model: ModelConfig,
        *,
        require_encoded_speech: bool = True,
        validate_manifest: bool = True,
    ) -> None:
        self.shards = str(Path(shards).expanduser())
        self.data = data
        self.model = model
        self.require_encoded_speech = require_encoded_speech
        self.manifest = (
            load_manifest(data.manifest) if validate_manifest and data.manifest else None
        )

    def __iter__(self) -> Iterator[Episode]:
        shard_paths = sorted(
            {path for expanded in braceexpand(self.shards) for path in glob.glob(expanded)}
        )
        if not shard_paths:
            raise FileNotFoundError(f"no WebDataset shards match {self.shards}")
        seen: set[str] = set()
        for shard_index, shard_path in enumerate(shard_paths):
            dataset = wds.WebDataset([shard_path], shardshuffle=False, empty_check=True)
            for sample_index, sample in enumerate(dataset):
                episode_id = str(sample["__key__"])
                if episode_id in seen:
                    raise ValueError(f"duplicate episode {episode_id} in shards")
                seen.add(episode_id)
                self._validate_sample_identity(sample)
                yield self._decode(sample, shard_index, sample_index)
        if self.manifest is not None and seen != self.manifest.keys():
            missing = sorted(self.manifest.keys() - seen)[:3]
            extra = sorted(seen - self.manifest.keys())[:3]
            raise ValueError(f"manifest/shard episode mismatch: missing={missing}, extra={extra}")

    def _validate_sample_identity(self, sample: dict[str, Any]) -> None:
        metadata = json.loads(sample["meta.json"])
        episode_id = str(sample["__key__"])
        if int(metadata.get("schema_version", -1)) != self.data.schema_version:
            raise ValueError(f"schema version mismatch for {episode_id}")
        if int(metadata.get("schema_version", -1)) != 5:
            raise ValueError(f"unsupported schema version for {episode_id}")
        required_v5 = (
            "stage",
            "dataset_scale",
            "sample_kind",
            "supervision_kind",
            "action_source",
            "task_id",
            "environment_id",
            "environment_version",
            "protocol_version",
            "action_vocabulary_id",
            "codec_id",
            "codec_revision",
            "runtime_identity",
        )
        missing = [key for key in required_v5 if key not in metadata]
        if missing:
            raise ValueError(f"schema v5 metadata is missing for {episode_id}: {missing}")
        for key in ("codec_id", "codec_weight_hash", "codec_revision"):
            if metadata.get(key) != getattr(self.data, key):
                raise ValueError(f"{key} mismatch for {episode_id}")
        if self.require_encoded_speech and not metadata.get("speech_codes_encoded", True):
            raise ValueError(f"speech codes are not encoded for {episode_id}")
        if self.manifest is not None:
            entry = self.manifest.get(episode_id)
            if entry is None:
                raise ValueError(f"episode {episode_id} is absent from the manifest")
            if entry.get("content_sha256") != _sample_sha256(sample):
                raise ValueError(f"content hash mismatch for {episode_id}")

    def _decode(self, sample: dict[str, Any], shard_index: int, sample_index: int) -> Episode:
        metadata = json.loads(sample["meta.json"])
        turns = json.loads(sample["turns.json"])
        if "controls.json" in sample:
            metadata["decoded_controls"] = json.loads(sample["controls.json"])
        if "receipts.json" in sample:
            metadata["receipts"] = json.loads(sample["receipts.json"])
        screens = _load_numpy(sample["screen.npz"], compressed=True)
        timeline = _load_timeline(sample["timeline.npz"])
        codes = _load_numpy(sample["speech_codes.npy"])
        audio, sample_rate = sf.read(io.BytesIO(sample["mic.flac"]), dtype="float32")
        target_speech, target_sample_rate = sf.read(
            io.BytesIO(sample["target_speech.flac"]), dtype="float32"
        )
        if sample_rate != self.data.audio_sample_rate or target_sample_rate != sample_rate:
            raise ValueError("sample audio rate does not match configuration")
        unit_count = int(metadata["unit_count"])
        metadata["turns"] = turns
        audio = audio.reshape(unit_count, int(metadata["unit_audio_samples"]))
        units: list[StreamUnit] = []
        for index in range(unit_count):
            screen_index = int(timeline["screen_indices"][index])
            screen_valid = screen_index >= 0
            if screen_valid:
                if screen_index >= len(screens):
                    raise ValueError("screen timeline index is outside screen.npz")
                screen = screens[screen_index]
            else:
                screen = np.zeros(
                    (3, self.data.screen_height, self.data.screen_width), dtype=np.float32
                )
            units.append(
                StreamUnit(
                    timestamp_ms=torch.tensor([timeline["timestamps_ms"][index]], dtype=torch.long),
                    delta_ms=torch.tensor([timeline["delta_ms"][index]], dtype=torch.long),
                    mic_audio=torch.from_numpy(audio[index]).float()[None],
                    screen=torch.from_numpy(screen).float()[None],
                    screen_valid=torch.tensor([screen_valid]),
                    screen_revision=torch.tensor([timeline["screen_revision"][index]]),
                    speech_mode=torch.tensor([timeline["speech_mode"][index]], dtype=torch.long),
                    speech_mode_mask=torch.tensor(
                        [timeline["speech_mode_mask"][index]], dtype=torch.bool
                    ),
                    speech_codes=torch.from_numpy(codes[index]).long()[None],
                    speech_codec_mask=torch.from_numpy(timeline["speech_codec_mask"][index]).bool()[
                        None
                    ],
                    action_tokens=torch.from_numpy(timeline["action_tokens"][index]).long()[None],
                    action_token_mask=torch.from_numpy(timeline["action_token_mask"][index]).bool()[
                        None
                    ],
                )
            )
        episode = Episode(
            sample["__key__"],
            units,
            metadata,
            target_speech=torch.from_numpy(target_speech).float(),
            ordered_shard_index=shard_index,
            sample_index_in_shard=sample_index,
        )
        episode.validate(
            audio_samples=self.data.unit_audio_samples,
            speech_frames=self.model.speech_frames_per_unit,
            speech_codebooks=self.model.speech_codebooks,
            speech_codebook_size=self.model.speech_codebook_size,
        )
        return episode
