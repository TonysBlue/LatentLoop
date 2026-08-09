from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from model.action_tokens import ActionEvent, ActionTokenizer
from model.types import ActionType, Episode, SpeechMode, StreamUnit
from runtime.config import DataConfig, ModelConfig


def _read_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    values, actual_rate = sf.read(Path(path).expanduser(), dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(f"{path} must be pre-resampled to {sample_rate} Hz")
    if values.shape[1] != 1:
        raise ValueError(f"{path} must be mono")
    waveform = torch.from_numpy(values[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{path} contains a non-finite audio sample")
    if waveform.abs().max() > 1.0:
        raise ValueError(f"{path} contains audio outside [-1, 1]")
    return waveform


def _speech_activity(target: torch.Tensor, frame_samples: int) -> list[bool]:
    frames = target.reshape(-1, frame_samples)
    return [bool(active) for active in (frames.square().mean(dim=1) > 1e-5).tolist()]


def _segment_activity(segments: list[dict[str, Any]], ticks: int, frame_samples: int) -> list[bool]:
    active = [False] * ticks
    previous_end = -1
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"target_segments[{index}] must be an object")
        try:
            start = int(segment["start_sample"])
            end = int(segment["end_sample"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"target_segments[{index}] requires integer start_sample and end_sample"
            ) from error
        if not isinstance(segment.get("turn_id"), str) or not segment["turn_id"].strip():
            raise ValueError(f"target_segments[{index}].turn_id must be a non-empty string")
        if start < 0 or start % frame_samples:
            raise ValueError("assistant segment starts must be on an 80 ms boundary")
        if end <= start:
            raise ValueError("assistant segment end must be greater than its start")
        start_frame = start // frame_samples
        final_frame = (end - 1) // frame_samples
        if final_frame >= ticks:
            raise ValueError("assistant segment is outside the episode timeline")
        if start_frame <= previous_end:
            raise ValueError("assistant segments must not overlap")
        for frame in range(start_frame, final_frame + 1):
            active[frame] = True
        previous_end = final_frame
    return active


def _load_screens(
    path: str | None, model: ModelConfig, data: DataConfig
) -> dict[int, torch.Tensor]:
    if path is None:
        return {}
    loaded = np.load(Path(path).expanduser(), allow_pickle=False)
    try:
        ticks = loaded["ticks"]
        frames = loaded["frames"]
    finally:
        loaded.close()
    if len(ticks) != len(frames):
        raise ValueError("screen ticks and frames must have identical lengths")
    expected = (3, data.screen_height, data.screen_width)
    result: dict[int, torch.Tensor] = {}
    for tick, frame in zip(ticks.tolist(), frames, strict=True):
        if tuple(frame.shape) != expected:
            raise ValueError(f"screen frame must have shape {expected}")
        if int(tick) in result:
            raise ValueError("screen tick is duplicated")
        result[int(tick)] = torch.from_numpy(frame.copy()).float()
    return result


def import_speech_manifest(
    path: str | Path, data: DataConfig, model: ModelConfig
) -> Iterator[Episode]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                yield _build_episode(record, manifest_path.parent, data, model)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid speech source manifest line {line_number}: {error}"
                ) from error


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _build_episode(
    record: dict[str, Any], base: Path, data: DataConfig, model: ModelConfig
) -> Episode:
    required = (
        "episode_id",
        "mic_audio",
        "target_speech",
        "source",
        "source_license",
        "redistribution_allowed",
        "language",
        "split",
        "session_id_hash",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for name in ("episode_id", "source", "source_license", "language", "split"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(record["redistribution_allowed"], bool):
        raise ValueError("redistribution_allowed must be a boolean")
    mic = _read_audio(_resolve(base, record["mic_audio"]), data.audio_sample_rate)
    target = _read_audio(_resolve(base, record["target_speech"]), data.audio_sample_rate)
    explicit_segments = record.get("target_segments")
    if explicit_segments is not None and not isinstance(explicit_segments, list):
        raise ValueError("target_segments must be a list")
    segment_timeline = 0
    if explicit_segments:
        try:
            segment_timeline = (
                max(int(segment["end_sample"]) for segment in explicit_segments)
                + data.unit_audio_samples
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("target_segments contain an invalid end_sample") from error
    samples = max(mic.numel(), target.numel(), segment_timeline)
    ticks = -(-samples // data.unit_audio_samples)
    timeline_samples = ticks * data.unit_audio_samples
    mic = torch.nn.functional.pad(mic, (0, timeline_samples - mic.numel()))
    target = torch.nn.functional.pad(target, (0, timeline_samples - target.numel()))
    active = (
        _segment_activity(explicit_segments, ticks, data.unit_audio_samples)
        if explicit_segments is not None
        else _speech_activity(target, data.unit_audio_samples)
    )
    screen_path = record.get("screens")
    screens = _load_screens(str(_resolve(base, screen_path)) if screen_path else None, model, data)
    if any(tick < 0 or tick >= ticks for tick in screens):
        raise ValueError("screen tick is outside the episode timeline")
    tokenizer = ActionTokenizer(model.max_action_duration_ms, model.action_burst_tokens)
    action_by_tick = _load_action_targets(record.get("actions"), ticks, tokenizer, model)
    units: list[StreamUnit] = []
    screen_revision = -1
    empty_screen = torch.zeros(3, data.screen_height, data.screen_width)
    for tick in range(ticks):
        screen_valid = tick in screens
        if screen_valid:
            screen_revision += 1
        speaking = active[tick]
        units.append(
            StreamUnit(
                timestamp_ms=torch.tensor([tick * data.unit_ms]),
                delta_ms=torch.tensor([data.unit_ms]),
                mic_audio=mic[
                    None, tick * data.unit_audio_samples : (tick + 1) * data.unit_audio_samples
                ],
                screen=screens.get(tick, empty_screen)[None],
                screen_valid=torch.tensor([screen_valid]),
                screen_revision=torch.tensor([screen_revision]),
                speech_mode=torch.tensor(
                    [int(SpeechMode.SPEECH if speaking else SpeechMode.SILENCE)]
                ),
                speech_mode_mask=torch.tensor([True]),
                speech_codes=torch.zeros(
                    1, model.speech_frames_per_unit, model.speech_codebooks, dtype=torch.long
                ),
                speech_codec_mask=torch.full(
                    (1, model.speech_frames_per_unit), speaking, dtype=torch.bool
                ),
                action_tokens=action_by_tick[tick][0],
                action_token_mask=action_by_tick[tick][1],
            )
        )
    metadata = {
        **record,
        "schema_version": data.schema_version,
        "sample_rate": data.audio_sample_rate,
        "unit_ms": data.unit_ms,
        "codec_frame_rate": data.codec_frame_rate,
        "codec_id": data.codec_id,
        "codec_weight_hash": data.codec_weight_hash,
        "codec_revision": data.codec_revision,
        "speech_codes_encoded": False,
        "scenario": record.get("scenario", "spoken-response"),
        "device_id_hash": record.get("device_id_hash", "speech-import"),
        "turns": record.get("turns", []),
        "stage": record.get("stage", "pretrain"),
        "dataset_scale": data.dataset,
        "sample_kind": record.get("sample_kind", "supervised_episode"),
        "supervision_kind": record.get("supervision_kind", "speech"),
        "action_source": "expert" if record.get("actions") else "none",
        "task_id": record.get("task_id", record["episode_id"]),
        "environment_id": record.get("environment_id", "recorded"),
        "environment_version": record.get("environment_version", "1"),
        "protocol_version": record.get("protocol_version", "realtime-v1"),
        "action_vocabulary_id": record.get("action_vocabulary_id", "unified-action-v4"),
        "action_schema_version": 4,
    }
    episode = Episode(str(record["episode_id"]), units, metadata, target_speech=target)
    episode.validate(
        audio_samples=data.unit_audio_samples,
        speech_frames=model.speech_frames_per_unit,
        speech_codebooks=model.speech_codebooks,
        speech_codebook_size=model.speech_codebook_size,
        action_vocab_size=tokenizer.vocab_size,
    )
    return episode


def _load_action_targets(
    values: Any, ticks: int, tokenizer: ActionTokenizer, model: ModelConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result = [
        (
            torch.zeros(1, model.action_burst_tokens, dtype=torch.long),
            torch.zeros(1, model.action_burst_tokens, dtype=torch.bool),
        )
        for _ in range(ticks)
    ]
    if values is None:
        return result
    if not isinstance(values, list):
        raise ValueError("actions must be a list")
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise ValueError(f"actions[{index}] must be an object")
        tick = int(raw["tick"])
        if tick < 0 or tick >= ticks:
            raise ValueError("action tick is outside the episode timeline")
        if result[tick][1].any():
            raise ValueError("action tick is duplicated")
        try:
            kind = ActionType[str(raw["type"]).upper()]
        except (KeyError, TypeError) as error:
            raise ValueError(f"actions[{index}] has an invalid type") from error
        event = ActionEvent(
            kind,
            coordinates=tuple(raw["coordinates"]) if raw.get("coordinates") is not None else None,
            scroll_delta=tuple(raw["scroll_delta"])
            if raw.get("scroll_delta") is not None
            else None,
            duration_ms=raw.get("duration_ms"),
            text=raw.get("text"),
            keys=tuple(raw["keys"]) if raw.get("keys") is not None else None,
        )
        encoded = tokenizer.encode(event)
        if len(encoded) > model.action_burst_tokens:
            raise ValueError("expert action exceeds one action burst")
        tokens, mask = result[tick]
        tokens[0, : len(encoded)] = torch.tensor(encoded)
        mask[0, : len(encoded)] = True
    return result
