from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
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


def _speech_controls(target: torch.Tensor, frame_samples: int) -> list[SpeechControl]:
    frames = target.reshape(-1, frame_samples)
    active = frames.square().mean(dim=1) > 1e-5
    controls: list[SpeechControl] = []
    speaking = False
    for is_active in active.tolist():
        if is_active and not speaking:
            controls.append(SpeechControl.START)
            speaking = True
        elif is_active:
            controls.append(SpeechControl.CONTINUE)
        elif speaking:
            controls.append(SpeechControl.STOP)
            speaking = False
        else:
            controls.append(SpeechControl.SILENT)
    return controls


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
    samples = max(mic.numel(), target.numel())
    ticks = -(-samples // data.unit_audio_samples)
    timeline_samples = ticks * data.unit_audio_samples
    mic = torch.nn.functional.pad(mic, (0, timeline_samples - mic.numel()))
    target = torch.nn.functional.pad(target, (0, timeline_samples - target.numel()))
    controls = _speech_controls(target, data.unit_audio_samples)
    if controls and controls[-1] in {SpeechControl.START, SpeechControl.CONTINUE}:
        mic = torch.nn.functional.pad(mic, (0, data.unit_audio_samples))
        target = torch.nn.functional.pad(target, (0, data.unit_audio_samples))
        controls.append(SpeechControl.STOP)
        ticks += 1
    screen_path = record.get("screens")
    screens = _load_screens(
        str(_resolve(base, screen_path)) if screen_path else None, model, data
    )
    if any(tick < 0 or tick >= ticks for tick in screens):
        raise ValueError("screen tick is outside the episode timeline")
    units: list[StreamUnit] = []
    screen_revision = -1
    for tick in range(ticks):
        screen_valid = tick in screens
        if screen_valid:
            screen_revision += 1
        screen = screens.get(
            tick, torch.zeros(3, data.screen_height, data.screen_width)
        )
        units.append(
            StreamUnit(
                timestamp_ms=torch.tensor([tick * data.unit_ms]),
                delta_ms=torch.tensor([data.unit_ms]),
                mic_audio=mic[
                    None,
                    tick * data.unit_audio_samples : (tick + 1) * data.unit_audio_samples,
                ],
                screen=screen[None],
                screen_valid=torch.tensor([screen_valid]),
                screen_revision=torch.tensor([screen_revision]),
                speech_codes=torch.zeros(
                    1, model.speech_frames_per_unit, model.speech_codebooks, dtype=torch.long
                ),
                speech_mask=torch.ones(1, model.speech_frames_per_unit, dtype=torch.bool),
                action_mask=torch.tensor([False]),
                speech_control_mask=torch.tensor([True]),
                action_control_mask=torch.tensor([False]),
                cognitive_control_mask=torch.tensor([False]),
                memory_mask=torch.tensor([False]),
                action_target=_empty_action(model),
                control_target=ControlTarget(
                    speech=torch.tensor([int(controls[tick])]),
                    action=torch.tensor([int(ActionControl.NOOP)]),
                    cognitive=torch.tensor([int(CognitiveControl.OBSERVE)]),
                ),
                memory_target=torch.zeros(1, dtype=torch.long),
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
    }
    episode = Episode(str(record["episode_id"]), units, metadata, target_speech=target)
    episode.validate(
        audio_samples=data.unit_audio_samples,
        speech_frames=model.speech_frames_per_unit,
        speech_codebooks=model.speech_codebooks,
        speech_codebook_size=model.speech_codebook_size,
    )
    return episode


def _empty_action(model: ModelConfig) -> ActionTarget:
    return ActionTarget(
        type=torch.tensor([int(ActionType.NOOP)]),
        coordinates=torch.zeros(1, 4),
        coordinate_mask=torch.zeros(1, 4, dtype=torch.bool),
        scroll_delta=torch.zeros(1, 2),
        scroll_mask=torch.tensor([False]),
        duration_ms=torch.zeros(1),
        duration_mask=torch.tensor([False]),
        text_tokens=torch.zeros(1, model.action_text_tokens, dtype=torch.long),
        text_mask=torch.zeros(1, model.action_text_tokens, dtype=torch.bool),
        key_mask=torch.zeros(1, model.action_key_vocab_size, dtype=torch.bool),
    )
