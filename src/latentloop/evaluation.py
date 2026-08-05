from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from latentloop.checkpoint import file_sha256
from latentloop.config import ProjectConfig
from latentloop.data import EpisodeShardReader
from latentloop.model import StreamingLatentLoop
from latentloop.types import SpeechSamplingConfig


@dataclass(frozen=True, slots=True)
class OverfitEvaluation:
    episodes: int
    speech_frames: int
    teacher_codec_accuracy: list[float]
    autoregressive_codec_accuracy: list[float]
    speech_control_macro_f1: float
    speech_control_accuracy: float
    speech_control_per_class_f1: list[float]
    speech_control_confusion: list[list[int]]
    passed: bool


@dataclass(frozen=True, slots=True)
class CanaryEvaluation:
    split: str
    episodes: int
    speech_frames: int
    teacher_codec_accuracy: list[float]
    speech_control_macro_f1: float
    speech_control_accuracy: float
    speech_control_per_class_f1: list[float]
    speech_control_confusion: list[list[int]]


class _ClassificationCounts:
    def __init__(self, classes: int) -> None:
        self.matrix = torch.zeros(classes, classes, dtype=torch.long)

    def add(self, predictions: Tensor, targets: Tensor) -> None:
        predictions = predictions.detach().long().cpu().flatten()
        targets = targets.detach().long().cpu().flatten()
        indices = targets * self.matrix.shape[0] + predictions
        self.matrix += torch.bincount(
            indices, minlength=self.matrix.numel()
        ).reshape_as(self.matrix)

    def accuracy(self) -> float:
        return float(self.matrix.diag().sum() / self.matrix.sum().clamp_min(1))

    def macro_f1(self) -> float:
        values, included = self.f1()
        return float(values[included].mean()) if included.any() else 0.0

    def f1(self) -> tuple[Tensor, Tensor]:
        true_positive = self.matrix.diag().float()
        predicted = self.matrix.sum(dim=0).float()
        actual = self.matrix.sum(dim=1).float()
        included = (predicted + actual) > 0
        f1 = 2 * true_positive / (predicted + actual).clamp_min(1)
        return f1, included


def load_evaluation_model(
    config: ProjectConfig,
    checkpoint: str | Path,
    device: torch.device,
    *,
    require_data_identity: bool = True,
) -> StreamingLatentLoop:
    payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
    if payload.get("format_version") != 3:
        raise ValueError("overfit evaluation requires a format version 3 checkpoint")
    metadata = payload.get("metadata", {})
    for field, expected in (
        ("codec_id", config.data.codec_id),
        ("codec_weight_hash", config.data.codec_weight_hash),
        ("codec_revision", config.data.codec_revision),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"checkpoint {field} does not match the evaluation config")
    if require_data_identity and config.data.manifest:
        manifest = Path(config.data.manifest).expanduser()
        if not manifest.is_absolute():
            manifest = config.runtime.root_path() / manifest
        if metadata.get("data_identity") != file_sha256(manifest):
            raise ValueError("checkpoint data identity does not match the evaluation manifest")
    model = StreamingLatentLoop(config.model)
    model.load_state_dict(payload["model"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def evaluate_overfit_checkpoint(
    config: ProjectConfig,
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
    codec_threshold: float = 0.9,
    control_f1_threshold: float = 0.9,
) -> OverfitEvaluation:
    selected_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = load_evaluation_model(config, checkpoint, selected_device)
    if config.data.shards is None:
        raise ValueError("overfit evaluation requires WebDataset shards")
    teacher_correct = torch.zeros(config.model.speech_codebooks, dtype=torch.long)
    autoregressive_correct = torch.zeros_like(teacher_correct)
    valid_frames = 0
    control = _ClassificationCounts(5)
    episodes = 0

    with torch.inference_mode():
        for episode in EpisodeShardReader(
            config.data.shards, config.data, config.model
        ):
            episodes += 1
            teacher_state = model.initial_state(1, selected_device)
            autoregressive_state = model.initial_state(1, selected_device)
            for raw_unit in episode.units:
                unit = raw_unit.to(selected_device)
                if selected_device.type == "cuda":
                    unit.mic_audio = unit.mic_audio.half()
                    unit.screen = unit.screen.half()

                teacher = model(unit, teacher_state, unit.speech_codes)
                predicted_control = model._select_speech_control(
                    teacher.controls.speech_logits,
                    teacher_state.speech_local.utterance_active,
                )
                control.add(predicted_control, unit.control_target.speech)
                teacher_state = teacher.state

                generated = model.generate_step(
                    unit,
                    autoregressive_state,
                    SpeechSamplingConfig(greedy=True),
                )
                autoregressive_state = generated.output.state
                mask = unit.speech_mask[:, :, None]
                if mask.any():
                    targets = unit.speech_codes
                    teacher_predictions = teacher.speech_logits.argmax(dim=-1)
                    teacher_correct += (
                        teacher_predictions.eq(targets) & mask
                    ).sum(dim=(0, 1)).cpu()
                    autoregressive_correct += (
                        generated.speech_codes.eq(targets) & mask
                    ).sum(dim=(0, 1)).cpu()
                    valid_frames += int(unit.speech_mask.sum().item())

    if episodes == 0 or valid_frames == 0:
        raise ValueError("overfit evaluation dataset contains no valid speech frames")
    teacher_accuracy = (teacher_correct.float() / valid_frames).tolist()
    autoregressive_accuracy = (autoregressive_correct.float() / valid_frames).tolist()
    control_f1 = control.macro_f1()
    control_per_class, _ = control.f1()
    passed = min(teacher_accuracy) >= codec_threshold and control_f1 >= control_f1_threshold
    return OverfitEvaluation(
        episodes=episodes,
        speech_frames=valid_frames,
        teacher_codec_accuracy=teacher_accuracy,
        autoregressive_codec_accuracy=autoregressive_accuracy,
        speech_control_macro_f1=control_f1,
        speech_control_accuracy=control.accuracy(),
        speech_control_per_class_f1=control_per_class.tolist(),
        speech_control_confusion=control.matrix.tolist(),
        passed=passed,
    )


def evaluate_canary_checkpoint(
    config: ProjectConfig,
    checkpoint: str | Path,
    *,
    split: str = "validation",
    device: str | torch.device | None = None,
) -> CanaryEvaluation:
    if split not in {"validation", "test"}:
        raise ValueError("Canary evaluation split must be validation or test")
    selected_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = load_evaluation_model(
        config, checkpoint, selected_device, require_data_identity=False
    )
    if config.data.shards is None:
        raise ValueError("Canary evaluation requires WebDataset shards")
    teacher_correct = torch.zeros(config.model.speech_codebooks, dtype=torch.long)
    valid_frames = 0
    control = _ClassificationCounts(5)
    episodes = 0
    with torch.inference_mode():
        for episode in EpisodeShardReader(config.data.shards, config.data, config.model):
            episodes += 1
            state = model.initial_state(1, selected_device)
            for raw_unit in episode.units:
                unit = raw_unit.to(selected_device)
                if selected_device.type == "cuda":
                    unit.mic_audio = unit.mic_audio.half()
                    unit.screen = unit.screen.half()
                output = model(unit, state, unit.speech_codes)
                predicted_control = model._select_speech_control(
                    output.controls.speech_logits, state.speech_local.utterance_active
                )
                control.add(predicted_control, unit.control_target.speech)
                state = output.state
                mask = unit.speech_mask[:, :, None]
                if mask.any():
                    teacher_correct += (
                        output.speech_logits.argmax(dim=-1).eq(unit.speech_codes) & mask
                    ).sum(dim=(0, 1)).cpu()
                    valid_frames += int(unit.speech_mask.sum().item())
    if episodes == 0 or valid_frames == 0:
        raise ValueError("Canary evaluation dataset contains no valid speech frames")
    per_class, _ = control.f1()
    return CanaryEvaluation(
        split=split,
        episodes=episodes,
        speech_frames=valid_frames,
        teacher_codec_accuracy=(teacher_correct.float() / valid_frames).tolist(),
        speech_control_macro_f1=control.macro_f1(),
        speech_control_accuracy=control.accuracy(),
        speech_control_per_class_f1=per_class.tolist(),
        speech_control_confusion=control.matrix.tolist(),
    )
