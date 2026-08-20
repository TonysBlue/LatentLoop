"""Model output loss helpers shared by supervised Training stages."""

from __future__ import annotations

import torch
from contracts import ActionKind
from torch import Tensor
from torch.nn import functional as F

from model.action import action_log_prob_components
from model.types import SpeechMode, StepOutput, StreamUnit


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    selected = torch.where(mask, values, torch.zeros_like(values))
    return selected.sum() / weights.sum().clamp_min(1)


def structured_action_loss(output: StepOutput, target: StreamUnit) -> Tensor:
    components = action_log_prob_components(output.action, target.action)
    supervised = target.action_supervision_mask
    losses = [_masked_mean(-components["kind"], supervised)]
    conditions = (
        (ActionKind.POINTER_MOVE, "pointer_move"),
        (ActionKind.POINTER_BUTTON, "pointer_button"),
        (ActionKind.SCROLL, "scroll"),
        (ActionKind.TYPE, "type"),
        (ActionKind.HOTKEY, "hotkey"),
    )
    for kind, name in conditions:
        mask = supervised & target.action.kind.eq(int(kind))
        if bool(mask.any()):
            losses.append(_masked_mean(-components[name], mask))
    return sum(losses) / len(losses)


def compute_losses(
    output: StepOutput,
    target: StreamUnit,
    speech_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
) -> dict[str, Tensor]:
    mode_values = F.cross_entropy(output.speech_mode_logits, target.speech_mode, reduction="none")
    mode_loss = _masked_mean(mode_values, target.speech_mode_mask)
    batch, frames, codebooks, vocab = output.speech_codec_logits.shape
    codec_values = F.cross_entropy(
        output.speech_codec_logits.reshape(batch * frames * codebooks, vocab),
        target.speech_codes.reshape(-1),
        reduction="none",
    ).view(batch, frames, codebooks)
    codec_mask = (
        target.speech_codec_mask & target.speech_mode.eq(int(SpeechMode.SPEECH))[:, None]
    )[:, :, None].expand_as(codec_values)
    codec_loss = _masked_mean(codec_values, codec_mask)
    action_loss = structured_action_loss(output, target)
    total = speech_loss_weight * (mode_loss + codec_loss) + action_loss_weight * action_loss
    return {
        "total": total,
        "speech": mode_loss + codec_loss,
        "speech_mode": mode_loss,
        "speech_codec": codec_loss,
        "action": action_loss,
    }
