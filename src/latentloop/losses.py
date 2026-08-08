from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from latentloop.types import SpeechMode, StepOutput, StreamUnit


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


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
    action_values = F.cross_entropy(
        output.action_logits.reshape(-1, output.action_logits.shape[-1]),
        target.action_tokens.reshape(-1),
        reduction="none",
    ).view_as(target.action_tokens)
    action_loss = _masked_mean(action_values, target.action_token_mask)
    total = speech_loss_weight * (mode_loss + codec_loss) + action_loss_weight * action_loss
    return {
        "total": total,
        "speech": mode_loss + codec_loss,
        "speech_mode": mode_loss,
        "speech_codec": codec_loss,
        "action": action_loss,
    }
