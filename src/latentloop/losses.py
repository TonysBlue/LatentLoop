from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from latentloop.types import StepOutput, StreamUnit


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_losses(output: StepOutput, target: StreamUnit) -> dict[str, Tensor]:
    batch, frames, codebooks, codebook_size = output.speech_logits.shape
    speech = F.cross_entropy(
        output.speech_logits.reshape(batch * frames * codebooks, codebook_size),
        target.speech_codes.reshape(-1),
        reduction="none",
    ).view(batch, frames, codebooks)
    speech_mask = target.speech_mask[:, :, None].expand_as(speech)
    speech_loss = _masked_mean(speech, speech_mask)
    action = target.action_target
    controls = target.control_target
    action_type = _masked_mean(
        F.cross_entropy(output.action.type_logits, action.type, reduction="none"),
        target.action_mask,
    )
    action_confidence = _masked_mean(
        F.binary_cross_entropy(
            output.action.confidence,
            (action.type != 0).to(output.action.confidence.dtype),
            reduction="none",
        ),
        target.action_mask,
    )
    action_coord = _masked_mean(
        F.smooth_l1_loss(
            output.action.coordinates,
            action.coordinates.to(output.action.coordinates.dtype),
            reduction="none",
        ),
        action.coordinate_mask & target.action_mask[:, None],
    )
    action_scroll = _masked_mean(
        F.smooth_l1_loss(
            output.action.scroll_delta,
            action.scroll_delta.to(output.action.scroll_delta.dtype),
            reduction="none",
        ),
        (action.scroll_mask & target.action_mask)[:, None].expand_as(
            output.action.scroll_delta
        ),
    )
    action_duration = _masked_mean(
        F.smooth_l1_loss(
            output.action.duration_ms / 1_000.0,
            action.duration_ms.to(output.action.duration_ms.dtype) / 1_000.0,
            reduction="none",
        ),
        action.duration_mask & target.action_mask,
    )
    text_shape = output.action.text_logits.shape
    action_text_values = F.cross_entropy(
        output.action.text_logits.reshape(-1, text_shape[-1]),
        action.text_tokens.reshape(-1),
        reduction="none",
    ).view_as(action.text_tokens)
    action_text = _masked_mean(
        action_text_values, action.text_mask & target.action_mask[:, None]
    )
    action_keys = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.action.key_logits,
            action.key_mask.to(output.action.key_logits.dtype),
            reduction="none",
        ),
        ((action.type == 7) & target.action_mask)[:, None].expand_as(
            output.action.key_logits
        ),
    )
    speech_control = _masked_mean(
        F.cross_entropy(output.controls.speech_logits, controls.speech, reduction="none"),
        target.speech_control_mask,
    )
    action_control = _masked_mean(
        F.cross_entropy(output.controls.action_logits, controls.action, reduction="none"),
        target.action_control_mask,
    )
    cognitive_control = _masked_mean(
        F.cross_entropy(output.controls.cognitive_logits, controls.cognitive, reduction="none"),
        target.cognitive_control_mask,
    )
    memory = _masked_mean(
        F.cross_entropy(output.memory_logits, target.memory_target, reduction="none"),
        target.memory_mask,
    )
    write_budget = output.latent_gate.mean()
    total = (
        speech_loss
        + action_type
        + 0.1 * action_confidence
        + action_coord
        + action_scroll
        + action_duration
        + action_text
        + action_keys
        + 0.25 * (speech_control + action_control + cognitive_control)
        + memory
        + 0.01 * write_budget
    )
    return {
        "total": total,
        "speech": speech_loss,
        "action_type": action_type,
        "action_confidence": action_confidence,
        "action_coord": action_coord,
        "action_scroll": action_scroll,
        "action_duration": action_duration,
        "action_text": action_text,
        "action_keys": action_keys,
        "control_speech": speech_control,
        "control_action": action_control,
        "control_cognitive": cognitive_control,
        "memory": memory,
        "latent_write": write_budget,
    }
