from __future__ import annotations

import math

import torch

from latentloop.config import ProjectConfig
from latentloop.data import SyntheticEpisodeDataset
from latentloop.losses import compute_losses
from latentloop.model import StreamingLatentLoop


def test_recurrent_state_is_bounded_and_heads_receive_gradients(
    smoke_config: ProjectConfig,
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    state = model.initial_state(1, "cpu")
    total = torch.tensor(0.0)
    output = None
    for unit in episode.units:
        output = model(unit, state)
        state = output.state
        total = total + compute_losses(output, unit)["total"]

    assert output is not None
    max_tokens = smoke_config.model.kv_units * smoke_config.model.tokens_per_unit
    assert all(cache.key.shape[2] == max_tokens for cache in state.layer_kv)
    assert state.latent.shape == (
        1,
        smoke_config.model.latent_slots,
        smoke_config.model.latent_dim,
    )
    total.backward()
    assert model.audio_encoder.conv.weight.grad is not None
    assert model.vision_encoder.encoder[0].weight.grad is not None
    assert model.latent_updater.gate.weight.grad is not None
    assert model.speech_heads[0].weight.grad is not None
    assert model.action_type_head.weight.grad is not None
    assert model.action_coord_head.weight.grad is not None
    assert model.action_scroll_head.weight.grad is not None
    assert model.action_duration_head.weight.grad is not None
    assert model.action_text_head.weight.grad is not None
    assert model.action_key_head.weight.grad is not None
    assert model.action_confidence_head.weight.grad is not None
    assert model.speech_control_head.weight.grad is not None
    assert model.action_control_head.weight.grad is not None
    assert model.cognitive_control_head.weight.grad is not None
    assert model.memory_probe.weight.grad is not None


def test_detach_breaks_tbptt_graph(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model(unit, model.initial_state(1, "cpu")).state.detach()
    assert state.latent.grad_fn is None
    assert all(cache.key.grad_fn is None for cache in state.layer_kv)


def test_activation_checkpointing_supports_backward(smoke_config: ProjectConfig) -> None:
    smoke_config.model.activation_checkpointing = True
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"))

    compute_losses(output, unit)["total"].backward()

    assert model.layers[0].self_attention.qkv.weight.grad is not None


def test_speech_loss_averages_over_valid_codec_tokens(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"))
    output.speech_logits = torch.zeros_like(output.speech_logits)

    speech_loss = compute_losses(output, unit)["speech"]

    assert torch.isclose(
        speech_loss,
        torch.tensor(math.log(smoke_config.model.speech_codebook_size)),
    )
