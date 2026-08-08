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
        output = model(unit, state, unit.speech_codes)
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
    assert model.speech_head.depth_embeddings[0].weight.grad is not None
    assert model.action_head.output.weight.grad is not None
    assert model.speech_head.mode.weight.grad is not None


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
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)

    compute_losses(output, unit)["total"].backward()

    assert model.layers[0].self_attention.qkv.weight.grad is not None


def test_speech_loss_averages_over_valid_codec_tokens(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)
    output.speech_codec_logits = torch.zeros_like(output.speech_codec_logits)
    output.speech_mode_logits = torch.zeros_like(output.speech_mode_logits)

    speech_loss = compute_losses(output, unit)["speech"]

    expected = math.log(smoke_config.model.speech_codebook_size) + math.log(2)
    assert torch.isclose(speech_loss, torch.tensor(expected), atol=1e-4)


def test_speech_loss_follows_model_dtype(smoke_config: ProjectConfig) -> None:
    model = StreamingLatentLoop(smoke_config.model).half()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    unit.mic_audio = unit.mic_audio.half()
    unit.screen = unit.screen.half()
    output = model(unit, model.initial_state(1, "cpu"), unit.speech_codes)

    losses = compute_losses(output, unit)

    assert torch.isfinite(losses["speech"])
