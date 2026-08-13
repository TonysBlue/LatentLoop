from __future__ import annotations

import inspect

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.latentloop import DeltaTimeEncoder, WorldStateUpdate


def test_world_state_update_is_single_state_transition(smoke_config) -> None:
    updater = WorldStateUpdate(
        smoke_config.model.model_dim,
        smoke_config.model.latent_dim,
        smoke_config.model.num_heads,
        smoke_config.model.latent_slots,
    )
    latent = torch.zeros(2, smoke_config.model.latent_slots, smoke_config.model.latent_dim)
    hidden = torch.randn(2, smoke_config.model.tokens_per_unit, smoke_config.model.model_dim)

    output = updater(latent, hidden)

    assert output.shape == latent.shape
    assert list(inspect.signature(updater.forward).parameters) == ["latent", "previous_hidden"]
    assert not hasattr(updater, "predicted_latent")
    assert not hasattr(updater, "corrected_latent")


def test_world_state_update_gate_is_per_slot_and_per_dimension(smoke_config) -> None:
    updater = WorldStateUpdate(
        smoke_config.model.model_dim,
        smoke_config.model.latent_dim,
        smoke_config.model.num_heads,
        smoke_config.model.latent_slots,
    )
    assert updater.gate.out_features == smoke_config.model.latent_dim
    assert updater.gate_bias.item() == -2.0


def test_delta_time_encoder_ignores_absolute_timestamp(smoke_config) -> None:
    encoder = DeltaTimeEncoder(
        smoke_config.model.model_dim,
        bands=smoke_config.model.delta_time_fourier_bands,
        base_period_ms=smoke_config.model.delta_time_base_period_ms,
    )
    delta = torch.tensor([80.0, 500.0])

    first = encoder(delta)
    second = encoder(delta)

    assert first.shape == (2, 1, smoke_config.model.model_dim)
    assert torch.equal(first, second)


def test_delta_time_changes_backbone_but_not_world_state_update(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    unit = episode.units[0]
    state = model.initial_state(1, "cpu")
    state.hidden = torch.randn_like(state.hidden)
    state.latent = torch.randn_like(state.latent)

    short = unit.to("cpu")
    long = unit.to("cpu")
    short.delta_ms = torch.tensor([80])
    long.delta_ms = torch.tensor([1_000])

    short_output = model(short, state)
    long_output = model(long, state)

    assert torch.equal(short_output.state.latent, long_output.state.latent)
    assert not torch.equal(short_output.hidden, long_output.hidden)
