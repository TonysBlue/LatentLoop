from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop


def test_vision_encoder_emits_sixteen_spatial_tokens(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    tokens = model.vision_encoder(episode.units[0].screen)

    assert tokens.shape == (1, smoke_config.model.vision_tokens, smoke_config.model.model_dim)
    assert smoke_config.model.vision_tokens == 16


def test_static_and_dynamic_frames_use_the_same_visual_path(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    static = torch.zeros(1, 3, 224, 224)
    dynamic = static.clone()
    dynamic[:, :, 40:80, 80:120] = 1

    static_tokens = model.vision_encoder(static)
    dynamic_tokens = model.vision_encoder(dynamic)

    assert static_tokens.shape == dynamic_tokens.shape == (1, 16, 64)
    assert not torch.equal(static_tokens, dynamic_tokens)


def test_unit_token_layout_contains_all_visual_positions(smoke_config) -> None:
    assert smoke_config.model.tokens_per_unit == smoke_config.model.audio_tokens + 18


def test_action_head_consumes_backbone_visual_hidden(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    state = model.initial_state(1, "cpu")
    output = model(episode.units[0], state)

    assert output.hidden.shape[1] == smoke_config.model.tokens_per_unit
    assert model.action_head.visual_attention is not None
    output.action.kind_logits.sum().backward()
    assert model.action_head.visual_attention.query_proj.weight.grad is not None
    assert model.vision_encoder.encoder[0].weight.grad is not None


def test_visual_and_temporal_kv_horizons_are_independent(smoke_config) -> None:
    assert smoke_config.model.temporal_kv_units == 4
    assert smoke_config.model.vision_kv_units == 2
    assert smoke_config.model.temporal_kv_units != smoke_config.model.vision_kv_units


def test_action_cell_mapping_uses_four_by_four_visual_positions(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    assert model.action_head.visual_grid_size == 4
    assert model.action_head.local_coordinate_grid_size == 8
    assert model.action_head.coordinate_cell_output.out_features == 8**2
