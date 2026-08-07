from __future__ import annotations

import pytest

from latentloop.config import load_config


def test_load_config_with_override() -> None:
    config = load_config("configs/smoke.yaml", ["model.latent_slots=6"])
    assert config.model.latent_slots == 6
    assert config.model.tokens_per_unit == config.model.audio_tokens + 3
    assert config.data.unit_ms == 80
    assert config.data.codec_frame_rate == 12.5


def test_local_dev_and_production_profiles_are_explicit() -> None:
    local = load_config("configs/local-dev.yaml")
    production = load_config("configs/production.yaml")
    assert local.data.dataset == "synthetic"
    assert production.data.dataset == "production"
    assert production.model.model_dim == 896


def test_config_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config("configs/smoke.yaml", ["model.model_dim=63"])


def test_direct_speech_requires_one_codec_frame_per_tick() -> None:
    with pytest.raises(ValueError, match="exactly one speech frame"):
        load_config("configs/smoke.yaml", ["model.speech_frames_per_unit=2"])


def test_config_rejects_invalid_speech_control_weights() -> None:
    with pytest.raises(ValueError, match="five positive"):
        load_config(
            "configs/smoke.yaml",
            ["training.speech_control_class_weights=[1,1,1]"],
        )


def test_config_rejects_non_positive_speech_control_loss_weight() -> None:
    with pytest.raises(ValueError, match="speech_control_loss_weight must be positive"):
        load_config(
            "configs/smoke.yaml",
            ["training.speech_control_loss_weight=0"],
        )
