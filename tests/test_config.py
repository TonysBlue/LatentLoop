from __future__ import annotations

import pytest

from latentloop.config import load_config


def test_load_config_with_override() -> None:
    config = load_config("configs/smoke.yaml", ["model.latent_slots=6"])
    assert config.model.latent_slots == 6
    assert config.model.tokens_per_unit == config.model.audio_tokens + 3
    assert config.data.unit_ms == 80
    assert config.data.codec_frame_rate == 12.5


def test_config_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config("configs/smoke.yaml", ["model.model_dim=63"])


def test_e2_config_requires_one_codec_frame_per_tick() -> None:
    with pytest.raises(ValueError, match="exactly one speech frame"):
        load_config("configs/smoke.yaml", ["model.speech_frames_per_unit=2"])
