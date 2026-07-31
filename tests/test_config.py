from __future__ import annotations

import pytest

from latentloop.config import load_config


def test_load_config_with_override() -> None:
    config = load_config("configs/smoke.yaml", ["model.latent_slots=6"])
    assert config.model.latent_slots == 6
    assert config.model.tokens_per_unit == config.model.audio_tokens + 3


def test_config_rejects_incompatible_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config("configs/smoke.yaml", ["model.model_dim=63"])
