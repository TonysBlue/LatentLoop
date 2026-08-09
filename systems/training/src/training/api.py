"""Training System API for Pretrain, SFT, and Online GRPO."""

from __future__ import annotations

from training.recipe import run_recipe as _run_recipe
from training.training import train as _train


def run(config, **kwargs):
    """Shared training entry point for Pretrain, SFT, and Online GRPO."""
    return _train(config, **kwargs)


def run_recipe(path, overrides=None, **kwargs):
    return _run_recipe(path, overrides, **kwargs)
