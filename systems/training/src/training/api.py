"""Training System API.

The legacy tensor implementation remains the source of the shared optimizer
until its files are physically relocated; this package is the only public
system entry point and exposes the full Pretrain/SFT/Online-GRPO dispatcher.
"""

from __future__ import annotations

from latentloop.recipe import run_recipe as _run_recipe
from latentloop.training import train as _train


def run(config, **kwargs):
    """Shared training entry point for Pretrain, SFT, and Online GRPO."""
    return _train(config, **kwargs)


def run_recipe(path, overrides=None, **kwargs):
    return _run_recipe(path, overrides, **kwargs)
