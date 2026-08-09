"""Training System public API with lazy imports to keep Model Core acyclic."""

from typing import Any


def run(config: Any, **kwargs: Any) -> Any:
    from training.api import run as implementation

    return implementation(config, **kwargs)


def run_recipe(path: Any, overrides: Any = None, **kwargs: Any) -> Any:
    from training.recipe import run_recipe as implementation

    return implementation(path, overrides=overrides, **kwargs)


def train(config: Any, **kwargs: Any) -> Any:
    from training.training import train as implementation

    return implementation(config, **kwargs)


def evaluate_checkpoint(config: Any, checkpoint: Any, **kwargs: Any) -> Any:
    from training.evaluation import evaluate_checkpoint as implementation

    return implementation(config, checkpoint, **kwargs)


def evaluate_overfit_checkpoint(config: Any, checkpoint: Any, **kwargs: Any) -> Any:
    from training.evaluation import evaluate_overfit_checkpoint as implementation

    return implementation(config, checkpoint, **kwargs)


__all__ = ["run", "run_recipe", "train", "evaluate_checkpoint", "evaluate_overfit_checkpoint"]
