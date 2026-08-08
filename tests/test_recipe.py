from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from latentloop.config import load_config
from latentloop.recipe import load_recipe, run_recipe


@dataclass
class _Result:
    split: str
    episodes: int = 1


def test_stage_configs_inherit_complete_profiles() -> None:
    canary = load_config("configs/stages/canary-pretrain.yaml")
    pilot = load_config("configs/stages/pilot-sft.yaml")
    production = load_config("configs/stages/production-rl.yaml")

    assert canary.data.dataset == "canary"
    assert pilot.data.dataset == "pilot"
    assert pilot.training.backbone_train_mode == "all"
    assert pilot.training.stage == "sft"
    assert production.training.objective == "grpo"
    assert production.data.dataset == "production"
    assert production.model.model_dim == 896


@pytest.mark.parametrize("scale", ["canary", "pilot", "production"])
def test_formal_recipes_have_the_same_three_stages(scale: str) -> None:
    recipe = load_recipe(f"configs/recipes/{scale}.yaml")
    assert [stage.name for stage in recipe.stages] == ["pretrain", "sft", "rl"]
    configs = [load_config(Path("configs/recipes") / stage.config) for stage in recipe.stages]
    assert [(config.training.stage, config.training.objective) for config in configs] == [
        ("pretrain", "supervised"),
        ("sft", "supervised"),
        ("rl", "grpo"),
    ]
    assert all(config.training.backbone_train_mode == "all" for config in configs)


def test_recipe_requires_unique_stages(tmp_path: Path) -> None:
    recipe = tmp_path / "duplicate.yaml"
    recipe.write_text(
        "name: bad\ndataset: pilot\nstages:\n"
        "  - {name: head, config: a.yaml}\n"
        "  - {name: head, config: b.yaml}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_recipe(recipe)


def test_recipe_chains_stage_checkpoint_and_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = tmp_path / "configs"
    recipes = configs / "recipes"
    stages = configs / "stages"
    recipes.mkdir(parents=True)
    stages.mkdir()
    base = Path("configs/smoke.yaml").resolve()
    for name in ("first", "second"):
        (stages / f"{name}.yaml").write_text(
            f"defaults:\n  - {base}\n  - _self_\n"
            "data:\n  dataset: direct-speech-overfit\n  source: webdataset\n"
            "  shards: /unused/train-*.tar\n  manifest: /unused/train-manifest.jsonl\n"
            f"training:\n  max_updates: {1 if name == 'first' else 2}\n"
            "runtime:\n  data_root: " + str(tmp_path / "datasets") + "\n",
            encoding="utf-8",
        )
    recipe = recipes / "test.yaml"
    recipe.write_text(
        "name: gate\ndataset: direct-speech-overfit\nstages:\n"
        "  - {name: first, config: ../stages/first.yaml}\n"
        "  - {name: second, config: ../stages/second.yaml}\n"
        "evaluation:\n  validation_after_each_stage: true\n  test_after_final_stage: true\n",
        encoding="utf-8",
    )
    calls: list[tuple[str | None, str | None]] = []

    def fake_train(config, *, resume=None, init_from=None):
        calls.append((resume, init_from))
        checkpoint = (
            config.runtime.root_path()
            / "checkpoints"
            / f"step-{config.training.max_updates:08d}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "train_state": {"update": config.training.max_updates},
            "metrics": {},
            "tracking": {},
        }

    monkeypatch.setattr("latentloop.recipe.train", fake_train)
    monkeypatch.setattr(
        "latentloop.recipe.evaluate_checkpoint",
        lambda config, checkpoint, split: _Result(split),
    )

    result = run_recipe(recipe)

    assert calls[0] == (None, None)
    assert calls[1][0] is None
    assert calls[1][1] == result["stages"][0]["checkpoint"]
    assert result["stages"][0]["validation"]["split"] == "validation"
    assert result["stages"][1]["test"]["split"] == "test"


def test_recipe_run_id_isolated_and_resume_validates_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "stage.yaml"
    base = Path("configs/smoke.yaml").resolve()
    config.write_text(
        f"defaults:\n  - {base}\n  - _self_\n"
        "data:\n  dataset: direct-speech-overfit\n  source: webdataset\n"
        "  shards: /unused/train-*.tar\n  manifest: /unused/train-manifest.jsonl\n"
        "training:\n  max_updates: 1\n"
        "runtime:\n  data_root: " + str(tmp_path / "datasets") + "\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: isolated\ndataset: direct-speech-overfit\n"
        f"stages:\n  - {{name: train, config: {config.name}}}\n"
        "evaluation:\n  validation_after_each_stage: false\n  test_after_final_stage: false\n",
        encoding="utf-8",
    )
    calls: list[tuple[str | None, str | None]] = []

    def fake_train(stage_config, *, resume=None, init_from=None):
        calls.append((resume, init_from))
        checkpoint = stage_config.runtime.root_path() / "checkpoints" / "step-00000001.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"train_state": {"update": 1}, "metrics": {}, "tracking": {}}

    monkeypatch.setattr("latentloop.recipe.train", fake_train)
    first = run_recipe(recipe, run_id="one")
    second = run_recipe(recipe, run_id="two")
    assert first["run_id"] == "one"
    assert second["run_id"] == "two"
    assert first["stages"][0]["checkpoint"] != second["stages"][0]["checkpoint"]
    assert calls == [(None, None), (None, None)]


def test_recipe_rejects_mismatched_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "stage.yaml"
    base = Path("configs/smoke.yaml").resolve()
    config.write_text(
        f"defaults:\n  - {base}\n  - _self_\n"
        "data:\n  dataset: direct-speech-overfit\n  source: webdataset\n"
        "  shards: /unused/train-*.tar\n  manifest: /unused/train-manifest.jsonl\n"
        "training:\n  max_updates: 1\n"
        "runtime:\n  data_root: " + str(tmp_path / "datasets") + "\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: mismatch\ndataset: direct-speech-overfit\n"
        f"stages:\n  - {{name: train, config: {config.name}}}\n"
        "evaluation:\n  validation_after_each_stage: false\n  test_after_final_stage: false\n",
        encoding="utf-8",
    )
    stage_root = tmp_path / "experiments" / "mismatch" / "run" / "train"
    checkpoint = stage_root / "checkpoints" / "step-00000001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong")
    monkeypatch.setattr(
        "latentloop.recipe.train", lambda *args, **kwargs: pytest.fail("must reject")
    )
    with pytest.raises(ValueError, match="does not match"):
        run_recipe(recipe, run_id="run")
