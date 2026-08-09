from __future__ import annotations

import json
from pathlib import Path

import pytest
from data.ray import generate_synthetic_with_ray, write_ray_report
from runtime.config import ProjectConfig


def test_ray_report_declares_no_gpu_workers(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    write_ray_report(path, [{"units": 3}, {"units": 4}])
    report = json.loads(path.read_text())
    assert report == {"episodes": 2, "units": 7, "gpu_workers": 0}


def test_ray_generation_rejects_empty_batches(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    with pytest.raises(ValueError, match="episodes_per_batch"):
        generate_synthetic_with_ray(
            smoke_config,
            str(tmp_path / "train-%06d.tar"),
            episodes_per_batch=0,
        )
