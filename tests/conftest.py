from __future__ import annotations

from pathlib import Path

import pytest

from latentloop.config import ProjectConfig, load_config


@pytest.fixture
def smoke_config(tmp_path: Path) -> ProjectConfig:
    return load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path}",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
