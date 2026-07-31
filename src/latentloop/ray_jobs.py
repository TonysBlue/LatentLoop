from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from latentloop.config import ProjectConfig


def generate_synthetic_with_ray(
    config: ProjectConfig,
    output_pattern: str,
    *,
    object_store_bytes: int = 1_073_741_824,
    episodes_per_batch: int = 8,
) -> list[dict[str, Any]]:
    """Generate independent episodes on CPU actors; never reserve a GPU."""
    if episodes_per_batch < 1:
        raise ValueError("episodes_per_batch must be positive")
    # Ray 2.56's uv hook drops optional extras when recreating worker environments.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        import ray
    except ImportError as error:
        raise RuntimeError("install the ray extra with: uv sync --extra ray") from error

    from latentloop.data.synthetic import SyntheticEpisodeDataset
    from latentloop.data.webdataset import write_episode_shards

    ray.init(
        num_cpus=min(os.cpu_count() or 1, 4),
        num_gpus=0,
        object_store_memory=object_store_bytes,
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    @ray.remote(num_cpus=1, num_gpus=0)
    def make_episode(index: int):
        return SyntheticEpisodeDataset(config.data, config.model).make_episode(index)

    try:

        def episodes():
            for start in range(0, config.data.train_episodes, episodes_per_batch):
                stop = min(start + episodes_per_batch, config.data.train_episodes)
                references = [make_episode.remote(index) for index in range(start, stop)]
                yield from ray.get(references)

        return write_episode_shards(episodes(), output_pattern)
    finally:
        ray.shutdown()


def write_ray_report(path: str | Path, manifest: list[dict[str, Any]]) -> None:
    report = {
        "episodes": len(manifest),
        "units": sum(int(item["units"]) for item in manifest),
        "gpu_workers": 0,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
