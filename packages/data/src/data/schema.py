from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryIdentity:
    dataset_scale: str
    sample_kind: str
    task_id: str
    environment_id: str
    environment_version: str
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not all((self.dataset_scale, self.sample_kind, self.task_id)):
            raise ValueError("trajectory identity fields are required")


def validate_metadata(metadata: dict[str, Any]) -> None:
    required = ("dataset_scale", "sample_kind", "task_id", "environment_id", "environment_version")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"trajectory metadata is missing: {missing}")
