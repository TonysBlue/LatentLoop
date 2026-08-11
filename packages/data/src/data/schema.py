from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 6


@dataclass(frozen=True, slots=True)
class TrajectoryIdentity:
    schema_version: int
    dataset_scale: str
    sample_kind: str
    task_id: str
    environment_id: str
    environment_version: str
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("trajectory data must use schema v6")
        if not all((self.dataset_scale, self.sample_kind, self.task_id)):
            raise ValueError("trajectory identity fields are required")


def validate_metadata(metadata: dict[str, Any]) -> None:
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported trajectory schema; rebuild from source")
    required = ("dataset_scale", "sample_kind", "task_id", "environment_id", "environment_version")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"schema v6 metadata is missing: {missing}")
