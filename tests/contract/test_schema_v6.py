from __future__ import annotations

import pytest
from data.schema import TrajectoryIdentity, validate_metadata


def test_schema_v7_is_required_without_flat_migration() -> None:
    identity = TrajectoryIdentity(7, "canary", "supervised_episode", "task", "qemu", "1")
    assert identity.schema_version == 7
    with pytest.raises(ValueError, match="schema v7"):
        TrajectoryIdentity(6, "canary", "supervised_episode", "task", "qemu", "1")


def test_schema_v7_metadata_validation_fails_closed() -> None:
    validate_metadata(
        {
            "schema_version": 7,
            "dataset_scale": "canary",
            "sample_kind": "supervised_episode",
            "task_id": "task",
            "environment_id": "qemu",
            "environment_version": "1",
        }
    )
    with pytest.raises(ValueError, match="rebuild from source"):
        validate_metadata({"schema_version": 6})
