from __future__ import annotations

import pytest
from data.schema import TrajectoryIdentity, validate_metadata


def test_current_trajectory_identity_has_no_schema_version() -> None:
    identity = TrajectoryIdentity("canary", "supervised_episode", "task", "qemu", "1")
    assert identity.dataset_scale == "canary"


def test_current_metadata_validation_fails_closed() -> None:
    validate_metadata(
        {
            "dataset_scale": "canary",
            "sample_kind": "supervised_episode",
            "task_id": "task",
            "environment_id": "qemu",
            "environment_version": "1",
        }
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_metadata({})
