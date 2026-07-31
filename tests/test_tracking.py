from __future__ import annotations

from latentloop.config import ProjectConfig
from latentloop.tracking import Tracker


def test_tracker_falls_back_to_offline_without_server(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.tracking.enabled = True
    smoke_config.tracking.mode = "online"
    smoke_config.tracking.base_url = "http://127.0.0.1:1"
    tracker = Tracker(
        smoke_config,
        "test",
        "smoke",
        parameter_count=1,
        data_identity="test-data",
    )
    try:
        assert tracker.effective_mode == "offline"
        assert tracker.media_allowed()
    finally:
        tracker.finish()
