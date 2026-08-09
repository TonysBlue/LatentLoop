from __future__ import annotations

import pytest
from harness.deployment.qemu import create_backend


def test_production_deployment_requires_real_device_endpoints() -> None:
    with pytest.raises(ValueError, match="base_image"):
        create_backend({})


def test_production_deployment_does_not_accept_missing_evaluator(tmp_path) -> None:
    config = {
        "base_image": str(tmp_path / "base.qcow2"),
        "runtime_root": str(tmp_path / "runtime"),
        "spice_screen_command": '["screen-bridge"]',
        "spice_audio_capture_command": '["audio-bridge"]',
        "spice_audio_playback_command": '["audio-bridge"]',
    }
    with pytest.raises(ValueError, match="evaluator_command"):
        create_backend(config)
