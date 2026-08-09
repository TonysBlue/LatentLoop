from __future__ import annotations

import base64
import threading
import time

import pytest
from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    ScreenSignal,
    SpeechSignal,
)
from contracts.protocol import actuation_to_payload
from harness.transport.control import HarnessControlClient, HarnessControlServer


class FakeBackend:
    environment_id = "isolated-qemu-v1"
    environment_version = "1"

    def reset(self, task_id: str, seed: int, session_id: str) -> ObservationSignal:
        return ObservationSignal(
            session_id,
            0,
            0,
            80,
            MicSignal(b"x" * 7680),
            ScreenSignal(b"rgb", 1, 1, 0),
        )

    def apply(self, output: ActuationSignal):
        return (
            ObservationSignal(
                output.session_id,
                output.unit_index + 1,
                80,
                80,
                MicSignal(b"x" * 7680),
                ScreenSignal(b"rgb", 1, 1, output.unit_index + 1),
            ),
            EnvironmentReceipt(output.session_id, output.unit_index, True),
        )

    def evaluate(self, task_id: str):
        from contracts import RewardBreakdown

        return RewardBreakdown(1, 0, 0, 0, 0)

    def close(self):
        return None


class CountingBackend(FakeBackend):
    def __init__(self, closed: list[str]) -> None:
        self.closed = closed

    def close(self):
        self.closed.append("closed")


def test_harness_control_socket_uses_physical_signals(tmp_path) -> None:
    path = tmp_path / "harness.sock"
    thread = threading.Thread(
        target=HarnessControlServer(lambda: FakeBackend(), str(path)).serve_forever,
        daemon=True,
    )
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.01)
    client = HarnessControlClient(str(path))
    assert client.identity()["environment_id"] == "isolated-qemu-v1"
    assert client.reset("task", 7, "session").unit_index == 0
    output = ActuationSignal("session", 0, SpeechSignal(b"y" * 7680, silent=True))
    next_observation, receipt = client.apply(output)
    assert next_observation.unit_index == 1
    assert receipt.accepted
    assert client.evaluate("task", "session").task == 1
    client.close("session")


def test_harness_control_rejects_order_errors_and_closes_replaced_session() -> None:
    closed: list[str] = []
    server = HarnessControlServer(lambda: CountingBackend(closed), "/tmp/unused.sock")
    server._handle({"operation": "reset", "task_id": "task", "seed": 1, "session_id": "session"})
    server._handle({"operation": "reset", "task_id": "task", "seed": 2, "session_id": "session"})
    assert closed == ["closed"]
    with pytest.raises(ValueError, match="out of order"):
        server._handle(
            {
                "operation": "apply",
                "session_id": "session",
                "actuation": base64.b64encode(
                    actuation_to_payload(
                        ActuationSignal("session", 4, SpeechSignal(b"y" * 7680, silent=True))
                    )
                ).decode(),
            }
        )
    server.close()
    assert closed == ["closed", "closed"]
