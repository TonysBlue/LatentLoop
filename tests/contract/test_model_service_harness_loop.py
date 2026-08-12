from __future__ import annotations

import threading
import time

from contracts import (
    ActuationSignal,
    ControlKind,
    ControlSignal,
    MicSignal,
    ObservationSignal,
    ScreenSignal,
    SpeechSignal,
)
from model_service.client import UnixModelServiceClient
from model_service.transport.server import UnixModelServer


class FakeModelService:
    def identity(self) -> dict[str, str]:
        return {"service": "fake", "protocol_version": "realtime-v2"}

    def open_session(self, session_id: str) -> str:
        return session_id

    def close_session(self, session_id: str) -> None:
        return None

    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        return ActuationSignal(
            observation.session_id,
            observation.unit_index,
            SpeechSignal(b"y" * 7680, silent=True),
            (ControlSignal(ControlKind.NOOP, f"{observation.session_id}-event"),),
        )


def test_model_service_and_harness_use_one_physical_signal_frame(tmp_path) -> None:
    socket_path = tmp_path / "model.sock"
    server = UnixModelServer(FakeModelService(), str(socket_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if socket_path.exists():
            break
        time.sleep(0.01)
    client = UnixModelServiceClient(str(socket_path))
    observation = ObservationSignal(
        "session",
        0,
        0,
        80,
        MicSignal(b"x" * 7680),
        ScreenSignal(b"rgb", 1, 1),
    )
    assert client.infer(observation).controls[0].kind is ControlKind.NOOP
    assert client.open_session("session") == "session"
    assert client.identity()["protocol_version"] == "realtime-v2"
    client.close_session("session")
