from __future__ import annotations

import socket
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    RewardBreakdown,
    ScreenSignal,
)
from harness.transport.control import HarnessControlServer
from model import StreamingLatentLoop
from runtime.codec import CodecIdentity
from runtime.codec_worker import receive_message, send_message
from runtime.config import load_config
from training.checkpoint import CheckpointManager, CheckpointMetadata, DataCursor
from training.training import train_online_grpo


class _CodecWorker:
    def __init__(self, path: Path, identity: CodecIdentity) -> None:
        self.path = path
        self.identity = identity
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(16)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                try:
                    header, payload = receive_message(connection)
                except ConnectionError:
                    return
                operation = header["operation"]
                if operation == "health":
                    send_message(connection, {"ok": True, "identity": asdict(self.identity)})
                elif operation == "reset":
                    send_message(connection, {"ok": True})
                elif operation == "decode_step":
                    codes = np.frombuffer(payload, dtype=np.uint16)
                    values = np.zeros(self.identity.frame_samples, dtype=np.float32)
                    values += float(codes.sum() % 3) / 10
                    send_message(
                        connection,
                        {"ok": True, "dtype": "float32", "shape": [1, self.identity.frame_samples]},
                        values.tobytes(),
                    )
                else:
                    send_message(connection, {"ok": False, "error": "unsupported"})

    def close(self) -> None:
        self.server.close()
        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(str(self.path))
        except OSError:
            pass
        self.thread.join(timeout=1)


class _PhysicalBackend:
    environment_id = "test-physical"
    environment_version = "1"
    seen: list[ActuationSignal] = []

    def reset(self, task_id: str, seed: int, session_id: str):
        self.task_id, self.session_id = task_id, session_id
        self._episode_number = int(session_id.rsplit("-", 1)[-1])
        return ObservationSignal(
            session_id,
            0,
            0,
            80,
            MicSignal(b"x" * 7680),
            ScreenSignal(b"x" * (224 * 224 * 3), 224, 224),
        )

    def apply(self, output: ActuationSignal):
        assert isinstance(output, ActuationSignal)
        self.seen.append(output)
        receipt = EnvironmentReceipt(output.session_id, output.unit_index, True)
        return (
            ObservationSignal(
                output.session_id, output.unit_index + 1, (output.unit_index + 1) * 80, 80,
                MicSignal(b"x" * 7680),
                ScreenSignal(b"x" * (224 * 224 * 3), 224, 224),
            ),
            receipt,
        )

    def evaluate(self, task_id: str):
        task_reward = float(self._episode_number)
        return RewardBreakdown(task_reward, 0, 0, 0, 0)

    def close(self):
        return None


def test_formal_physical_online_grpo_runs_one_update(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path / 'datasets'}",
            f"runtime.experiment_root={tmp_path / 'experiment'}",
            "data.dataset=synthetic",
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.objective=grpo",
            "training.stage=rl",
            "training.max_updates=1",
            "training.rl.group_size=2",
            "training.rl.rollout_horizon_units=1",
            "training.rl.groups_per_update=1",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
    task_manifest = tmp_path / "tasks.jsonl"
    task_manifest.write_text('{"task_id":"task"}\n', encoding="utf-8")
    config.training.rl.task_manifest = str(task_manifest)
    config.training.rl.environment_socket = str(tmp_path / "harness.sock")
    config.training.rl.codec_socket = str(tmp_path / "codec.sock")
    config.training.rl.environment_id = "test-physical"
    config.training.rl.environment_version = "1"
    config.training.rl.environment_protocol_version = "realtime-v2"
    identity = CodecIdentity(
        "synthetic-codec-v1", "synthetic", "synthetic", sample_rate=24000,
        frame_samples=1920, codebooks=2, codebook_size=32
    )
    codec = _CodecWorker(Path(config.training.rl.codec_socket), identity)
    backend = _PhysicalBackend()
    server = HarnessControlServer(
        lambda: backend,
        config.training.rl.environment_socket,
        expected_environment_id="test-physical",
        expected_environment_version="1",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(100):
        if Path(config.training.rl.environment_socket).exists():
            break
    init = StreamingLatentLoop(config.model)
    init_path = tmp_path / "sft.pt"
    manager = CheckpointManager(tmp_path / "init")
    manager.save(
        "sft",
        model=init,
        optimizer=torch.optim.AdamW(init.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata(
            "synthetic", identity.codec_id, identity.weight_sha256, "test",
            codec_revision=identity.revision, stage="sft"
        ),
        config=config.as_dict(),
    )
    init_path = tmp_path / "init" / "sft.pt"
    try:
        result = train_online_grpo(
            config,
            init_from=str(init_path),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=1,
        )
    finally:
        server.close()
        codec.close()
    assert result["train_state"]["update"] == 1
    assert backend.seen
    assert all(isinstance(output, ActuationSignal) for output in backend.seen)
    assert all(not hasattr(output, "action_tokens") for output in backend.seen)
