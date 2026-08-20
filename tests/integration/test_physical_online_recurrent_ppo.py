from __future__ import annotations

import base64
import json
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
    ScreenSignal,
)
from contracts.framing import frame, read_frame
from harness.transport.control import HarnessControlServer
from model import StreamingLatentLoop
from runtime.codec import CodecIdentity
from runtime.codec_worker import receive_message, send_message
from runtime.config import load_config
from training.checkpoint import CheckpointManager, CheckpointMetadata, DataCursor
from training.training import train_online_ppo


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

    def __init__(self) -> None:
        self.start_count = 0

    def start_lifetime_session(self, initial_snapshot_id: str, seed: int, session_id: str):
        self.start_count += 1
        self.session_id = session_id
        return ObservationSignal(
            session_id,
            0,
            0,
            80,
            MicSignal(np.zeros(1920, dtype=np.float32).tobytes()),
            ScreenSignal(b"x" * (224 * 224 * 3), 224, 224),
        )

    def apply(self, output: ActuationSignal):
        assert isinstance(output, ActuationSignal)
        self.seen.append(output)
        receipt = EnvironmentReceipt(output.session_id, output.unit_index, True)
        return (
            ObservationSignal(
                output.session_id, output.unit_index + 1, (output.unit_index + 1) * 80, 80,
                MicSignal(np.zeros(1920, dtype=np.float32).tobytes()),
                ScreenSignal(b"x" * (224 * 224 * 3), 224, 224),
            ),
            receipt,
        )

    def close(self):
        return None


class _RewardWorker:
    def __init__(self, path: Path, config, *, delayed: bool = False) -> None:
        self.path = path
        self.config = config
        self.delayed = delayed
        self.chain_by_unit: dict[int, str] = {}
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(16)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            block = connection.recv(size - len(output))
            if not block:
                raise ConnectionError
            output.extend(block)
        return bytes(output)

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                try:
                    payload = read_frame(
                        lambda size, active=connection: self._read_exact(active, size)
                    )
                    request = json.loads(payload)
                    if request["operation"] == "identity":
                        response = {
                            "ok": True,
                            "identity": {
                                "spec_id": self.config.training.rl.reward_spec_id,
                                "judge_model_id": self.config.training.rl.judge_model_id,
                                "judge_revision": self.config.training.rl.judge_revision,
                                "rubric_sha256": self.config.training.rl.rubric_sha256,
                            },
                        }
                    else:
                        assert base64.b64decode(request["observation"], validate=True)
                        unit_index = int(request["unit_index"])
                        self.chain_by_unit[unit_index] = request[
                            "observation_chain_sha256"
                        ]
                        events = []
                        event_unit = 9 if self.delayed else 7
                        if unit_index == event_unit:
                            events.append(
                                {
                                    "event_id": "goal-success",
                                    "lineage_id": request["lineage_id"],
                                    "session_id": request["session_id"],
                                    "goal_id": "goal",
                                    "goal_start_unit": 0,
                                    "outcome_unit": 7,
                                    "evidence_start_unit": 6,
                                    "evidence_end_unit": 7,
                                    "status": "finalized",
                                    "outcome": "success",
                                    "reward": {
                                        "task": 1.0,
                                        "speech_quality": 0.5,
                                        "latency_quality": 0.5,
                                        "action_efficiency": 0.5,
                                        "safety_quality": 0.0,
                                    },
                                    "spec_id": self.config.training.rl.reward_spec_id,
                                    "judge_model_id": self.config.training.rl.judge_model_id,
                                    "judge_revision": self.config.training.rl.judge_revision,
                                    "rubric_sha256": self.config.training.rl.rubric_sha256,
                                    "observation_chain_end_sha256": self.chain_by_unit[7],
                                }
                            )
                        response = {
                            "ok": True,
                            "finalized_through_unit": (
                                min(unit_index, 6)
                                if self.delayed and unit_index < event_unit
                                else unit_index
                            ),
                            "events": events,
                        }
                    connection.sendall(frame(json.dumps(response).encode()))
                except (ConnectionError, OSError):
                    return

    def close(self) -> None:
        self.server.close()
        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(str(self.path))
        except OSError:
            pass
        self.thread.join(timeout=1)


def test_formal_physical_online_recurrent_ppo_runs_one_update(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path / 'datasets'}",
            f"runtime.experiment_root={tmp_path / 'experiment'}",
            "data.dataset=synthetic",
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.objective=ppo",
            "training.stage=rl",
            "training.max_updates=1",
            "training.rl.ppo_window_units=8",
            "training.rl.candidate_max_reference_kl=1000000000",
            "training.rl.candidate_max_eval_loss_ratio=1000000000",
            "training.rl.judge_revision=test-revision",
            "training.rl.rubric_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
    session_manifest = tmp_path / "sessions.jsonl"
    session_manifest.write_text(
        '{"session_id":"life","initial_snapshot_id":"snapshot","seed":17}\n',
        encoding="utf-8",
    )
    config.training.rl.session_manifest = str(session_manifest)
    config.training.rl.timeline_root = str(tmp_path / "timeline")
    config.training.rl.environment_socket = str(tmp_path / "harness.sock")
    config.training.rl.codec_socket = str(tmp_path / "codec.sock")
    config.training.rl.reward_socket = str(tmp_path / "reward.sock")
    config.training.rl.environment_id = "test-physical"
    config.training.rl.environment_version = "1"
    config.training.rl.environment_protocol_version = "realtime-v2"
    identity = CodecIdentity(
        "synthetic-codec-v1", "synthetic", "synthetic", sample_rate=24000,
        frame_samples=1920, codebooks=2, codebook_size=32
    )
    codec = _CodecWorker(Path(config.training.rl.codec_socket), identity)
    reward = _RewardWorker(
        Path(config.training.rl.reward_socket), config, delayed=True
    )
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
    sft_config = load_config(
        "configs/smoke.yaml",
        [
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=sft",
            "training.objective=supervised",
        ],
    )
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
        config=sft_config.as_dict(),
    )
    init_path = tmp_path / "init" / "sft.pt"
    try:
        result = train_online_ppo(
            config,
            init_from=str(init_path),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=1,
        )
    finally:
        server.close()
        codec.close()
        reward.close()
    assert result["train_state"]["update"] == 1
    assert backend.seen
    assert all(isinstance(output, ActuationSignal) for output in backend.seen)
    assert all(not hasattr(output, "action_tokens") for output in backend.seen)
    assert result["train_state"]["consumed_units"] > config.training.rl.ppo_window_units
    assert result["train_state"]["dropped_windows"] >= 1
    assert result["metrics"]["rl/reward_mean"] > 0


def test_online_recurrent_ppo_resumes_the_same_lifetime_session(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(
        "configs/smoke.yaml",
        [
            f"runtime.data_root={tmp_path / 'datasets'}",
            f"runtime.experiment_root={tmp_path / 'experiment'}",
            "data.dataset=synthetic",
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.objective=ppo",
            "training.stage=rl",
            "training.max_updates=2",
            "training.rl.ppo_window_units=8",
            "training.rl.candidate_max_reference_kl=1000000000",
            "training.rl.candidate_max_eval_loss_ratio=1000000000",
            "training.rl.judge_revision=test-revision",
            "training.rl.rubric_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "tracking.enabled=false",
            "tracking.mode=disabled",
        ],
    )
    session_manifest = tmp_path / "sessions.jsonl"
    session_manifest.write_text(
        '{"session_id":"life","initial_snapshot_id":"snapshot","seed":17}\n',
        encoding="utf-8",
    )
    config.training.rl.session_manifest = str(session_manifest)
    config.training.rl.timeline_root = str(tmp_path / "timeline")
    config.training.rl.environment_socket = str(tmp_path / "harness.sock")
    config.training.rl.codec_socket = str(tmp_path / "codec.sock")
    config.training.rl.reward_socket = str(tmp_path / "reward.sock")
    config.training.rl.environment_id = "test-physical"
    config.training.rl.environment_version = "1"
    config.training.rl.environment_protocol_version = "realtime-v2"
    identity = CodecIdentity(
        "synthetic-codec-v1",
        "synthetic",
        "synthetic",
        sample_rate=24000,
        frame_samples=1920,
        codebooks=2,
        codebook_size=32,
    )
    codec = _CodecWorker(Path(config.training.rl.codec_socket), identity)
    reward = _RewardWorker(Path(config.training.rl.reward_socket), config)
    backend = _PhysicalBackend()
    server = HarnessControlServer(
        lambda: backend,
        config.training.rl.environment_socket,
        expected_environment_id="test-physical",
        expected_environment_version="1",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(100):
        if Path(config.training.rl.environment_socket).exists():
            break
    init = StreamingLatentLoop(config.model)
    init_manager = CheckpointManager(tmp_path / "init")
    sft_config = load_config(
        "configs/smoke.yaml",
        [
            "data.audio_sample_rate=24000",
            "data.unit_audio_samples=1920",
            "training.stage=sft",
            "training.objective=supervised",
        ],
    )
    init_manager.save(
        "sft",
        model=init,
        optimizer=torch.optim.AdamW(init.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
        recurrent_state=None,
        train_state={},
        data_cursor=DataCursor(),
        metadata=CheckpointMetadata(
            "synthetic",
            identity.codec_id,
            identity.weight_sha256,
            "test",
            codec_revision=identity.revision,
            stage="sft",
        ),
        config=sft_config.as_dict(),
    )
    sft_path = tmp_path / "init" / "sft.pt"
    try:
        first = train_online_ppo(
            config,
            init_from=str(sft_path),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=1,
        )
        checkpoint = tmp_path / "experiment" / "checkpoints" / "step-00000001.pt"
        first_units = first["train_state"]["consumed_units"]
        resumed = train_online_ppo(
            config,
            resume=str(checkpoint),
            model=StreamingLatentLoop(config.model),
            stop_after_updates=2,
        )
    finally:
        server.close()
        codec.close()
        reward.close()
    assert resumed["train_state"]["update"] == 2
    assert resumed["train_state"]["consumed_units"] > first_units
    assert backend.start_count == 1
