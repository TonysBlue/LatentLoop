from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal
from harness.environment.adapters import ActuatorAdapter, SensorAdapter
from harness.environment.backend import ComputerBackend


@dataclass(slots=True)
class QemuConfig:
    base_image: Path
    runtime_root: Path
    qemu_binary: str = "qemu-system-x86_64"
    qemu_img_binary: str = "qemu-img"
    kvm: bool = True
    memory_mb: int = 4096
    cpus: int = 2
    readiness_timeout_s: float = 30.0
    qmp_timeout_s: float = 2.0
    remove_overlay_on_close: bool = True
    extra_args: tuple[str, ...] = ()


class QemuBackend(ComputerBackend):
    """QEMU/KVM lifecycle boundary.

    Device-specific screen/audio/input adapters are injected at deployment;
    this class owns snapshot identity and fail-closed process lifecycle.
    """

    environment_id = "isolated-qemu-v1"
    environment_version = "1"

    def __init__(
        self,
        config: QemuConfig,
        observation_factory: SensorAdapter,
        executor: ActuatorAdapter,
    ) -> None:
        if not config.base_image.is_file():
            raise FileNotFoundError(f"QEMU base image is absent: {config.base_image}")
        self.config = config
        self.observation_factory = observation_factory
        self.executor = executor
        self.process: subprocess.Popen[bytes] | None = None
        self.overlay: Path | None = None
        self.qmp_socket: Path | None = None
        self.initial_snapshot_id: str | None = None
        self.session_id: str | None = None
        self._last_unit = -1

    def start_lifetime_session(
        self, initial_snapshot_id: str, seed: int, session_id: str | None = None
    ) -> ObservationSignal:
        reset = getattr(self.executor, "reset", None)
        if reset is not None:
            reset()
        self._terminate_process()
        session_key = session_id or f"{initial_snapshot_id}-{seed}"
        digest = hashlib.sha256(session_key.encode()).hexdigest()[:20]
        overlay = self.config.runtime_root / f"session-{digest}.qcow2"
        qmp_socket = self.config.runtime_root / f"session-{digest}.qmp.sock"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        for stale in (overlay, qmp_socket):
            if stale.exists():
                stale.unlink()
        try:
            subprocess.run(
                [
                    self.config.qemu_img_binary,
                    "create",
                    "-f",
                    "qcow2",
                    "-F",
                    "qcow2",
                    "-b",
                    str(self.config.base_image),
                    str(overlay),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot create QEMU overlay: {error}") from error
        command = [
            self.config.qemu_binary,
            "-accel",
            "kvm" if self.config.kvm else "tcg",
            "-m",
            str(self.config.memory_mb),
            "-smp",
            str(self.config.cpus),
            "-drive",
            f"file={overlay},if=virtio,format=qcow2",
            "-display",
            "none",
            "-nodefaults",
            "-qmp",
            f"unix:{qmp_socket},server=on,wait=off",
            *self.config.extra_args,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"cannot start QEMU: {error}") from error
        self.initial_snapshot_id = initial_snapshot_id
        self.session_id = session_key
        self.overlay = overlay
        self.qmp_socket = qmp_socket
        self._last_unit = -1
        try:
            self._wait_ready()
        except Exception:
            self._terminate_process()
            raise
        return self.observation_factory.capture(self.session_id, 0)

    def apply(self, output: ActuationSignal) -> tuple[ObservationSignal, EnvironmentReceipt]:
        if self.process is None or self.initial_snapshot_id is None:
            raise RuntimeError("QEMU session is not active")
        if self.process.poll() is not None:
            raise RuntimeError("QEMU process exited before control application")
        if output.unit_index != self._last_unit + 1:
            raise ValueError("control unit index is out of order")
        started = time.perf_counter()
        receipt = self.executor.apply(output)
        if receipt.session_id != output.session_id or receipt.unit_index != output.unit_index:
            raise RuntimeError("actuator returned an invalid receipt identity")
        observation = self.observation_factory.capture(
            self.session_id or output.session_id, output.unit_index + 1
        )
        self._last_unit = output.unit_index
        return observation, EnvironmentReceipt(
            session_id=output.session_id,
            unit_index=output.unit_index,
            accepted=receipt.accepted,
            execution_latency_ms=(time.perf_counter() - started) * 1000,
            safety_violation=receipt.safety_violation,
            terminated=False,
            infrastructure_failure=receipt.infrastructure_failure,
        )

    def close(self) -> None:
        reset = getattr(self.executor, "reset", None)
        if reset is not None:
            reset()
        self._terminate_process()
        if hasattr(self.observation_factory, "close"):
            self.observation_factory.close()
        if hasattr(self.executor, "close"):
            self.executor.close()

    def _terminate_process(self) -> None:
        process, self.process = self.process, None
        qmp_socket, self.qmp_socket = self.qmp_socket, None
        overlay, self.overlay = self.overlay, None
        self.initial_snapshot_id = None
        self.session_id = None
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if qmp_socket is not None:
            try:
                qmp_socket.unlink()
            except FileNotFoundError:
                pass
        if overlay is not None and self.config.remove_overlay_on_close:
            try:
                overlay.unlink()
            except FileNotFoundError:
                pass

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.readiness_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is None:
                raise RuntimeError("QEMU process is not running")
            if self.process.poll() is not None:
                diagnostics = self.process.stderr.read().decode(errors="replace")[-4000:]
                raise RuntimeError(f"QEMU exited during readiness: {diagnostics}")
            try:
                status = self._qmp_command("query-status")
                if status.get("status") in {"running", "inmigrate", "postmigrate"}:
                    return
                last_error = RuntimeError(f"QEMU status is not running: {status}")
            except (OSError, TimeoutError, ValueError, RuntimeError) as error:
                last_error = error
            time.sleep(0.05)
        raise RuntimeError(f"QEMU readiness timed out: {last_error}")

    def _qmp_command(self, command: str) -> dict[str, Any]:
        if self.qmp_socket is None:
            raise RuntimeError("QMP socket is not configured")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.config.qmp_timeout_s)
            connection.connect(str(self.qmp_socket))
            greeting = self._qmp_read(connection)
            if "QMP" not in greeting:
                raise RuntimeError("invalid QMP greeting")
            connection.sendall((json.dumps({"execute": "qmp_capabilities"}) + "\r\n").encode())
            capabilities = self._qmp_read(connection)
            if "error" in capabilities:
                raise RuntimeError(str(capabilities["error"]))
            connection.sendall((json.dumps({"execute": command}) + "\r\n").encode())
            response = self._qmp_read(connection)
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            return response.get("return", {})

    @staticmethod
    def _qmp_read(connection: socket.socket) -> dict[str, Any]:
        data = bytearray()
        while b"\r\n" not in data and len(data) < 1_048_576:
            block = connection.recv(64 * 1024)
            if not block:
                raise ConnectionError("QMP closed the connection")
            data.extend(block)
        line = bytes(data).split(b"\r\n", 1)[0]
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("QMP response must be an object")
        return value
