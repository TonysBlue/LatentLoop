from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, RewardBreakdown
from harness.environment.adapters import ActuatorAdapter, EvaluatorAdapter, SensorAdapter
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
        evaluator: EvaluatorAdapter,
    ) -> None:
        if not config.base_image.is_file():
            raise FileNotFoundError(f"QEMU base image is absent: {config.base_image}")
        self.config = config
        self.observation_factory = observation_factory
        self.executor = executor
        self.evaluator = evaluator
        self.process: subprocess.Popen[bytes] | None = None
        self.task_id: str | None = None
        self.session_id: str | None = None
        self._last_unit = -1

    def reset(
        self, task_id: str, seed: int, session_id: str | None = None
    ) -> ObservationSignal:
        self._terminate_process()
        overlay = self.config.runtime_root / f"{task_id}-{seed}.qcow2"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        if overlay.exists():
            overlay.unlink()
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
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"cannot start QEMU: {error}") from error
        self.task_id = task_id
        self.session_id = session_id or f"{task_id}-{seed}"
        self._last_unit = -1
        time.sleep(0.01)
        return self.observation_factory.capture(self.session_id, 0)

    def apply(
        self, output: ActuationSignal, *, current_revision: int | None = None
    ) -> tuple[ObservationSignal, EnvironmentReceipt]:
        if self.process is None or self.task_id is None:
            raise RuntimeError("QEMU session is not active")
        if output.unit_index != self._last_unit + 1:
            raise ValueError("control unit index is out of order")
        started = time.perf_counter()
        receipt = self.executor.apply(output, current_revision=current_revision or 0)
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
        )

    def evaluate(self, task_id: str) -> RewardBreakdown:
        if self.task_id != task_id or self.process is None:
            raise RuntimeError("QEMU session is not active")
        return self.evaluator.evaluate(task_id)

    def close(self) -> None:
        self._terminate_process()
        if hasattr(self.observation_factory, "close"):
            self.observation_factory.close()
        if hasattr(self.executor, "close"):
            self.executor.close()

    def _terminate_process(self) -> None:
        process, self.process = self.process, None
        self.task_id = None
        self.session_id = None
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
