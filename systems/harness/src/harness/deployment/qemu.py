"""Real QEMU/SPICE deployment for the physical Harness boundary.

The deployment is deliberately endpoint based.  Device integration is kept
outside the training/model processes and is supplied by the host deployment:
screen and audio endpoints speak a small JSON-lines protocol, while input is
sent through QMP.  Missing endpoints fail at construction; there are no silent
or in-memory fallbacks in this module.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contracts import (
    ActuationSignal,
    ControlKind,
    ControlSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    RewardBreakdown,
    ScreenSignal,
)

from harness.environment.adapters import ActuatorAdapter, EvaluatorAdapter, SensorAdapter
from harness.environment.qemu import QemuBackend, QemuConfig


def _required(config: Mapping[str, Any], key: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"production Harness config requires {key}")
    return value


def _command_args(value: str) -> list[str]:
    # A deployment command is intentionally an argv list in YAML.  Accepting
    # a shell string would make quoting and privilege boundaries ambiguous.
    try:
        args = json.loads(value)
    except json.JSONDecodeError:
        args = [value]
    if not isinstance(args, list) or not args or not all(isinstance(item, str) for item in args):
        raise ValueError("device command must be a non-empty JSON argv list")
    return [os.path.expanduser(os.path.expandvars(item)) for item in args]


class JsonLineDevice:
    """Invoke a host/SPICE device bridge using one JSON request/response."""

    def __init__(self, command: str, *, name: str, timeout_s: float = 5.0) -> None:
        self.command = _command_args(command)
        self.name = name
        self.timeout_s = timeout_s
        self._closed = False

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"{self.name} device is closed")
        try:
            completed = subprocess.run(
                self.command,
                input=(json.dumps(payload, separators=(",", ":")) + "\n").encode(),
                capture_output=True,
                timeout=self.timeout_s,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(f"{self.name} device request failed: {error}") from error
        try:
            value = json.loads(completed.stdout.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{self.name} device returned invalid JSON") from error
        if not isinstance(value, dict) or value.get("ok", True) is False:
            raise RuntimeError(f"{self.name} device returned an error: {value}")
        return value

    def health(self) -> None:
        self.request({"operation": "health"})

    def close(self) -> None:
        self._closed = True


class SpiceScreenCapture:
    """Captures RGB frames from the configured SPICE display bridge."""

    def __init__(self, command: str, *, width: int, height: int, timeout_s: float = 5.0) -> None:
        self.device = JsonLineDevice(command, name="SPICE screen", timeout_s=timeout_s)
        self.width = width
        self.height = height

    def health(self) -> None:
        self.device.health()

    def capture(self, session_id: str, unit_index: int) -> tuple[int, int, bytes]:
        value = self.device.request(
            {"operation": "capture", "session_id": session_id, "unit_index": unit_index}
        )
        image = base64.b64decode(str(value.get("image_b64", "")), validate=True)
        width = int(value.get("width", self.width))
        height = int(value.get("height", self.height))
        if width != self.width or height != self.height:
            raise RuntimeError("SPICE screen dimensions differ from deployment config")
        expected = width * height * 3
        if len(image) != expected:
            raise RuntimeError("SPICE screen bridge did not return an RGB frame")
        return int(value["timestamp_ms"]), int(value.get("delta_ms", 80)), image

    def close(self) -> None:
        self.device.close()


class SpiceAudioCapture(SpiceScreenCapture):
    """Named audio/display bridge variant for deployments with separate audio."""

    def capture_pcm(self, session_id: str, unit_index: int) -> bytes:
        value = self.device.request(
            {"operation": "capture", "session_id": session_id, "unit_index": unit_index}
        )
        try:
            pcm = base64.b64decode(str(value["pcm_f32le_b64"]), validate=True)
        except (KeyError, ValueError, TypeError) as error:
            raise RuntimeError("SPICE audio bridge did not return PCM") from error
        return pcm


class SpicePhysicalSensor(SensorAdapter):
    def __init__(self, screen: SpiceScreenCapture, audio: SpiceAudioCapture) -> None:
        self.screen = screen
        self.audio = audio

    def health(self) -> None:
        self.screen.health()
        self.audio.health()

    def capture(self, session_id: str, unit_index: int) -> ObservationSignal:
        timestamp_ms, delta_ms, image = self.screen.capture(session_id, unit_index)
        return ObservationSignal(
            session_id=session_id,
            unit_index=unit_index,
            timestamp_ms=timestamp_ms,
            delta_ms=delta_ms,
            mic=MicSignal(self.audio.capture_pcm(session_id, unit_index)),
            screen=ScreenSignal(image, self.screen.width, self.screen.height),
        )

    def close(self) -> None:
        self.screen.close()
        self.audio.close()


class SpiceAudioActuator(ActuatorAdapter):
    """Plays speech through the SPICE/virtual audio playback bridge."""

    def __init__(self, command: str, *, timeout_s: float = 5.0) -> None:
        self.device = JsonLineDevice(command, name="SPICE audio playback", timeout_s=timeout_s)

    def health(self) -> None:
        self.device.health()

    def apply(self, output: ActuationSignal) -> EnvironmentReceipt:
        value = self.device.request(
            {
                "operation": "play",
                "session_id": output.session_id,
                "unit_index": output.unit_index,
                "pcm_b64": base64.b64encode(output.speech.pcm).decode(),
                "silent": output.speech.silent,
            }
        )
        if value.get("accepted", True) is not True:
            return EnvironmentReceipt(
                output.session_id,
                output.unit_index,
                accepted=False,
                infrastructure_failure=str(value.get("error", "audio playback rejected")),
            )
        return EnvironmentReceipt(output.session_id, output.unit_index, accepted=True)

    def close(self) -> None:
        self.device.close()


class QmpInputInjector:
    """Executes validated controls using QMP input-send-event."""

    def __init__(self, qmp_socket_getter, *, timeout_s: float = 2.0) -> None:
        self._qmp_socket_getter = qmp_socket_getter
        self.timeout_s = timeout_s
        self._held_buttons: set[int] = set()
        self._held_keys: set[int] = set()

    def apply(self, output: ActuationSignal) -> EnvironmentReceipt:
        started = time.perf_counter()
        try:
            for control in output.controls:
                if control.kind is ControlKind.NOOP:
                    continue
                self._send_control(control)
        except (OSError, RuntimeError, ValueError) as error:
            return EnvironmentReceipt(
                output.session_id,
                output.unit_index,
                accepted=False,
                execution_latency_ms=(time.perf_counter() - started) * 1000,
                infrastructure_failure=f"QMP input failed: {error}",
            )
        return EnvironmentReceipt(
            output.session_id,
            output.unit_index,
            accepted=True,
            execution_latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _send_control(self, control: ControlSignal) -> None:
        events: list[dict[str, Any]] = []
        if control.kind is ControlKind.POINTER_MOVE:
            assert control.x is not None and control.y is not None
            events.extend(
                [
                    {"type": "abs", "axis": "x", "value": round(control.x * 32767)},
                    {"type": "abs", "axis": "y", "value": round(control.y * 32767)},
                ]
            )
        elif control.kind is ControlKind.POINTER_BUTTON:
            if control.button_phase is None:
                raise ValueError("pointer button has no phase")
            if control.button_phase.value == "click":
                if int(control.button or 0) in self._held_buttons:
                    raise ValueError("held pointer button requires an UP control")
                events.append({"type": "btn", "button": int(control.button or 0), "down": True})
                events.append({"type": "btn", "button": int(control.button or 0), "down": False})
            else:
                button = int(control.button or 0)
                down = control.button_phase.value == "down"
                if down and button in self._held_buttons:
                    raise ValueError("pointer button is already held")
                if not down and button not in self._held_buttons:
                    raise ValueError("pointer button is not held")
                events.append(
                    {
                        "type": "btn",
                        "button": button,
                        "down": down,
                    }
                )
        elif control.kind is ControlKind.SCROLL:
            events.append({"type": "rel", "axis": "x", "value": round(control.dx or 0)})
            events.append({"type": "rel", "axis": "y", "value": round(control.dy or 0)})
        elif control.kind in {ControlKind.KEY_PRESS, ControlKind.KEY_RELEASE}:
            if control.key is None:
                raise ValueError("key control has no key")
            down = control.kind is ControlKind.KEY_PRESS
            if down and control.key in self._held_keys:
                raise ValueError("key is already held")
            if not down and control.key not in self._held_keys:
                raise ValueError("key is not held")
            events.append(
                {
                    "type": "key",
                    "key": int(control.key),
                    "down": down,
                }
            )
        elif control.kind is ControlKind.TEXT_INPUT:
            events.append({"type": "text", "text": str(control.text or "")})
        for event in events:
            self._qmp({"execute": "input-send-event", "arguments": {"events": [event]}})
            if event["type"] == "btn" and control.button_phase.value != "click":
                (self._held_buttons.add if event["down"] else self._held_buttons.discard)(
                    int(event["button"])
                )
            elif event["type"] == "key":
                (self._held_keys.add if event["down"] else self._held_keys.discard)(
                    int(event["key"])
                )

    def _qmp(self, request: dict[str, Any]) -> None:
        path = self._qmp_socket_getter()
        if not path:
            raise RuntimeError("QMP socket is unavailable")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_s)
            connection.connect(str(path))
            _read_qmp(connection)
            connection.sendall((json.dumps({"execute": "qmp_capabilities"}) + "\r\n").encode())
            _read_qmp(connection)
            connection.sendall((json.dumps(request) + "\r\n").encode())
            result = _read_qmp(connection)
            if "error" in result:
                raise RuntimeError(str(result["error"]))

    def close(self) -> None:
        self.reset()

    def reset(self) -> None:
        events = [
            {"type": "key", "key": key, "down": False}
            for key in sorted(self._held_keys, reverse=True)
        ]
        events.extend(
            {"type": "btn", "button": button, "down": False}
            for button in sorted(self._held_buttons)
        )
        try:
            for event in events:
                self._qmp({"execute": "input-send-event", "arguments": {"events": [event]}})
        finally:
            self._held_keys.clear()
            self._held_buttons.clear()


def _read_qmp(connection: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while b"\r\n" not in data:
        block = connection.recv(65536)
        if not block:
            raise ConnectionError("QMP closed connection")
        data.extend(block)
    value = json.loads(bytes(data).split(b"\r\n", 1)[0])
    if not isinstance(value, dict):
        raise ValueError("QMP response must be an object")
    return value


class TaskEvaluator(EvaluatorAdapter):
    """External evaluator adapter; task success is never fabricated locally."""

    def __init__(self, command: str, *, timeout_s: float = 30.0) -> None:
        self.device = JsonLineDevice(command, name="task evaluator", timeout_s=timeout_s)

    def health(self) -> None:
        self.device.health()

    def evaluate(self, task_id: str) -> RewardBreakdown:
        value = self.device.request({"operation": "evaluate", "task_id": task_id})
        required = ("task", "speech_quality", "latency_quality", "action_efficiency", "safety")
        if any(key not in value for key in required):
            raise RuntimeError("task evaluator response is missing reward fields")
        return RewardBreakdown(
            *(float(value[key]) for key in required),
            spec_id=str(value.get("spec_id", "realtime-v2")),
        )

    def terminated(self, task_id: str) -> bool:
        value = self.device.request({"operation": "terminated", "task_id": task_id})
        return bool(value.get("terminated", False))

    def close(self) -> None:
        self.device.close()


class _CompositeActuator(ActuatorAdapter):
    def __init__(self, playback: SpiceAudioActuator, input_injector: QmpInputInjector) -> None:
        self.playback = playback
        self.input = input_injector

    def apply(self, output: ActuationSignal) -> EnvironmentReceipt:
        speech_receipt = self.playback.apply(output)
        if speech_receipt.infrastructure_failure:
            return speech_receipt
        input_receipt = self.input.apply(output)
        if input_receipt.infrastructure_failure:
            return input_receipt
        return EnvironmentReceipt(
            output.session_id,
            output.unit_index,
            accepted=True,
            execution_latency_ms=(
                speech_receipt.execution_latency_ms + input_receipt.execution_latency_ms
            ),
        )

    def health(self) -> None:
        self.playback.health()

    def reset(self) -> None:
        self.input.reset()

    def close(self) -> None:
        self.input.close()
        self.playback.close()


class ProductionQemuBackend(QemuBackend):
    """QEMU backend with real SPICE/audio/input/evaluator deployment adapters."""

    def __init__(self, config: QemuConfig, screen, actuator, evaluator) -> None:
        super().__init__(config, screen, actuator, evaluator)
        self._evaluator = evaluator

    def health(self) -> None:
        for adapter in (self.observation_factory, self.executor, self._evaluator):
            check = getattr(adapter, "health", None)
            if check is None:
                raise RuntimeError(
                    f"deployment adapter {type(adapter).__name__} has no health check"
                )
            check()

    def reset(self, task_id: str, seed: int, session_id: str | None = None) -> ObservationSignal:
        observation = super().reset(task_id, seed, session_id)
        # QMP input is resolved only after reset created the per-session socket.
        self.health()
        return observation


def create_backend(config: Mapping[str, Any]) -> ProductionQemuBackend:
    """Build the only formal Harness backend from a resolved YAML mapping."""
    base_image = Path(_required(config, "base_image")).expanduser()
    runtime_root = Path(_required(config, "runtime_root")).expanduser()
    width = int(config.get("screen_width", 224))
    height = int(config.get("screen_height", 224))
    screen = SpiceScreenCapture(
        _required(config, "spice_screen_command"), width=width, height=height
    )
    audio = SpiceAudioCapture(
        _required(config, "spice_audio_capture_command"), width=width, height=height
    )
    playback = SpiceAudioActuator(_required(config, "spice_audio_playback_command"))
    evaluator = TaskEvaluator(_required(config, "evaluator_command"))
    spice_socket = runtime_root / "spice.sock"
    qemu_extra_args = tuple(str(arg) for arg in config.get("qemu_extra_args", ()))
    if not any("spice" in argument for argument in qemu_extra_args):
        qemu_extra_args += ("-spice", f"unix=on,addr={spice_socket},disable-ticketing=on")
    qemu_config = QemuConfig(
        base_image=base_image,
        runtime_root=runtime_root,
        qemu_binary=str(config.get("qemu_binary", "qemu-system-x86_64")),
        qemu_img_binary=str(config.get("qemu_img_binary", "qemu-img")),
        kvm=bool(config.get("kvm", True)),
        memory_mb=int(config.get("memory_mb", 4096)),
        cpus=int(config.get("cpus", 2)),
        readiness_timeout_s=float(config.get("readiness_timeout_s", 30.0)),
        qmp_timeout_s=float(config.get("qmp_timeout_s", 2.0)),
        extra_args=qemu_extra_args,
    )
    backend: ProductionQemuBackend
    # The injector resolves QMP lazily, after QemuBackend.reset has created the socket.
    injector = QmpInputInjector(lambda: backend.qmp_socket)
    actuator = _CompositeActuator(playback, injector)
    backend = ProductionQemuBackend(
        qemu_config, SpicePhysicalSensor(screen, audio), actuator, evaluator
    )
    backend.health()
    return backend


__all__ = [
    "ProductionQemuBackend",
    "QmpInputInjector",
    "SpiceAudioActuator",
    "SpiceScreenCapture",
    "TaskEvaluator",
    "create_backend",
]
