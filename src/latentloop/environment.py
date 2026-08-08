from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    environment_id: str
    version: str
    protocol_version: str
    action_vocabulary_id: str

    def __post_init__(self) -> None:
        if not all((self.environment_id, self.version, self.protocol_version)):
            raise ValueError("environment identity fields are required")
        if not self.action_vocabulary_id:
            raise ValueError("action vocabulary identity is required")


@dataclass(slots=True)
class Observation:
    timestamp_ms: int
    delta_ms: int
    mixed_microphone: Tensor
    screen: Tensor
    screen_valid: bool
    screen_revision: int
    terminated: bool


@dataclass(frozen=True, slots=True)
class EnvironmentReceipt:
    task_id: str
    unit_index: int
    accepted: bool
    safety_violation: str | None = None


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    task: float
    speech_quality: float
    latency_quality: float
    action_efficiency: float
    safety: float
    spec_id: str = "realtime-v1"

    @property
    def interaction(self) -> float:
        return 0.4 * self.speech_quality + 0.3 * self.latency_quality + 0.3 * self.action_efficiency

    @property
    def total(self) -> float:
        return self.task + 0.2 * self.interaction + self.safety


class IsolatedComputerEnvironment(Protocol):
    def identity(self) -> EnvironmentIdentity: ...
    def reset(self, task_id: str, seed: int) -> Observation: ...
    def submit_unit(
        self,
        task_id: str,
        unit_index: int,
        speech_mode: int,
        speech_codes: Tensor,
        action_tokens: Tensor,
    ) -> tuple[Observation, EnvironmentReceipt]: ...
    def evaluate(self, task_id: str) -> RewardBreakdown: ...
    def close(self) -> None: ...


class UnixSocketEnvironmentClient:
    """Versioned JSON-line client for the external isolated desktop harness."""

    def __init__(
        self,
        socket_path: str,
        expected_identity: EnvironmentIdentity,
        timeout: float = 30.0,
    ) -> None:
        self.socket_path = socket_path
        self.expected_identity = expected_identity
        self.timeout = timeout
        self._identity: EnvironmentIdentity | None = None

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(self.socket_path)
                connection.sendall(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
                data = bytearray()
                while not data.endswith(b"\n"):
                    block = connection.recv(1024 * 1024)
                    if not block:
                        raise ConnectionError("environment closed the connection")
                    data.extend(block)
        except (OSError, TimeoutError, ConnectionError) as error:
            raise RuntimeError(f"environment request failed: {error}") from error
        response = json.loads(bytes(data))
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "environment request failed")))
        return response

    @staticmethod
    def _identity(payload: dict[str, object]) -> EnvironmentIdentity:
        value = payload.get("identity")
        if not isinstance(value, dict):
            raise ValueError("environment identity response is missing")
        return EnvironmentIdentity(
            str(value["environment_id"]),
            str(value["version"]),
            str(value["protocol_version"]),
            str(value["action_vocabulary_id"]),
        )

    @staticmethod
    def _observation(payload: dict[str, object]) -> Observation:
        def tensor(value: object) -> Tensor:
            if isinstance(value, list):
                return torch.tensor(value)
            if isinstance(value, str):
                return torch.tensor(json.loads(base64.b64decode(value)))
            raise ValueError("environment tensor field is invalid")

        return Observation(
            timestamp_ms=int(payload["timestamp_ms"]),
            delta_ms=int(payload["delta_ms"]),
            mixed_microphone=tensor(payload["mixed_microphone"]).float(),
            screen=tensor(payload["screen"]).float(),
            screen_valid=bool(payload["screen_valid"]),
            screen_revision=int(payload["screen_revision"]),
            terminated=bool(payload.get("terminated", False)),
        )

    def identity(self) -> EnvironmentIdentity:
        if self._identity is None:
            response = self._request({"operation": "identity"})
            identity = self._identity(response)
            if identity != self.expected_identity:
                raise ValueError("environment identity does not match configuration")
            self._identity = identity
        return self._identity

    def reset(self, task_id: str, seed: int) -> Observation:
        self.identity()
        return self._observation(
            self._request({"operation": "reset", "task_id": task_id, "seed": seed})
        )

    def submit_unit(
        self,
        task_id: str,
        unit_index: int,
        speech_mode: int,
        speech_codes: Tensor,
        action_tokens: Tensor,
    ) -> tuple[Observation, EnvironmentReceipt]:
        response = self._request(
            {
                "operation": "submit_unit",
                "task_id": task_id,
                "unit_index": unit_index,
                "speech_mode": speech_mode,
                "speech_codes": speech_codes.detach().cpu().tolist(),
                "action_tokens": action_tokens.detach().cpu().tolist(),
            }
        )
        receipt = response.get("receipt", {})
        if not isinstance(receipt, dict):
            raise ValueError("environment receipt is missing")
        return self._observation(response), EnvironmentReceipt(
            task_id=str(receipt.get("task_id", task_id)),
            unit_index=int(receipt.get("unit_index", unit_index)),
            accepted=bool(receipt.get("accepted", False)),
            safety_violation=receipt.get("safety_violation"),
        )

    def evaluate(self, task_id: str) -> RewardBreakdown:
        response = self._request({"operation": "evaluate", "task_id": task_id})
        value = response.get("reward")
        if not isinstance(value, dict):
            raise ValueError("environment reward is missing")
        return RewardBreakdown(
            task=float(value["task"]),
            speech_quality=float(value["speech_quality"]),
            latency_quality=float(value["latency_quality"]),
            action_efficiency=float(value["action_efficiency"]),
            safety=float(value["safety"]),
            spec_id=str(value.get("spec_id", "realtime-v1")),
        )

    def close(self) -> None:
        try:
            self._request({"operation": "close"})
        except RuntimeError:
            pass


def validate_observation(observation: Observation, *, audio_samples: int) -> None:
    if observation.delta_ms <= 0:
        raise ValueError("environment observation delta_ms must be positive")
    if observation.mixed_microphone.shape != (audio_samples,):
        raise ValueError("environment microphone has wrong shape")
    if observation.screen.ndim != 3 or observation.screen.shape[0] != 3:
        raise ValueError("environment screen must have shape [3,H,W]")


def validate_identity(
    identity: EnvironmentIdentity, expected_id: str, expected_action_vocab: str
) -> None:
    if identity.environment_id != expected_id:
        raise ValueError("environment identity does not match configuration")
    if identity.action_vocabulary_id != expected_action_vocab:
        raise ValueError("environment action vocabulary does not match configuration")
