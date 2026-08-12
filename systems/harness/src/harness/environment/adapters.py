"""Deployment adapters for the physical QEMU/KVM Harness boundary."""

from __future__ import annotations

from typing import Protocol

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, RewardBreakdown


class SensorAdapter(Protocol):
    """Captures only physical microphone/screen/time signals."""

    def capture(self, session_id: str, unit_index: int) -> ObservationSignal: ...

    def close(self) -> None: ...


class ActuatorAdapter(Protocol):
    """Plays speech and executes already-decoded, validated controls."""

    def apply(self, output: ActuationSignal) -> EnvironmentReceipt: ...

    def close(self) -> None: ...


class EvaluatorAdapter(Protocol):
    def evaluate(self, task_id: str) -> RewardBreakdown: ...

    def terminated(self, task_id: str) -> bool: ...
