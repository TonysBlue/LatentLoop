from __future__ import annotations

from typing import Protocol

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, RewardBreakdown


class ComputerBackend(Protocol):
    environment_id: str
    environment_version: str

    def reset(
        self, task_id: str, seed: int, session_id: str | None = None
    ) -> ObservationSignal: ...
    def apply(
        self, output: ActuationSignal, *, current_revision: int | None = None
    ) -> tuple[ObservationSignal, EnvironmentReceipt]: ...
    def evaluate(self, task_id: str) -> RewardBreakdown: ...
    def close(self) -> None: ...
