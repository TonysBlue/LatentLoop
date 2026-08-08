from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentReceipt:
    session_id: str
    unit_index: int
    accepted: bool
    execution_latency_ms: float = 0.0
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
