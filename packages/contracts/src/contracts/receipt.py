from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentReceipt:
    session_id: str
    unit_index: int
    accepted: bool
    execution_latency_ms: float = 0.0
    safety_violation: str | None = None
    terminated: bool = False
    infrastructure_failure: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or self.unit_index < 0:
            raise ValueError("invalid receipt identity")
        if self.execution_latency_ms < 0:
            raise ValueError("execution latency must be non-negative")
        if self.accepted and (self.safety_violation or self.infrastructure_failure):
            raise ValueError("an accepted receipt cannot report a failure")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    task: float
    speech_quality: float
    latency_quality: float
    action_efficiency: float
    safety: float
    spec_id: str = "realtime-v2"

    def __post_init__(self) -> None:
        if not self.spec_id:
            raise ValueError("reward spec_id is required")

    @property
    def interaction(self) -> float:
        return 0.4 * self.speech_quality + 0.3 * self.latency_quality + 0.3 * self.action_efficiency

    @property
    def total(self) -> float:
        return self.task + 0.2 * self.interaction + self.safety
